"""Runnable scheduler/KV composition with an injected paged step runner.

This is the resident, single-device path. The executor adapter owns tickets and
completion state; an injected ``StepWorker`` owns the device context, streams,
runner and fences, and must cover every producer/consumer it enqueues. Dense
model runners that ignore memory_view do not satisfy the paged runner contract.
"""

from __future__ import annotations

from ayaka.configs.assembly import (
    build_execution_plan,
    build_parallel_plan,
)
from ayaka.configs.cache import CacheConfig
from ayaka.configs.memory import MemoryConfig, MemoryProfile
from ayaka.configs.model import ArchitectureConfig
from ayaka.configs.parallel import ParallelConfig
from ayaka.configs.scheduler import (
    PreemptionMode,
    ResolvedSchedulerPlan,
    SchedulerCapabilities,
    SchedulerConfig,
    SchedulingPolicy,
)
from ayaka.executor.base import Executor
from ayaka.executor.completion import CompletionCoordinator, ShutdownResult
from ayaka.executor.ticket import CompletionFence, ExecutionTicket
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.memory.workspace import WorkspaceManager
from ayaka.plan import EMPTY_MEMORY_PLAN
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.request.schema import Request
from ayaka.runner.buffers import RunnerBuffers
from ayaka.runner.sampling_runner import SampleRunner
from ayaka.runtime.engine import Engine
from ayaka.runtime.kv import (
    KVRequestPreparer,
    KVSequenceAllocator,
    KVStepRuntime,
    PrefixContextProvider,
)
from ayaka.runtime.output import OutputProcessor
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sched.factory import create_scheduler
from ayaka.types import DType
from ayaka.worker.base import StepWorker, WorkerStep
from ayaka.worker.local import LocalWorker
from ayaka.worker.resources import build_configured_resources


class ResidentKVExecutor(Executor):
    """Thin ticket adapter over a device worker.

    The executor owns tickets, submission state and retirement tracking; the
    worker owns device context, streams, runner invocation, COW ordering and
    fence creation. This class never imports a CUDA stream, an event or a
    tensor layout. The read-only shims keep existing callers on the worker's
    device objects without making the executor a device owner.
    """

    def __init__(self, worker: StepWorker, *, max_inflight: int = 1) -> None:
        super().__init__(max_inflight=max_inflight)
        self._worker = worker

    @property
    def worker(self) -> StepWorker:
        return self._worker

    @property
    def device(self):
        return self._worker.device

    @property
    def stream(self):
        return getattr(self._worker, "stream", None)

    @property
    def kv_stream(self):
        return getattr(self._worker, "kv_stream", None)

    @property
    def runner(self):
        return getattr(self._worker, "runner", None)

    @runner.setter
    def runner(self, value) -> None:
        self._worker.runner = value  # type: ignore[attr-defined]

    def initialize(self) -> None:
        self._worker.initialize()
        super().initialize()

    def has_submission_capacity(self) -> bool:
        return super().has_submission_capacity() and self._worker.accepting

    def shutdown(self) -> bool:
        # Stop worker admission first: a later retry must never enqueue.
        self._worker.request_closing()
        return super().shutdown()

    def _enqueue(self, ticket: ExecutionTicket) -> None:
        outcome = self._worker.execute(WorkerStep.from_ticket(ticket))
        self.set_samples(ticket, outcome.samples)
        self.track_fence(ticket, outcome.fence)

    def _begin_drain(self, ticket: ExecutionTicket) -> CompletionFence:
        return self._worker.drain(WorkerStep.from_ticket(ticket))

    def _close(self) -> None:
        if not self._worker.shutdown():
            raise RuntimeError("worker still owns active flights at close")
        self._worker.close()


class ResidentKVEngine(Engine):
    """Engine plus resident adapters with explicit borrowed or owned KV storage.

    close cancels all requests, settles tickets and aborts unadopted preparation.
    LogicalKVManager.close() drops prefix-cache ownership and its slab pins; the
    caller then closes borrowed storage leases. ``from_configs`` and serving
    inject a WorkerResources owner that closes its slabs and ledger itself.
    """

    def __init__(
        self,
        plan: ResolvedSchedulerPlan,
        kv: LogicalKVManager,
        runner: SampleRunner,
        *,
        kind: str = "continuous",
        output: OutputProcessor | None = None,
        sampling: SamplingCoordinator | None = None,
        prefix_context: PrefixContextProvider | None = None,
        workspace: WorkspaceManager | None = None,
        buffers: RunnerBuffers | None = None,
        worker: StepWorker | None = None,
    ) -> None:
        kind = kind.strip().lower().replace("-", "_")
        if (
            kind in ("eager", "reference", "debug")
            and plan.scheduling_policy is SchedulingPolicy.LONGEST_PREFIX_MATCH
        ):
            raise ValueError("LPM ranking requires the continuous scheduler")
        if plan.workspace != EMPTY_MEMORY_PLAN and workspace is None:
            raise ValueError(
                "resident engine requires a workspace manager for a non-empty workspace plan"
            )
        if (
            plan.scheduling_policy is SchedulingPolicy.LONGEST_PREFIX_MATCH
            and prefix_context is None
        ):
            raise ValueError("LPM requires a prefix context provider")
        if (
            kind in ("eager", "reference", "debug")
            and plan.preemption_mode is not PreemptionMode.NONE
        ):
            raise ValueError("recompute preemption requires the continuous scheduler")
        if prefix_context is not None and not plan.capabilities.prefix_cache:
            raise ValueError("prefix reuse requires declared runner support")
        requests = LifecycleManager()
        allocator = KVSequenceAllocator(kv)
        self.preparer = KVRequestPreparer(allocator, prefix_context=prefix_context)
        # The worker owns the device context, streams, runner and fences; the
        # engine only composes it and the executor adapter drives tickets.
        self.worker = worker or LocalWorker(kv, runner, max_inflight=plan.max_inflight)
        self.executor = ResidentKVExecutor(self.worker, max_inflight=plan.max_inflight)
        self.coordinator = CompletionCoordinator(self.executor, requests)
        self.runtime = KVStepRuntime(
            plan.execution,
            allocator,
            self.coordinator,
            requests=self.preparer,
            graph_planner=getattr(runner, "graph_pool", None),
            workspace=workspace,
            buffers=buffers,
        )
        output = output or OutputProcessor()
        scheduler = create_scheduler(
            kind,
            plan,
            requests,
            self.runtime,
            allocator,
            text_stops=output.text_stops_supported,
            sampling=sampling,
            request_preparer=self.preparer,
            prefix_hints=self.preparer,
            preemption=self.preparer if plan.preemption_mode is PreemptionMode.RECOMPUTE else None,
            allow_mixed_batches=bool(getattr(runner, "supports_mixed_batches", False)),
        )
        super().__init__(
            plan,
            requests=requests,
            scheduler=scheduler,
            coordinator=self.coordinator,
            output=output,
            allocator=allocator,
        )
        self.kv = kv
        self.allocator = allocator
        self.workspace = workspace
        self._closing = False
        self._runner_requests: set[str] = set()
        set_source = getattr(runner, "set_request_source", None)
        set_bans = getattr(runner, "set_bans", None)
        self._forget_runner_request = getattr(runner, "forget_request", lambda _: None)
        if set_source is not None:
            set_source(self.requests.get)
        if set_bans is not None:
            set_bans(self.sampling_masks)
        self.executor.initialize()

    @classmethod
    def from_configs(
        cls,
        architecture: ArchitectureConfig,
        cache_config: CacheConfig,
        *,
        runner: SampleRunner,
        device_total_bytes: int,
        plan_id: str,
        model_id: str,
        model_revision: str,
        weights_revision: str,
        scheduler_config: SchedulerConfig | None = None,
        memory_config: MemoryConfig | None = None,
        profile: MemoryProfile | None = None,
        parallel_config: ParallelConfig | None = None,
        compute_dtype: DType = DType.BF16,
        device: str = "cpu",
        device_index: int = 0,
        world_rank: int = 0,
        kind: str = "continuous",
        max_model_len: int | None = None,
        zero_initialize: bool = False,
        sampling: SamplingCoordinator | None = None,
        prefix_context: PrefixContextProvider | None = None,
    ) -> ResidentKVEngine:
        """Resolve model/cache/memory/scheduler/parallel configs into an engine.

        Owns the materialization order: plan -> ledger -> leases -> manager ->
        logical facade -> execution identity -> resolved scheduler plan. On any
        failure after materialization the logical manager and every lease are
        closed, so a caller never inherits an uncharged or half-owned slab.
        """
        resolved_scheduler = scheduler_config or SchedulerConfig()
        parallel = parallel_config or ParallelConfig()
        kv_plan, resources = build_configured_resources(
            architecture,
            cache_config,
            scheduler=resolved_scheduler,
            device_total_bytes=device_total_bytes,
            model_id=model_id,
            model_revision=model_revision,
            weights_revision=weights_revision,
            memory_config=memory_config,
            profile=profile,
            parallel=parallel,
            max_model_len=max_model_len,
            compute_dtype=compute_dtype,
            device=device,
            device_index=device_index,
            zero_initialize=zero_initialize,
        )
        kv, ledger = resources.kv, resources.ledger
        worker = LocalWorker(
            kv,
            runner,
            max_inflight=resolved_scheduler.max_inflight,
            resources=resources,
        )
        try:
            execution = build_execution_plan(
                architecture,
                cache_config,
                kv_plan,
                plan_id=plan_id,
                model_id=model_id,
                model_revision=model_revision,
                weights_revision=weights_revision,
                max_num_batched_tokens=resolved_scheduler.max_num_batched_tokens,
                enable_chunked_prefill=resolved_scheduler.enable_chunked_prefill,
                compute_dtype=compute_dtype,
                parallel=(
                    build_parallel_plan(kv_plan.parallel, world_rank)
                    if kv_plan.parallel is not None
                    else None
                ),
            )
            capabilities = SchedulerCapabilities(
                max_num_seqs=resolved_scheduler.max_num_seqs,
                chunked_prefill=resolved_scheduler.enable_chunked_prefill,
                prefix_cache=prefix_context is not None,
                recompute_preemption=(
                    resolved_scheduler.preemption_mode is PreemptionMode.RECOMPUTE
                ),
            )
            plan = resolved_scheduler.resolve(
                kv_plan.max_model_len,
                execution=execution,
                capabilities=capabilities,
                memory=ledger.snapshot(),
                workspace=EMPTY_MEMORY_PLAN,
            )

            return cls(
                plan,
                kv,
                runner,
                kind=kind,
                sampling=sampling,
                prefix_context=prefix_context,
                worker=worker,
            )
        except BaseException:
            try:
                worker.close()
            finally:
                resources.close()
            raise

    def submit(self, request: Request) -> RequestLifecycle:
        if self._closing:
            raise RuntimeError("resident engine is closing")
        prepare = getattr(self.executor.runner, "prepare_request", None)
        if prepare is not None:
            prepare(request)
        try:
            lifecycle = super().submit(request)
        except BaseException:
            if prepare is not None:
                self._forget_runner_request(str(request.request_id))
            raise
        if prepare is not None:
            self._runner_requests.add(str(request.request_id))
        return lifecycle

    def step(self) -> bool:
        if self._closing:
            raise RuntimeError("resident engine is closing; poll close() instead")
        did = super().step()
        for request_id in tuple(self._runner_requests):
            lifecycle = self.requests.find(request_id)
            if lifecycle is None or lifecycle.is_terminal:
                self._forget_runner_request(request_id)
                self._runner_requests.remove(request_id)
        return did

    def close(self) -> ShutdownResult:
        self._closing = True
        self.runtime.close()
        for lifecycle in self.requests:
            if not lifecycle.is_terminal:
                self.abort(lifecycle.request_id)
        result = super().close()
        for completed in result.completed:
            self.scheduler.update_from_output(completed)
        if result.closed:
            self.scheduler.flush_reports()
            if not self.kv.closed:
                self.kv.reclaim_deferred()
        return result
