"""Runnable scheduler/KV composition with an injected paged step runner.

This is the resident, single-device, single-flight path. The runner writes every
reserved KV slot and returns packed samples; dense model runners that ignore
memory_view do not satisfy this contract. CUDA work must use the current stream,
or join its side streams back to that stream before returning or raising.
"""

from __future__ import annotations

from contextlib import nullcontext, suppress

import torch

from ayaka.configs.assembly import (
    bind_resident_kv,
    build_execution_plan,
    build_parallel_plan,
    materialize_resident_kv,
    plan_resident_kv,
    resident_ledger,
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
from ayaka.executor.ticket import CompletionFence, ExecutionTicket, FenceResult, WorkState
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.memory.capacity import (
    MemoryLane,
    build_capacity_snapshot,
    mint_generation,
    reconcile_actual_usage,
)
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


class _CPUFence:
    def query(self) -> FenceResult:
        return FenceResult(WorkState.SUCCEEDED, quiescent=True)


class _CUDAFence:
    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.event = torch.cuda.Event()
        self.event.record(stream)

    def query(self) -> FenceResult:
        if self.event.query():
            return FenceResult(WorkState.SUCCEEDED, quiescent=True)
        return FenceResult(WorkState.PENDING)


class ResidentKVExecutor(Executor):
    """Order COW before the runner and fence all compute, including partial failure.

    Storage and runner inputs must be initialized before construction, or explicitly
    ordered onto this executor's stream. No global device synchronization occurs.
    The CPU runner must complete synchronously and return CPU samples.
    """

    def __init__(
        self, kv: LogicalKVManager, runner: SampleRunner, *, max_inflight: int = 1
    ) -> None:
        super().__init__(max_inflight=max_inflight)
        self.kv = kv
        self.runner = runner
        devices = {
            tensor.device
            for lease in kv.storages.values()
            for family in lease.storage.buffers()
            for tensor in family
        }
        if len(devices) != 1:
            raise ValueError("resident executor requires KV on exactly one device")
        self.device = next(iter(devices))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("resident executor supports only CPU or CUDA")
        self.stream = None
        self.kv_stream = None
        self._transfer_event = None
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(device=self.device)
            self.kv_stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def _enqueue(self, ticket: ExecutionTicket) -> None:
        context = nullcontext() if self.stream is None else torch.cuda.stream(self.stream)
        with context, torch.inference_mode():
            self._apply_cow(ticket)
            samples = self.runner(ticket.prepared)
            if samples.token_ids.device != self.device:
                raise ValueError("runner samples must reside on the KV device")
            self.set_samples(ticket, samples)
            self.track_fence(ticket, self._fence())

    def _copy_pages(self, ticket: ExecutionTicket) -> None:
        for copy in ticket.prepared.memory_view.copies:
            source = self.kv.physical_page(copy.group_name, copy.source)
            destination = self.kv.physical_page(copy.group_name, copy.destination)
            storage = self.kv.storages[copy.group_name].storage
            if not 0 < copy.valid_tokens <= storage.page_size:
                raise ValueError("invalid COW copy length")
            # Copy the physical representation bit-for-bit, including quantized planes.
            for family in storage.buffers():
                for tensor in family:
                    tensor[destination, : copy.valid_tokens].copy_(
                        tensor[source, : copy.valid_tokens]
                    )

    def _apply_cow(self, ticket: ExecutionTicket) -> None:
        """Run lease-owned KV copies on the transfer stream at the batch boundary."""
        if self.kv_stream is None or self.stream is None:
            self._copy_pages(ticket)
            return
        self.kv_stream.wait_stream(self.stream)
        try:
            with torch.cuda.stream(self.kv_stream):
                self._copy_pages(ticket)
                self._transfer_event = torch.cuda.Event()
                self._transfer_event.record(self.kv_stream)
            self.stream.wait_event(self._transfer_event)
        finally:
            # Cover partial-copy failure too, before the compute drain fence.
            self.stream.wait_stream(self.kv_stream)

    def _fence(self) -> CompletionFence:
        return _CPUFence() if self.stream is None else _CUDAFence(self.stream)

    def _begin_drain(self, ticket: ExecutionTicket) -> CompletionFence:
        # Join any partial transfer before proving compute/transfer quiescence.
        if self.stream is not None and self.kv_stream is not None:
            self.stream.wait_stream(self.kv_stream)
        return self._fence()

    def _close(self) -> None:
        self.runner.close()


class ResidentKVEngine(Engine):
    """Engine plus resident adapters; the caller retains ownership of KV slabs.

    close cancels all requests, settles tickets and aborts unadopted preparation.
    LogicalKVManager.close() drops prefix-cache ownership and its slab pins; the
    caller then closes its storage leases.
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
        self.executor = ResidentKVExecutor(kv, runner, max_inflight=plan.max_inflight)
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
        kv_plan = plan_resident_kv(
            architecture,
            cache_config,
            device_total_bytes=device_total_bytes,
            max_num_seqs=resolved_scheduler.max_num_seqs,
            memory_config=memory_config,
            profile=profile,
            parallel=parallel,
            max_model_len=max_model_len,
        )
        ledger = resident_ledger(
            kv_plan.memory,
            device=device,
            device_total_bytes=device_total_bytes,
            device_index=device_index,
        )
        leases = materialize_resident_kv(
            kv_plan, ledger=ledger, device=device, zero_initialize=zero_initialize
        )
        kv: LogicalKVManager | None = None
        try:
            _, kv = bind_resident_kv(kv_plan, leases)
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
            lane = MemoryLane.CPU if str(device).strip().lower() == "cpu" else MemoryLane.CUDA
            measured = profile
            generation = mint_generation(
                model_id=model_id,
                model_revision=model_revision,
                weights_revision=weights_revision,
                kv_storage=kv.fingerprint,
                backend=type(kv.backend).__name__,
            )
            snapshot = build_capacity_snapshot(
                generation=generation,
                lane=lane,
                dtype=compute_dtype.label,
                kv_dtype=cache_config.kv_dtype.label,
                page_size=kv_plan.storage_specs[0][1].page_size,
                group_pages={group.group_id: group.num_pages for group in kv_plan.physical.groups},
                max_model_len=kv_plan.max_model_len,
                max_num_seqs=resolved_scheduler.max_num_seqs,
                max_num_batched_tokens=resolved_scheduler.max_num_batched_tokens,
                max_inflight=resolved_scheduler.max_inflight,
                activation_bytes=measured.activation_bytes if measured else 0,
                workspace_ceiling_bytes=measured.kernel_workspace_bytes if measured else 0,
                graph_bytes=measured.graph_pool_bytes if measured else 0,
                staging_bytes=0,
                budget_bytes=kv_plan.memory.policy_budget_bytes,
                kv_budget_bytes=kv_plan.memory.kv_cache_bytes,
                weights_bytes=measured.weights_bytes if measured else 0,
                ledger=ledger,
            )
            kv.bind_capacity(snapshot)
            reconcile_actual_usage(ledger, snapshot)
            return cls(
                plan,
                kv,
                runner,
                kind=kind,
                sampling=sampling,
                prefix_context=prefix_context,
            )
        except BaseException:
            if kv is not None:
                with suppress(Exception):
                    kv.close()
            for lease in reversed(leases):
                with suppress(Exception):
                    lease.close()
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
            self.kv.reclaim_deferred()
        return result
