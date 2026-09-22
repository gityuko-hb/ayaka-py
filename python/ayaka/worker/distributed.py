"""Distributed StepWorker: one rank-local worker driven

`LocalStepHost` adapts :class:`~ayaka.worker.base.StepWorker` (the
rank-local device owner, typically `LocalWorker`) onto
:class:`~ayaka.distributed.execution.RankLocalStepHost`. `DistributedStepWorker`
implements the same `StepWorker` surface the resident executor already drives,
so an engine can swap its worker without changing ticket handling:

- `execute(step)` stages the rank's own step and runs the group protocol. The
  source rank broadcasts its envelope; every other rank cross-checks the
  staged local step against it (request incarnation, phases, query ranges,
  sampling shape) — physical pages and local generations stay rank-local and
  are never compared.
- The rank's own outcome is returned only after the global success proof and
  commit acknowledgements have landed, so the engine's local commit and
  external publication always happen after the group protocol completed.
- Device ownership stays with the local worker: streams, fences, drain proofs
  and lifecycle states delegate unchanged.
- Shutdown is coordinated, not local: every rank votes through the group
  agreement, the barrier orders the teardown, and only then does the facade
  close the coordinator's KV worker and the rank-local worker. A rank that
  still owns an active flight refuses (no resource is freed before
  communication last-use); a failed or wedged peer makes the healthy ranks
  raise the coordinator's fail-closed error instead of reporting a clean
  shutdown.

The rank-local KV/storage binding and any per-rank reservation hook stay
inside the local worker's resources (R09); the host adds no second ledger.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ayaka.configs.distributed import ResolvedDistributedPlan
from ayaka.distributed.completion import CompletionFence
from ayaka.distributed.execution import (
    DistributedControlPlane,
    DistributedStepCoordinator,
    RankLocalStepHost,
)
from ayaka.distributed.step_envelope import DistributedStepEnvelope
from ayaka.distributed.worker import RankLocalKVWorkerState
from ayaka.executor.ticket import CompletionFence as ExecutorCompletionFence
from ayaka.executor.ticket import FenceResult, WorkState
from ayaka.plan import ExecutionPlan
from ayaka.worker.base import StepWorker, WorkerOutcome, WorkerStep
from ayaka.worker.lifecycle import WorkerState

__all__ = [
    "DistributedStepWorker",
    "LocalStepHost",
    "QueryFenceWaitBridge",
    "build_distributed_worker",
    "execution_plan_identity",
]


def execution_plan_identity(execution: ExecutionPlan) -> dict[str, Any]:
    """Global plan identity the envelope must match on every rank.

    Dotted keys mirror the envelope's canonical payload; per-rank placement
    (``tp_rank``/device) is deliberately excluded because ranks legitimately
    disagree about it.
    """
    return {
        "execution_plan_id": execution.plan_id,
        "model_id": execution.model_id,
        "model_revision": execution.model_revision,
        "weights_revision": execution.weights_revision,
        "dtype": execution.compute.dtype.label,
        "kv_dtype": execution.compute.kv_dtype.label,
        "layer_start": int(execution.compute.layer_range[0]),
        "layer_stop": int(execution.compute.layer_range[1]),
        "parallel.tp_size": int(execution.parallel.tp_size),
        "parallel.pp_size": int(execution.parallel.pp_size),
        "parallel.dp_size": int(execution.parallel.dp_size),
    }


@dataclass(frozen=True, slots=True)
class QueryFenceWaitBridge:
    """Bridge a query-based executor fence into the wait-based protocol.

    The coordinator waits on a rank's completion proof with a deadline; R09
    executor fences answer ``query()`` non-blockingly. The bridge polls until
    the fence stops reporting pending, so a wedged rank turns into a timeout
    (fail-closed) rather than an unbounded wait. Query errors propagate —
    they prove neither completion nor safety.
    """

    fence: ExecutorCompletionFence
    poll_interval_s: float = 0.001

    def wait(self, timeout_s: float | None = None) -> bool:
        result = self.fence.query()
        if not isinstance(result, FenceResult):
            raise TypeError("fence query must return a FenceResult")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while result.state is WorkState.PENDING:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            time.sleep(min(self.poll_interval_s, remaining or self.poll_interval_s))
            result = self.fence.query()
        return result.state is WorkState.SUCCEEDED


def _wait_fence(fence: ExecutorCompletionFence) -> CompletionFence:
    """Adapt an executor fence to the coordinator's wait-based protocol."""
    if callable(getattr(fence, "wait", None)):
        return cast(CompletionFence, fence)
    return QueryFenceWaitBridge(fence)


class LocalStepHost(RankLocalStepHost):
    """Rank-local host: stage a ``WorkerStep``, validate it, execute locally."""

    def __init__(self, local: StepWorker, *, plan_identity: Mapping[str, Any]) -> None:
        self.local = local
        self._plan_identity = dict(plan_identity)
        self._staged: WorkerStep | None = None
        self.last_outcome: WorkerOutcome | None = None

    def plan_identity(self) -> Mapping[str, Any]:
        return dict(self._plan_identity)

    def stage(self, step: WorkerStep) -> None:
        self._staged = step

    def prepare(self, envelope: DistributedStepEnvelope) -> None:
        staged = self._require_staged()
        self._cross_check(envelope, staged)

    def rollback(self, envelope: DistributedStepEnvelope) -> None:
        self.clear()

    def clear(self) -> None:
        """Drop any staged step; safe between flights and after failures."""
        self._staged = None

    def execute(self, envelope: DistributedStepEnvelope) -> Any:
        staged = self._require_staged()
        outcome = self.local.execute(staged)
        self._staged = None
        self.last_outcome = outcome
        return outcome

    def fence(self, result: Any) -> CompletionFence:
        if not isinstance(result, WorkerOutcome):
            raise TypeError("distributed host execute must return a WorkerOutcome")
        return _wait_fence(result.fence)

    def commit(self, envelope: DistributedStepEnvelope, result: Any) -> None:
        """No-op at worker level: commit rights stay with the engine's ticket.

        The protocol guarantees this runs after the global success proof, so
        the engine's local commit and external publication already happen
        after every rank proved quiescence and the commit acks landed.
        """

    def publish(self, envelope: DistributedStepEnvelope, result: Any) -> None:
        """No-op at worker level: external publication belongs to the engine."""

    def fail(self, error: BaseException | str) -> None:
        fail_hook = getattr(self.local, "fail", None)
        if fail_hook is None:
            raise RuntimeError("rank-local worker does not expose fail() for fail-closed marking")
        fail_hook(error)

    def _require_staged(self) -> WorkerStep:
        staged = self._staged
        if staged is None:
            raise RuntimeError("rank has no staged worker step for this flight")
        return staged

    def _cross_check(self, envelope: DistributedStepEnvelope, staged: WorkerStep) -> None:
        """Reject any divergence between the broadcast plan and the local one."""
        payload = envelope.payload
        if staged.step_id != envelope.step_id:
            raise ValueError(
                f"staged step {staged.step_id} does not match the broadcast step {envelope.step_id}"
            )
        broadcast = payload["sequences"]
        slices = staged.prepared.step.slices
        inputs = staged.prepared.step.inputs
        if len(broadcast) != len(slices):
            raise ValueError(
                f"broadcast batch has {len(broadcast)} slices; the rank staged {len(slices)}"
            )
        for entry, scheduled, value in zip(broadcast, slices, inputs, strict=True):
            self._require(entry["request_id"] == scheduled.request_id, "request_id")
            self._require(
                int(entry["sequence_epoch"]) == scheduled.sequence_epoch, "sequence_epoch"
            )
            self._require(
                int(entry["state_version"]) == scheduled.expected_state_version,
                "state_version",
            )
            self._require(entry["phase"] == scheduled.phase.value, "phase")
            self._require(int(entry["query_start"]) == scheduled.query_start, "query_start")
            self._require(int(entry["query_count"]) == scheduled.query_count, "query_count")
            self._require(
                bool(entry["sample_last_query"]) == scheduled.sample_last_query,
                "sample_last_query",
            )
            del value
        requirements = payload["kv_requirements"]
        local_requirements = staged.prepared.step.kv_requirements
        if len(requirements) != len(local_requirements):
            raise ValueError("broadcast KV requirements disagree with the staged plan")
        for entry, requirement in zip(requirements, local_requirements, strict=True):
            if (
                int(entry["group_id"]) != requirement.group_id
                or int(entry["append_tokens"]) != requirement.append_tokens
                or int(entry["cow_pages"]) != requirement.cow_pages
                or int(entry["restore_bytes"]) != requirement.restore_bytes
                or int(entry["growth_bytes"]) != requirement.growth_bytes
            ):
                raise ValueError("broadcast KV requirements disagree with the staged plan")
            attributed = tuple(
                (str(request_id), int(tokens)) for request_id, tokens in entry["request_tokens"]
            )
            if attributed != tuple(
                (request_id, int(tokens)) for request_id, tokens in requirement.request_tokens
            ):
                raise ValueError("broadcast KV request attribution disagrees with the staged plan")

    @staticmethod
    def _require(condition: bool, field_name: str) -> None:
        if not condition:
            raise ValueError(f"staged plan disagrees with the broadcast step on {field_name}")


class DistributedStepWorker:
    """``StepWorker`` facade for one rank of a distributed step group.

    Every rank constructs its own facade over its local worker plus a shared
    control plane. ``execute`` runs the full group protocol; the returned
    outcome is this rank's own local samples and fence.
    """

    def __init__(
        self,
        control: DistributedControlPlane,
        local: StepWorker,
        *,
        plan_identity: Mapping[str, Any],
        allow_graph: bool = False,
        step_timeout_s: float | None = None,
        shutdown_timeout_s: float | None = None,
        publisher_rank: int = 0,
    ) -> None:
        if step_timeout_s is not None and not isinstance(step_timeout_s, int | float):
            raise TypeError("step_timeout_s must be a number or None")
        if shutdown_timeout_s is not None and not isinstance(shutdown_timeout_s, int | float):
            raise TypeError("shutdown_timeout_s must be a number or None")
        if shutdown_timeout_s is not None and shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        self.host = LocalStepHost(local, plan_identity=plan_identity)
        self.coordinator = DistributedStepCoordinator(
            control,
            self.host,
            publisher_rank=publisher_rank,
            allow_graph=allow_graph,
        )
        self.local = local
        self._step_timeout_s = step_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._group_shutdown_done = False
        self._closed = False

    # ------------------------------------------------------------------
    # StepWorker surface
    # ------------------------------------------------------------------

    @property
    def device(self):
        return self.local.device

    @property
    def state(self) -> WorkerState:
        if self.local.state is WorkerState.FAILED:
            return WorkerState.FAILED
        if self.coordinator.worker.state is RankLocalKVWorkerState.FAILED:
            return WorkerState.FAILED
        return self.local.state

    @property
    def accepting(self) -> bool:
        return self.local.accepting and self.coordinator.worker.state is not (
            RankLocalKVWorkerState.FAILED
        )

    @property
    def incarnation(self) -> int:
        return self.local.incarnation

    @property
    def generation(self) -> int:
        """The coordinator's worker incarnation (fresh after recover)."""
        return self.coordinator.generation

    @property
    def failure(self) -> str | None:
        coordinator_failure = self.coordinator.failure
        if coordinator_failure is not None:
            return coordinator_failure
        return getattr(self.local, "failure", None)

    @property
    def stream(self):
        return getattr(self.local, "stream", None)

    @property
    def kv_stream(self):
        return getattr(self.local, "kv_stream", None)

    @property
    def runner(self):
        return getattr(self.local, "runner", None)

    @runner.setter
    def runner(self, value) -> None:
        self.local.runner = value  # type: ignore[attr-defined]

    def initialize(self) -> None:
        self.local.initialize()

    def execute(self, step: WorkerStep) -> WorkerOutcome:
        self.host.stage(step)
        prepared = (
            step.prepared if self.coordinator.control.rank == self.coordinator.source_rank else None
        )
        try:
            return self.coordinator.run(prepared, timeout_s=self._step_timeout_s)
        finally:
            self.host.clear()

    def drain(self, step: WorkerStep) -> ExecutorCompletionFence:
        return self.local.drain(step)

    def request_closing(self) -> WorkerState:
        return self.local.request_closing()

    @property
    def closed(self) -> bool:
        """Whether a clean coordinated shutdown closed every rank-local owner."""
        return self._closed

    def shutdown(self, *, timeout_s: float | None = None) -> bool:
        """Stop admission and coordinate the group shutdown.

        Returns True only when this rank drained, every rank accepted the
        shutdown agreement, the final barrier completed, and both the
        coordinator's KV worker and the rank-local worker are closed. A rank
        that still owns an active flight returns False without entering the
        collective and without freeing anything: the peers fail closed on
        their deadline, exactly like the R09 "worker still owns active
        flights" contract. A failed incarnation or a wedged peer raises the
        coordinator's fail-closed error instead of reporting success.

        Safe to repeat: a completed shutdown is a no-op, and a rank that was
        already failed stays failed and observable through :attr:`failure`.
        """
        if self._closed:
            return True
        self.local.request_closing()
        if not self._local_drained():
            return False
        self._mark_local_failure()
        self._group_shutdown(self._resolve_shutdown_timeout(timeout_s))
        self._close_rank()
        return True

    def close(self) -> None:
        """Close rank-local resources after a coordinated group shutdown.

        This is the fail-closed teardown path: an active flight makes the
        rank-local worker refuse (``cannot close a worker with active flights``)
        and nothing is freed; a failed or wedged peer surfaces the
        coordinator's shutdown error before any resource is released. Repeat
        calls after a successful close are no-ops.
        """
        if self._closed:
            return
        self.local.request_closing()
        if not self._local_drained():
            # R09 contract: never free before the last communication use. The
            # local worker raises for an active flight; keep the refusal
            # fail-closed even if an implementation silently allows close.
            self.local.close()
            raise RuntimeError("cannot close a distributed worker with active flights")
        self._mark_local_failure()
        self._group_shutdown(self._shutdown_timeout_s)
        self._close_rank()

    def _resolve_shutdown_timeout(self, timeout_s: float | None) -> float | None:
        return self._shutdown_timeout_s if timeout_s is None else timeout_s

    def _local_drained(self) -> bool:
        """Whether the rank-local worker proved every flight quiescent."""
        num_flights = getattr(self.local, "num_flights", None)
        if num_flights is None:
            raise RuntimeError(
                "distributed shutdown requires rank-local flight accounting (num_flights)"
            )
        return int(num_flights) == 0

    def _mark_local_failure(self) -> None:
        """Propagate a rank-local failure into the coordinator's incarnation.

        A failed local worker must vote no in the shutdown agreement even when
        the coordinator never observed the failure itself: a clean shutdown
        cannot be reported while any rank-local owner is FAILED.
        """
        if self.local.state is not WorkerState.FAILED:
            return
        if self.coordinator.worker.state is RankLocalKVWorkerState.FAILED:
            return
        failure = getattr(self.local, "failure", None)
        self.coordinator.worker.fail(failure or "rank-local worker failed before shutdown")

    def _group_shutdown(self, timeout_s: float | None) -> None:
        """Run the group agreement/barrier once; raises fail-closed on refusal."""
        if self._group_shutdown_done:
            return
        self.coordinator.shutdown(timeout_s=timeout_s)
        self._group_shutdown_done = True

    def _close_rank(self) -> None:
        """Close the coordinator's KV worker, then the rank-local device owner."""
        self.host.clear()
        self.coordinator.worker.close()
        self.local.close()
        self._closed = True


def build_distributed_worker(
    plan: ResolvedDistributedPlan,
    local: StepWorker,
    *,
    plan_identity: Mapping[str, Any],
    allow_graph: bool = False,
    step_timeout_s: float | None = None,
    shutdown_timeout_s: float | None = None,
    publisher_rank: int = 0,
    process_group: DistributedControlPlane | None = None,
    group: Any | None = None,
) -> DistributedStepWorker:
    """Build the rank facade over a process group initialized from the plan.

    Lifecycle: call :meth:`DistributedStepWorker.shutdown`/:meth:`~DistributedStepWorker.close`
    on every rank to run the coordinated group shutdown, then
    :func:`ayaka.distributed.runtime.shutdown_distributed_process_group` on every
    rank for the final barrier and to destroy a group this process initialized.
    """
    if process_group is None:
        from ayaka.distributed.runtime import init_distributed_process_group

        process_group = init_distributed_process_group(plan, group=group)
    return DistributedStepWorker(
        process_group,
        local,
        plan_identity=plan_identity,
        allow_graph=allow_graph,
        step_timeout_s=step_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
        publisher_rank=publisher_rank,
    )
