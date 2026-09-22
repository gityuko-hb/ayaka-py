"""Distributed WorkerStep protocol: prepare, all-grant, launch, commit, publish.

The coordinator extends the R09 worker seam with the R11 group contract while
the scheduler stays free of CUDA/distributed ownership:

1. The source rank broadcasts one :class:`~ayaka.distributed.step_envelope.
   DistributedStepEnvelope`; every rank validates it, reserves rank-local
   storage and votes through an all-grant collective. A single refusal rolls
   the whole candidate back before any rank enqueues device work.
2. After a unanimous grant each rank executes on its own resources. Logical
   ordering is shared through the envelope; physical pages, pointers and local
   generations may differ per rank and are validated by the rank itself.
3. Completion is proven rank-locally (a fence/drain that covers the rank's
   last use of every submitted collective), then a second all-grant provides
   the global success proof, ranks commit locally, a third all-grant carries
   the commit acknowledgements, and only then does the publisher rank publish
   output. Any failure after the grant is fail-closed: the group is FAILED,
   nothing is optimistically rolled back, and no rank serves the next step
   until a recovery rebuilds resources and a fresh incarnation.
4. Shutdown is a fourth group agreement: every rank votes from its own
   lifecycle state, a unanimous accept is followed by a barrier, and only then
   does a rank close its KV worker or free device state. A refusal or deadline
   failure is fail-closed — the group is marked FAILED and no rank reports a
   clean shutdown.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, runtime_checkable

from ayaka.distributed.completion import CompletionFence
from ayaka.distributed.process_group import (
    DistributedCollectiveTimeout,
    DistributedStepError,
)
from ayaka.distributed.step_envelope import DistributedStepEnvelope
from ayaka.distributed.worker import RankLocalKVWorker, RankLocalKVWorkerState
from ayaka.exceptions import InvariantViolationError

if TYPE_CHECKING:
    from ayaka.sched.plan import PreparedStep

__all__ = [
    "DistributedControlPlane",
    "DistributedStepCoordinator",
    "DistributedStepFailed",
    "DistributedStepRefused",
    "RankLocalStepHost",
]


class DistributedStepRefused(DistributedStepError):
    """The candidate step was rolled back before any rank launched work."""


class DistributedStepFailed(DistributedStepError):
    """A post-launch failure: the group is fail-closed until recovery."""


@runtime_checkable
class DistributedControlPlane(Protocol):
    """Collective surface the coordinator drives, with explicit deadlines.

    :class:`~ayaka.distributed.process_group.TorchDistributedKVProcessGroup`
    implements it over NCCL/Gloo; tests may substitute a loopback plane.
    """

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def all_grant(self, local_grant: bool, *, timeout_s: float | None = None) -> bool: ...

    def broadcast_text(
        self,
        text: str | None,
        *,
        source_rank: int,
        timeout_s: float | None = None,
    ) -> str: ...

    def barrier(self, *, timeout_s: float | None = None) -> None: ...

    def shutdown(self, local_accepting: bool, *, timeout_s: float | None = None) -> bool: ...


@runtime_checkable
class RankLocalStepHost(Protocol):
    """One rank's device-owned half of a distributed step.

    The host owns its context, streams, runner and KV binding; the coordinator
    never touches them directly. ``prepare`` may refuse (raise) — a refused
    step must leave every tentative reference clean through ``rollback``,
    which is also called when another rank refuses.
    """

    def plan_identity(self) -> Mapping[str, Any]: ...

    def prepare(self, envelope: DistributedStepEnvelope) -> None: ...

    def rollback(self, envelope: DistributedStepEnvelope) -> None: ...

    def execute(self, envelope: DistributedStepEnvelope) -> Any: ...

    def fence(self, result: Any) -> CompletionFence: ...

    def commit(self, envelope: DistributedStepEnvelope, result: Any) -> None: ...

    def publish(self, envelope: DistributedStepEnvelope, result: Any) -> None: ...

    def fail(self, error: BaseException | str) -> None: ...


class DistributedStepCoordinator:
    """Group protocol around rank-local workers for one step at a time.

    The coordinator owns no device resource and no request state: the envelope
    is the only thing that crosses ranks, and every collective carries an
    explicit deadline so a wedged rank becomes a fail-closed error instead of
    an unbounded hang.
    """

    def __init__(
        self,
        control: DistributedControlPlane,
        host: RankLocalStepHost,
        *,
        source_rank: int = 0,
        publisher_rank: int = 0,
        allow_graph: bool = False,
    ) -> None:
        if not 0 <= source_rank < control.world_size:
            raise ValueError("source_rank outside process group")
        if not 0 <= publisher_rank < control.world_size:
            raise ValueError("publisher_rank outside process group")
        if type(allow_graph) is not bool:
            raise TypeError("allow_graph must be a boolean")
        self.control = control
        self.host = host
        self.source_rank = source_rank
        self.publisher_rank = publisher_rank
        self.allow_graph = allow_graph
        self._last_attempt = -1
        self.worker = RankLocalKVWorker(control.rank)
        self.worker.start()

    @property
    def rank(self) -> int:
        return self.control.rank

    @property
    def generation(self) -> int:
        """This rank's worker incarnation; recover() advances it."""
        return self.worker.generation

    @property
    def failure(self) -> str | None:
        return self.worker.failure

    # ------------------------------------------------------------------
    # step protocol
    # ------------------------------------------------------------------

    def run(
        self,
        prepared: PreparedStep | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Run one distributed step; returns the rank-local execution result.

        Only the source rank may pass ``prepared``; every other rank receives
        the envelope by broadcast. The return value is opaque host state —
        output publication is exclusively the publisher rank's job and only
        after the commit acknowledgements have landed.
        """
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("step timeout must be positive")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        try:
            envelope = self._broadcast(prepared, deadline)
            # ---- prepare phase: admit + reserve, then exactly one vote ----
            refusal: BaseException | None = None
            try:
                self._admit(envelope)
                self.host.prepare(envelope)
            except BaseException as exc:
                refusal = exc
            if not self._agree(refusal is None, deadline):
                self.host.rollback(envelope)
                self._end_step(envelope)
                if refusal is not None:
                    raise DistributedStepRefused(
                        f"rank {self.control.rank} refused step {envelope.step_id}: {refusal}"
                    ) from refusal
                raise DistributedStepRefused(
                    f"step {envelope.step_id} was refused by another rank; "
                    "no rank launched and every tentative reference is clean"
                )
            # ---- launch phase: execute, then prove quiescence locally ----
            failure: BaseException | None = None
            result: Any = None
            try:
                result = self.host.execute(envelope)
                fence = self.host.fence(result)
                if not fence.wait(self._budget(deadline)):
                    failure = DistributedStepFailed(
                        f"rank {self.control.rank} completion timed out before the success proof"
                    )
            except BaseException as exc:
                failure = exc
            # ---- global success proof: a failed rank still votes exactly ----
            # once with grant=False; enqueued state is never rolled back or
            # freed before this proof — the group fails closed instead.
            if not self._agree(failure is None, deadline):
                if failure is None:
                    failure = DistributedStepFailed(
                        f"step {envelope.step_id} failed the global execution proof"
                    )
                self.worker.fail(failure)
                self.host.fail(failure)
                if isinstance(failure, DistributedStepError):
                    raise failure
                raise DistributedStepFailed(
                    f"rank {self.control.rank} failed step {envelope.step_id}: {failure}"
                ) from failure
            # ---- commit phase: global proof first, then local commit ----
            commit_failure: BaseException | None = None
            try:
                self.host.commit(envelope, result)
            except BaseException as exc:
                commit_failure = exc
            if not self._agree(commit_failure is None, deadline):
                if commit_failure is None:
                    commit_failure = DistributedStepFailed(
                        f"step {envelope.step_id} failed the group commit acknowledgement"
                    )
                self.worker.fail(commit_failure)
                self.host.fail(commit_failure)
                raise DistributedStepFailed(
                    f"rank {self.control.rank} failed step {envelope.step_id}: {commit_failure}"
                ) from commit_failure
            # ---- publication: only the publisher rank, only after acks ----
            if self.control.rank == self.publisher_rank:
                try:
                    self.host.publish(envelope, result)
                except BaseException as exc:
                    self._fail_closed(exc)
            self.control.barrier(timeout_s=self._budget(deadline))
            self._end_step(envelope)
            return result
        except DistributedStepRefused:
            raise
        except BaseException as exc:
            self.worker.fail(exc)
            self.host.fail(exc)
            if isinstance(exc, DistributedStepError):
                raise
            raise DistributedStepFailed(
                f"rank {self.control.rank} failed step {self._last_attempt}: {exc}"
            ) from exc

    def recover(
        self,
        reinitialize: Callable[[], DistributedControlPlane | None],
    ) -> None:
        """Rebuild rank-local resources and optionally replace the plane.

        Recovery is the only path out of FAILED: it must produce a fresh
        incarnation (the worker generation increments) before the rank may
        rejoin collectives, and stale steps from before the failure are
        rejected by the monotonic step check plus the generation guard.
        """
        replacement: DistributedControlPlane | None = None

        def rebuild() -> None:
            nonlocal replacement
            replacement = reinitialize()

        self.worker.recover(rebuild)
        if replacement is not None:
            if replacement.rank != self.control.rank:
                self.worker.fail("replacement control plane changed local rank")
                raise InvariantViolationError("replacement control plane changed local rank")
            self.control = replacement

    # ------------------------------------------------------------------
    # shutdown protocol
    # ------------------------------------------------------------------

    def shutdown(self, *, timeout_s: float | None = None) -> bool:
        """Coordinate the group shutdown; True only on a clean group-wide close.

        Every rank votes exactly once through the control plane's shutdown
        handshake. This rank accepts only from READY: an active flight
        (``RUNNING``) means device work may still be in use, and a ``FAILED``
        incarnation can never be proven clean, so both vote no. The plane then
        orders the barrier, and only after it returns does this rank close its
        KV worker — no rank frees state while a peer still expects a
        collective.

        Failure semantics are fail-closed:

        - a refusal by any rank (or a peer that never votes) raises
          :class:`DistributedStepFailed` / :class:`DistributedCollectiveTimeout`
          instead of reporting a clean shutdown;
        - a READY rank that observed a refusal or deadline failure moves to
          ``FAILED`` and stays observable through :attr:`failure`; a rank with
          an active flight keeps ``RUNNING`` (the flight is not poisoned), and
          an already ``FAILED`` rank keeps its original failure;
        - the process group is never destroyed here; only
          :func:`ayaka.distributed.runtime.shutdown_distributed_process_group`
          may destroy an owned group after this handshake.

        Repeating a completed shutdown is a no-op that returns True.
        """
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("shutdown timeout must be positive")
        if self.worker.state is RankLocalKVWorkerState.CLOSED:
            return True
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        accepting = self.worker.state is RankLocalKVWorkerState.READY
        try:
            agreed = self.control.shutdown(accepting, timeout_s=self._budget(deadline))
        except BaseException as exc:
            # A rank with a pending flight keeps RUNNING: the flight may still
            # complete safely and must not be poisoned by a shutdown attempt.
            if self.worker.state is RankLocalKVWorkerState.READY:
                self.worker.fail(f"distributed shutdown agreement failed: {exc}")
            raise
        if not agreed:
            if self.worker.state is RankLocalKVWorkerState.READY:
                self.worker.fail("distributed shutdown was refused by at least one rank")
            raise DistributedStepFailed(
                f"rank {self.control.rank} shutdown was refused by at least one rank; "
                "no rank reported a clean shutdown"
            )
        self.worker.close()
        return True

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------

    def _broadcast(
        self,
        prepared: PreparedStep | None,
        deadline: float | None,
    ) -> DistributedStepEnvelope:
        text: str | None = None
        built: DistributedStepEnvelope | None = None
        if self.control.rank == self.source_rank:
            if prepared is None:
                raise ValueError("the source rank must provide a prepared step")
            built = DistributedStepEnvelope.from_prepared(prepared)
            text = built.encode()
        elif prepared is not None:
            raise ValueError("only the source rank may pass a prepared step")
        received = self.control.broadcast_text(
            text,
            source_rank=self.source_rank,
            timeout_s=self._budget(deadline),
        )
        envelope = DistributedStepEnvelope.decode(received)
        if built is not None and built.sha256 != envelope.sha256:
            raise InvariantViolationError("source envelope changed during broadcast")
        self._last_attempt = envelope.step_id
        return envelope

    def _admit(self, envelope: DistributedStepEnvelope) -> None:
        """Validate the envelope against this rank before any reservation.

        Only the *global* plan/model identity is compared — per-rank physical
        placement (page ids, local generations) is the rank's own business and
        never crosses the wire. Raises for a global mismatch, an uncertified
        graph mode, out-of-group ranks or a stale incarnation/step; the caller
        turns any raise into a refusal vote so the group rolls back together.
        """
        payload = envelope.payload
        local_identity = dict(self.host.plan_identity())
        mismatched = [
            key
            for key, expected in local_identity.items()
            if _resolve_identity(payload, key) != expected
        ]
        if mismatched:
            raise RuntimeError(f"rank {self.control.rank} plan identity disagrees on {mismatched}")
        graph_mode = payload["graph"]["mode"]
        if graph_mode in ("capture", "replay") and not self.allow_graph:
            raise RuntimeError(
                f"rank {self.control.rank} rejects graph mode {graph_mode!r}: "
                "distributed graph replay is not certified"
            )
        distributed = payload["distributed"]
        if distributed is not None:
            ranks = set(distributed["participating_ranks"])
            if not ranks <= set(range(self.control.world_size)):
                raise RuntimeError(
                    f"rank {self.control.rank} rejects ranks {sorted(ranks)} "
                    f"outside the {self.control.world_size}-rank group"
                )
            if distributed["worker_generation"] != self.worker.generation:
                raise RuntimeError(
                    f"rank {self.control.rank} holds generation {self.worker.generation} "
                    f"but step {envelope.step_id} targets {distributed['worker_generation']}"
                )
        self.worker.begin(envelope)

    def _agree(self, local_grant: bool, deadline: float | None) -> bool:
        return self.control.all_grant(local_grant, timeout_s=self._budget(deadline))

    def _fail_closed(self, exc: BaseException) -> NoReturn:
        self.worker.fail(exc)
        self.host.fail(exc)
        if isinstance(exc, DistributedStepError):
            raise
        raise DistributedStepFailed(
            f"rank {self.control.rank} failed step {self._last_attempt}: {exc}"
        ) from exc

    def _end_step(self, envelope: DistributedStepEnvelope) -> None:
        if self.worker.state is RankLocalKVWorkerState.RUNNING:
            self.worker.complete(envelope.step_id)

    def _budget(self, deadline: float | None) -> float | None:
        """Remaining wall-clock budget; the watchdog never waits forever."""
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DistributedCollectiveTimeout(
                f"rank {self.control.rank} step watchdog expired before the collective"
            )
        return remaining


def _resolve_identity(payload: Mapping[str, Any], dotted_key: str) -> Any:
    """Resolve a dotted plan-identity key (``parallel.tp_size``) into the payload."""
    node: Any = payload["plan"]
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node
