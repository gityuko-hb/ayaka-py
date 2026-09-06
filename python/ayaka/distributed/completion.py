from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from time import monotonic, sleep
from typing import Any, Protocol, runtime_checkable

from ayaka.distributed.process_group import DistributedStepError
from ayaka.distributed.topology import _optional_torch
from ayaka.exceptions import StorageUnavailableError
from ayaka.handles import KVReservationHandle, StepMemoryLeaseHandle
from ayaka.memory.manager import RuntimeMemoryManager


class DistributedStepTimeout(DistributedStepError):
    """Rank completion timed out while the least deliberately stays in flight."""

    def __init__(self, message: str, *, ticket: TensorParallelStepTicket) -> None:
        super().__init__(message)
        self.ticket = ticket


class RankDecision(Enum):
    """How one rank voted on a proposed group action."""

    GRANT = auto()
    REFUSE = auto()


@dataclass(frozen=True, slots=True)
class RankVote:
    """One rank's vote, with the reason when it refuses."""

    rank: int
    decision: RankDecision
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.reason is None) != (self.decision is RankDecision.GRANT):
            raise ValueError("a refusal must carry a reason and a grant must not")

    @classmethod
    def grant(cls, rank: int) -> RankVote:
        return cls(rank=rank, decision=RankDecision.GRANT)

    @classmethod
    def refuse(cls, rank: int, reason: str) -> RankVote:
        return cls(rank=rank, decision=RankDecision.REFUSE, reason=reason)


@dataclass(frozen=True, slots=True)
class AgreementOutcome:
    """The group's decision, and every vote that produced it."""

    granted: bool
    votes: tuple[RankVote, ...]

    @property
    def refusals(self) -> tuple[RankVote, ...]:
        return tuple(vote for vote in self.votes if vote.decision is RankDecision.REFUSE)

    def describe(self) -> str:
        if self.granted:
            return f"granted by all {len(self.votes)} ranks"
        return "; ".join(f"rank {vote.rank}: {vote.reason}" for vote in self.refusals)


def decide_reservation(votes: Sequence[RankVote]) -> AgreementOutcome:
    """Agree a reservation across ranks by unanimity (A12-03)."""
    collected = tuple(votes)
    if not collected:
        raise ValueError("a reservation agreement needs at least one vote")
    seen = {vote.rank for vote in collected}
    if len(seen) != len(collected):
        raise ValueError("each rank votes exactly once")
    granted = all(vote.decision is RankDecision.GRANT for vote in collected)
    return AgreementOutcome(granted=granted, votes=collected)


@runtime_checkable
class CompletionFence(Protocol):
    """A completion proof for one rank's submitted device work."""

    def wait(self, timeout_s: float | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class SynchronousCompletion:
    """Explicit proof that a launcher completed all work before returning."""

    def wait(self, timeout_s: float | None = None) -> bool:
        del timeout_s
        return True


@dataclass(slots=True)
class CudaEventCompletion:
    """Completion fence backed by a recorded ``torch.cuda.Event``."""

    event: Any
    poll_interval_s: float = 0.001

    def __post_init__(self) -> None:
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if not callable(getattr(self.event, "query", None)) or not callable(
            getattr(self.event, "synchronize", None)
        ):
            raise TypeError("event must provide query() and synchronize()")

    def wait(self, timeout_s: float | None = None) -> bool:
        if timeout_s is None:
            self.event.synchronize()
            return True
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = monotonic() + timeout_s
        while not bool(self.event.query()):
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            sleep(min(self.poll_interval_s, remaining))
        self.event.synchronize()
        return True


@dataclass(frozen=True, slots=True)
class DeviceSynchronizeCompletion:
    """Safe compatibility fence for legacy CUDA launchers returning ``None``."""

    device: str

    def wait(self, timeout_s: float | None = None) -> bool:
        del timeout_s
        torch = _optional_torch()
        if torch is None or not torch.cuda.is_available():
            return True
        torch.cuda.synchronize(self.device)
        return True


def record_cuda_completion(
    *,
    device: str | int | None = None,
    stream: Any | None = None,
) -> CudaEventCompletion:
    """Record and return an event after the caller's CUDA submissions."""
    torch = _optional_torch()
    if torch is None or not torch.cuda.is_available():
        raise StorageUnavailableError("CUDA completion requested without CUDA")
    resolved_device = torch.cuda.current_device() if device is None else device
    with torch.cuda.device(resolved_device):
        event = torch.cuda.Event(blocking=False, interprocess=False)
        if stream is None:
            event.record(torch.cuda.current_stream(resolved_device))
        else:
            event.record(stream)
    return CudaEventCompletion(event)


class TensorParallelStepState(Enum):
    """Lifecycle of a launched multi-rank step completion ticket."""

    PENDING = auto()
    COMPLETED = auto()
    FAILED = auto()


class TensorParallelStepTicket:
    """Own rank completion fences until a TP lease can be finalized safely."""

    def __init__(
        self,
        *,
        manager: RuntimeMemoryManager,
        lease_handle: StepMemoryLeaseHandle,
        agreement: AgreementOutcome,
        completions: Mapping[int, CompletionFence],
        safe_epoch: int,
        written_tokens: Mapping[KVReservationHandle, int] | None,
        launch_errors: Mapping[int, BaseException] | None = None,
    ) -> None:
        self._manager = manager
        self._lease_handle = lease_handle
        self._agreement = agreement
        self._completions = dict(completions)
        self._safe_epoch = int(safe_epoch)
        self._written_tokens = written_tokens
        self._errors: dict[int, BaseException] = dict(launch_errors or {})
        self._resolved: set[int] = set()
        self._state = TensorParallelStepState.PENDING
        self._lock = RLock()

    @property
    def state(self) -> TensorParallelStepState:
        with self._lock:
            return self._state

    @property
    def lease_handle(self) -> StepMemoryLeaseHandle:
        return self._lease_handle

    @property
    def pending_ranks(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(set(self._completions) - self._resolved))

    def wait(self, timeout_s: float | None = None) -> AgreementOutcome:
        """Wait for every rank and publish or fail the lease exactly once."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = None if timeout_s is None else monotonic() + timeout_s
        with self._lock:
            if self._state is TensorParallelStepState.COMPLETED:
                return self._agreement
            if self._state is TensorParallelStepState.FAILED:
                raise DistributedStepError("tensor-parallel step already failed")

            for rank in sorted(self._completions):
                if rank in self._resolved:
                    continue
                remaining = None if deadline is None else max(0.0, deadline - monotonic())
                try:
                    complete = self._completions[rank].wait(remaining)
                except BaseException as exc:
                    self._errors.setdefault(rank, exc)
                    self._resolved.add(rank)
                    continue
                if not complete:
                    raise DistributedStepTimeout(
                        "tensor-parallel completion timed out; lease remains in flight for "
                        f"ranks {self.pending_ranks}",
                        ticket=self,
                    )
                self._resolved.add(rank)

            if self._errors:
                safe_epoch = max(self._safe_epoch, self._manager.current_epoch)
                self._manager.fail_in_flight_step(
                    self._lease_handle,
                    safe_epoch=safe_epoch,
                )
                self._state = TensorParallelStepState.FAILED
                detail = "; ".join(
                    f"rank {rank}: {error}" for rank, error in sorted(self._errors.items())
                )
                raise DistributedStepError(
                    f"tensor-parallel step failed after launch: {detail}"
                )

            self._manager.complete_step(
                self._lease_handle,
                written_tokens=self._written_tokens,
            )
            self._state = TensorParallelStepState.COMPLETED
            return self._agreement


def _normalize_completion(
    result: Any,
    *,
    rank: int,
    rank_devices: Sequence[str] | None,
) -> CompletionFence:
    if isinstance(result, CompletionFence):
        return result
    if callable(getattr(result, "query", None)) and callable(
        getattr(result, "synchronize", None)
    ):
        return CudaEventCompletion(result)
    if result is not None:
        raise TypeError(
            "rank launcher must return CompletionFence, torch.cuda.Event, or None"
        )
    torch = _optional_torch()
    if torch is None or not torch.cuda.is_available():
        return SynchronousCompletion()
    device = (
        str(rank_devices[rank])
        if rank_devices is not None
        else f"cuda:{rank}"
    )
    return DeviceSynchronizeCompletion(device)


def _preflight_votes(
    preflight: Callable[[int], RankVote],
    num_ranks: int,
) -> tuple[RankVote, ...]:
    votes: list[RankVote] = []
    for rank in range(num_ranks):
        try:
            votes.append(preflight(rank))
        except Exception as error:
            votes.append(RankVote.refuse(rank, f"preflight raised: {error}"))
    return tuple(votes)


def launch_tensor_parallel_step(
    manager: RuntimeMemoryManager,
    lease_handle: StepMemoryLeaseHandle,
    launchers: Sequence[Callable[[int], Any]],
    *,
    safe_epoch: int,
    preflight: Callable[[int], RankVote] | None = None,
    written_tokens: Mapping[KVReservationHandle, int] | None = None,
    rank_devices: Sequence[str] | None = None,
) -> TensorParallelStepTicket:
    """Launch one prepared TP step and return its completion owner."""
    if not launchers:
        raise ValueError("a tensor-parallel step needs at least one rank launcher")
    if rank_devices is not None and len(rank_devices) != len(launchers):
        raise ValueError("rank_devices must contain exactly one device per launcher")

    votes = (
        _preflight_votes(preflight, len(launchers))
        if preflight is not None
        else tuple(RankVote.grant(rank) for rank in range(len(launchers)))
    )
    outcome = decide_reservation(votes)
    if not outcome.granted:
        manager.abort_prepared_step(lease_handle)
        raise DistributedStepError(f"step refused before launch: {outcome.describe()}")

    manager.mark_step_in_flight(lease_handle)
    completions: dict[int, CompletionFence] = {}
    launch_errors: dict[int, BaseException] = {}
    for rank, launch in enumerate(launchers):
        try:
            result = launch(rank)
            completions[rank] = _normalize_completion(
                result,
                rank=rank,
                rank_devices=rank_devices,
            )
        except BaseException as exc:
            launch_errors[rank] = exc
            completions[rank] = _normalize_completion(
                None,
                rank=rank,
                rank_devices=rank_devices,
            )
            break

    return TensorParallelStepTicket(
        manager=manager,
        lease_handle=lease_handle,
        agreement=outcome,
        completions=completions,
        safe_epoch=safe_epoch,
        written_tokens=written_tokens,
        launch_errors=launch_errors,
    )


def execute_tensor_parallel_step(
    manager: RuntimeMemoryManager,
    lease_handle: StepMemoryLeaseHandle,
    launchers: Sequence[Callable[[int], Any]],
    *,
    safe_epoch: int,
    preflight: Callable[[int], RankVote] | None = None,
    written_tokens: Mapping[KVReservationHandle, int] | None = None,
    rank_devices: Sequence[str] | None = None,
    completion_timeout_s: float | None = None,
) -> AgreementOutcome:
    """Launch, prove rank completion, then publish KV metadata."""
    ticket = launch_tensor_parallel_step(
        manager,
        lease_handle,
        launchers,
        safe_epoch=safe_epoch,
        preflight=preflight,
        written_tokens=written_tokens,
        rank_devices=rank_devices,
    )
    return ticket.wait(timeout_s=completion_timeout_s)
