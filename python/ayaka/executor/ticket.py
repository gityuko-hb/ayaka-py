"""Torch-free ownership and completion contracts for asynchronous execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ayaka.handles import SequenceHandle
from ayaka.sched.plan import PreparedStep
from ayaka.utils.validation import require_frozen, require_int


@dataclass(frozen=True, slots=True)
class TicketId:
    """Process-local executor incarnation plus monotonically increasing ordinal."""

    executor: int
    ordinal: int

    def __post_init__(self) -> None:
        require_int(self.executor, "executor", minimum=1)
        require_int(self.ordinal, "ordinal", minimum=1)


class TicketState(StrEnum):
    ADOPTED = "adopted"
    SUBMITTED = "submitted"
    DRAINING = "draining"
    QUARANTINED = "quarantined"
    COMPLETED = "completed"
    RETIRED = "retired"


class WorkState(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FenceResult:
    """Failure and quiescence are independent; a query error proves neither."""

    state: WorkState
    quiescent: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, WorkState) or type(self.quiescent) is not bool:
            raise TypeError("invalid fence state/quiescence")
        if self.state is WorkState.PENDING and self.quiescent:
            raise ValueError("pending work cannot prove quiescence")
        if self.state is WorkState.SUCCEEDED and not self.quiescent:
            raise ValueError("success must cover every use represented by the fence")
        if self.state is WorkState.FAILED and not self.error:
            raise ValueError("a failed fence requires an error")


class CompletionFence(Protocol):
    """Nonblocking query. Exceptions imply unknown completion, never safe release."""

    def query(self) -> FenceResult: ...


class ExecutionResources(Protocol):
    """One exclusive bundle retained by a ticket until all consumers stop.

    adopt is atomic: failure leaves ownership with the caller. After adoption
    the caller must never roll back/free this bundle. mark_submitted runs before
    any backend enqueue. commit publishes KV only, returning sequence versions
    in packed order; it must not retire execution leases. retire releases last-use
    leases after logical publication/discard and is idempotent by ticket identity.
    discarded_sequences contains original generation-safe handles from this step;
    real resources release those requests behind their active KV lease.

    P1 supplies only a fake implementation. Physical KV/workspace/staging
    accounting and real allocator validation are implemented in P2. Exceptions
    after adoption quarantine the bundle; callbacks are not blindly retried.
    """

    def adopt(self, ticket_id: TicketId, prepared: PreparedStep) -> None: ...
    def mark_submitted(self, ticket_id: TicketId) -> None: ...
    def commit(self, ticket_id: TicketId) -> tuple[int, ...]: ...
    def retire(
        self,
        ticket_id: TicketId,
        *,
        succeeded: bool,
        discarded_sequences: tuple[SequenceHandle, ...] = (),
    ) -> None: ...


class TerminalStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """Executor-authored, quiescent result; not itself permission to free pages."""

    ticket_id: TicketId
    status: TerminalStatus
    samples: tuple[int, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        require_frozen(self, "terminal outcome")
        if not isinstance(self.status, TerminalStatus):
            raise TypeError("status must be TerminalStatus")
        for sample in self.samples:
            require_int(sample, "sample token")
        if self.status is not TerminalStatus.SUCCEEDED and self.samples:
            raise ValueError("failed/cancelled work cannot publish samples")


@dataclass(slots=True)
class ExecutionTicket:
    """Live runtime owner, unlike the immutable PreparedStep it retains.

    Underscored fields are mutated only by the executor/completion coordinator.
    Retired tickets may remain in diagnostics; the active executor registry
    retains every non-retired ticket, including drain and quarantine failures.
    """

    id: TicketId
    prepared: PreparedStep
    _resources: ExecutionResources = field(repr=False)
    _state: TicketState = TicketState.ADOPTED
    _fences: list[CompletionFence] = field(default_factory=list, repr=False)
    _drain_fence: CompletionFence | None = field(default=None, repr=False)
    _needs_drain: bool = False
    _host_failure: bool = False
    _cancelled: bool = False
    _samples: tuple[int, ...] = ()
    _error: str = ""
    _terminal: TerminalOutcome | None = None
    _settling: bool = False
    _retirement_complete: bool = False
    _execution_failed: bool = False
    _adopted_ns: int = 0
    _submitted_ns: int | None = None
    _drain_started_ns: int | None = None
    _terminal_ns: int | None = None

    @property
    def state(self) -> TicketState:
        return self._state

    @property
    def error(self) -> str:
        return self._error

    @property
    def terminal(self) -> TerminalOutcome | None:
        return self._terminal
