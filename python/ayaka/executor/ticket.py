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
    """Runtime lifecycle states of an ExecutionTicket within an Executor.

    Reflects the ownership transfer and execution phase of a resource bundle,
    tracking work from initial plan adoption through device execution to safe,
    quiescent retirement.
    """

    ADOPTED = "adopted"
    """The ticket has atomically accepted ownership of both the PreparedStep
    and its ExecutionResources bundle, but work has not yet been enqueued
    to backend execution streams.
    """

    SUBMITTED = "submitted"
    """Resources have been marked for submission and operations are actively
    enqueued on the device; completion fences are registered and awaiting
    evaluation.
    """

    DRAINING = "draining"
    """Work is in-flight or partially submitted and must be drained to prove
    hardware quiescence before completion. Triggered by cancellation,
    submission faults, or executor shutdown.
    """

    QUARANTINED = "quarantined"
    """Fault-isolation state reached when ambiguous errors occur (e.g., host
    exceptions, fence query failures, or non-quiescent device faults). The ticket
    and its resource leases are frozen indefinitely to prevent use-after-free bugs,
    halting new admissions.
    """

    COMPLETED = "completed"
    """Execution has reached verifiable hardware quiescence. A terminal outcome
    has been authored and is ready for the completion coordinator to settle
    (commit KV cache, publish sample tokens, or discard).
    """

    RETIRED = "retired"
    """Terminal lifecycle state. Logical commitments and physical resource leases
    have been acknowledged and released via `retire()`. The ticket is pruned from
    the executor's active registry.
    """

class WorkState(StrEnum):
    """Physical execution state reported by a CompletionFence."""

    PENDING = "pending"
    """The underlying backend work is actively running or scheduled in a device queue.
    Cannot coexist with a quiescent status.
    """

    SUCCEEDED = "succeeded"
    """Work completed successfully on the device. By contract, must guarantee
    that all associated memory operations have achieved hardware quiescence.
    """

    FAILED = "failed"
    """Work encountered a backend or device error. Requires an accompanying error
    message; does not automatically guarantee that the hardware is quiescent.
    """

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
    """Terminal completion status authored by an Executor for an ExecutionTicket.

    Represents the final logical outcome of a quiescent unit of work, instructing
    the CompletionCoordinator how to settle KV cache commits, deliver sampled tokens,
    and transition request lifecycle.
    """

    SUCCEEDED = "succeeded"
    """The execution step completed successfully with all device operations quiescent.
    Authorizes the coordinator to commit KV cache state versions and publish
    generated token samples to participating requests.
    """

    FAILED = "failed"
    """Execution encountered an unrecoverable host, kernel, or hardware error.
    Prohibits sample publication and directs the coordinator to discard computed
    slices and fail active requests.
    """

    CANCELLED = "cancelled"
    """Execution was aborted before launch or drained mid-flight due to client
    cancellation or executor shutdown. Discards generated outputs and triggers
    clean resource retirement.
    """

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

    @property
    def state(self) -> TicketState:
        return self._state

    @property
    def error(self) -> str:
        return self._error

    @property
    def terminal(self) -> TerminalOutcome | None:
        return self._terminal
