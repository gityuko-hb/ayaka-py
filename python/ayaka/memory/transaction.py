from __future__ import annotations

from dataclasses import dataclass, field

from ayaka.exceptions import InvalidStateTransitionError, TransactionClosedError
from ayaka.handles import (
    KVPageHandle,
    KVReservationHandle,
    MemoryTransactionHandle,
    SequenceHandle,
    StepMemoryLeaseHandle,
)
from ayaka.memory.sequence import PageTableEntry
from ayaka.memory.state import LeaseState, ReservationFailure, TransactionState
from ayaka.memory.views import KVWriteSlot


@dataclass(frozen=True, slots=True)
class ReservationResult:
    """Structured outcome of one tentative reservation.

    On success ``handle`` carries the opaque reservation identity the scheduler
    stores in its plan; on failure ``reason`` carries one of
    ``ReservationFailure`` codes so the scheduler can react without page
    internals. ``required_pages`` is the number of new pages the append needs
    (regardless of success); ``allocated_pages`` counts only pages actually
    handed out. ``attached_prefix_tokens`` reports how many leading tokens came
    from a reused cache prefix.
    """
    ok: bool
    handle: KVReservationHandle | None
    required_pages: int
    allocated_pages: int
    attached_prefix_tokens: int = 0
    reason: ReservationFailure | None = None

    @classmethod
    def success(
        cls,
        handle: KVReservationHandle,
        *,
        required_pages: int,
        allocated_pages: int,
        attached_prefix_tokens: int = 0,
    ) -> ReservationResult:
        """Build a success result carrying the opaque reservation handle."""
        return cls(
            ok=True,
            handle=handle,
            required_pages=required_pages,
            allocated_pages=allocated_pages,
            attached_prefix_tokens=attached_prefix_tokens,
        )

    @classmethod
    def failure(
        cls,
        reason: ReservationFailure,
        *,
        required_pages: int = 0,
    ) -> ReservationResult:
        """Build a failure result with no handle and no allocated pages."""
        return cls(
            ok=False,
            handle=None,
            required_pages=required_pages,
            allocated_pages=0,
            reason=reason,
        )


@dataclass(slots=True)
class ReservationRecord:
    """Everything the manager needs to plan and later execute one reservation.

    Captures the sequence version/length at planning time (the rollback
    baseline), the tentative page table, the pages newly allocated as
    ``RESERVED``, prefix pages tentatively acquired, and the write-slot map the
    batch builder will materialize into kernel-facing metadata.
    """

    handle: KVReservationHandle
    sequence: SequenceHandle
    base_sequence_version: int
    """Sequence version at planning time; must be unchanged to execute."""
    base_committed_tokens: int
    """Committed token count before the append; a successful step advances it."""
    attached_prefix_tokens: int
    """Leading tokens served by the cache; execution_base_tokens includes them."""
    num_new_tokens: int
    """Tokens the step will write; A1 requires all of them to be written."""
    planned_page_table: tuple[PageTableEntry, ...]
    """Full planned table: existing committed entries plus new pages."""
    allocated_pages: tuple[KVPageHandle, ...]
    """Newly reserved pages; rolled back, committed, or abandoned with the step."""
    acquired_prefix_pages: tuple[KVPageHandle, ...]
    """Tentative request refs on cache pages; released on rollback/failure."""
    write_slots: tuple[KVWriteSlot, ...]
    """One slot per written token, mapping logical position to flat slot."""

    @property
    def execution_base_tokens(self) -> int:
        """Attention-visible start position for this step's writes."""
        return self.base_committed_tokens + self.attached_prefix_tokens

    @property
    def touched_pages(self) -> tuple[KVPageHandle, ...]:
        """Every page the step may read or write (planned table pages)."""
        return tuple(entry.page for entry in self.planned_page_table)


@dataclass(slots=True)
class TransactionRecord:
    """Open transaction state: its handle and the reservations it owns."""

    handle: MemoryTransactionHandle
    state: TransactionState = TransactionState.OPEN
    reservation_handles: list[KVReservationHandle] = field(default_factory=list)


@dataclass(slots=True)
class LeaseRecord:
    """Prepared execution lease: frozen reservations plus lease state."""

    handle: StepMemoryLeaseHandle
    transaction: MemoryTransactionHandle
    reservation_handles: tuple[KVReservationHandle, ...]
    state: LeaseState = LeaseState.PREPARED

class LifecycleTransitions:
    """Shared transaction/lease state machine for homogeneous and grouped KV.

    Reservation payloads differ by layout, but legal state transitions do not.
    Keeping them here prevents the two managers from silently drifting on
    rollback, launch, completion, or failure behavior.
    """

    @staticmethod
    def require_open(transaction: TransactionRecord) -> None:
        if transaction.state is not TransactionState.OPEN:
            raise TransactionClosedError(
                f"transaction {transaction.handle} is not open"
            )

    @staticmethod
    def prepare(transaction: TransactionRecord) -> None:
        LifecycleTransitions.require_open(transaction)
        transaction.state = TransactionState.PREPARED

    @staticmethod
    def rollback(transaction: TransactionRecord) -> None:
        LifecycleTransitions.require_open(transaction)
        transaction.state = TransactionState.ROLLED_BACK

    @staticmethod
    def require_lease(lease: LeaseRecord, expected: LeaseState) -> None:
        if lease.state is not expected:
            raise InvalidStateTransitionError(
                f"lease {lease.handle} is {lease.state.name}, expected {expected.name}"
            )

    @staticmethod
    def transition_lease(
        lease: LeaseRecord,
        *,
        expected: LeaseState,
        target: LeaseState,
    ) -> None:
        LifecycleTransitions.require_lease(lease, expected)
        lease.state = target