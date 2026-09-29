"""Transactional records for tentative KV plans.

A scheduler step opens a transaction, reserves pages tentatively, freezes the
plan into an execution lease, then commits or rolls it back.  The records and
the small state machine (:class:`LifecycleTransitions`) are shared by the
homogeneous and grouped managers; the managers own the behaviour, these records
own the truth.  Nothing in this module touches the allocator or the ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field

from ayaka.exceptions import InvalidStateTransitionError, TransactionClosedError
from ayaka.handles import (
    KVPageHandle,
    KVReservationHandle,
    MemoryTransactionHandle,
    SequenceHandle,
    StepMemoryLeaseHandle,
)
from ayaka.memory.sequence import GroupPageTableEntry, PageTableEntry
from ayaka.memory.state import LeaseState, ReservationFailure, TransactionState
from ayaka.memory.views import GroupKVWriteSlot, KVPageCopy, KVWriteSlot


@dataclass(frozen=True, slots=True)
class ReservationResult:
    """Structured outcome of one tentative reservation.

    On success ``handle`` carries the opaque reservation identity the scheduler
    stores in its plan; on failure ``reason`` carries one of
    ``ReservationFailure`` codes so the scheduler can react without page
    internals. ``required_pages`` is the number of new pages the append needs
    (regardless of success); ``allocated_pages`` counts only pages actually
    handed out.
    """

    ok: bool
    handle: KVReservationHandle | None
    required_pages: int
    allocated_pages: int
    reason: ReservationFailure | None = None

    @classmethod
    def success(
        cls,
        handle: KVReservationHandle,
        *,
        required_pages: int,
        allocated_pages: int,
    ) -> ReservationResult:
        """Build a success result carrying the opaque reservation handle."""
        return cls(
            ok=True,
            handle=handle,
            required_pages=required_pages,
            allocated_pages=allocated_pages,
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
    ``RESERVED``, and the write-slot map the batch builder will materialize
    into kernel-facing metadata. Cached prefix pages are already committed
    request ownership before a reservation is planned, so no prefix field
    exists here.
    """

    handle: KVReservationHandle
    sequence: SequenceHandle
    base_sequence_version: int
    """Sequence version at planning time; must be unchanged to execute."""
    base_committed_tokens: int
    """Committed token count before the append; a successful step advances it."""
    num_new_tokens: int
    """Tokens the step will write; A1 requires all of them to be written."""
    planned_page_table: tuple[PageTableEntry, ...]
    """Full planned table: existing committed entries plus new pages."""
    allocated_pages: tuple[KVPageHandle, ...]
    """Newly reserved pages; rolled back, committed, or abandoned with the step."""
    committed: bool = field(default=False, init=False)
    """Whether tentative refs have transferred into committed sequence tables."""
    write_slots: tuple[KVWriteSlot, ...]
    """One slot per written token, mapping logical position to flat slot."""
    copies: tuple[KVPageCopy, ...] = ()
    """Source refs stay in the old table until commit; the lease protects last use."""
    min_commit_tokens: int | None = None
    """Smallest accepted written-token report; ``None`` requires every token.

    Only a speculative reservation lowers it: its trailing rows are tentative
    and commit only when verification accepts them.
    """

    @property
    def commit_floor(self) -> int:
        """Fewest written tokens a commit may report for this reservation."""
        return self.num_new_tokens if self.min_commit_tokens is None else self.min_commit_tokens

    @property
    def execution_base_tokens(self) -> int:
        """Attention-visible start position for this step's writes."""
        return self.base_committed_tokens

    @property
    def touched_pages(self) -> tuple[KVPageHandle, ...]:
        """Every page the step may read or write (planned table pages)."""
        return tuple(entry.page for entry in self.planned_page_table) + tuple(
            copy.source for copy in self.copies
        )


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
            raise TransactionClosedError(f"transaction {transaction.handle} is not open")

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


@dataclass(slots=True)
class GroupReservationPlan:
    """Per-group portion of one reservation: table, pages, and write slots."""

    group_name: str
    planned_page_table: tuple[GroupPageTableEntry, ...]
    allocated_pages: tuple[KVPageHandle, ...]
    write_slots: tuple[GroupKVWriteSlot, ...]
    copies: tuple[KVPageCopy, ...] = ()

    @property
    def touched_pages(self) -> tuple[KVPageHandle, ...]:
        """Every page of this group the step may read or write."""
        return tuple(entry.page for entry in self.planned_page_table) + tuple(
            copy.source for copy in self.copies
        )


@dataclass(slots=True)
class GroupedReservationRecord:
    """Full reservation: baseline version/length plus per-group plans."""

    handle: KVReservationHandle
    sequence: SequenceHandle
    base_sequence_version: int
    base_committed_tokens: int
    num_new_tokens: int
    group_plans: tuple[GroupReservationPlan, ...]
    committed: bool = field(default=False, init=False)
    min_commit_tokens: int | None = None
    """Smallest accepted written-token report; ``None`` requires every token."""

    @property
    def commit_floor(self) -> int:
        """Fewest written tokens a commit may report for this reservation."""
        return self.num_new_tokens if self.min_commit_tokens is None else self.min_commit_tokens


# Compatibility imports share the exact lifecycle records.
GroupedTransactionRecord = TransactionRecord
GroupedLeaseRecord = LeaseRecord


def resolve_written_counts(
    records: Sequence[ReservationRecord | GroupedReservationRecord],
    written_tokens: Mapping[KVReservationHandle, int] | None,
) -> dict[KVReservationHandle, int]:
    """Resolve one commit's written-token report against its reservations.

    ``None`` means every reserved token was written; it is refused when a
    reservation has tentative rows, because committing those implicitly would
    make unverified KV sequence-owned. An explicit report must name exactly the
    step's reservations and keep each count within
    ``[commit_floor, num_new_tokens]``.

    Raises:
        InvalidStateTransitionError: On a missing, extra or out-of-range count.
    """
    if written_tokens is None:
        for record in records:
            if record.commit_floor != record.num_new_tokens:
                raise InvalidStateTransitionError(
                    "a speculative reservation needs an explicit written-token report"
                )
        return {record.handle: record.num_new_tokens for record in records}
    if set(written_tokens) != {record.handle for record in records}:
        raise InvalidStateTransitionError(
            "written-token report does not match the step reservations"
        )
    resolved: dict[KVReservationHandle, int] = {}
    for record in records:
        count = written_tokens[record.handle]
        if type(count) is not int or not record.commit_floor <= count <= record.num_new_tokens:
            if record.commit_floor == record.num_new_tokens:
                raise InvalidStateTransitionError(
                    "a non-speculative reservation requires every reserved token to be written"
                )
            raise InvalidStateTransitionError(
                "written-token report is outside the reservation's commit range"
            )
        resolved[record.handle] = count
    return resolved


class TransactionOrchestrator:
    """Common prepare/rollback ordering with manager-specific strategy hooks.

    Callers hold their manager lock. Validation may fail and is rolled back
    before any lease is published. Activation and undo hooks only mutate
    already-validated host metadata; they must not enqueue device work.
    If activation fails part way through, deactivate restores transaction
    ownership before rollback releases reservations. Cleanup hooks must not fail.
    """

    @staticmethod
    def prepare(
        transaction: TransactionRecord,
        lease_handle: StepMemoryLeaseHandle,
        *,
        validate: Callable[[], None],
        activate: Callable[[LeaseRecord], None],
        deactivate: Callable[[LeaseRecord], None],
        rollback: Callable[[], None],
        transactions: MutableMapping[int, TransactionRecord],
        leases: MutableMapping[int, LeaseRecord],
    ) -> StepMemoryLeaseHandle:
        LifecycleTransitions.require_open(transaction)
        try:
            if not transaction.reservation_handles:
                raise InvalidStateTransitionError("cannot prepare an empty transaction")
            validate()
        except BaseException:
            rollback()
            raise
        lease = LeaseRecord(
            lease_handle, transaction.handle, tuple(transaction.reservation_handles)
        )
        try:
            leases[lease_handle.index] = lease
            activate(lease)
        except BaseException:
            deactivate(lease)
            leases.pop(lease_handle.index, None)
            rollback()
            raise
        LifecycleTransitions.prepare(transaction)
        transactions.pop(transaction.handle.index)
        return lease_handle

    @staticmethod
    def rollback(
        transaction: TransactionRecord,
        *,
        undo: Callable[[], None],
        finish: Callable[[], None],
        transactions: MutableMapping[int, TransactionRecord],
    ) -> None:
        LifecycleTransitions.require_open(transaction)
        undo()
        LifecycleTransitions.rollback(transaction)
        transactions.pop(transaction.handle.index)
        finish()
