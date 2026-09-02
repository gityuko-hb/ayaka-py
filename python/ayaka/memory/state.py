from __future__ import annotations

from enum import Enum, auto


class PageAllocationState(Enum):
    """Allocation lifecycle of one physical page.

    Transitions follow ``FREE -> RESERVED -> LIVE -> RECLAIM_PENDING -> FREE``
    with ``PERMANENT`` as a side branch never returned to the free pool.
    """

    FREE = auto()
    """Ready to allocate: zero refs, zero valid tokens, no pending epoch."""
    RESERVED = auto()
    """Owned by exactly one transaction reservation; not yet committed."""
    LIVE = auto()
    """Committed and owned by request/cache references; KV is readable."""
    RECLAIM_PENDING = auto()
    """Unowned and unrefed but awaiting its safe free epoch (deferred free)."""
    PERMANENT = auto()
    """Terminal state for the padding page; excluded from usable capacity."""


class PageResidency(Enum):
    """Placement is independent from allocation and sharing state.

    A page can be resident on device, host, or unmapped regardless of its
    allocation lifecycle; v1 only exercises ``DEVICE``.
    """

    DEVICE = auto()
    """KV bytes live in the device (GPU/HBM) tier."""
    HOST = auto()
    """KV bytes live in the host (CPU pinned) tier."""
    UNMAPPED = auto()
    """KV bytes are not currently backed by any storage tier."""


class TransactionState(Enum):
    """Lifecycle of one tentative-planning transaction."""

    OPEN = auto()
    """Reservations may still be added; the plan is still tentative."""
    PREPARED = auto()
    """Frozen into an execution lease; no further reservations are allowed."""
    ROLLED_BACK = auto()
    """Terminal: every reservation was undone and state restored."""


class LeaseState(Enum):
    """Execution-lease lifecycle after a transaction is prepared."""

    PREPARED = auto()
    """Lease is frozen but no GPU work has been launched."""
    IN_FLIGHT = auto()
    """A GPU step may read or write the leased pages."""
    COMPLETED = auto()
    """Terminal: the step succeeded and KV metadata was published."""
    ABORTED = auto()
    """Terminal: cancelled before launch; reservations were rolled back."""
    FAILED = auto()
    """Terminal: failed after launch; pages were abandoned to deferred free."""


class ReservationFailure(Enum):
    """Structured reasons a tentative reservation may fail.

    These codes let the scheduler react without seeing page internals.
    """

    NO_CAPACITY = auto()
    """Not enough free pages for the requested append."""
    PREFIX_NOT_RESIDENT = auto()
    """The cached prefix vanished or changed before attachment."""
    SEQUENCE_INVALID = auto()
    """The sequence handle is stale or out of range."""
    SEQUENCE_BUSY = auto()
    """The sequence is already in a transaction or execution lease."""
    REQUEST_TOO_LARGE = auto()
    """The final length exceeds max_sequence_tokens or total usable capacity."""
    TRANSACTION_INVALID = auto()
    """The transaction handle is stale or not open."""
    INVALID_TOKEN_COUNT = auto()
    """num_new_tokens is zero or negative."""


class ReleaseStatus(Enum):
    """Outcome of requesting a sequence release."""

    RELEASED = auto()
    """Request ownership was dropped immediately."""
    DEFERRED = auto()
    """Release was scheduled; it will run when the active step completes."""
