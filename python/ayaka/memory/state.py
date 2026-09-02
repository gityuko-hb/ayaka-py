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

