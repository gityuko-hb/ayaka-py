"""Mutable per-page allocator metadata.

Split out of the protocol-level lifecycle enums on purpose: the enums are
contract vocabulary that crosses into the scheduler, while ``PageMetadata`` is
allocator-private state that nothing outside :mod:`ayaka.cache.block` may
mutate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ayaka.handles import PhysicalPageId
from ayaka.kv.state import PageAllocationState, PageResidency


@dataclass(slots=True)
class PageMetadata:
    """Authoritative metadata for one physical page slot.

    Refcounts are split into four ownership dimensions: ``request_refs``
    (sequence page tables), ``cache_refs`` (prefix cache), ``reservation_refs``
    (tentative transactions), and ``inflight_refs`` (launched steps). A page
    can only return to ``FREE`` with every counter at zero after its safe
    completion epoch. Only memory-manager methods may mutate these counters.
    """
    
    generation: int
    """Generation of the current ownership lifetime; 0 means never used."""
    physical_id: PhysicalPageId
    """Stable storage index; the handle generation, not the ID, tracks reuse."""
    allocation_state: PageAllocationState = PageAllocationState.FREE
    """Current allocation lifecycle state; see ``PageAllocationState``."""
    residency: PageResidency = PageResidency.DEVICE
    """Current storage-tier placement; v1 only uses ``DEVICE``."""

    request_refs: int = 0
    """Sequences whose committed page tables reference this page."""
    cache_refs: int = 0
    """Prefix-cache nodes that own this page (one per cached chain position)."""
    reservation_refs: int = 0
    """Tentative reservations that own this page; at most 1 in v1."""
    inflight_refs: int = 0
    """Launched execution steps that may still read or write this page."""
    pin_refs: int = 0
    """Reference count for explicit pins (e.g. static system prompts, active LoRA caches)."""

    valid_tokens: int = 0
    """Number of valid KV tokens in this page; bounded by the page size."""
    last_access_epoch: int = 0
    """Epoch of the most recent allocation/release touch (LRU bookkeeping)."""
    pending_free_epoch: int | None = None
    """Earliest epoch at which this unowned page may safely become free."""

    @property
    def is_pinned(self) -> bool:
        """Check whether the page is pinned in memory."""
        return self.pin_refs > 0

    @property
    def is_evictable(self) -> bool:
        """Conditions for safely selecting this page for eviction (LRU algorithm)."""
        return (
            self.allocation_state == PageAllocationState.LIVE
            and not self.is_pinned
            and self.request_refs == 0
            and self.reservation_refs == 0
            and self.cache_refs > 0
        )
        
    @property
    def can_mutate(self) -> bool:
        """Directly writing a new token is permitted only if it is neither shared nor pinned."""
        return (
            self.request_refs == 1
            and self.cache_refs == 0
            and self.reservation_refs <= 1
            and not self.is_pinned
        )

    @property
    def ownership_refs(self) -> int:
        """Ownership refs without transient step ownership."""
        return (
            self.request_refs
            + self.cache_refs
            + self.reservation_refs
            + self.pin_refs
        )
        
    @property
    def total_refs(self) -> int:
        """Every ref dimension; must be zero before the page can be freed."""
        return self.ownership_refs + self.inflight_refs

    @property
    def is_shared(self) -> bool:
        """True when more than one durable owner (request or cache) exists."""
        return self.request_refs + self.cache_refs > 1
    
    def pin(self) -> None:
        """Lock page, prevent LRU eviction and disable in-place mutation."""
        self.pin_refs += 1
        
    def unpin(self) -> None: 
        """Unlock page"""
        if self.pin_refs <= 0:
            raise ValueError(f"Page {self.physical_id} is not pinned.")
        self.pin_refs -= 1
        
    def snapshot(self) -> PageMetadata:
        """Return a copy safe to inspect outside the allocator lock."""
        return replace(self)