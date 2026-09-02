from __future__ import annotations

from dataclasses import dataclass

from ayaka.handles import KVPageHandle, KVReservationHandle, PrefixHandle, SequenceHandle, StepMemoryLeaseHandle

@dataclass(frozen=True, slots=True)
class KVWriteSlot:
    """One kernel-facing write target for a single token.

    ``physical_page`` is the storage page index, ``page_offset`` the position
    within that page, and ``flat_slot`` the linear ``page * page_size + offset``
    slot address used for scatter writes.
    """
    
    logical_position: int
    """Logical sequence position the token is written to."""
    physical_page: int
    """Storage-side physical page index (no generation; kernel identity)."""
    page_offset: int
    """Token offset inside the physical page; in ``[0, page_size)``."""
    flat_slot: int
    """Linear storage slot address for the write."""

@dataclass(frozen=True, slots=True)
class SequenceExecutionView:
    """Per-sequence portion of an execution lease, ready for batch building."""
    sequence: SequenceHandle
    reservation: KVReservationHandle
    base_committed_tokens: int
    """Attention-visible start position (includes attached prefix tokens)."""
    num_reserved_tokens: int
    """Tokens the step is allowed to write for this sequence."""
    block_table: tuple[int, ...]
    """Physical page IDs per logical block, in logical-block order."""
    write_slots: tuple[KVWriteSlot, ...]
    """One slot per reserved token in logical-position order."""
    
@dataclass(frozen=True, slots=True)
class ExecutionMemoryView: 
    """Full execution view for one prepared step."""

    step_id: int
    lease: StepMemoryLeaseHandle
    sequences: tuple[SequenceExecutionView, ...]
    page_size: int
    padding_page: int
    """Physical page reserved for graph-padding writes; never a live page."""
    padding_slot: int
    """Flat slot address of the padding page's first position."""
    
@dataclass(frozen=True, slots=True)
class SequenceCacheView:
    """Read-only attention contract for one sequence's committed KV."""

    sequence: SequenceHandle
    committed_tokens: int
    """Tokens whose KV is safe for attention to read."""
    pages: tuple[KVPageHandle, ...]
    """Generation-safe handles of committed pages, in logical-block order."""
    block_table: tuple[int, ...]
    """Physical page IDs for the same blocks; the kernel-facing page table."""


    
@dataclass(frozen=True, slots=True)
class CacheView:
    """Read-only attention contract for a set of sequences."""

    page_size: int
    sequences: tuple[SequenceCacheView, ...]
    padding_page: int
    padding_slot: int

@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Approximate scheduler-facing capacity feedback.

    Page counts partition physical capacity; token-derived fields are
    heuristic (e.g. ``immediately_available_tokens`` assumes the tail page
    fills entirely). Actual allocations are always validated transitionally.
    """

    
    total_pages: int
    """All physical pages including the permanent padding page."""
    usable_pages: int
    """Pages allocatable by the runtime (total minus permanent)."""
    free_pages: int
    """Pages currently in the ready-free queue."""
    reserved_pages: int
    """Pages held by open tentative reservations."""
    live_pages: int
    """Pages committed with request/cache ownership."""
    deferred_free_pages: int
    """Pages in RECLAIM_PENDING awaiting their safe epoch."""
    permanent_pages: int
    """Padding page(s) excluded from usable capacity."""

    request_owned_pages: int
    """Pages with at least one request reference."""
    cache_owned_pages: int
    """Pages with at least one prefix-cache reference."""
    shared_pages: int
    """Pages with more than one durable owner (request + cache)."""
    inflight_pages: int
    """Pages currently covered by an in-flight execution lease."""

    page_size: int
    committed_tokens: int
    """Sum of committed tokens across live sequences."""
    internal_fragmentation_tokens: int
    """Tail-page waste across sequences: sum of (P - len mod P) mod P."""
    live_sequences: int
    open_transactions: int
    active_leases: int
    cached_prefix_blocks: int
    """Number of cached prefix blocks (pages held by the cache)."""
    cached_prefix_entries: int
    """Number of retained terminal prefix entries."""
    cached_prefix_tokens: int
    pending_prefix_matches: int
    """Outstanding one-shot prefix-match handles issued to the scheduler."""
    cache_evictable_pages: int
    """Pages the prefix cache could release under pressure (cache-only refs)."""

    @property
    def immediately_available_tokens(self) -> int:
        """Tokens instantly allocatable without reclaiming or evicting."""
        return self.free_pages * self.page_size

    @property
    def immediately_available_pages(self) -> int:
        """Pages instantly allocatable without reclaiming or evicting."""
        return self.free_pages

@dataclass(frozen=True, slots=True)
class LeakReport:
    """Full accounting report for deterministic leak detection.

    ``clean`` is true only when every sequence, transaction, lease, prefix
    entry, and page has returned to the idle baseline described in the runtime
    memory contract.
    """
    
    snapshot: MemorySnapshot
    live_sequence_handles: tuple[SequenceHandle, ...]
    open_transaction_ids: tuple[int, ...]
    active_lease_ids: tuple[int, ...]
    cached_prefix_handles: tuple[PrefixHandle, ...]
    
    @property
    def clean(self) -> bool:
        """True when all usable pages are free and no ownership remains."""
        return (
            not self.live_sequence_handles
            and not self.open_transaction_ids
            and not self.active_lease_ids
            and not self.cached_prefix_handles
            and self.snapshot.pending_prefix_matches == 0
            and self.snapshot.cache_owned_pages == 0
            and self.snapshot.reserved_pages == 0
            and self.snapshot.live_pages == 0
            and self.snapshot.deferred_free_pages == 0
            and self.snapshot.free_pages == self.snapshot.usable_pages
        )
