from __future__ import annotations

from dataclasses import dataclass

from ayaka.handles import (
    KVPageHandle,
    KVReservationHandle,
    PrefixHandle,
    SequenceHandle,
    StepMemoryLeaseHandle,
)


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
class GroupKVWriteSlot:
    """One kernel-facing write target for one token in one group.

    Extends ``KVWriteSlot`` with the group-local ``logical_block`` so each
    group's execution view can build its own block table and slot mapping.
    """

    logical_position: int
    """Logical sequence position the token is written to."""
    logical_block: int
    """Sequence-local block index for this position."""
    physical_page: int
    """Group-local physical page index (group page namespaces are independent)."""
    page_offset: int
    """Token offset inside the physical page; in ``[0, page_size)``."""
    flat_slot: int
    """Linear slot address within this group's storage."""


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
class KVPageCopy:
    """Immutable COW copy under the enclosing lease, before any destination writer."""

    source: KVPageHandle
    destination: KVPageHandle
    valid_tokens: int
    group_name: str = "default"


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
    copies: tuple[KVPageCopy, ...] = ()


@dataclass(frozen=True, slots=True)
class KVCacheGroupExecutionView:
    """Per-group execution metadata for one sequence in a prepared lease.

    Exposes each group's retention window: ``retained_token_start/stop`` is the
    group's full live interval and ``attention_token_start`` the start after
    retention for the first query position. Evicted leading blocks resolve to
    the group's shared zeroed padding page.
    """

    group_name: str
    layer_ids: tuple[int, ...]
    storage_kind: str
    dtype: str
    page_size: int
    logical_blocks: tuple[int, ...]
    """Sequence-local block indexes covered by this group's block table."""
    block_table: tuple[int, ...]
    """Group-local physical page IDs in the same block order."""
    write_slots: tuple[GroupKVWriteSlot, ...]
    attention_token_start: int
    attention_token_stop: int
    retained_token_start: int
    retained_token_stop: int
    padding_page: int
    padding_slot: int


@dataclass(frozen=True, slots=True)
class GroupedSequenceExecutionView:
    """Per-sequence portion of a grouped execution lease."""

    sequence: SequenceHandle
    reservation: KVReservationHandle
    base_committed_tokens: int
    num_reserved_tokens: int
    groups: tuple[KVCacheGroupExecutionView, ...]


@dataclass(frozen=True, slots=True)
class GroupedExecutionMemoryView:
    """Full grouped execution view for one prepared step."""

    step_id: int
    lease: StepMemoryLeaseHandle
    sequences: tuple[GroupedSequenceExecutionView, ...]
    copies: tuple[KVPageCopy, ...] = ()


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


@dataclass(frozen=True, slots=True)
class KVCacheGroupSnapshot:
    """Capacity accounting for one cache group."""

    group_name: str
    total_pages: int
    usable_pages: int
    free_pages: int
    reserved_pages: int
    live_pages: int
    deferred_free_pages: int
    permanent_pages: int
    request_owned_pages: int
    inflight_pages: int
    reclaimed_pages_total: int
    page_size: int
    bytes_per_page: int


@dataclass(frozen=True, slots=True)
class GroupedMemorySnapshot:
    """Cross-group capacity aggregates plus per-group accounting.

    Sum properties aggregate the groups; ``page_size`` is the conservative
    minimum across groups (a scheduler-facing estimate only), and
    ``immediately_available_tokens`` is the minimum across groups because
    every group must have capacity for a step.
    """

    groups: tuple[KVCacheGroupSnapshot, ...]
    committed_tokens: int
    live_sequences: int
    open_transactions: int
    active_leases: int

    @property
    def total_pages(self) -> int:
        return sum(group.total_pages for group in self.groups)

    @property
    def usable_pages(self) -> int:
        return sum(group.usable_pages for group in self.groups)

    @property
    def free_pages(self) -> int:
        return sum(group.free_pages for group in self.groups)

    @property
    def reserved_pages(self) -> int:
        return sum(group.reserved_pages for group in self.groups)

    @property
    def live_pages(self) -> int:
        return sum(group.live_pages for group in self.groups)

    @property
    def deferred_free_pages(self) -> int:
        return sum(group.deferred_free_pages for group in self.groups)

    @property
    def permanent_pages(self) -> int:
        return sum(group.permanent_pages for group in self.groups)

    @property
    def request_owned_pages(self) -> int:
        return sum(group.request_owned_pages for group in self.groups)

    @property
    def inflight_pages(self) -> int:
        return sum(group.inflight_pages for group in self.groups)

    @property
    def immediately_available_tokens(self) -> int:
        """Minimum free-token capacity across groups.

        A step writes every group in parallel, so the binding constraint is
        the group with the least headroom.
        """
        return min(
            (group.free_pages * group.page_size for group in self.groups),
            default=0,
        )

    @property
    def page_size(self) -> int:
        """Conservative scheduler-facing page size estimate across all groups.

        Per-group geometry stays inside execution views; the scheduler only
        needs one deterministic page size for capacity heuristics.
        """

        return min(group.page_size for group in self.groups)


@dataclass(frozen=True, slots=True)
class GroupedLeakReport:
    """Full accounting report for grouped leak detection.

    ``clean`` requires every group's pages to return to the free pool after
    deferred reclaim, plus no live sequences, transactions, or leases.
    """

    snapshot: GroupedMemorySnapshot
    live_sequence_handles: tuple[SequenceHandle, ...]
    open_transaction_ids: tuple[int, ...]
    active_lease_ids: tuple[int, ...]

    @property
    def clean(self) -> bool:
        return (
            not self.live_sequence_handles
            and not self.open_transaction_ids
            and not self.active_lease_ids
            and self.snapshot.reserved_pages == 0
            and self.snapshot.live_pages == 0
            and self.snapshot.deferred_free_pages == 0
            and all(group.free_pages == group.usable_pages for group in self.snapshot.groups)
        )
