"""Structured outcomes for the memory-pressure paths.

The manager never returns bare counters from reclaim, eviction or preemption.
It returns one of these immutable results so a caller can distinguish "nothing
was reclaimable" from "the action gained capacity" without inspecting
internals, and so monitoring gets exact cumulative totals alongside the
attempt/progress pairs.  Each result validates its own consistency: a
``PROGRESSED`` status with zero reclaimed pages is rejected at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ayaka.handles import SequenceHandle


class PressureAction(Enum):
    """The single pressure-resolution step this result reports."""

    RECLAIM_DEFERRED = auto()
    """Merge/reclaim pages whose safe epoch already completed."""
    EVICT_PREFIX = auto()
    """Evict LRU cache-only pages to gain free capacity."""


class PressureStatus(Enum):
    """Whether one pressure action changed any capacity accounting."""

    NO_PROGRESS = auto()
    """Nothing was reclaimed or evicted; repeated calls are idempotent."""
    PROGRESSED = auto()
    """At least one page was evicted or reclaimed."""


@dataclass(frozen=True, slots=True)
class MemoryPressureResult:
    """Capacity-only result of one idempotent pressure action.

    All counts are non-negative; ``status`` must agree with the outcome
    counts, so a result claiming progress while reporting zero evictions and
    zero reclamations is rejected at construction.
    """

    action: PressureAction
    status: PressureStatus
    requested_pages: int
    """Additional free pages the caller asked the action to produce."""
    cache_pages_evicted: int
    """Cache-only pages released by the prefix cache."""
    pages_reclaimed: int
    """Pages returned to the free pool from deferred reclaim."""
    free_pages_before: int
    free_pages_after: int
    deferred_pages_before: int
    deferred_pages_after: int
    cached_pages_before: int
    cached_pages_after: int

    def __post_init__(self) -> None:
        counts = (
            self.requested_pages,
            self.cache_pages_evicted,
            self.pages_reclaimed,
            self.free_pages_before,
            self.free_pages_after,
            self.deferred_pages_before,
            self.deferred_pages_after,
            self.cached_pages_before,
            self.cached_pages_after,
        )
        if any(value < 0 for value in counts):
            raise ValueError("memory-pressure counts must be non-negative")
        progressed = self.cache_pages_evicted > 0 or self.pages_reclaimed > 0
        if (self.status is PressureStatus.PROGRESSED) != progressed:
            raise ValueError("memory-pressure status disagrees with its outcome counts")

    @property
    def made_progress(self) -> bool:
        """True when the action changed capacity accounting."""
        return self.status is PressureStatus.PROGRESSED

    @property
    def capacity_gained_pages(self) -> int:
        """Net free-page growth; clamped so preemption noise never goes negative."""
        return max(self.free_pages_after - self.free_pages_before, 0)


class PressureStopReason(Enum):
    """Why a bounded pressure-relief request stopped."""

    SATISFIED = auto()
    """The requested capacity was reached."""
    NO_PROGRESS = auto()
    """No action could change capacity; a retry cannot help without new state."""
    HOST_BACKPRESSURE = auto()
    """Demotion is blocked by the host watermark; eviction ran out of work."""
    ROUND_LIMIT = auto()
    """The policy round bound was reached with capacity still missing."""


@dataclass(frozen=True, slots=True)
class PressurePolicy:
    """Bounds and watermarks for one pressure-relief request.

    ``host_low_watermark_slots`` keeps free host slots in reserve so a
    promotion can land without an immediate demotion, which is the hysteresis
    that stops spill/promote churn under a persistent watermark.
    """

    max_rounds: int = 4
    max_pages_per_round: int = 64
    host_low_watermark_slots: int = 0

    def __post_init__(self) -> None:
        for name in ("max_rounds", "max_pages_per_round", "host_low_watermark_slots"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")


@dataclass(frozen=True, slots=True)
class PressureOutcome:
    """Aggregate result of one bounded pressure-relief sequence.

    The individual :class:`MemoryPressureResult` entries stay in order, so a
    caller can see exactly which actions ran and how each changed capacity.
    """

    requested_pages: int
    target_pages: int
    reached_pages: int
    results: tuple[MemoryPressureResult, ...]
    stop_reason: PressureStopReason
    host_demotions: int = 0

    def __post_init__(self) -> None:
        counts = (self.requested_pages, self.target_pages, self.reached_pages, self.host_demotions)
        if any(value < 0 for value in counts):
            raise ValueError("pressure-outcome counts must be non-negative")
        if not isinstance(self.stop_reason, PressureStopReason):
            raise TypeError("stop_reason must be a PressureStopReason")
        for result in self.results:
            if not isinstance(result, MemoryPressureResult):
                raise TypeError("pressure outcomes hold MemoryPressureResult entries")

    @property
    def made_progress(self) -> bool:
        return any(result.made_progress for result in self.results)

    @property
    def satisfied(self) -> bool:
        return self.stop_reason is PressureStopReason.SATISFIED

    @property
    def capacity_gained_pages(self) -> int:
        if not self.results:
            return 0
        return max(
            0,
            self.results[-1].free_pages_after - self.results[0].free_pages_before,
        )


class PreemptionStatus(Enum):
    """Logical outcome of dropping one request's resident KV."""

    RELEASED = auto()
    """Request ownership was dropped and pages were freed or deferred."""
    NO_RESIDENT_STATE = auto()
    """The sequence had no resident KV; the call was a harmless no-op."""
    BUSY = auto()
    """The sequence is busy (transaction/lease/writable) and was left intact."""


@dataclass(frozen=True, slots=True)
class SequencePreemptionResult:
    """Logical result of dropping one request's resident KV ownership.

    A non-``RELEASED`` outcome must report zero released/deferred state, so
    callers can treat a non-``RELEASED`` result as "nothing changed".
    """

    sequence: SequenceHandle
    status: PreemptionStatus
    released_tokens: int
    """Committed tokens whose KV was dropped (recompute will restore them)."""
    released_request_pages: int
    """Request-owned pages released across the sequence page table."""
    pages_reclaimed: int
    """Pages already returned to the free pool during this preemption."""
    pages_deferred: int
    """Pages queued for deferred free, awaiting their safe epoch."""
    free_pages_before: int
    free_pages_after: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, SequenceHandle):
            raise TypeError("sequence must be a SequenceHandle")
        counts = (
            self.released_tokens,
            self.released_request_pages,
            self.pages_reclaimed,
            self.pages_deferred,
            self.free_pages_before,
            self.free_pages_after,
        )
        if any(value < 0 for value in counts):
            raise ValueError("preemption counts must be non-negative")
        if self.status is not PreemptionStatus.RELEASED and (
            self.released_tokens
            or self.released_request_pages
            or self.pages_reclaimed
            or self.pages_deferred
        ):
            raise ValueError("a non-release preemption outcome cannot report released state")

    @property
    def made_progress(self) -> bool:
        """True only when resident KV was actually released."""
        return self.status is PreemptionStatus.RELEASED


@dataclass(frozen=True, slots=True)
class SequenceTruncationResult:
    """Logical result of releasing one sequence's committed suffix.

    Suffix release is not a pressure action: the caller asks for an exact new
    length. The result reports what left this sequence's table and how much
    capacity that returned immediately. ``released_request_pages`` counts the
    request references dropped, including pages a fork sibling, prefix cache,
    or pin still owns; those pages stay live under their remaining owners.
    """

    sequence: SequenceHandle
    retained_tokens: int
    """Committed length this sequence keeps (the requested boundary)."""
    released_tokens: int
    """Committed tokens removed from the tail."""
    released_request_pages: int
    """Request-owned page entries dropped from the sequence table."""
    pages_reclaimed: int
    """Released pages already returned to the free pool during this call."""
    pages_deferred: int
    """Released pages queued for deferred free, awaiting their safe epoch."""
    free_pages_before: int
    free_pages_after: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, SequenceHandle):
            raise TypeError("sequence must be a SequenceHandle")
        counts = (
            self.retained_tokens,
            self.released_tokens,
            self.released_request_pages,
            self.pages_reclaimed,
            self.pages_deferred,
            self.free_pages_before,
            self.free_pages_after,
        )
        if any(value < 0 for value in counts):
            raise ValueError("truncation counts must be non-negative")

    @property
    def released_any(self) -> bool:
        """True when the call changed the sequence's committed length."""
        return self.released_tokens > 0


@dataclass(frozen=True, slots=True)
class MemoryPressureMetrics:
    """Cumulative memory-owned counters; page counters count physical pages.

    Exposed to monitoring as exact totals (``kv_reclaims_total``,
    ``kv_prefix_evictions_total``) plus pressure-path attempt/progress pairs so
    operators can measure how often pressure handling actually gains capacity.
    """

    kv_reclaims_total: int
    """Cumulative pages returned to free via deferred reclaim."""
    kv_prefix_evictions_total: int
    """Cumulative cache-only pages evicted by the prefix cache."""
    pressure_reclaim_attempts_total: int
    pressure_reclaim_progress_total: int
    pressure_prefix_eviction_attempts_total: int
    pressure_prefix_eviction_progress_total: int
    pressure_preemptions_total: int


@dataclass(frozen=True, slots=True)
class PressureSnapshot:
    """One advisory capacity reading across the device and host tiers.

    A snapshot is a *report*, never a reservation: capacity decisions always
    re-validate through the authoritative prepare path, so a stale snapshot is
    safe by construction. ``generation`` is the allocator completion epoch and
    ``timestamp_ns`` the monotonic reading time; a policy that acts on a
    snapshot must treat both as advisory.

    Page counts partition physical capacity exactly:
    ``free + reserved + live + reclaim_pending + permanent == total``, and
    ``usable + permanent == total``. Shared pages are counted once under the
    ownership field that owns them, never summed into a second total.
    """

    generation: int
    timestamp_ns: int
    page_size: int
    total_pages: int
    usable_pages: int
    free_pages: int
    reserved_pages: int
    live_pages: int
    reclaim_pending_pages: int
    permanent_pages: int
    request_owned_pages: int
    cache_owned_pages: int
    shared_pages: int
    inflight_pages: int
    cache_evictable_pages: int
    open_transactions: int
    active_leases: int
    host_capacity_pages: int
    host_used_slots: int
    host_free_slots: int
    host_pinned: bool
    host_mirror_bytes: int
    host_pin_limit_bytes: int
    transfer_limit_bytes: int
    transfer_held_bytes: int
    inflight_transfers: int
    quarantined_blocks: int

    def __post_init__(self) -> None:
        counts = (
            self.generation,
            self.timestamp_ns,
            self.total_pages,
            self.usable_pages,
            self.free_pages,
            self.reserved_pages,
            self.live_pages,
            self.reclaim_pending_pages,
            self.permanent_pages,
            self.request_owned_pages,
            self.cache_owned_pages,
            self.shared_pages,
            self.inflight_pages,
            self.cache_evictable_pages,
            self.open_transactions,
            self.active_leases,
            self.host_capacity_pages,
            self.host_used_slots,
            self.host_free_slots,
            self.host_mirror_bytes,
            self.host_pin_limit_bytes,
            self.transfer_limit_bytes,
            self.transfer_held_bytes,
            self.inflight_transfers,
            self.quarantined_blocks,
        )
        if any(value < 0 for value in counts):
            raise ValueError("pressure-snapshot counts must be non-negative")
        if self.page_size < 1:
            raise ValueError("page_size must be positive")
        if self.usable_pages + self.permanent_pages != self.total_pages:
            raise ValueError("usable + permanent pages must equal total pages")
        if (
            self.free_pages
            + self.reserved_pages
            + self.live_pages
            + self.reclaim_pending_pages
            + self.permanent_pages
            != self.total_pages
        ):
            raise ValueError("the device page partition must sum to total pages")
        if self.host_used_slots + self.host_free_slots != self.host_capacity_pages:
            raise ValueError("host used + free slots must equal host capacity")
        if self.host_used_slots > self.host_capacity_pages:
            raise ValueError("host used slots exceed host capacity")
        if self.transfer_limit_bytes > 0 and self.transfer_held_bytes > self.transfer_limit_bytes:
            raise ValueError("transfer credits held exceed the transfer limit")
        if self.cache_evictable_pages > self.cache_owned_pages:
            raise ValueError("evictable pages cannot exceed cache-owned pages")
        if self.shared_pages > self.request_owned_pages:
            raise ValueError("shared pages cannot exceed request-owned pages")

    @property
    def available_pages(self) -> int:
        """Free pages plus every page the prepare path may reclaim or evict."""
        return self.free_pages + self.reclaim_pending_pages + self.cache_evictable_pages

    @property
    def cow_headroom_pages(self) -> int:
        """Pages a private-tail copy can take without waiting on eviction.

        A COW destination must be independent of cache ownership, so deferred
        and cache-only pages are excluded even though prepare may release them.
        """
        return self.free_pages

    @property
    def host_utilization(self) -> float:
        if self.host_capacity_pages == 0:
            return 0.0
        return self.host_used_slots / self.host_capacity_pages

    def as_dict(self) -> dict[str, int | bool | float]:
        """Flat mapping for status reporting and tests."""
        return {
            "generation": self.generation,
            "timestamp_ns": self.timestamp_ns,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "usable_pages": self.usable_pages,
            "free_pages": self.free_pages,
            "reserved_pages": self.reserved_pages,
            "live_pages": self.live_pages,
            "reclaim_pending_pages": self.reclaim_pending_pages,
            "permanent_pages": self.permanent_pages,
            "request_owned_pages": self.request_owned_pages,
            "cache_owned_pages": self.cache_owned_pages,
            "shared_pages": self.shared_pages,
            "inflight_pages": self.inflight_pages,
            "cache_evictable_pages": self.cache_evictable_pages,
            "open_transactions": self.open_transactions,
            "active_leases": self.active_leases,
            "available_pages": self.available_pages,
            "cow_headroom_pages": self.cow_headroom_pages,
            "host_capacity_pages": self.host_capacity_pages,
            "host_used_slots": self.host_used_slots,
            "host_free_slots": self.host_free_slots,
            "host_pinned": self.host_pinned,
            "host_mirror_bytes": self.host_mirror_bytes,
            "host_pin_limit_bytes": self.host_pin_limit_bytes,
            "host_utilization": self.host_utilization,
            "transfer_limit_bytes": self.transfer_limit_bytes,
            "transfer_held_bytes": self.transfer_held_bytes,
            "inflight_transfers": self.inflight_transfers,
            "quarantined_blocks": self.quarantined_blocks,
        }


@dataclass(frozen=True, slots=True)
class GroupedPressureSnapshot:
    """Per-group pressure readings for the grouped cache.

    Every group executes a step in parallel, so capacity is bound by the group
    with the least headroom. Byte-level host budgets are shared across groups
    (one transfer budget, one pin ceiling), so the aggregate takes the maximum
    rather than the sum for those fields.
    """

    generation: int
    timestamp_ns: int
    groups: tuple[PressureSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("a grouped pressure snapshot needs at least one group")
        if self.generation < 0 or self.timestamp_ns < 0:
            raise ValueError("grouped pressure snapshot counters must be non-negative")

    @property
    def page_size(self) -> int:
        """Conservative minimum across groups (a scheduler-facing estimate)."""
        return min(group.page_size for group in self.groups)

    @property
    def total_pages(self) -> int:
        return sum(group.total_pages for group in self.groups)

    @property
    def free_pages(self) -> int:
        return sum(group.free_pages for group in self.groups)

    @property
    def reclaim_pending_pages(self) -> int:
        return sum(group.reclaim_pending_pages for group in self.groups)

    @property
    def cache_evictable_pages(self) -> int:
        return sum(group.cache_evictable_pages for group in self.groups)

    @property
    def binding_available_pages(self) -> int:
        """The group with the least reclaimable page capacity."""
        return min(group.available_pages for group in self.groups)

    @property
    def binding_cow_headroom_pages(self) -> int:
        return min(group.cow_headroom_pages for group in self.groups)

    @property
    def host_capacity_pages(self) -> int:
        return sum(group.host_capacity_pages for group in self.groups)

    @property
    def host_used_slots(self) -> int:
        return sum(group.host_used_slots for group in self.groups)

    @property
    def host_mirror_bytes(self) -> int:
        return sum(group.host_mirror_bytes for group in self.groups)

    @property
    def host_pin_limit_bytes(self) -> int:
        return max(group.host_pin_limit_bytes for group in self.groups)

    @property
    def transfer_limit_bytes(self) -> int:
        return max(group.transfer_limit_bytes for group in self.groups)

    @property
    def transfer_held_bytes(self) -> int:
        return max(group.transfer_held_bytes for group in self.groups)

    @property
    def inflight_transfers(self) -> int:
        return sum(group.inflight_transfers for group in self.groups)

    @property
    def quarantined_blocks(self) -> int:
        return sum(group.quarantined_blocks for group in self.groups)
