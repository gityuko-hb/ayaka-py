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
