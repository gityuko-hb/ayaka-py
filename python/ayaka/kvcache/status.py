"""Capacity/status report for one resident KV pool, sized for a UI slider.

Every number here is either measured from the live allocator snapshot or
derived from the same byte math the resize fit-check uses
(:mod:`ayaka.kvcache.resize`). The report is read-only: a caller may display
``max_pages`` as the largest capacity :func:`~ayaka.kvcache.resize.validate_resize`
would currently accept, and re-query after traffic drains.
"""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.kvcache.resize import (
    LEDGER_OVERHEAD_BYTES,
    minimum_pages,
    storage_bytes,
)
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec
from ayaka.kvcache.storage.layout import DEFAULT_KV_ALIGNMENT_BYTES
from ayaka.memory.views import MemorySnapshot

__all__ = ["CachePoolStatus", "CacheStatus", "build_cache_status"]


@dataclass(frozen=True, slots=True)
class CachePoolStatus:
    """One pool family's current and admissible capacity.

    Attributes:
        name: Group/pool name (``"default"`` for the homogeneous manager).
        page_size: Tokens per page.
        current_pages: Capacity the pool is serving with right now.
        min_pages: Smallest capacity that still holds one full model context
            plus the padding page.
        max_pages: Largest capacity the budget would currently accept, or None
            when the device budget is not measurable (CPU diagnostics).
        bytes_per_page: Payload bytes of one page across every layer.
        current_bytes: Reserved device bytes at ``current_pages``, padding and
            ledger overhead included.
        max_bytes: Reserved device bytes at ``max_pages``, or None.
        free_pages: Pages immediately allocatable.
        used_pages: Pages committed to requests/cache or awaiting reclamation.
        total_pages: All physical pages including the permanent padding page.
    """

    name: str
    page_size: int
    current_pages: int
    min_pages: int
    max_pages: int | None
    bytes_per_page: int
    current_bytes: int
    max_bytes: int | None
    free_pages: int
    used_pages: int
    total_pages: int

    def as_dict(self) -> dict[str, object]:
        """JSON-ready projection for the HTTP status endpoint."""
        return {
            "name": self.name,
            "page_size": self.page_size,
            "current_pages": self.current_pages,
            "min_pages": self.min_pages,
            "max_pages": self.max_pages,
            "bytes_per_page": self.bytes_per_page,
            "current_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "free_pages": self.free_pages,
            "used_pages": self.used_pages,
            "total_pages": self.total_pages,
        }


@dataclass(frozen=True, slots=True)
class CacheStatus:
    """Whole-runtime cache report across every resident pool."""

    pools: tuple[CachePoolStatus, ...]
    headroom_bytes: int | None
    safety_bytes: int

    @property
    def total_pages(self) -> int:
        """Current capacity summed across pools."""
        return sum(pool.current_pages for pool in self.pools)

    @property
    def free_pages(self) -> int:
        """Immediately allocatable pages summed across pools."""
        return sum(pool.free_pages for pool in self.pools)

    def as_dict(self) -> dict[str, object]:
        """JSON-ready projection for the HTTP status endpoint."""
        return {
            "headroom_bytes": self.headroom_bytes,
            "safety_bytes": self.safety_bytes,
            "total_pages": self.total_pages,
            "free_pages": self.free_pages,
            "pools": [pool.as_dict() for pool in self.pools],
        }


def build_cache_status(
    *,
    name: str,
    spec: BaseKVStorageSpec,
    snapshot: MemorySnapshot,
    max_sequence_tokens: int | None = None,
    headroom_bytes: int | None = None,
    safety_bytes: int = 0,
    budget_bytes: int | None = None,
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES,
    ledger_overhead_bytes: int = LEDGER_OVERHEAD_BYTES,
) -> CacheStatus:
    """Assemble a status report without touching allocator or storage state.

    Args:
        name: Pool name.
        spec: Live storage geometry.
        snapshot: Backend capacity snapshot for the same pool.
        max_sequence_tokens: Context length used for the floor.
        headroom_bytes: Measured headroom, or None when not measurable.
        safety_bytes: Reserve retained during a resize.
        budget_bytes: Frozen KV budget capping the admissible maximum.
        alignment_bytes: Alignment used by the materializer.
        ledger_overhead_bytes: Fixed bookkeeping bytes charged on top.

    Raises:
        ValueError: If ``safety_bytes`` is negative.
    """
    if safety_bytes < 0:
        raise ValueError("safety_bytes must be non-negative")
    current_pages = spec.capacity_pages
    floor = minimum_pages(spec.page_size, max_sequence_tokens)
    current_bytes = storage_bytes(
        spec,
        current_pages,
        alignment_bytes=alignment_bytes,
        ledger_overhead_bytes=ledger_overhead_bytes,
    )
    max_pages: int | None = None
    max_bytes: int | None = None
    if headroom_bytes is not None:
        budget = headroom_bytes + current_bytes - safety_bytes
        if budget_bytes is not None:
            budget = min(budget, budget_bytes)
        max_pages = _max_pages(
            spec,
            floor=floor,
            budget=budget,
            alignment_bytes=alignment_bytes,
            ledger_overhead_bytes=ledger_overhead_bytes,
        )
        max_bytes = storage_bytes(
            spec,
            max_pages,
            alignment_bytes=alignment_bytes,
            ledger_overhead_bytes=ledger_overhead_bytes,
        )
    pool = CachePoolStatus(
        name=name,
        page_size=spec.page_size,
        current_pages=current_pages,
        min_pages=floor,
        max_pages=max_pages,
        bytes_per_page=spec.bytes_per_page,
        current_bytes=current_bytes,
        max_bytes=max_bytes,
        free_pages=snapshot.free_pages,
        used_pages=snapshot.total_pages - snapshot.free_pages,
        total_pages=snapshot.total_pages,
    )
    return CacheStatus(
        pools=(pool,),
        headroom_bytes=headroom_bytes,
        safety_bytes=safety_bytes,
    )


def _max_pages(
    spec: BaseKVStorageSpec,
    *,
    floor: int,
    budget: int,
    alignment_bytes: int,
    ledger_overhead_bytes: int,
) -> int:
    """Largest capacity whose aligned bytes fit ``budget``.

    Doubling search followed by a binary search over the monotone cost
    function; the floor is returned even when the pool is already over budget so
    a UI never shows a maximum below the current minimum.
    """

    def cost(pages: int) -> int:
        return storage_bytes(
            spec,
            pages,
            alignment_bytes=alignment_bytes,
            ledger_overhead_bytes=ledger_overhead_bytes,
        )

    if cost(floor) > budget:
        return floor
    high = floor
    while cost(high * 2) <= budget and high < (1 << 40):
        high *= 2
    low = high
    high = high * 2
    while low < high:
        mid = (low + high + 1) // 2
        if cost(mid) <= budget:
            low = mid
        else:
            high = mid - 1
    return low
