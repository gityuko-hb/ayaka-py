"""Caching allocator — size-class byte pool over one raw source.

The shape of the problem: an inference engine allocates a small number of
*shapes* over and over. Every decode step wants the same logits buffer, the same
attention workspace, the same all-reduce staging buffer. Returning those to the
driver and re-requesting them is the worst possible trade, because
``cudaMalloc`` and ``cudaFree`` both synchronise the device — a free in the
middle of a step stalls every stream on it.

So: never return to the source. Round the request up to a size class, keep freed
blocks on a per-class free list, and hand the same block back next step.

Size classes are the design decision that matters. Rounding to powers of two is
simple and wastes up to 50%; at a 2 GiB activation footprint that is a gigabyte
of nothing. These classes are 256 B granular below 8 KiB (where the count of
distinct sizes is large and the sizes are small) and geometric at 1.25x above
(where the sizes are large and the count is small), which bounds internal waste
at 25% in the worst case and around 10% in practice, while keeping the number of
classes small enough that free lists stay warm.

Alignment is the second one. A caller may ask for a pointer aligned more
strongly than the source's floor (a 4 KiB page boundary, say), and a block
allocated without that in mind cannot be made to satisfy it after the fact. So
the class is chosen for ``nbytes + (alignment - SOURCE_ALIGNMENT)`` and the
returned region starts at the first suitably aligned byte inside it. The extra
bytes are visible as class rounding — the ledger never under-counts — and the
region itself reports the caller's ``nbytes`` so a bounds check stays strict.

Rule 10 is enforced structurally: :meth:`CachingAllocator.allocate` has no
default owner. There is no way to allocate anonymously, so "how much HBM is
activation versus KV" is always answerable.

KV paging is deliberately not this allocator's job. A KV page must survive until
its safe completion epoch, while a scratch block is reusable the moment its step
releases it — that difference is why :mod:`ayaka.memory.allocator` exists and
why the two never share a free path.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from typing import Any

from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.memory.region import MemoryRegion, first_aligned_offset
from ayaka.memory.source import SOURCE_ALIGNMENT, RawMemorySource
from ayaka.types import MemoryOwner, MemoryTier
from ayaka.utils.math_utils import align_down, align_up

__all__ = [
    "SIZE_CLASSES",
    "AllocatorStats",
    "CachingAllocator",
    "size_class",
]


def _build_size_classes() -> tuple[int, ...]:
    classes: list[int] = []
    # Fine and linear where allocations are small and numerous.
    step = 256
    value = 256
    while value < 8192:
        classes.append(value)
        value += step
    # Geometric where they are large and few. 1.25x bounds internal waste at
    # 20% rather than the 50% a power-of-two ladder allows.
    value = 8192
    while value <= (16 << 30):
        classes.append(value)
        value = max(value + 256, align_down(int(value * 1.25), 256))
    return tuple(classes)


SIZE_CLASSES: tuple[int, ...] = _build_size_classes()


def size_class(nbytes: int) -> int:
    """Smallest class that holds ``nbytes``.

    Binary search rather than a loop: this is on the allocation path, and the
    ladder has a few hundred entries.
    """
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")
    lo, hi = 0, len(SIZE_CLASSES)
    while lo < hi:
        mid = (lo + hi) // 2
        if SIZE_CLASSES[mid] < nbytes:
            lo = mid + 1
        else:
            hi = mid
    if lo == len(SIZE_CLASSES):
        # Beyond the ladder: round to 2 MiB, the large-page granularity, so a
        # 30 GiB weight buffer does not get a class of its own on every call.
        return align_up(nbytes, 2 << 20)
    return SIZE_CLASSES[lo]


@dataclass(frozen=True, slots=True)
class AllocatorStats:
    """What the allocator is doing, in the terms that diagnose it."""

    live_bytes: int
    """Bytes handed out and not yet freed, at class-rounded size.

    Class rounding includes any bytes held so a strong alignment request can
    start on the right boundary, because those bytes left the source too.
    """
    requested_bytes: int
    """Bytes callers actually asked for; the gap is internal slack."""
    cached_bytes: int
    """Bytes on free lists: held from the source, available for reuse."""
    held_bytes: int
    """live + cached — everything taken from the source and not returned."""
    peak_live_bytes: int
    held_at_peak_bytes: int
    """held_bytes at the instant live_bytes was highest."""
    num_live: int
    num_cached: int
    num_allocs: int
    num_reuses: int
    num_source_allocs: int
    num_returns: int
    """Surplus blocks handed back to the source rather than cached deeper."""

    @property
    def internal_fragmentation(self) -> float:
        """Waste from rounding to a size class. Bounded by the ladder."""
        if not self.live_bytes:
            return 0.0
        return 1.0 - self.requested_bytes / self.live_bytes

    @property
    def peak_fragmentation(self) -> float:
        """Bytes held but not in use, measured **at peak demand**.

        This is the fragmentation number that means something. Measuring it at
        an arbitrary moment is meaningless — right after a drain, live is zero
        and every allocator looks 100% fragmented. What matters is the instant
        the engine needs memory most: how much was held and not usable then.
        """
        if not self.held_at_peak_bytes:
            return 0.0
        return 1.0 - self.peak_live_bytes / self.held_at_peak_bytes

    @property
    def overhead_ratio(self) -> float:
        """Final pool size relative to peak demand.

        NOT fragmentation, and it is worth being precise about why. A per-class
        cache ends up holding the *sum* of each class's peak concurrency, while
        the working set is the *peak of the sum*; on a multi-modal workload
        those differ by ~10% purely because the per-class maxima do not co-occur.
        That is a property of the arrival pattern, not waste the allocator
        created, and it is bounded by the number of size classes in play rather
        than growing with time. Use :attr:`peak_fragmentation` to judge the
        allocator; use this to notice unbounded growth.
        """
        if not self.peak_live_bytes:
            return 0.0
        return self.held_bytes / self.peak_live_bytes - 1.0

    @property
    def reuse_rate(self) -> float:
        return self.num_reuses / self.num_allocs if self.num_allocs else 0.0


@dataclass(slots=True)
class _Block:
    """One block taken from the source, plus the last region handed out from it.

    ``base_ptr`` is the raw pointer the source returned; the region's
    ``base_ptr`` may start later, at the first byte that satisfies the caller's
    alignment. The region is kept so ``free`` can subtract the request that is
    actually live without trusting the caller to hand back the same object.
    """

    base_ptr: int
    keepalive: Any
    class_bytes: int
    requested_bytes: int = 0
    region: MemoryRegion | None = None


class CachingAllocator:
    """Size-class caching allocator over one :class:`RawMemorySource`.

    Thread-safe. Freed blocks stay on class free lists and are never returned to
    the source, so a steady-state engine stops calling the source entirely.
    """

    __slots__ = (
        "_cached_bytes",
        "_device_index",
        "_free_lists",
        "_held_at_peak",
        "_ids",
        "_ledger",
        "_ledger_label",
        "_live",
        "_lock",
        "_max_cached_per_class",
        "_num_allocs",
        "_num_returns",
        "_num_reuses",
        "_num_source_allocs",
        "_peak_live",
        "_requested",
        "_source",
        "_tier",
    )

    def __init__(
        self,
        source: RawMemorySource,
        *,
        ledger: MemoryLedger | None = None,
        tier: MemoryTier = MemoryTier.DEVICE,
        device_index: int = 0,
        ledger_label: str = "",
        max_cached_per_class: int = 0,
    ) -> None:
        self._source = source
        self._ledger = ledger
        self._tier = tier
        self._device_index = device_index
        self._lock = threading.RLock()
        self._free_lists: dict[int, list[_Block]] = {}
        self._live: dict[int, _Block] = {}
        self._ids = itertools.count(1)
        self._requested = 0
        self._cached_bytes = 0
        self._peak_live = 0
        self._held_at_peak = 0
        self._num_allocs = 0
        self._num_reuses = 0
        self._num_source_allocs = 0
        self._num_returns = 0
        if max_cached_per_class < 0:
            raise ValueError("max_cached_per_class must be >= 0 (0 = unlimited)")
        self._max_cached_per_class = max_cached_per_class
        self._ledger_label = ledger_label or f"pool.{source.name}.{device_index}"
        if self._ledger is not None:
            # admit(), not reserve(): the pool starts at zero bytes, so there is
            # nothing that can fail between deciding and having.
            self._ledger.admit(
                Reservation.backed(
                    MemoryOwner.WORKSPACE,
                    self._ledger_label,
                    0,
                    tier=tier,
                    device_index=device_index,
                )
            )

    # ── allocate / free ──────────────────────────────────────────────────────

    def allocate(
        self,
        nbytes: int,
        *,
        owner: MemoryOwner,
        alignment: int = 256,
    ) -> MemoryRegion:
        """Hand out a region. ``owner`` is required — Rule 10 has no default.

        The returned region's ``nbytes`` is what was *requested*, not the class
        size: a caller that writes past its request is out of bounds even though
        the block behind it is larger, and reporting the class size would make
        that overrun invisible. ``class_bytes`` carries the block size so a
        ledger/arena reconciliation can tell rounding from a leak.

        ``alignment`` may exceed what the source guarantees; the allocator pays
        for that with a larger class and an offset into the block, never with a
        misaligned pointer.
        """
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        if alignment & (alignment - 1):
            raise ValueError(f"alignment {alignment} is not a power of two")

        over = alignment - SOURCE_ALIGNMENT if alignment > SOURCE_ALIGNMENT else 0
        wanted = size_class(nbytes + over)
        with self._lock:
            self._num_allocs += 1
            block = self._take_cached(wanted)
            if block is not None:
                self._cached_bytes -= block.class_bytes
                self._num_reuses += 1
            else:
                keepalive, base_ptr = self._source.alloc(wanted)
                if base_ptr % SOURCE_ALIGNMENT:
                    # The class was sized on this promise; fail closed and give
                    # the block back rather than hand out what cannot be fixed.
                    self._source.free(keepalive)
                    raise MemoryError(
                        f"source {self._source.name!r} returned {base_ptr:#x}, not "
                        f"aligned to the {SOURCE_ALIGNMENT}-byte floor"
                    )
                self._num_source_allocs += 1
                block = _Block(base_ptr=base_ptr, keepalive=keepalive, class_bytes=wanted)

            offset = first_aligned_offset(block.base_ptr, alignment)
            if offset + nbytes > block.class_bytes:  # pragma: no cover - sizing guard
                raise MemoryError(
                    f"block of {block.class_bytes} B cannot hold {nbytes} B at "
                    f"{alignment}-byte alignment (offset {offset}); the source or "
                    "the size-class ladder disagreed with the request"
                )

            region = MemoryRegion(
                region_id=next(self._ids),
                base_ptr=block.base_ptr + offset,
                nbytes=nbytes,
                tier=self._tier,
                owner=owner,
                device_index=self._device_index,
                alignment=alignment,
                class_bytes=block.class_bytes,
            )
            block.region = region
            block.requested_bytes = nbytes
            self._live[region.region_id] = block
            self._requested += nbytes
            live = self._live_bytes()
            if live > self._peak_live:
                self._peak_live = live
                self._held_at_peak = live + self._cached_bytes
            self._sync_ledger()
            return region

    def free(self, region: MemoryRegion) -> None:
        with self._lock:
            block = self._live.pop(region.region_id, None)
            if block is None:
                raise RuntimeError(
                    f"free of region {region.region_id} this allocator does not hold "
                    "(double free, or a region from another pool)"
                )
            # Subtract what the live allocation actually asked for, not what the
            # caller passed in: a sliced region shares the parent's id, and the
            # accounting must not drift because of it.
            self._requested -= block.requested_bytes
            pool = self._free_lists.setdefault(block.class_bytes, [])
            if self._max_cached_per_class and len(pool) >= self._max_cached_per_class:
                # Hand it back rather than deepening the cache.
                #
                # This is the one thing that beats segregated free lists at their
                # own weakness. We cannot coalesce: the source returns
                # independent blocks with gaps between them, so two adjacent
                # class-4 KiB blocks can never re-form a class-8 KiB block, and
                # splitting a large block into small ones is irreversible. But
                # the source *is* a pool -- torch's caching allocator, which does
                # split and coalesce inside its own arena. Giving a surplus block
                # back lets the layer that can reuse it across sizes do so.
                #
                # The cap is off by default (0 = unlimited) because there is no
                # setting that is right for every workload, and the measured
                # trade is steep. On a 4-mode churn at 10k cycles:
                #
                #   depth   peak fragmentation   reuse rate
                #     2           2.3%              71%
                #     4           4.5%              85%
                #     8           6.8%              94%
                #   unlimited    10.0%              98%
                #
                # Ayaka's own workload does not need it: the workspace manager
                # collapses every per-step scratch buffer into ONE growing
                # allocation, and weights, KV and comm buffers are one-shot. What
                # reaches this allocator is a handful of long-lived blocks, where
                # unlimited caching fragments at 0% and pays nothing. The knob is
                # for a caller whose workload is genuinely multi-modal churn, and
                # trim() is the pressure valve for everyone else.
                block.region = None
                self._source.free(block.keepalive)
                self._num_returns += 1
            else:
                pool.append(block)
                self._cached_bytes += block.class_bytes
            self._sync_ledger()

    def _take_cached(self, wanted: int) -> _Block | None:
        """Exact class first, then a bounded search upward.

        Without the upward search the pool holds the *sum* of each class's peak
        concurrency, while the working set is only the *peak of the sum* — those
        differ by 10% on a mixed workload, and the gap is memory held and
        unusable rather than anything the caller did wrong.

        The bound is purely relative -- 1.5x the request, never an absolute
        slack.  An absolute term looks harmless and is not: "or within 64 KiB"
        lets a 4 KiB request consume a cached 66 KiB block, which trades a
        little cross-class waste for 94% internal waste and inflates every
        "live bytes" number that depends on it.
        """
        pool = self._free_lists.get(wanted)
        if pool:
            return pool.pop()
        ceiling = wanted + wanted // 2
        for candidate in sorted(self._free_lists):
            if candidate <= wanted:
                continue
            if candidate > ceiling:
                break
            pool = self._free_lists[candidate]
            if pool:
                return pool.pop()
        return None

    def _live_bytes(self) -> int:
        return sum(b.class_bytes for b in self._live.values())

    # ── maintenance ──────────────────────────────────────────────────────────

    def trim(self) -> int:
        """Return cached blocks to the source. Returns bytes released.

        Only worth calling under real pressure, and never inside a step: for
        ``TorchDeviceSource`` this drops references, and reclaiming them is
        torch's business; for a raw CUDA source it would be ``cudaFree``, which
        synchronises the device.
        """
        with self._lock:
            released = 0
            for pool in self._free_lists.values():
                for block in pool:
                    self._source.free(block.keepalive)
                    released += block.class_bytes
                pool.clear()
            self._cached_bytes = 0
            self._sync_ledger()
            return released

    def close(self) -> None:
        """Release everything. Refuses while blocks are live."""
        with self._lock:
            if self._live:
                raise RuntimeError(f"close() with {len(self._live)} live regions; free them first")
            self.trim()
            if self._ledger is not None:
                self._ledger.release(self._ledger_label)

    def _sync_ledger(self) -> None:
        """The pool is ONE ledger entry, restated — never a sum of regions.

        Summing regions would double-count against what the source already holds,
        which for torch is precisely the ``memory_allocated`` vs
        ``memory_reserved`` trap the ledger exists to avoid.
        """
        if self._ledger is None:
            return
        self._ledger.update(self._ledger_label, backed_bytes=self._held_bytes())

    def _held_bytes(self) -> int:
        return self._live_bytes() + self._cached_bytes

    # ── introspection ────────────────────────────────────────────────────────

    def stats(self) -> AllocatorStats:
        with self._lock:
            live = self._live_bytes()
            return AllocatorStats(
                live_bytes=live,
                requested_bytes=self._requested,
                cached_bytes=self._cached_bytes,
                held_bytes=live + self._cached_bytes,
                peak_live_bytes=self._peak_live,
                held_at_peak_bytes=self._held_at_peak,
                num_live=len(self._live),
                num_cached=sum(len(p) for p in self._free_lists.values()),
                num_allocs=self._num_allocs,
                num_reuses=self._num_reuses,
                num_source_allocs=self._num_source_allocs,
                num_returns=self._num_returns,
            )

    def bytes_by_owner(self) -> dict[MemoryOwner, int]:
        with self._lock:
            totals: dict[MemoryOwner, int] = {}
            for block in self._live.values():
                region = block.region
                if region is None:  # pragma: no cover - a live block always has one
                    continue
                owner = region.owner
                totals[owner] = totals.get(owner, 0) + block.class_bytes
            return totals

    @property
    def num_live(self) -> int:
        with self._lock:
            return len(self._live)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"<CachingAllocator {self._source.name} live={s.live_bytes >> 20}MiB "
            f"cached={s.cached_bytes >> 20}MiB reuse={s.reuse_rate:.0%}>"
        )
