from __future__ import annotations

import threading
from dataclasses import dataclass

from ayaka.memory.region import MemoryRegion
from ayaka.types import MemoryOwner

__all__ = ["Arena", "ArenaExhausted", "ArenaStats", "SlabAllocator", "SlabStats"]


class ArenaExhausted(MemoryError):
    """The arena's backing region cannot satisfy a request.

    Distinct from a general ``MemoryError`` because the remedy is different: an
    arena is sized once for a worst-case step, so exhaustion means the step is
    bigger than the profile predicted, not that the device is full.
    """


@dataclass(frozen=True, slots=True)
class ArenaStats:
    capacity_bytes: int
    used_bytes: int
    requested_bytes: int
    high_water_bytes: int
    num_allocs: int
    num_resets: int

    @property
    def alignment_waste(self) -> float:
        """Bytes lost to padding between objects."""
        if not self.used_bytes:
            return 0.0
        return 1.0 - self.requested_bytes / self.used_bytes

    @property
    def utilization(self) -> float:
        return self.high_water_bytes / self.capacity_bytes if self.capacity_bytes else 0.0


class Arena:
    """Bump allocator over one region. Freed all at once, never individually."""

    __slots__ = (
        "_generation",
        "_high_water",
        "_lock",
        "_num_allocs",
        "_num_resets",
        "_offset",
        "_region",
        "_requested",
    )

    def __init__(self, region: MemoryRegion) -> None:
        if not region.backed:
            raise ValueError("an arena needs a backed region; VA alone is not memory")
        self._region = region
        self._offset = 0
        self._requested = 0
        self._high_water = 0
        self._num_allocs = 0
        self._num_resets = 0
        # Bumped on every reset, so a stale offset handed out before the reset
        # can be rejected instead of silently pointing at another step's data.
        self._generation = 1
        self._lock = threading.Lock()

    @property
    def capacity_bytes(self) -> int:
        return self._region.nbytes

    @property
    def used_bytes(self) -> int:
        return self._offset

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def owner(self) -> MemoryOwner:
        return self._region.owner

    def allocate(self, nbytes: int, *, alignment: int = 256) -> MemoryRegion:
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        if alignment & (alignment - 1):
            raise ValueError(f"alignment {alignment} is not a power of two")
        with self._lock:
            base = self._region.base_ptr + self._offset
            padding = (-base) % alignment
            start = self._offset + padding
            if start + nbytes > self._region.nbytes:
                raise ArenaExhausted(
                    f"arena for {self._region.owner.value} has "
                    f"{self._region.nbytes - start} B left, needs {nbytes} B "
                    f"(capacity {self._region.nbytes >> 20} MiB, high water "
                    f"{self._high_water >> 20} MiB) — the step is larger than the "
                    "profile it was sized from"
                )
            self._offset = start + nbytes
            self._requested += nbytes
            self._high_water = max(self._high_water, self._offset)
            self._num_allocs += 1
            return self._region.slice(start, nbytes)

    def reset(self) -> None:
        """Free everything. O(1), and the only free this allocator has.

        Every region handed out before this call is dangling afterwards. The
        generation bump is what lets a checker catch that; it does not prevent
        the write.
        """
        with self._lock:
            self._offset = 0
            self._requested = 0
            self._generation += 1
            self._num_resets += 1

    def stats(self) -> ArenaStats:
        with self._lock:
            return ArenaStats(
                capacity_bytes=self._region.nbytes,
                used_bytes=self._offset,
                requested_bytes=self._requested,
                high_water_bytes=self._high_water,
                num_allocs=self._num_allocs,
                num_resets=self._num_resets,
            )

    def __repr__(self) -> str:
        return (
            f"<Arena {self._region.owner.value} "
            f"{self._offset >> 10}/{self._region.nbytes >> 10} KiB "
            f"gen={self._generation}>"
        )


@dataclass(frozen=True, slots=True)
class SlabStats:
    object_bytes: int
    stride_bytes: int
    capacity: int
    live: int

    @property
    def free(self) -> int:
        return self.capacity - self.live

    @property
    def stride_waste(self) -> float:
        """Padding between objects, as a fraction of the slab."""
        if not self.stride_bytes:
            return 0.0
        return 1.0 - self.object_bytes / self.stride_bytes


class SlabAllocator:
    """Fixed-size objects from one contiguous region.

    ``stride`` is the object size rounded up to ``alignment``. Keeping objects
    strided rather than packed costs a little memory and buys the property that
    matters: object *n* is at ``base + n * stride``, so an index is an address
    and a scan over live objects walks memory in order.
    """

    __slots__ = ("_capacity", "_free", "_live", "_lock", "_object_bytes", "_region", "_stride")

    def __init__(self, region: MemoryRegion, *, object_bytes: int, alignment: int = 256) -> None:
        if object_bytes <= 0:
            raise ValueError("object_bytes must be positive")
        if alignment & (alignment - 1):
            raise ValueError(f"alignment {alignment} is not a power of two")
        if region.base_ptr % alignment:
            raise ValueError(
                f"slab base {region.base_ptr:#x} is not {alignment}-aligned; every "
                "object would inherit the misalignment"
            )
        self._region = region
        self._object_bytes = object_bytes
        self._stride = (object_bytes + alignment - 1) // alignment * alignment
        self._capacity = region.nbytes // self._stride
        if self._capacity == 0:
            raise ValueError(
                f"region of {region.nbytes} B holds no object of stride {self._stride} B"
            )
        # Reversed so pop() yields index 0 first: deterministic reuse order makes
        # a failing test reproduce.
        self._free: list[int] = list(reversed(range(self._capacity)))
        self._live: set[int] = set()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def stride_bytes(self) -> int:
        return self._stride

    @property
    def num_free(self) -> int:
        with self._lock:
            return len(self._free)

    def allocate(self) -> tuple[int, MemoryRegion]:
        """Return ``(index, region)``. The index is the addressing identity."""
        with self._lock:
            if not self._free:
                raise MemoryError(
                    f"slab exhausted: all {self._capacity} objects of "
                    f"{self._object_bytes} B are live"
                )
            index = self._free.pop()
            self._live.add(index)
            return index, self._region.slice(index * self._stride, self._object_bytes)

    def free(self, index: int) -> None:
        with self._lock:
            if index not in self._live:
                raise RuntimeError(f"slab index {index} is not live (double free?)")
            self._live.discard(index)
            self._free.append(index)

    def address_of(self, index: int) -> int:
        if not 0 <= index < self._capacity:
            raise IndexError(f"slab index {index} out of range [0, {self._capacity})")
        return self._region.base_ptr + index * self._stride

    def stats(self) -> SlabStats:
        with self._lock:
            return SlabStats(
                object_bytes=self._object_bytes,
                stride_bytes=self._stride,
                capacity=self._capacity,
                live=len(self._live),
            )

    def __repr__(self) -> str:
        return f"<SlabAllocator {self._object_bytes}B x{self._capacity} live={len(self._live)}>"
