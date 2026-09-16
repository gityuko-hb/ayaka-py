"""The one address vocabulary in the memory package: an owned byte range.

A :class:`MemoryRegion` is what byte allocators return and what an
:class:`~ayaka.memory.arena.Arena` carves up.  It is an integer pointer plus
metadata rather than a tensor, so the ledger, the allocator and the workspace
manager can all speak it, and so it can describe a reserved VA range that has no
backing pages yet.  ``class_bytes`` carries the size-class allocator's block
size as metadata, which is how class rounding stays visible instead of looking
like a leak.
"""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.types import MemoryOwner, MemoryTier

__all__ = ["MemoryRegion", "first_aligned_offset"]


def first_aligned_offset(base_ptr: int, alignment: int) -> int:
    """Bytes from ``base_ptr`` to the next ``alignment``-aligned address.

    Zero when the pointer is already aligned.  Callers validate that
    ``alignment`` is a positive power of two once, at their own boundary; this
    is the arithmetic on the allocation path and deliberately repeats neither
    check.
    """
    return (-base_ptr) % alignment


@dataclass(frozen=True, slots=True)
class MemoryRegion:
    """A contiguous byte range with exactly one owner.

    ``base_ptr`` is a raw integer address (device or host).  Keeping it an int
    rather than a torch storage means the allocator, the arena, and the workspace
    manager all speak the same language, and a region can describe a
    ``cuMemAddressReserve`` VA range that has no backing pages yet.

    ``class_bytes`` is metadata, not address space: when the region came out of a
    size-class caching allocator it records the size of the block that was taken
    from the source, which is ``>= nbytes``.  Zero means "not class-managed or
    unknown".  Callers allocate against ``nbytes`` — the gap is internal slack
    (:attr:`slack_bytes`), and it exists so numbers reported by an arena and by
    the ledger can be reconciled instead of looking like a leak.
    """

    region_id: int
    base_ptr: int
    nbytes: int
    tier: MemoryTier
    owner: MemoryOwner
    device_index: int = -1
    alignment: int = 256
    backed: bool = True  # False for a reserved-but-unmapped VA range (VMM)
    class_bytes: int = 0

    def __post_init__(self) -> None:
        if self.nbytes <= 0:
            raise ValueError(f"region {self.region_id}: nbytes must be > 0")
        if self.backed and self.base_ptr % self.alignment:
            raise ValueError(
                f"region {self.region_id}: base_ptr {self.base_ptr:#x} "
                f"not aligned to {self.alignment}"
            )
        if self.class_bytes < 0:
            raise ValueError(f"region {self.region_id}: class_bytes must be non-negative")
        if self.class_bytes and self.class_bytes < self.nbytes:
            raise ValueError(
                f"region {self.region_id}: class_bytes {self.class_bytes} cannot be "
                f"smaller than nbytes {self.nbytes}"
            )

    @property
    def slack_bytes(self) -> int:
        """Bytes held by the backing block but outside this region's request.

        Only meaningful on a region returned directly by an allocator; a slice
        inherits its parent's ``class_bytes`` and the parent's slack would be
        misattributed if summed across children.
        """
        return max(self.class_bytes - self.nbytes, 0)

    @property
    def end_ptr(self) -> int:
        return self.base_ptr + self.nbytes

    def contains(self, ptr: int, nbytes: int = 0) -> bool:
        return self.base_ptr <= ptr and ptr + nbytes <= self.end_ptr

    def slice(self, offset: int, nbytes: int) -> MemoryRegion:
        if offset < 0 or offset + nbytes > self.nbytes:
            raise ValueError(
                f"region {self.region_id}: slice [{offset}, {offset + nbytes}) "
                f"escapes [0, {self.nbytes})"
            )
        return MemoryRegion(
            region_id=self.region_id,
            base_ptr=self.base_ptr + offset,
            nbytes=nbytes,
            tier=self.tier,
            owner=self.owner,
            device_index=self.device_index,
            alignment=1,
            backed=self.backed,
            class_bytes=self.class_bytes,
        )
