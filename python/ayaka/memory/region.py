
from __future__ import annotations

from dataclasses import dataclass

from ayaka.types import MemoryOwner, MemoryTier

__all__ = ["MemoryRegion"]

@dataclass(frozen=True, slots=True)
class MemoryRegion:
    """A contiguous byte range with exactly one owner.

    ``base_ptr`` is a raw integer address (device or host).  Keeping it an int
    rather than a torch storage means the allocator, the arena, and the workspace
    manager all speak the same language, and a region can describe a
    ``cuMemAddressReserve`` VA range that has no backing pages yet.
    """

    region_id: int
    base_ptr: int
    nbytes: int
    tier: MemoryTier
    owner: MemoryOwner
    device_index: int = -1
    alignment: int = 256
    backed: bool = True  # False for a reserved-but-unmapped VA range (VMM)

    def __post_init__(self) -> None:
        if self.nbytes <= 0:
            raise ValueError(f"region {self.region_id}: nbytes must be > 0")
        if self.backed and self.base_ptr % self.alignment:
            raise ValueError(
                f"region {self.region_id}: base_ptr {self.base_ptr:#x} "
                f"not aligned to {self.alignment}"
            )

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
        )
