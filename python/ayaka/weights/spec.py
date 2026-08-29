from __future__ import annotations

import math
from dataclasses import dataclass

from ayaka.types import DType, Layout, MemoryOwner, MemoryTier


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Pure description of a buffer."""
    
    shape: tuple[int, ...]
    dtype: DType
    layout: Layout = Layout.ROW_MAJOR
    tier: MemoryTier = MemoryTier.DEVICE
    owner: MemoryOwner = MemoryOwner.ACTIVATION
    alignment: int = 256
    name: str = ""
    
    def __post_init__(self) -> None:
        if any(d < 0 for d in self.shape):
            raise ValueError(f"{self.name}: negative dim in {self.shape}")
        if self.dtype.sub_byte and self.layout is not Layout.PACKED_SUB_BYTE:
            raise ValueError(f"{self.name}: {self.dtype.label} must use Layout.PACKED_SUB_BYTE")

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.dtype.nbytes(self.numel)
    
    @property
    def padded_nbytes(self) -> int:
        a = self.alignment
        return (self.nbytes + a - 1) // a * a
    
    def contiguous_strides(self) -> tuple[int, ...]:
        """Element strides for the declared layout.  Sub-byte dtypes are packed
        along the last dim, so the last stride is still 1 *in elements* — a
        caller converting to bytes must go through ``dtype.nbytes``."""
        if self.layout is Layout.COL_MAJOR:
            strides: list[int] = []
            acc = 1
            for d in self.shape:
                strides.append(acc)
                acc *= d
            return tuple(strides)
        strides = [1] * len(self.shape)
        acc = 1
        for i in range(len(self.shape) - 1, -1, -1):
            strides[i] = acc
            acc *= self.shape[i]
        return tuple(strides)

    def with_shape(self, shape: tuple[int, ...]) -> TensorSpec:
        return TensorSpec(
            shape=shape,
            dtype=self.dtype,
            layout=self.layout,
            tier=self.tier,
            owner=self.owner,
            alignment=self.alignment,
            name=self.name,
        )