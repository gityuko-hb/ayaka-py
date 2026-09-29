"""Owned CPU weight IR; checkpoint, canonical and device storage are distinct."""

from dataclasses import dataclass

import torch

from ayaka.lora.variant import AdapterIdentity


@dataclass(frozen=True, slots=True)
class LoRAWeights:
    """Caller-owned full logical tensors, A[rank,input], B[output,rank]."""

    a: torch.Tensor
    b: torch.Tensor
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class LoRALayerWeights:
    """Private cloned CPU tensors; consumers must not mutate this owned IR.

    Shapes are global logical projection shapes, before TP. Scaling is applied
    once when materializing B into runtime dtype, never by the forward kernel.
    """

    target: str
    rank: int
    alpha: float
    scale: float
    a: torch.Tensor
    b: torch.Tensor
    source_dtype: torch.dtype
    runtime_dtype: torch.dtype
    logical_shape: tuple[int, int]
    local_shape: tuple[int, int]
    output_offset: int

    @property
    def nbytes(self) -> int:
        return self.a.numel() * self.a.element_size() + self.b.numel() * self.b.element_size()


@dataclass(frozen=True, slots=True)
class AdapterWeights:
    identity: AdapterIdentity
    layers: tuple[LoRALayerWeights, ...]
    content_digest: str

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)
