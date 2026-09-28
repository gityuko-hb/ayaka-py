"""Typed, non-owning descriptions of canonical RunnerBuffers backing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ayaka.runner.buffers import RunnerBufferLease


@dataclass(frozen=True, slots=True)
class TensorBinding:
    pointer: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, tensor: torch.Tensor) -> TensorBinding:
        return cls(
            tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device
        )


@dataclass(frozen=True, slots=True)
class InputBuffers:
    """Stamp a leased slot without allocating or making another buffer owner.

    The underlying FlightBuffers stage methods own dtype and padding rules:
    int64 model input/rows, int32 addressing, zeroed stale tails and explicit
    backend padding for all rows read by a decode graph.
    """

    lease: RunnerBufferLease
    generation: int
    bindings: tuple[TensorBinding, ...]

    @classmethod
    def borrow(cls, lease: RunnerBufferLease) -> InputBuffers:
        return cls(lease, lease.generation, tuple(map(TensorBinding.of, lease.buffers.tensors())))

    def validate(self) -> None:
        if self.lease.generation != self.generation:
            raise ValueError("input buffer generation changed after preparation")
        if tuple(map(TensorBinding.of, self.lease.buffers.tensors())) != self.bindings:
            raise ValueError("input buffer backing changed after preparation")
