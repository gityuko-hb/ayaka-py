"""Ayaka adapters for the reference MoE Triton kernels.

This preview reuses Ayaka's host-side guards and arithmetic helpers while
keeping operation-specific validation beside the kernels.
"""

from __future__ import annotations

import struct
from typing import Any

import torch
import triton

from ayaka.kernel.triton._host import (
    require_contiguous,
    require_cuda,
    require_dtype,
    require_last_dim_stride1,
    require_ndim,
    require_same_device,
    require_tensor,
)
from ayaka.kernel.triton._triton_compat import autotune
from ayaka.utils.math_utils import div_ceil, next_power_of_2

__all__ = [
    "autotune",
    "cdiv",
    "check",
    "check_tensor",
    "device_guard",
    "f32",
    "next_pow2",
    "same_device",
]

#: Row-reduction tiles (``moe_sum``, ``moe_sum_reduce``,
#: ``apply_shuffle_mul_sum``) -- all three kernels shared one private list.
ROW_CONFIGS = [
    triton.Config({"BLOCK_D": 256}, num_warps=1),
    triton.Config({"BLOCK_D": 512}, num_warps=2),
    triton.Config({"BLOCK_D": 1024}, num_warps=4),
    triton.Config({"BLOCK_D": 2048}, num_warps=8),
    triton.Config({"BLOCK_D": 1024}, num_warps=2),
    triton.Config({"BLOCK_D": 512}, num_warps=1),
]

#: Byte-copy tiles for :func:`triton_kernels._internal.rows.gather_rows`.
SHUFFLE_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
]


def check(condition: object, message: str) -> None:
    """Raise Ayaka's standard value error for a failed precondition."""
    if not condition:
        raise ValueError(message)


def check_tensor(
    tensor: object,
    name: str,
    *,
    dtypes: tuple[torch.dtype, ...] | None = None,
    ndim: int | None = None,
    cuda: bool = True,
    contiguous: bool | None = None,
    last_dim_contiguous: bool | None = None,
) -> None:
    """Compose Ayaka's existing tensor guards for a MoE contract."""
    require_tensor(tensor, name)
    assert isinstance(tensor, torch.Tensor)
    if cuda:
        require_cuda(tensor, name)
    if dtypes is not None:
        require_dtype(tensor, name, dtypes)
    if ndim is not None:
        require_ndim(tensor, name, ndim)
    if contiguous:
        require_contiguous(tensor, name)
    if last_dim_contiguous:
        require_last_dim_stride1(tensor, name)


def same_device(*tensors: torch.Tensor | None) -> None:
    """Require every supplied tensor to share a device."""
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        raise ValueError("at least one tensor is required")
    reference = present[0]
    for tensor in present[1:]:
        require_same_device(tensor, "tensor", reference, "reference")


def cdiv(a: int, b: int) -> int:
    """Use Ayaka's integer ceiling-division helper."""
    return div_ceil(a, b)


def next_pow2(value: int) -> int:
    """Return at least one, safe for Triton tile sizes."""
    return max(1, next_power_of_2(int(value)))


def f32(value: float) -> float:
    """Round a Python scalar to IEEE-754 binary32 before passing to Triton."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def device_guard(tensor: torch.Tensor) -> Any:
    """Use the PyTorch CUDA device context used by Ayaka's launchers."""
    return torch.cuda.device(tensor.device)
