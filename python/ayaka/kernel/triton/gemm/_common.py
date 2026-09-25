"""Host-side contracts shared by the GEMM/GEMV kernel modules.

The tiled GEMMs, the split-K GEMVs and the weight-only path all accept the
same operand/scale/bias contract, so the checks and the autotune ladder live
here once instead of being copied per module. Kernel modules keep their own
launch code; this module is host-only and must not import Triton kernels.
"""

from __future__ import annotations

from typing import cast

import torch
import triton

from ayaka.kernel.triton._host import (
    dtype_list,
    require_cuda,
    require_dtype,
    require_same_device,
    require_tensor,
)
from ayaka.utils.math_utils import div_ceil

#: Dtypes accepted for scales, biases and GEMM outputs.
FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

#: Alias kept for call sites that read "output" instead of "float".
OUTPUT_DTYPES = FLOAT_DTYPES

#: K width of one quantization block in the blockwise FP8 checkpoint contract.
SCALE_BLOCK_K = 128

#: Autotune ladder shared by the tiled E4M3 and INT8 GEMMs.
GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4
    ),
]

_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)


def require_fp8_dtype() -> torch.dtype:
    """Return the E4M3 dtype, or fail on a build that lacks it."""
    if _FP8_E4M3 is None:
        raise RuntimeError("this PyTorch build does not provide torch.float8_e4m3fn")
    return cast(torch.dtype, _FP8_E4M3)


def validate_operand(
    tensor: torch.Tensor,
    name: str,
    *,
    dtype: torch.dtype,
    allow_transposed: bool,
) -> None:
    """Validate a 2-D GEMM operand with a required dtype and layout.

    Args:
        tensor: Candidate operand.
        name: Argument name used in error messages.
        dtype: Required element dtype.
        allow_transposed: Accept ``stride(0) == 1`` in addition to row-major.

    Raises:
        TypeError: If ``tensor`` is not a tensor or has the wrong dtype.
        ValueError: If it is not CUDA, not 2-D, or badly strided.
    """
    require_tensor(tensor, name)
    require_dtype(tensor, name, (dtype,))
    require_cuda(tensor, name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if tensor.stride(1) != 1 and not (allow_transposed and tensor.stride(0) == 1):
        layout = "stride(1) == 1" if not allow_transposed else "stride(1) == 1 or stride(0) == 1"
        raise ValueError(f"{name} must have {layout}")


def validate_scale(
    scale: torch.Tensor,
    name: str,
    reference: torch.Tensor,
    size: int,
    *,
    dtypes: tuple[torch.dtype, ...] = FLOAT_DTYPES,
    scalar_ok: bool = True,
    reference_name: str = "a",
) -> None:
    """Validate a scalar or ``[size]`` per-row/column dequantization scale.

    Raises:
        TypeError: If ``scale`` is not a tensor or its dtype is unsupported.
        ValueError: If the device, shape or stride contract is violated.
    """
    require_tensor(scale, name)
    require_same_device(scale, name, reference, reference_name)
    if scale.dtype not in dtypes:
        raise TypeError(f"{name} must have dtype {dtype_list(dtypes)}")
    if scale.ndim == 0:
        if scalar_ok:
            return
        raise ValueError(f"{name} must have shape [{size}]")
    if scale.ndim != 1 or scale.numel() != size:
        if scalar_ok:
            raise ValueError(f"{name} must be a scalar or a vector of shape [{size}]")
        raise ValueError(f"{name} must have shape [{size}]")
    if scale.stride(0) != 1:
        raise ValueError(f"{name} must be contiguous")


def validate_block_scale(
    scale: torch.Tensor,
    name: str,
    reference: torch.Tensor,
    shape: tuple[int, int],
    *,
    dtypes: tuple[torch.dtype, ...] = FLOAT_DTYPES,
    reference_name: str = "a",
) -> None:
    """Validate a 2-D block-scale tensor of exactly ``shape``.

    Raises:
        TypeError: If ``scale`` is not a tensor or its dtype is unsupported.
        ValueError: If the device, shape or contiguity contract is violated.
    """
    require_tensor(scale, name)
    require_same_device(scale, name, reference, reference_name)
    if scale.dtype not in dtypes:
        raise TypeError(f"{name} must have dtype {dtype_list(dtypes)}")
    if tuple(scale.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_bias(
    bias: torch.Tensor | None,
    reference: torch.Tensor,
    size: int,
    *,
    dtypes: tuple[torch.dtype, ...] = FLOAT_DTYPES,
    reference_name: str = "a",
) -> None:
    """Validate an optional ``[size]`` additive bias.

    Raises:
        TypeError: If ``bias`` is not a tensor or its dtype is unsupported.
        ValueError: If the device or shape contract is violated.
    """
    if bias is None:
        return
    require_tensor(bias, "bias")
    require_same_device(bias, "bias", reference, reference_name)
    if bias.dtype not in dtypes:
        raise TypeError(f"bias must have dtype {dtype_list(dtypes)}")
    if tuple(bias.shape) != (size,):
        raise ValueError(f"bias must have shape [{size}]")


def validate_w8a16_scale(
    weight_scale: torch.Tensor,
    reference: torch.Tensor,
    *,
    scale_dtype: str,
    block_size_y: int,
    scale_row_stride: int,
    columns: int,
    reference_name: str = "input",
) -> None:
    """Validate the packed E8M0/float32 weight-scale storage of W8A16.

    Raises:
        TypeError: If the dtype does not match ``scale_dtype``.
        ValueError: If the device mismatches or the storage is undersized.
    """
    require_tensor(weight_scale, "weight_scale")
    require_same_device(weight_scale, "weight_scale", reference, reference_name)
    expected = {"e8m0": torch.uint8, "float32": torch.float32, "bfloat16": torch.bfloat16}[
        scale_dtype
    ]
    if weight_scale.dtype is not expected:
        raise TypeError(f"{scale_dtype} weight_scale must have dtype {expected}")
    if weight_scale.numel() < div_ceil(columns, block_size_y) * scale_row_stride:
        raise ValueError("weight_scale storage is too small for the requested block layout")


def weight_bytes(weight: torch.Tensor) -> torch.Tensor:
    """Return the raw ``uint8`` view of an E4M3 weight operand.

    Raises:
        TypeError: If ``weight`` is not a tensor or not uint8/E4M3.
        ValueError: If ``weight`` is not CUDA, not 2-D, or badly strided.
    """
    require_tensor(weight, "weight")
    require_cuda(weight, "weight")
    if weight.ndim != 2:
        raise ValueError("weight must be 2-D")
    if weight.dtype is torch.uint8:
        byte_view = weight
    elif _FP8_E4M3 is not None and weight.dtype is _FP8_E4M3:
        byte_view = weight.view(torch.uint8)
    else:
        raise TypeError("weight must have dtype uint8 or torch.float8_e4m3fn")
    if byte_view.stride(1) != 1:
        raise ValueError("weight must have stride(1) == 1")
    return byte_view
