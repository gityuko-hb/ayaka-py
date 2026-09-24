"""Plain-torch oracles for the GEMM/GEMV and quantization kernels."""

from __future__ import annotations

import torch

from ayaka.kernel.triton.gemm._common import SCALE_BLOCK_K, require_fp8_dtype
from ayaka.utils.math_utils import div_ceil


def e8m0_values(storage: torch.Tensor) -> torch.Tensor:
    """Host-side mirror of ``e8m0_u8_to_f32``: bit trick, ``0xFF`` -> NaN."""
    values = (storage.to(torch.int32) << 23).view(torch.float32)
    return torch.where(storage == 0xFF, torch.full_like(values, float("nan")), values)


def fp8_scaled_mm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    backend: str = "triton",
) -> torch.Tensor:
    """``(a @ b) * scale_a * scale_b + bias`` in FP32, scaled after the dot."""
    product = a.to(torch.float32) @ b.to(torch.float32)
    sa = scale_a.to(torch.float32)
    sb = scale_b.to(torch.float32)
    product = product * (sa.reshape(-1, 1) if sa.ndim != 0 else sa)
    product = product * sb
    if bias is not None:
        product = product + bias.to(torch.float32)
    if out is None:
        return product
    out.copy_(product.to(out.dtype))
    return out


def fp8_blockwise_mm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    a_scale_col_major: bool = False,
) -> torch.Tensor:
    """128-block-scaled ``sum_k ((a @ b^T) * scale_a * scale_b) + bias``."""
    rows, inner = int(a.shape[0]), int(a.shape[1])
    columns = int(b.shape[0])
    scale_k = div_ceil(inner, SCALE_BLOCK_K)
    a_wide = a.to(torch.float32)
    b_wide = b.to(torch.float32)
    sa_wide = scale_a.to(torch.float32)
    sb_wide = scale_b.to(torch.float32)
    product = torch.zeros((rows, columns), dtype=torch.float32, device=a.device)
    for sk in range(scale_k):
        start = sk * SCALE_BLOCK_K
        stop = min(start + SCALE_BLOCK_K, inner)
        partial = a_wide[:, start:stop] @ b_wide[:, start:stop].T
        sa = sa_wide[sk, :] if a_scale_col_major else sa_wide[:, sk]
        sb = sb_wide[:, sk].repeat_interleave(SCALE_BLOCK_K)[:columns]
        product = product + partial * sa.reshape(-1, 1) * sb.reshape(1, -1)
    if bias is not None:
        product = product + bias.to(torch.float32)
    if out is None:
        return product
    out.copy_(product.to(out.dtype))
    return out


def fp8_weight_only_gemm_ref(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_row_stride: int | None = None,
    block_size_y: int = 1,
    block_size_x: int = 128,
    scale_dtype: str = "e8m0",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``input @ dequant(weight).T``, rounding the dequantized weight to ``input.dtype``."""
    fp8 = require_fp8_dtype()
    inner = int(input.shape[1])
    columns = int(weight.shape[0])
    stride = div_ceil(inner, block_size_x) if scale_row_stride is None else int(scale_row_stride)
    weight_values = weight.view(fp8).float()
    if scale_dtype == "e8m0":
        scale_values = e8m0_values(weight_scale)
    else:
        scale_values = weight_scale.float()
    row_index = torch.arange(columns, device=input.device) // block_size_y
    column_index = torch.arange(inner, device=input.device) // block_size_x
    scale = scale_values.reshape(-1)[row_index[:, None] * stride + column_index[None, :]]
    # The kernel dequantizes into the activation dtype before the dot, so the
    # reference has to round there too.
    dequantized = (weight_values * scale).to(input.dtype)
    product = input.to(torch.float32) @ dequantized.to(torch.float32).t()
    result = product.to(out.dtype if out is not None else input.dtype)
    if out is None:
        return result
    out.copy_(result)
    return out


def fp8_weight_only_gemv_ref(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_row_stride: int | None = None,
    block_size_y: int = 1,
    block_size_x: int = 128,
    scale_dtype: str = "e8m0",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP64 M == 1 companion of :func:`fp8_weight_only_gemm_ref`."""
    fp8 = require_fp8_dtype()
    inner = int(weight.shape[1])
    columns = int(weight.shape[0])
    stride = div_ceil(inner, block_size_x) if scale_row_stride is None else int(scale_row_stride)
    weight_values = weight.view(fp8).double()
    if scale_dtype == "e8m0":
        scale_values = e8m0_values(weight_scale).double()
    else:
        scale_values = weight_scale.double()
    row_index = torch.arange(columns, device=input.device) // block_size_y
    column_index = torch.arange(inner, device=input.device) // block_size_x
    scale = scale_values.reshape(-1)[row_index[:, None] * stride + column_index[None, :]]
    product = (weight_values * scale) @ input.reshape(-1).double()
    result = product.float().to(out.dtype if out is not None else input.dtype)
    if out is None:
        return result
    out.copy_(result)
    return out


def fp8_blockwise_gemv_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """M == 1 case of :func:`fp8_blockwise_mm_ref` in FP64."""
    inner = int(a.numel())
    columns = int(b.shape[0])
    scale_k = div_ceil(inner, SCALE_BLOCK_K)
    a_wide = a.reshape(-1).double()
    b_wide = b.double()
    sa_wide = scale_a.reshape(-1).double()
    sb_wide = scale_b.double()
    product = torch.zeros(columns, dtype=torch.float64, device=a.device)
    for sk in range(scale_k):
        start = sk * SCALE_BLOCK_K
        stop = min(start + SCALE_BLOCK_K, inner)
        partial = a_wide[start:stop] @ b_wide[:, start:stop].T
        sb = sb_wide[:, sk].repeat_interleave(SCALE_BLOCK_K)[:columns]
        product = product + partial * sa_wide[sk] * sb
    if bias is not None:
        product = product + bias.double()
    result = product.float().to(out.dtype if out is not None else torch.float32)
    if out is None:
        return result
    out.copy_(result)
    return out


def int8_scaled_mm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``(a @ b) * scale_a * scale_b + bias`` with FP32 accumulation."""
    product = a.to(torch.float32) @ b.to(torch.float32)
    product = product * scale_a.to(torch.float32).reshape(-1, 1)
    product = product * scale_b.to(torch.float32).reshape(1, -1)
    if bias is not None:
        product = product + bias.to(torch.float32)
    result = product.to(out.dtype if out is not None else out_dtype)
    if out is None:
        return result
    out.copy_(result)
    return out
