"""Split-K FP8-E4M3 GEMV decode kernels.

At M == 1 the tiled GEMMs in :mod:`ayaka.kernel.triton.gemm.fp8_gemm` and
:mod:`ayaka.kernel.triton.gemm.fp8_w8a16_gemm` launch one program per N tile and
walk the whole K axis inside each: the grid is ``cdiv(N, BLOCK_N)`` CTAs -- 8 for
N = 1024 on a 188-SM part -- so decode latency tracks K and ignores N. These GEMV
kernels partition K instead (``split_k`` CTAs per N tile): the weight traffic is
the same, but the grid covers the device.

Two entry points:

* :func:`fp8_weight_only_gemv` -- W8A16 weight-only dequantization for the
  ``[N, K]`` fp8 + block-scale storage of ``fp8_weight_only_gemm``.
* :func:`fp8_blockwise_gemv` -- W8A8, the M == 1 case of the 128-block contract
  consumed by :func:`ayaka.kernel.triton.gemm.fp8_gemm.fp8_blockwise_mm`.

Both reduce over the split axis with a power-of-two split, so the accumulation
order is fixed and repeated launches are bit-identical: a request's decode
trajectory must not depend on a tuning heuristic. The split is derived from
shapes only, never from tensor values, so the launch stays CUDA-graph safe.

Numerics differ from the tiled GEMMs on purpose: the GEMV applies each scale to
the fp32 products and stores that, while the GEMM rounds a dequantized weight
to the activation dtype before the dot.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton.fp8_compat import (
    e4m3_kernel_view,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    e8m0_u8_to_f32,
)
from ayaka.utils.math_utils import div_ceil, require_cuda

from .fp8_gemm import _validate_bias, _validate_block_scale, _validate_scale
from .fp8_w8a16_gemm import (
    _e8m0_values,
    _weight_bytes,
)
from .fp8_w8a16_gemm import (
    _validate_scale as _validate_w8a16_scale,
)

__all__ = ["fp8_blockwise_gemv", "fp8_weight_only_gemv"]

FP8 = torch.float8_e4m3fn
_OUTPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}
_SCALE_FORMATS = ("e8m0", "float32")

#: One weight scale block: BLOCK_K == 128 so a program's K iteration is exactly
#: one weight-scale block.
_BLOCK_K = 128

#: Outputs per program. Small on purpose: at bs=1 the grid is the only source of
#: parallelism, so more, narrower programs beat fewer wide ones.
_BLOCK_N = 16

#: Rough ceiling on the split axis: total CTAs stay bounded so the reduce kernel
#: and the partial buffer do not grow without bound on wide N.
_SPLIT_BUDGET = 1536

_REDUCE_BLOCK = 256


@triton.jit
def _fp8_weight_only_gemv_splitk_kernel(
    a_ptr,
    w_ptr,
    s_ptr,
    part_ptr,
    N,
    K,
    n_kb,
    kb_per,
    stride_ak,
    stride_wn,
    stride_wk,
    scale_row_stride,
    stride_pk,
    stride_pn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    SCALE_E8M0: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(
                    tl.load(
                        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=0,
                    )
                )
            if BLOCK_SIZE_X % BLOCK_K == 0:
                # One scale per (output row, K block): apply it to the block sum.
                s_ptrs = s_ptr + (offs_n // BLOCK_SIZE_Y) * scale_row_stride
                s_ptrs += (kb * BLOCK_K) // BLOCK_SIZE_X
                raw = tl.load(s_ptrs, mask=n_mask, other=0)
                if SCALE_E8M0:
                    scale = e8m0_u8_to_f32(raw.to(tl.uint8))
                else:
                    scale = raw.to(tl.float32)
                acc += tl.sum(w * a[None, :], axis=1) * scale
            else:
                # The scale varies inside the tile; gather it per element.
                s_ptrs = s_ptr + (offs_n[:, None] // BLOCK_SIZE_Y) * scale_row_stride
                s_ptrs += offs_k[None, :] // BLOCK_SIZE_X
                raw = tl.load(s_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)
                if SCALE_E8M0:
                    scale = e8m0_u8_to_f32(raw.to(tl.uint8))
                else:
                    scale = raw.to(tl.float32)
                acc += tl.sum(w * a[None, :] * scale, axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _fp8_blockwise_gemv_splitk_kernel(
    a_ptr,
    w_ptr,
    sa_ptr,
    sb_ptr,
    part_ptr,
    N,
    K,
    n_kb,
    kb_per,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_sbn,
    stride_sbk,
    stride_pk,
    stride_pn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(
                    tl.load(
                        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=0,
                    )
                )
            sa = tl.load(sa_ptr + kb).to(tl.float32)
            sb = tl.load(
                sb_ptr + (offs_n // 128) * stride_sbn + kb * stride_sbk,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * a[None, :], axis=1) * (sa * sb)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(
    part_ptr,
    bias_ptr,
    out_ptr,
    N,
    SPLIT_K: tl.constexpr,
    stride_pk,
    stride_pn,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):  # type: ignore
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(OUT), mask=mask)


def _split_plan(n: int, k: int) -> tuple[int, int, int, int]:
    """``(n_tiles, split_k, kb_per, n_kb)`` from shapes only."""
    n_kb = div_ceil(k, _BLOCK_K)
    n_tiles = div_ceil(n, _BLOCK_N)
    split_k = max(1, min(_SPLIT_BUDGET // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # power of two: stable reduction order
    return n_tiles, split_k, div_ceil(n_kb, split_k), n_kb


def _reduce(
    part: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None,
    split_k: int,
) -> None:
    n = int(out.shape[0])
    bias_arg = bias if bias is not None else out
    cast(Any, _splitk_reduce_kernel)[(div_ceil(n, _REDUCE_BLOCK),)](
        part,
        bias_arg,
        out,
        n,
        split_k,
        part.stride(0),
        part.stride(1),
        BLOCK=_REDUCE_BLOCK,
        HAS_BIAS=bias is not None,
        OUT=_TL_DTYPE[out.dtype],
        num_warps=2,
    )


def _flatten_vector(x: torch.Tensor, name: str, size: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.numel() != size:
        raise ValueError(f"{name} must have exactly {size} elements (M == 1)")
    require_cuda(x, name)
    return x.reshape(-1)


def _prepare_vector_output(
    reference: torch.Tensor,
    out: torch.Tensor | None,
    size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if out is None:
        return torch.empty(size, device=reference.device, dtype=dtype)
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out.device != reference.device:
        raise ValueError("out and input must be on the same CUDA device")
    if out.dtype is not dtype:
        raise TypeError(f"out must have dtype {dtype}")
    if tuple(out.shape) != (size,):
        raise ValueError(f"out must have shape [{size}]")
    if out.stride(0) != 1:
        raise ValueError("out must be contiguous")
    return out


# ======================================================================================
# W8A16: weight-only dequantization of [N, K] fp8 with block scales.
# ======================================================================================
def _weight_only_gemv_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_row_stride: int | None = None,
    block_size_y: int = 1,
    block_size_x: int = 128,
    scale_dtype: str = "e8m0",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty(weight.shape[0], dtype=input.dtype, device=input.device)


def _weight_only_gemv_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_row_stride: int | None = None,
    block_size_y: int = 1,
    block_size_x: int = 128,
    scale_dtype: str = "e8m0",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    inner = int(weight.shape[1])
    columns = int(weight.shape[0])
    stride = div_ceil(inner, block_size_x) if scale_row_stride is None else int(scale_row_stride)
    weight_values = weight.view(FP8).double()
    if scale_dtype == "e8m0":
        scale_values = _e8m0_values(weight_scale).double()
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


@custom_op(
    namespace="ayaka",
    name="fp8_weight_only_gemv",
    mutates_args=["out"],
    reference=_weight_only_gemv_reference,
    fake_impl=_weight_only_gemv_fake,
    dispatch_key="CUDA",
)
def fp8_weight_only_gemv(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_row_stride: int | None = None,
    block_size_y: int = 1,
    block_size_x: int = 128,
    scale_dtype: str = "e8m0",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``input @ dequant(weight).T`` for an M == 1 activation.

    The M == 1 companion of :func:`ayaka.kernel.triton.gemm.fp8_w8a16_gemm
    .fp8_weight_only_gemm`, with the same scale layout and validation.

    Args:
        input: ``[K]`` (or any shape with K elements) fp16/bf16 CUDA tensor.
        weight: ``[N, K]`` fp8-e4m3 or raw ``uint8`` E4M3 bytes.
        weight_scale: ``scale[n // block_size_y, k // block_size_x]`` storage,
            E8M0 bytes or float32.
        scale_row_stride: Distance between scale rows. Defaults to
            ``ceil(K / block_size_x)``.
        block_size_y: Output rows sharing one scale value.
        block_size_x: K columns sharing one scale value.
        scale_dtype: ``"e8m0"`` or ``"float32"``.
        out: Optional ``[N]`` buffer with the same dtype as ``input``.

    Returns:
        The ``[N]`` output tensor, the same object as ``out`` when provided.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("input must have dtype float16 or bfloat16")
    require_cuda(input, "input")
    weight_bytes = _weight_bytes(weight)
    columns = int(weight_bytes.shape[0])
    inner = int(weight_bytes.shape[1])
    a = _flatten_vector(input, "input", inner)
    if block_size_x <= 0 or block_size_y <= 0:
        raise ValueError("block sizes must be positive")
    if scale_dtype not in _SCALE_FORMATS:
        raise ValueError("scale_dtype must be 'e8m0' or 'float32'")
    if scale_row_stride is None:
        scale_row_stride = div_ceil(inner, block_size_x)
    if scale_row_stride <= 0:
        raise ValueError("scale_row_stride must be positive")
    _validate_w8a16_scale(
        weight_scale,
        input,
        scale_dtype=scale_dtype,
        block_size_y=block_size_y,
        scale_row_stride=scale_row_stride,
        columns=columns,
    )
    result = _prepare_vector_output(input, out, columns, input.dtype)
    if columns == 0:
        return result

    n_tiles, split_k, kb_per, n_kb = _split_plan(columns, inner)
    part = torch.empty((split_k, columns), dtype=torch.float32, device=input.device)
    cast(Any, _fp8_weight_only_gemv_splitk_kernel)[(n_tiles, split_k)](
        a,
        e4m3_kernel_view(weight_bytes.view(FP8)),
        weight_scale,
        part,
        columns,
        inner,
        n_kb,
        kb_per,
        a.stride(0),
        weight_bytes.stride(0),
        weight_bytes.stride(1),
        scale_row_stride,
        part.stride(0),
        part.stride(1),
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        BLOCK_SIZE_Y=block_size_y,
        BLOCK_SIZE_X=block_size_x,
        SCALE_E8M0=scale_dtype == "e8m0",
        num_warps=1,
    )
    _reduce(part, result, None, split_k)
    return result


# ======================================================================================
# W8A8: the M == 1 case of the 128-block contract.
# ======================================================================================
def _blockwise_gemv_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty(b.shape[0], dtype=torch.float32, device=a.device)


def _blockwise_gemv_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    inner = int(a.numel())
    columns = int(b.shape[0])
    scale_k = div_ceil(inner, _BLOCK_K)
    a_wide = a.reshape(-1).double()
    b_wide = b.double()
    sa_wide = scale_a.reshape(-1).double()
    sb_wide = scale_b.double()
    product = torch.zeros(columns, dtype=torch.float64, device=a.device)
    for sk in range(scale_k):
        start = sk * _BLOCK_K
        stop = min(start + _BLOCK_K, inner)
        partial = a_wide[start:stop] @ b_wide[:, start:stop].T
        sb = sb_wide[:, sk].repeat_interleave(_BLOCK_K)[:columns]
        product = product + partial * sa_wide[sk] * sb
    if bias is not None:
        product = product + bias.double()
    result = product.float().to(out.dtype if out is not None else torch.float32)
    if out is None:
        return result
    out.copy_(result)
    return out


@custom_op(
    namespace="ayaka",
    name="fp8_blockwise_gemv",
    mutates_args=["out"],
    reference=_blockwise_gemv_reference,
    fake_impl=_blockwise_gemv_fake,
    dispatch_key="CUDA",
)
def fp8_blockwise_gemv(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the M == 1 case of ``fp8_blockwise_mm``.

    Args:
        a: ``[K]`` (or any shape with K elements) ``torch.float8_e4m3fn``.
        b: ``[N, K]`` ``torch.float8_e4m3fn``.
        scale_a: ``[ceil(K / 128)]`` fp32 activation scales.
        scale_b: ``[ceil(N / 128), ceil(K / 128)]`` weight scales.
        bias: Optional ``[N]`` additive term.
        out: Optional ``[N]`` buffer (fp32, fp16, or bf16).

    Returns:
        The ``[N]`` output tensor, the same object as ``out`` when provided.
    """

    _validate_scale_input(a, "a")
    if not isinstance(b, torch.Tensor):
        raise TypeError("b must be a torch.Tensor")
    if b.dtype is not FP8:
        raise TypeError("b must have dtype torch.float8_e4m3fn")
    require_cuda(b, "b")
    if b.ndim != 2:
        raise ValueError("b must be 2-D")
    if b.stride(1) != 1:
        raise ValueError("b must have stride(1) == 1")
    columns, inner = int(b.shape[0]), int(b.shape[1])
    a_vec = _flatten_vector(a, "a", inner)
    scale_k = div_ceil(inner, _BLOCK_K)
    _validate_scale(scale_a, "scale_a", a, scale_k)
    _validate_block_scale(scale_b, "scale_b", a, (div_ceil(columns, _BLOCK_K), scale_k))
    _validate_bias(bias, a, columns)
    dtype = out.dtype if out is not None else torch.float32
    if dtype not in _OUTPUT_DTYPES:
        raise TypeError("out must be float16, bfloat16, or float32")
    result = _prepare_vector_output(a, out, columns, dtype)
    if columns == 0:
        return result

    n_tiles, split_k, kb_per, n_kb = _split_plan(columns, inner)
    part = torch.empty((split_k, columns), dtype=torch.float32, device=a.device)
    cast(Any, _fp8_blockwise_gemv_splitk_kernel)[(n_tiles, split_k)](
        a_vec,
        e4m3_kernel_view(b),
        scale_a,
        scale_b,
        part,
        columns,
        inner,
        n_kb,
        kb_per,
        a_vec.stride(0),
        b.stride(0),
        b.stride(1),
        scale_b.stride(0),
        scale_b.stride(1),
        part.stride(0),
        part.stride(1),
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=1,
    )
    _reduce(part, result, bias, split_k)
    return result


def _validate_scale_input(a: torch.Tensor, name: str) -> None:
    if not isinstance(a, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if a.dtype is not FP8:
        raise TypeError(f"{name} must have dtype torch.float8_e4m3fn")
    require_cuda(a, name)
