from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import fake_output, prepare_output, require_last_dim_stride1
from ayaka.kernel.triton.fp8_compat import (
    e4m3_native_cx,
    e4m3_u8_to_f32,
    fp8_kernel_view,
)
from ayaka.kernel.triton.gemm._common import (
    GEMM_CONFIGS,
    OUTPUT_DTYPES,
    require_fp8_dtype,
    validate_bias,
    validate_block_scale,
    validate_operand,
    validate_scale,
)
from ayaka.kernel.triton.gemm._common import SCALE_BLOCK_K as _SCALE_BLOCK_K
from ayaka.kernel.triton.quant import fp8_cublaslt
from ayaka.kernel.triton.reference.gemm import fp8_blockwise_mm_ref, fp8_scaled_mm_ref
from ayaka.kernel.triton.swizzle import grouped_pid
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.math_utils import div_ceil

__all__ = ["fp8_blockwise_mm", "fp8_scaled_mm"]

#: ``backend`` values accepted by :func:`fp8_scaled_mm`.
_BACKENDS = ("triton", "auto", "cublaslt")


@triton.autotune(configs=GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fp8_scaled_mm_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    SA_SCALAR: tl.constexpr,
    SB_SCALAR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    B_TRANSPOSED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = grouped_pid(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    if B_TRANSPOSED:
        # [N, K] weight view: keep the contiguous K axis innermost for the load,
        # then transpose the tile into the [BLOCK_K, BLOCK_N] dot operand.
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    else:
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros(shape=(BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if K % BLOCK_K != 0:
            k = k0 + offs_k
            a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
            if B_TRANSPOSED:
                b_mask = (offs_n[:, None] < N) & (k[None, :] < K)
            else:
                b_mask = (k[:, None] < K) & (offs_n[None, :] < N)
        else:
            a_mask = offs_m[:, None] < M
            if B_TRANSPOSED:
                b_mask = offs_n[:, None] < N
            else:
                b_mask = offs_n[None, :] < N
        a_raw = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_raw = tl.load(b_ptrs, mask=b_mask, other=0.0)
        if B_TRANSPOSED:
            b_raw = tl.trans(b_raw)
        if e4m3_native_cx():
            a = a_raw
            b = b_raw
        else:
            a = e4m3_u8_to_f32(a_raw).to(tl.float16)
            b = e4m3_u8_to_f32(b_raw).to(tl.float16)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if SA_SCALAR:
        y = acc * tl.load(sa_ptr).to(tl.float32)
    else:
        sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        y = acc * sa[:, None]
    if SB_SCALAR:
        y = y * tl.load(sb_ptr).to(tl.float32)
    else:
        sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        y = y * sb[None, :]
    if HAS_BIAS:
        y = y + tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(configs=GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fp8_blockwise_mm_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    A_SCALE_COLMAJOR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
):
    tl.static_assert(SCALE_BLOCK_K % BLOCK_K == 0)
    pid_m, pid_n = grouped_pid(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    scale_k = tl.cdiv(K, SCALE_BLOCK_K)

    # B is the [N, K] source layout: load [BLOCK_N, BLOCK_K] coalesced along K,
    # then transpose the tile into the [BLOCK_K, BLOCK_N] dot operand.
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros(shape=(BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Split at SCALE_BLOCK_K boundaries so the checkpoint's block-scaling order
    # is kept: each block is accumulated on its own, then rescaled.
    for ks in range(0, K, SCALE_BLOCK_K):
        partial = tl.zeros(shape=(BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, SCALE_BLOCK_K, BLOCK_K):
            if K % SCALE_BLOCK_K == 0:
                a_mask = offs_m[:, None] < M
                b_mask = offs_n[:, None] < N
            else:
                k = ks + kk + offs_k
                a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
                b_mask = (offs_n[:, None] < N) & (k[None, :] < K)
            a_raw = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_raw = tl.load(b_ptrs, mask=b_mask, other=0.0)
            if e4m3_native_cx():
                a = a_raw
                b = b_raw
            else:
                a = e4m3_u8_to_f32(a_raw).to(tl.float16)
                b = e4m3_u8_to_f32(b_raw).to(tl.float16)
            partial = tl.dot(a, tl.trans(b), acc=partial, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        sk = ks // SCALE_BLOCK_K
        if A_SCALE_COLMAJOR:
            sa = tl.load(sa_ptr + sk * M + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        else:
            sa = tl.load(sa_ptr + offs_m * scale_k + sk, mask=offs_m < M, other=0.0).to(tl.float32)
        sb = tl.load(
            sb_ptr + (offs_n // SCALE_BLOCK_K) * scale_k + sk,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)
        acc += partial * sa[:, None] * sb[None, :]

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _mm_output(
    a: torch.Tensor,
    out: torch.Tensor | None,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Allocate or validate the ``[rows, columns]`` fp32 output."""
    result = prepare_output(
        a,
        out,
        shape=(rows, columns),
        dtype=torch.float32,
        dtypes=OUTPUT_DTYPES,
        like_name="a",
    )
    require_last_dim_stride1(result, "out")
    return result


def _fp8_scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    backend: str = "triton",
) -> torch.Tensor:
    return fake_output(a, out, shape=(a.shape[0], b.shape[1]), dtype=torch.float32)


@custom_op(
    namespace="ayaka",
    name="fp8_scaled_mm",
    mutates_args=["out"],
    reference=fp8_scaled_mm_ref,
    fake_impl=_fp8_scaled_mm_fake,
    dispatch_key="CUDA",
)
def fp8_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    backend: str = "triton",
) -> torch.Tensor:
    """Compute ``(a @ b) * scale_a * scale_b + bias`` for E4M3 operands.

    Args:
        a: ``[M, K]`` ``torch.float8_e4m3fn`` with ``stride(1) == 1``.
        b: ``[K, N]`` ``torch.float8_e4m3fn``, or an ``[N, K]`` transposed weight
            view with ``stride(0) == 1``.
        scale_a: Scalar or ``[M]`` dequantization scale applied after accumulation.
        scale_b: Scalar or ``[N]`` dequantization scale applied after accumulation.
        bias: Optional ``[N]`` additive term.
        out: Optional ``[M, N]`` output buffer (fp32, fp16, or bf16). A new fp32
            buffer is allocated when omitted.
        backend: ``"triton"`` (default) runs the tiled Triton kernel.
            ``"cublaslt"`` requires ``torch._scaled_mm`` and raises
            ``CapabilityError`` where the device or build cannot run it.
            ``"auto"`` uses cuBLASLt when available and the shapes satisfy its
            constraints (``K`` and ``N`` divisible by 16, plus the row-wise
            kernel for per-row/per-column scales), else the Triton kernel.

    Returns:
        The ``[M, N]`` output tensor, same object as ``out`` when provided.

    Raises:
        TypeError: On non-E4M3 operands, non-float scales, or unsupported ``out``.
        ValueError: On mismatched shapes, devices, unsupported layouts, or an
            unknown ``backend``.
        CapabilityError: When ``backend="cublaslt"`` and cuBLASLt cannot run.
    """

    fp8_dtype = require_fp8_dtype()
    validate_operand(a, "a", dtype=fp8_dtype, allow_transposed=False)
    validate_operand(b, "b", dtype=fp8_dtype, allow_transposed=True)
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions do not match: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    rows, inner, columns = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    validate_scale(scale_a, "scale_a", a, rows)
    validate_scale(scale_b, "scale_b", a, columns)
    validate_bias(bias, a, columns)
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if rows == 0 or columns == 0:
        return _mm_output(a, out, rows, columns)
    if out is not None:
        _mm_output(a, out, rows, columns)

    rowwise = scale_a.numel() > 1 or scale_b.numel() > 1
    if backend != "triton" and inner % 16 == 0 and columns % 16 == 0:
        if fp8_cublaslt.usable(rowwise):
            return fp8_cublaslt.scaled_mm(a, b, scale_a, scale_b, bias, out)
    if backend == "cublaslt":
        raise CapabilityError(
            "fp8_scaled_mm.cublaslt",
            detail=(
                "torch._scaled_mm needs K and N divisible by 16 and, for per-row or "
                f"per-column scales, the row-wise kernel; got K={inner}, N={columns}, "
                f"rowwise={rowwise}"
            ),
            remedy="use backend='auto' to fall back to the Triton kernel",
        )

    result = _mm_output(a, out, rows, columns)

    b_transposed = b.stride(0) == 1 and b.stride(1) != 1
    a_arg = fp8_kernel_view(a)
    b_arg = fp8_kernel_view(b)
    bias_arg = bias if bias is not None else a_arg

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(columns, meta["BLOCK_N"]),)

    with torch.cuda.device(a.device):
        cast(Any, _fp8_scaled_mm_kernel)[grid](
            a_arg,
            b_arg,
            scale_a,
            scale_b,
            bias_arg,
            result,
            rows,
            columns,
            inner,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            result.stride(0),
            SA_SCALAR=scale_a.ndim == 0,
            SB_SCALAR=scale_b.ndim == 0,
            HAS_BIAS=bias is not None,
            B_TRANSPOSED=b_transposed,
        )
    return result


def _fp8_blockwise_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    a_scale_col_major: bool = False,
) -> torch.Tensor:
    return fake_output(a, out, shape=(a.shape[0], b.shape[0]), dtype=torch.float32)


@custom_op(
    namespace="ayaka",
    name="fp8_blockwise_mm",
    mutates_args=["out"],
    reference=fp8_blockwise_mm_ref,
    fake_impl=_fp8_blockwise_mm_fake,
    dispatch_key="CUDA",
)
def fp8_blockwise_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    a_scale_col_major: bool = False,
) -> torch.Tensor:
    """Compute the 128-block-scaled ``sum_k ((a @ b^T) * scale_a * scale_b) + bias``.

    Args:
        a: ``[M, K]`` ``torch.float8_e4m3fn`` with ``stride(1) == 1``.
        b: ``[N, K]`` ``torch.float8_e4m3fn`` source layout with ``stride(1) == 1``.
        scale_a: ``[M, ceil(K / 128)]`` row-major, or ``[ceil(K / 128), M]`` when
            ``a_scale_col_major`` is set.
        scale_b: ``[ceil(N / 128), ceil(K / 128)]`` weight scales, row-major.
        bias: Optional ``[N]`` additive term.
        out: Optional ``[M, N]`` output buffer (fp32, fp16, or bf16). A new fp32
            buffer is allocated when omitted.
        a_scale_col_major: Select the col-major activation-scale layout. The flag
            is explicit because a square ``M == ceil(K / 128)`` makes the two
            layouts shape-ambiguous.

    Returns:
        The ``[M, N]`` output tensor, same object as ``out`` when provided.

    Raises:
        TypeError: On non-E4M3 operands, non-float scales, or unsupported ``out``.
        ValueError: On mismatched shapes, devices, or unsupported layouts.
    """

    validate_operand(a, "a", dtype=require_fp8_dtype(), allow_transposed=False)
    validate_operand(b, "b", dtype=require_fp8_dtype(), allow_transposed=False)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"inner dimensions do not match: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    rows, inner, columns = int(a.shape[0]), int(a.shape[1]), int(b.shape[0])
    scale_k = div_ceil(inner, _SCALE_BLOCK_K)
    scale_a_shape = (scale_k, rows) if a_scale_col_major else (rows, scale_k)
    validate_block_scale(scale_a, "scale_a", a, scale_a_shape)
    validate_block_scale(scale_b, "scale_b", a, (div_ceil(columns, _SCALE_BLOCK_K), scale_k))
    validate_bias(bias, a, columns)
    result = _mm_output(a, out, rows, columns)
    if rows == 0 or columns == 0:
        return result

    a_arg = fp8_kernel_view(a)
    b_arg = fp8_kernel_view(b)
    bias_arg = bias if bias is not None else a_arg

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(columns, meta["BLOCK_N"]),)

    with torch.cuda.device(a.device):
        cast(Any, _fp8_blockwise_mm_kernel)[grid](
            a_arg,
            b_arg,
            scale_a,
            scale_b,
            bias_arg,
            result,
            rows,
            columns,
            inner,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(0),
            result.stride(0),
            A_SCALE_COLMAJOR=a_scale_col_major,
            HAS_BIAS=bias is not None,
            SCALE_BLOCK_K=_SCALE_BLOCK_K,
        )
    return result
