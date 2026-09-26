"""MXFP8 W8A8 GEMM: E4M3 operands with UE8M0 1x32 scales.

The MXFP8 recipe stores both activation and weight as E4M3, each with one
power-of-two ``UE8M0`` scale per row and per 32-element K group. The two FP8
operands accumulate in FP32 and each 32-K partial product is rescaled by the
product of the activation and weight scales before it is added to the output
accumulator — the same block-rescale structure as the 128-block GEMM, with a
32-wide block and power-of-two scales.

Where native FP8 arithmetic is unavailable (sm < 89) the kernel decodes both
operands to BF16 in-register and uses the BF16 tensor-core dot; E4M3 values are
exact in BF16, so the emulated accumulation is equivalent for finite inputs.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import (
    fake_output,
    prepare_output,
    require_last_dim_stride1,
    require_tensor,
)
from ayaka.kernel.triton.fp8_compat import (
    e4m3_native_cx,
    e4m3_u8_to_f32,
    e8m0_u8_to_f32,
    fp8_kernel_view,
)
from ayaka.kernel.triton.gemm._common import (
    OUTPUT_DTYPES,
    require_fp8_dtype,
    validate_bias,
    validate_operand,
)
from ayaka.kernel.triton.reference.gemm import mxfp8_scaled_mm_ref

__all__ = ["mxfp8_scaled_mm"]

#: K width of one MXFP8 UE8M0 scale group.
MX_BLOCK = 32

_TL_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _mxfp8_scaled_mm_kernel(
    a_ptr,
    sa_ptr,
    w_ptr,
    sw_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_sam,
    stride_sak,
    stride_wn,
    stride_wk,
    stride_swn,
    stride_swk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        if e4m3_native_cx():
            product = tl.dot(
                tl.load(a_ptrs, mask=m_mask[:, None], other=0.0),
                tl.trans(tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)),
                out_dtype=tl.float32,
            )
        else:
            product = tl.dot(
                e4m3_u8_to_f32(tl.load(a_ptrs, mask=m_mask[:, None], other=0)).to(tl.bfloat16),
                tl.trans(
                    e4m3_u8_to_f32(tl.load(w_ptrs, mask=n_mask[:, None], other=0)).to(tl.bfloat16)
                ),
                out_dtype=tl.float32,
            )
        scale_a = e8m0_u8_to_f32(
            tl.load(
                sa_ptr + offs_m * stride_sam + (k // BLOCK_K) * stride_sak, mask=m_mask, other=0
            )
        ).to(tl.float32)
        scale_w = e8m0_u8_to_f32(
            tl.load(
                sw_ptr + offs_n * stride_swn + (k // BLOCK_K) * stride_swk, mask=n_mask, other=0
            )
        ).to(tl.float32)
        acc += product * scale_a[:, None] * scale_w[None, :]
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(OUT_TYPE), mask=m_mask[:, None] & n_mask[None, :])


def _validate_mx_scale(
    scale: torch.Tensor,
    name: str,
    reference: torch.Tensor,
    shape: tuple[int, int],
) -> None:
    require_tensor(scale, name)
    if scale.device != reference.device:
        raise ValueError(f"{name} and a must be on the same CUDA device")
    if scale.dtype is not torch.uint8:
        raise TypeError("scale must have dtype uint8 (UE8M0)")
    if tuple(scale.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _mxfp8_scaled_mm_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return fake_output(a, out, shape=(a.shape[0], weight.shape[0]), dtype=torch.float32)


@custom_op(
    namespace="ayaka",
    name="mxfp8_scaled_mm",
    mutates_args=["out"],
    reference=mxfp8_scaled_mm_ref,
    fake_impl=_mxfp8_scaled_mm_fake,
    dispatch_key="CUDA",
)
def mxfp8_scaled_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``(a @ weight^T)`` with E4M3 operands and UE8M0 1x32 scales.

    Args:
        a: ``[M, K]`` ``torch.float8_e4m3fn`` with ``stride(1) == 1``.
        a_scale: ``[M, K // 32]`` uint8 UE8M0 activation scales.
        weight: ``[N, K]`` ``torch.float8_e4m3fn`` with ``stride(1) == 1``.
        weight_scale: ``[N, K // 32]`` uint8 UE8M0 weight scales.
        bias: Optional ``[N]`` additive term, applied in the FP32 accumulator.
        out: Optional ``[M, N]`` output buffer (fp32, fp16, or bf16). A new fp32
            buffer is allocated when omitted.

    Returns:
        The ``[M, N]`` output tensor, the same object as ``out`` when provided.

    Raises:
        TypeError: On non-E4M3 operands, non-uint8 scales, or unsupported ``out``.
        ValueError: On mismatched shapes, devices, layouts, or ``K % 32 != 0``.
    """

    fp8_dtype = require_fp8_dtype()
    validate_operand(a, "a", dtype=fp8_dtype, allow_transposed=False)
    validate_operand(weight, "weight", dtype=fp8_dtype, allow_transposed=False)
    if a.shape[1] != weight.shape[1]:
        raise ValueError(
            f"inner dimensions do not match: a is {tuple(a.shape)}, weight is {tuple(weight.shape)}"
        )
    rows, inner, columns = int(a.shape[0]), int(a.shape[1]), int(weight.shape[0])
    if inner % MX_BLOCK:
        raise ValueError(f"K={inner} must be a multiple of the MXFP8 group size 32")
    scale_k = inner // MX_BLOCK
    _validate_mx_scale(a_scale, "a_scale", a, (rows, scale_k))
    _validate_mx_scale(weight_scale, "weight_scale", a, (columns, scale_k))
    validate_bias(bias, a, columns)
    result = prepare_output(
        a,
        out,
        shape=(rows, columns),
        dtype=torch.float32,
        dtypes=OUTPUT_DTYPES,
        like_name="a",
    )
    require_last_dim_stride1(result, "out")
    if rows == 0 or columns == 0:
        return result

    out_dtype = result.dtype if result.dtype in _TL_DTYPE else torch.float32
    a_arg = fp8_kernel_view(a)
    weight_arg = fp8_kernel_view(weight)
    bias_arg = bias if bias is not None else a_arg
    block_m = 64 if rows >= 64 else 16
    block_n = 64
    grid = (triton.cdiv(rows, block_m), triton.cdiv(columns, block_n))
    with torch.cuda.device(a.device):
        cast(Any, _mxfp8_scaled_mm_kernel)[grid](
            a_arg,
            a_scale,
            weight_arg,
            weight_scale,
            bias_arg,
            result,
            rows,
            columns,
            inner,
            a.stride(0),
            a.stride(1),
            a_scale.stride(0),
            a_scale.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            result.stride(0),
            result.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=MX_BLOCK,
            OUT_TYPE=_TL_DTYPE[out_dtype],
            num_warps=4,
            num_stages=3,
        )
    return result
