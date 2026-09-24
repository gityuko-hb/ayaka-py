"""FP8 W8A16 GEMM: E4M3 weights decoded onto 16-bit tensor cores."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import (
    fake_output,
    prepare_output,
    require_cuda,
    require_last_dim_stride1,
)
from ayaka.kernel.triton.fp8_compat import e4m3_u8_to_f32, e8m0_u8_to_f32
from ayaka.kernel.triton.gemm._common import validate_w8a16_scale, weight_bytes
from ayaka.kernel.triton.reference.gemm import fp8_weight_only_gemm_ref
from ayaka.kernel.triton.swizzle import grouped_pid
from ayaka.utils.math_utils import div_ceil

__all__ = ["fp8_matmul_channelwise", "fp8_weight_only_gemm"]

_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
_SCALE_FORMATS = ("e8m0", "float32")

_W8A16_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4
    ),
]


@triton.autotune(configs=_W8A16_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fp8_w8a16_kernel(
    a_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_om,
    scale_row_stride,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    SCALE_E8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = grouped_pid(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + kk
        a = tl.load(
            a_ptr + m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        raw = tl.load(
            w_ptr + n[:, None] * stride_wn + k[None, :] * stride_wk,
            mask=(n[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.uint8)
        w = e4m3_u8_to_f32(raw)
        sr = n[:, None] // BLOCK_SIZE_Y
        # A scale block that spans whole BLOCK_K tiles only needs one load per
        # iteration; otherwise the scale varies inside the tile.
        if BLOCK_SIZE_X % BLOCK_K == 0:
            sc = k0 // BLOCK_SIZE_X
            if SCALE_E8M0:
                sraw = tl.load(s_ptr + sr * scale_row_stride + sc, mask=n[:, None] < N, other=0).to(
                    tl.uint8
                )
                scale = e8m0_u8_to_f32(sraw)
            else:
                scale = tl.load(
                    s_ptr + sr * scale_row_stride + sc, mask=n[:, None] < N, other=0.0
                ).to(tl.float32)
        else:
            sc = k[None, :] // BLOCK_SIZE_X
            if SCALE_E8M0:
                sraw = tl.load(
                    s_ptr + sr * scale_row_stride + sc,
                    mask=(n[:, None] < N) & (k[None, :] < K),
                    other=0,
                ).to(tl.uint8)
                scale = e8m0_u8_to_f32(sraw)
            else:
                scale = tl.load(
                    s_ptr + sr * scale_row_stride + sc,
                    mask=(n[:, None] < N) & (k[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
        wdq = (w * scale).to(a.dtype)
        acc = tl.dot(a, tl.trans(wdq), acc=acc, out_dtype=tl.float32)
    tl.store(
        out_ptr + m[:, None] * stride_om + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def _fp8_weight_only_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_row_stride: int | None = None,
    block_size_y: int = 1,
    block_size_x: int = 128,
    scale_dtype: str = "e8m0",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return fake_output(input, out, shape=(input.shape[0], weight.shape[0]))


@custom_op(
    namespace="ayaka",
    name="fp8_weight_only_gemm",
    mutates_args=["out"],
    reference=fp8_weight_only_gemm_ref,
    fake_impl=_fp8_weight_only_gemm_fake,
    dispatch_key="CUDA",
)
def fp8_weight_only_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_row_stride: int | None = None,
    block_size_y: int = 1,
    block_size_x: int = 128,
    scale_dtype: str = "e8m0",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``input @ dequant(weight).T`` for fp16/bf16 activations.

    Args:
        input: ``[M, K]`` fp16 or bf16 CUDA tensor with ``stride(1) == 1``.
        weight: ``[N, K]`` CUDA tensor of raw E4M3 bytes (``uint8``) or a
            ``torch.float8_e4m3fn`` tensor, which is byte-viewed on the host.
        weight_scale: Flat or 2-D storage for ``scale[n // block_size_y,
            k // block_size_x]``. E8M0 bytes when ``scale_dtype="e8m0"``, float32
            otherwise.
        scale_row_stride: Distance between scale rows. Defaults to
            ``ceil(K / block_size_x)``, which packs rows without padding.
        block_size_y: Output rows sharing one scale value. Must be positive.
        block_size_x: K columns sharing one scale value. Must be positive.
        scale_dtype: ``"e8m0"`` or ``"float32"``.
        out: Optional ``[M, N]`` output buffer with the same dtype as ``input``.
            A new buffer is allocated when omitted.

    Returns:
        The ``[M, N]`` output tensor, same object as ``out`` when provided.

    Raises:
        TypeError: On unsupported input/weight/scale dtypes or bad ``out``.
        ValueError: On mismatched shapes, devices, or an undersized scale buffer.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("input must have dtype float16 or bfloat16")
    require_cuda(input, "input")
    if input.ndim != 2:
        raise ValueError("input must be 2-D")
    if input.stride(1) != 1:
        raise ValueError("input must have stride(1) == 1")
    weight_view = weight_bytes(weight)
    if weight_view.shape[1] != input.shape[1]:
        raise ValueError("weight must be [N, K] with the same K as input")
    rows, inner, columns = int(input.shape[0]), int(input.shape[1]), int(weight_view.shape[0])
    if block_size_x <= 0 or block_size_y <= 0:
        raise ValueError("block sizes must be positive")
    if scale_dtype not in _SCALE_FORMATS:
        raise ValueError("scale_dtype must be 'e8m0' or 'float32'")
    if scale_row_stride is None:
        scale_row_stride = div_ceil(inner, block_size_x)
    if scale_row_stride <= 0:
        raise ValueError("scale_row_stride must be positive")
    validate_w8a16_scale(
        weight_scale,
        input,
        scale_dtype=scale_dtype,
        block_size_y=block_size_y,
        scale_row_stride=scale_row_stride,
        columns=columns,
    )
    result = prepare_output(input, out, shape=(rows, columns), like_name="input")
    require_last_dim_stride1(result, "out")
    if rows == 0 or columns == 0:
        return result

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(columns, meta["BLOCK_N"]),)

    with torch.cuda.device(input.device):
        cast(Any, _fp8_w8a16_kernel)[grid](
            input,
            weight_view,
            weight_scale,
            result,
            rows,
            columns,
            inner,
            input.stride(0),
            input.stride(1),
            weight_view.stride(1),
            weight_view.stride(0),
            result.stride(0),
            scale_row_stride,
            BLOCK_SIZE_Y=block_size_y,
            BLOCK_SIZE_X=block_size_x,
            SCALE_E8M0=scale_dtype == "e8m0",
        )
    return result


def fp8_matmul_channelwise(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Per-output-channel float32 scales: ``weight_scale`` is ``[N]``."""
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if input.ndim != 2:
        raise ValueError("input must be 2-D")
    return fp8_weight_only_gemm(
        input,
        weight,
        weight_scale,
        scale_row_stride=1,
        block_size_y=1,
        block_size_x=int(input.shape[1]),
        scale_dtype="float32",
    )
