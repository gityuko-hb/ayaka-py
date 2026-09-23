"""INT8 scaled GEMM: INT32 tensor-core accumulation with an FP32 epilogue."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton.swizzle import grouped_pid
from ayaka.utils.math_utils import require_cuda, validate_output_dtype

__all__ = ["int8_scaled_mm"]

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_INT8_CONFIGS = [
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


@triton.autotune(configs=_INT8_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _int8_scaled_mm_kernel(
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        if B_TRANSPOSED:
            b_mask = (offs_n[:, None] < N) & (k[None, :] < K)
        else:
            b_mask = (k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int8)
        b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int8)
        if B_TRANSPOSED:
            b = tl.trans(b)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    y = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        y += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _validate_operand(tensor: torch.Tensor, name: str, *, allow_transposed: bool) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype is not torch.int8:
        raise TypeError(f"{name} must have dtype torch.int8")
    require_cuda(tensor, name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if tensor.stride(1) != 1 and not (allow_transposed and tensor.stride(0) == 1):
        layout = "stride(1) == 1" if not allow_transposed else "stride(1) == 1 or stride(0) == 1"
        raise ValueError(f"{name} must have {layout}")


def _validate_scale(scale: torch.Tensor, name: str, reference: torch.Tensor, size: int) -> None:
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if scale.device != reference.device:
        raise ValueError(f"{name} and a must be on the same CUDA device")
    if scale.dtype is not torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if scale.ndim != 1 or scale.numel() != size:
        raise ValueError(f"{name} must have shape [{size}]")
    if scale.stride(0) != 1:
        raise ValueError(f"{name} must be contiguous")


def _validate_bias(bias: torch.Tensor | None, reference: torch.Tensor, size: int) -> None:
    if bias is None:
        return
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias must be a torch.Tensor")
    if bias.device != reference.device:
        raise ValueError("bias and a must be on the same CUDA device")
    if bias.dtype not in _FLOAT_DTYPES:
        raise TypeError("bias must be float16, bfloat16, or float32")
    if tuple(bias.shape) != (size,):
        raise ValueError(f"bias must have shape [{size}]")


def _prepare_output(
    a: torch.Tensor,
    out: torch.Tensor | None,
    shape: tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if out is None:
        validate_output_dtype(out_dtype)
        return torch.empty(shape, device=a.device, dtype=out_dtype)
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out.device != a.device:
        raise ValueError("out and a must be on the same CUDA device")
    validate_output_dtype(out.dtype)
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}")
    if out.stride(1) != 1:
        raise ValueError("out must have stride(1) == 1")
    return out


def _int8_scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)


def _int8_scaled_mm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
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


@custom_op(
    namespace="ayaka",
    name="int8_scaled_mm",
    mutates_args=["out"],
    reference=_int8_scaled_mm_reference,
    fake_impl=_int8_scaled_mm_fake,
    dispatch_key="CUDA",
)
def int8_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute ``(a @ b) * scale_a * scale_b + bias`` for INT8 operands.

    Args:
        a: ``[M, K]`` ``torch.int8`` row-major with ``stride(1) == 1`` and ``K`` a
            multiple of 16.
        b: ``[K, N]`` ``torch.int8``, or a transposed ``[N, K]`` weight view with
            ``stride(0) == 1``.
        scale_a: ``[M]`` float32 dequantization scale applied to the INT32
            accumulator after the dot product.
        scale_b: ``[N]`` float32 dequantization scale.
        bias: Optional ``[N]`` additive term (fp32, fp16, or bf16).
        out: Optional ``[M, N]`` output buffer (fp32, fp16, or bf16). A new
            buffer is allocated when omitted.
        out_dtype: Output dtype used only when ``out`` is omitted.

    Returns:
        The ``[M, N]`` output tensor, same object as ``out`` when provided.

    Raises:
        TypeError: On non-INT8 operands, non-float32 scales, or unsupported bias.
        ValueError: On mismatched shapes, devices, K alignment, or bad ``out``.
    """

    _validate_operand(a, "a", allow_transposed=False)
    _validate_operand(b, "b", allow_transposed=True)
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions do not match: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    rows, inner, columns = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if inner % 16:
        raise ValueError("K must be a multiple of 16, matching the CUDA API")
    _validate_scale(scale_a, "scale_a", a, rows)
    _validate_scale(scale_b, "scale_b", a, columns)
    _validate_bias(bias, a, columns)
    result = _prepare_output(a, out, (rows, columns), out_dtype)
    if rows == 0 or columns == 0:
        return result

    b_transposed = b.stride(0) == 1 and b.stride(1) != 1
    bias_arg = bias if bias is not None else a

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(columns, meta["BLOCK_N"]),)

    with torch.cuda.device(a.device):
        cast(Any, _int8_scaled_mm_kernel)[grid](
            a,
            b,
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
            HAS_BIAS=bias is not None,
            B_TRANSPOSED=b_transposed,
        )
    return result
