"""FP8-E4M3 activation quantization.

Two kernels:

* :func:`per_token_group_quant_fp8` — dynamic quantization with one fp32 scale
  per token per ``group_size``-element K group. This is the activation scheme
  of the DeepSeek-V3-style block-FP8 checkpoints (``activation_scheme:
  dynamic``) and the producer of the ``[M, ceil(K / 128)]`` ``scale_a`` that
  :func:`ayaka.kernel.triton.gemm.fp8_gemm.fp8_blockwise_mm` expects.
* :func:`per_token_quant_fp8` — dynamic W8A8 rowwise activation quantization
  with exactly one fp32 scale per token over the whole K row. This is the
  activation scheme of the rowwise FP8 recipe; unlike the group kernel it has
  no power-of-two group-size or ``K % group_size == 0`` requirement.
* :func:`per_token_group_quant_mxfp8` — the MXFP8 1x32 scheme: one UE8M0
  (power-of-two, 8-bit exponent) scale per token per 32-element K group. The
  zero group is the only group that emits code ``0`` (which the shared E8M0
  decoder reads as ``+0.0``); non-zero groups stay in ``[-126, 127]`` so no
  real scale collides with the zero code or with ``0xFF`` (NaN).
* :func:`static_quant_fp8` — one broadcast per-tensor scale, read from device
  memory *inside* the kernel so the launch has no host-side dependency on tensor
  values and stays CUDA-graph safe. This is the W8A8 activation scheme of the
  per-tensor (modelopt ``MIXED_PRECISION``) checkpoints.

All three write ``torch.float8_e4m3fn`` storage on every architecture: where
native FP8 arithmetic is unavailable (sm < 89) the kernel packs E4M3 codes with
:func:`ayaka.kernel.triton.fp8_compat.e4m3_f32_to_u8` and the host views the
bytes as FP8. The storage format — and therefore the GEMM operand contract —
does not change with the device.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.caps import Cap
from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import (
    fake_output,
    fake_tensor,
    prepare_output,
    require_cuda,
    require_dtype,
    require_tensor,
)
from ayaka.kernel.triton.fp8_compat import (
    e4m3_f32_to_u8,
    e4m3_native_cx,
    fp8_kernel_view,
)
from ayaka.kernel.triton.reference.quant import (
    per_token_group_quant_mxfp8_ref,
    per_token_group_quant_ref,
    per_token_quant_ref,
    static_quant_ref,
)
from ayaka.types import DType
from ayaka.utils.math_utils import div_ceil

__all__ = [
    "per_token_group_quant_fp8",
    "per_token_group_quant_mxfp8",
    "per_token_quant_fp8",
    "static_quant_fp8",
]

FP8 = torch.float8_e4m3fn
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

#: E4M3 finite max, from the dtype registry so a format change cannot leave the
#: kernels clamping to a stale constant. Wrapped in ``tl.constexpr`` because
#: Triton 3.8 rejects plain module globals inside ``@triton.jit`` bodies.
_E4M3_MAX = tl.constexpr(float(DType.FP8_E4M3.max_finite or 448.0))

#: K width of one dynamic activation scale group.
_GROUP = 128

#: A group whose amax is zero would divide by zero; the floor keeps the scale
#: representable while mapping the group to exact zeros.
_GROUP_FLOOR = 1e-10

#: Rows per program in the dynamic kernel. One program covers a whole group
#: (no K mask is needed: the host requires K % group_size == 0).
_BLOCK_M = 32

#: Row tile and K chunk of the rowwise kernel. The K loop is walked twice
#: (amax, then quantize) so an arbitrary K never exceeds the register budget.
_ROW_BLOCK_M = 16

#: K elements loaded per iteration of the rowwise kernel.
_ROW_BLOCK_K = 1024

#: Elements per program in the static kernel.
_STATIC_BLOCK = 1024

#: K width of one MXFP8 UE8M0 scale group.
_MX_GROUP = 32

#: Exponent bounds of a real (non-zero) UE8M0 scale. Code ``0`` is reserved for
#: an all-zero group and code ``0xFF`` decodes to NaN, so a live scale stays in
#: ``[-126, 127]``.
_MX_MIN_EXP = -126.0
_MX_MAX_EXP = 127.0


@triton.jit
def _per_token_group_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_g * GROUP + tl.arange(0, GROUP)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    s = amax / _E4M3_MAX
    y = x / s[:, None]
    if e4m3_native_cx():
        yq = tl.clamp(y, -_E4M3_MAX, _E4M3_MAX).to(tl.float8e4nv)
    else:
        yq = e4m3_f32_to_u8(y)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk,
        yq,
        mask=m_mask[:, None],
    )
    tl.store(s_ptr + offs_m * stride_sm + pid_g * stride_sk, s, mask=m_mask)


@triton.jit
def _per_token_group_quant_mxfp8_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    e4m3_max,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_g * GROUP + tl.arange(0, GROUP)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    exponent = tl.ceil(tl.log2(tl.maximum(amax / e4m3_max, 1e-38)))
    exponent = tl.minimum(tl.maximum(exponent, -126.0), 127.0)
    scale = tl.exp2(exponent)
    # The division is pinned to fp32 because the runtime scalar keeps the
    # expression at Python float width otherwise.
    y = (x / scale[:, None]).to(tl.float32)
    # An all-zero group encodes code 0 (decoded as +0.0) and still divides by
    # 2**-126, which maps its zeros to exact zeros.
    code = tl.where(amax <= 0.0, 0.0, exponent + 127.0)  # type: ignore
    if e4m3_native_cx():
        yq = tl.clamp(y, -_E4M3_MAX, _E4M3_MAX).to(tl.float8e4nv)
    else:
        yq = e4m3_f32_to_u8(y)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk,
        yq,
        mask=m_mask[:, None],
    )
    tl.store(
        s_ptr + offs_m * stride_sm + pid_g * stride_sk,
        code.to(tl.uint8),
        mask=m_mask,
    )


@triton.jit
def _per_token_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    s = tl.maximum(amax, 1e-10) / _E4M3_MAX
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        y = x / s[:, None]
        if e4m3_native_cx():
            yq = tl.clamp(y, -_E4M3_MAX, _E4M3_MAX).to(tl.float8e4nv)
        else:
            yq = e4m3_f32_to_u8(y)
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk,
            yq,
            mask=m_mask[:, None] & k_mask[None, :],
        )
    tl.store(s_ptr + offs_m, s, mask=m_mask)


@triton.jit
def _static_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    inv = 1.0 / tl.load(s_ptr).to(tl.float32)
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv
    v = tl.minimum(tl.maximum(v, -_E4M3_MAX), _E4M3_MAX)
    if e4m3_native_cx():
        yq = v.to(tl.float8e4nv)
    else:
        yq = e4m3_f32_to_u8(v)
    tl.store(y_ptr + offs, yq, mask=mask)


def _validate_input(x: torch.Tensor, name: str) -> None:
    require_tensor(x, name)
    require_dtype(x, name, _FLOAT_DTYPES)
    require_cuda(x, name)
    if x.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")


def _per_token_group_quant_fake(
    x: torch.Tensor, group_size: int = _GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, k = x.shape
    rows = math.prod(lead) if lead else 1
    return (
        fake_tensor(x, shape=(rows, k), dtype=FP8),
        fake_tensor(x, shape=(rows, k // group_size), dtype=torch.float32),
    )


@custom_op(
    namespace="ayaka",
    name="per_token_group_quant_fp8",
    reference=per_token_group_quant_ref,
    fake_impl=_per_token_group_quant_fake,
    dispatch_key="CUDA",
)
def per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int = _GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to E4M3 with one fp32 scale per token per K group.

    Args:
        x: ``[..., K]`` fp16/bf16/fp32 CUDA tensor.
        group_size: K elements sharing one scale. Must be a power of two that
            divides ``K``.

    Returns:
        ``(x_fp8 [M, K], x_scale [M, K // group_size] fp32)`` with
        ``M = numel(x) // K``.

    Raises:
        TypeError: On a non-float or non-tensor ``x``.
        ValueError: On a CPU tensor, a bad ``group_size``, or ``K`` not
            divisible by ``group_size``.
    """

    _validate_input(x, "x")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be an integer")
    if group_size < 1 or group_size & (group_size - 1):
        raise ValueError("group_size must be a positive power of two")
    k = int(x.shape[-1])
    if k % group_size:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    rows = int(x2d.shape[0])
    if rows == 0:
        return (
            torch.empty((0, k), dtype=FP8, device=x.device),
            torch.empty((0, k // group_size), dtype=torch.float32, device=x.device),
        )
    y = torch.empty((rows, k), dtype=FP8, device=x.device)
    scale = torch.empty((rows, k // group_size), dtype=torch.float32, device=x.device)
    y_bytes = fp8_kernel_view(y)
    grid = (div_ceil(rows, _BLOCK_M), k // group_size)
    cast(Any, _per_token_group_quant_kernel)[grid](
        x2d,
        y_bytes,
        scale,
        rows,
        x2d.stride(0),
        x2d.stride(1),
        y_bytes.stride(0),
        y_bytes.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_M=_BLOCK_M,
        GROUP=group_size,
        num_warps=4,
    )
    return y, scale


def _per_token_group_quant_mxfp8_fake(
    x: torch.Tensor, group_size: int = _MX_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, k = x.shape
    rows = math.prod(lead) if lead else 1
    return (
        fake_tensor(x, shape=(rows, k), dtype=FP8),
        fake_tensor(x, shape=(rows, k // group_size), dtype=torch.uint8),
    )


@custom_op(
    namespace="ayaka",
    name="per_token_group_quant_mxfp8",
    reference=per_token_group_quant_mxfp8_ref,
    fake_impl=_per_token_group_quant_mxfp8_fake,
    dispatch_key="CUDA",
)
def per_token_group_quant_mxfp8(
    x: torch.Tensor, group_size: int = _MX_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to E4M3 with one UE8M0 scale per token per K group.

    Args:
        x: ``[..., K]`` fp16/bf16/fp32 CUDA tensor.
        group_size: K elements sharing one power-of-two scale. The MXFP8 recipe
            uses 32; the encoder is group-size agnostic and requires the group
            size to divide ``K``.

    Returns:
        ``(x_fp8 [M, K], x_scale [M, K // group_size] uint8)`` with
        ``M = numel(x) // K``. Scale code ``0`` marks an all-zero group and
        decodes to ``+0.0``; codes ``1..254`` decode to ``2 ** (code - 127)``.

    Raises:
        TypeError: On a non-float or non-tensor ``x``.
        ValueError: On a CPU tensor, a bad ``group_size``, or ``K`` not
            divisible by ``group_size``.
    """

    _validate_input(x, "x")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("group_size must be an integer")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    k = int(x.shape[-1])
    if k % group_size:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    rows = int(x2d.shape[0])
    if rows == 0:
        return (
            torch.empty((0, k), dtype=FP8, device=x.device),
            torch.empty((0, k // group_size), dtype=torch.uint8, device=x.device),
        )
    y = torch.empty((rows, k), dtype=FP8, device=x.device)
    scale = torch.empty((rows, k // group_size), dtype=torch.uint8, device=x.device)
    y_bytes = fp8_kernel_view(y)
    grid = (div_ceil(rows, _BLOCK_M), k // group_size)
    cast(Any, _per_token_group_quant_mxfp8_kernel)[grid](
        x2d,
        y_bytes,
        scale,
        rows,
        float(_E4M3_MAX.value),
        x2d.stride(0),
        x2d.stride(1),
        y_bytes.stride(0),
        y_bytes.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_M=_BLOCK_M,
        GROUP=group_size,
        num_warps=4,
    )
    return y, scale


def _per_token_quant_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, k = x.shape
    rows = math.prod(lead) if lead else 1
    return (
        fake_tensor(x, shape=(rows, k), dtype=FP8),
        fake_tensor(x, shape=(rows,), dtype=torch.float32),
    )


@custom_op(
    namespace="ayaka",
    name="per_token_quant_fp8",
    reference=per_token_quant_ref,
    fake_impl=_per_token_quant_fake,
    dispatch_key="CUDA",
)
def per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to E4M3 with one fp32 scale per token over the whole row.

    The rowwise companion of :func:`per_token_group_quant_fp8`: ``K`` may be any
    positive width and the scale covers all of it.

    Args:
        x: ``[..., K]`` fp16/bf16/fp32 CUDA tensor.

    Returns:
        ``(x_fp8 [M, K], x_scale [M] fp32)`` with ``M = numel(x) // K``.

    Raises:
        TypeError: On a non-float or non-tensor ``x``.
        ValueError: On a CPU tensor or a zero-width last dimension.
    """

    _validate_input(x, "x")
    k = int(x.shape[-1])
    if k < 1:
        raise ValueError("x must have a positive last dimension")
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    rows = int(x2d.shape[0])
    if rows == 0:
        return (
            torch.empty((0, k), dtype=FP8, device=x.device),
            torch.empty((0,), dtype=torch.float32, device=x.device),
        )
    y = torch.empty((rows, k), dtype=FP8, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    y_bytes = fp8_kernel_view(y)
    cast(Any, _per_token_quant_kernel)[(div_ceil(rows, _ROW_BLOCK_M),)](
        x2d,
        y_bytes,
        scale,
        rows,
        k,
        x2d.stride(0),
        x2d.stride(1),
        y_bytes.stride(0),
        y_bytes.stride(1),
        BLOCK_M=_ROW_BLOCK_M,
        BLOCK_K=_ROW_BLOCK_K,
        num_warps=4,
    )
    return y, scale


def _static_quant_fake(
    x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    return fake_output(x, out, dtype=FP8)


@custom_op(
    namespace="ayaka",
    name="static_quant_fp8",
    mutates_args=["out"],
    reference=static_quant_ref,
    fake_impl=_static_quant_fake,
    dispatch_key="CUDA",
    caps=Cap.CUDAGRAPH_SAFE,
)
def static_quant_fp8(
    x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize ``x`` to E4M3 under one broadcast per-tensor scale.

    ``scale`` is read from device memory inside the kernel, so no host value is
    ever needed: the op is CUDA-graph safe and allocation-shape stable.

    Args:
        x: fp16/bf16/fp32 CUDA tensor of any shape.
        scale: One-element CUDA tensor (the checkpoint's ``input_scale``).
        out: Optional same-shape E4M3 buffer written in place.

    Returns:
        The E4M3 tensor, the same object as ``out`` when provided.

    Raises:
        TypeError: On non-float ``x``, non-tensor ``scale``, or bad ``out``.
        ValueError: On CPU tensors, a non-scalar ``scale``, or shape mismatch.
    """

    _validate_input(x, "x")
    if not isinstance(scale, torch.Tensor):
        raise TypeError("scale must be a torch.Tensor")
    require_cuda(scale, "scale")
    if scale.numel() != 1:
        raise ValueError("scale must contain exactly one element")
    result = prepare_output(x, out, dtype=FP8, name="out", like_name="x")
    if x.numel() == 0:
        return result
    x_flat = x.reshape(-1)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    y_bytes = fp8_kernel_view(result)
    n = int(x_flat.numel())
    cast(Any, _static_quant_kernel)[(div_ceil(n, _STATIC_BLOCK),)](
        x_flat,
        y_bytes,
        scale,
        n,
        BLOCK=_STATIC_BLOCK,
        num_warps=4,
    )
    return result
