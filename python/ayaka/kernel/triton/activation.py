from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh as _tanh

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import (
    fake_output,
    prepare_output,
    require_contiguous,
    require_cuda,
    require_dtype,
    require_tensor,
)
from ayaka.kernel.triton.reference.activation import (
    gelu_and_mul_ref,
    gelu_quick_ref,
    gelu_ref,
    gelu_tanh_and_mul_ref,
    gelu_tanh_ref,
    silu_and_mul_ref,
)
from ayaka.utils.torch_utils import compute_torch_dtypes

# Triton's libdevice stubs also describe interpreter-only None results.
tanh: Any = _tanh

_ACT_SILU = 0
_ACT_GELU = 1
_ACT_GELU_TANH = 2
_NUM_WARPS = 8
_BLOCK_SIZE = 1024
_SUPPORTED_DTYPES = compute_torch_dtypes()

# 1 / sqrt(2), used by exact GELU (erf form).
_GELU_KALPHA = tl.constexpr(0.7071067811865475)
# tanh-approximate GELU constants.
_GELU_TANH_KALPHA = tl.constexpr(0.044715)
_GELU_TANH_KBETA = tl.constexpr(0.7978845608028654)
# Quick-GELU / gelu_quick_act constant.
_QUICK_GELU_ALPHA = tl.constexpr(1.702)


@triton.jit
def _act_and_mul_kernel(
    input_ptr,
    output_ptr,
    d,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    tile = tl.program_id(axis=1).to(tl.int64)
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    input_row = input_ptr + row * (2 * d)
    x = tl.load(input_row + offsets, mask=mask, other=0.0)
    gate = tl.load(input_row + d + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    if ACTIVATION == 0:
        activated_f32 = x_f32 / (1.0 + tl.exp(-x_f32))
    elif ACTIVATION == 1:
        activated_f32 = x_f32 * (0.5 * (1.0 + tl.erf(x_f32 * _GELU_KALPHA)))
    else:
        x_cubed = x_f32 * x_f32 * x_f32
        cdf = 0.5 * (1.0 + tanh(_GELU_TANH_KBETA * (x_f32 + _GELU_TANH_KALPHA * x_cubed)))
        activated_f32 = x_f32 * cdf

    # Match activation<T>: FP32 transcendental evaluation followed by T cast.
    activated = activated_f32.to(x.dtype)
    result = activated * gate
    output_row = output_ptr + row * d
    tl.store(output_row + offsets, result, mask=mask)


@triton.jit
def _gelu_quick_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(axis=0).to(tl.int64)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    activated_f32 = x_f32 / (1.0 + tl.exp(-_QUICK_GELU_ALPHA * x_f32))
    tl.store(output_ptr + offsets, activated_f32.to(x.dtype), mask=mask)


@triton.jit
def _gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(axis=0).to(tl.int64)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    if ACTIVATION == 1:
        activated_f32 = x_f32 * (0.5 * (1.0 + tl.erf(x_f32 * _GELU_KALPHA)))
    else:
        x_cubed = x_f32 * x_f32 * x_f32
        cdf = 0.5 * (1.0 + tanh(_GELU_TANH_KBETA * (x_f32 + _GELU_TANH_KALPHA * x_cubed)))
        activated_f32 = x_f32 * cdf

    tl.store(output_ptr + offsets, activated_f32.to(x.dtype), mask=mask)


def _validate_input(input: torch.Tensor) -> None:
    require_tensor(input, "input")
    require_cuda(input, "input")
    require_dtype(input, "input", _SUPPORTED_DTYPES)
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    require_contiguous(input, "input")


def _act_and_mul(
    input: torch.Tensor,
    activation: int,
    out: torch.Tensor | None,
) -> torch.Tensor:
    _validate_input(input=input)
    last_dim = input.shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(f"input.shape[-1] must be even; got {last_dim}")

    d = last_dim // 2
    output_shape = (*input.shape[:-1], d)
    output = prepare_output(input, out, shape=output_shape)
    require_contiguous(output, "out")
    if output.numel() == 0:
        return output

    num_rows = input.numel() // last_dim
    grid = (num_rows, triton.cdiv(d, _BLOCK_SIZE))
    with torch.cuda.device(input.device):
        cast(Any, _act_and_mul_kernel)[grid](
            input,
            output,
            d,
            ACTIVATION=activation,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
    return output


def _gelu_impl(
    input: torch.Tensor,
    activation: int,
    out: torch.Tensor | None,
) -> torch.Tensor:
    _validate_input(input=input)
    output = prepare_output(input, out, shape=tuple(input.shape))
    require_contiguous(output, "out")
    if input.numel() == 0:
        return output

    grid = (triton.cdiv(input.numel(), _BLOCK_SIZE),)
    with torch.cuda.device(input.device):
        cast(Any, _gelu_kernel)[grid](
            input,
            output,
            input.numel(),
            ACTIVATION=activation,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
    return output


# --- References and Meta implementations for torch.compile and fallback ---


def _act_and_mul_fake(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return fake_output(input, out, shape=(*input.shape[:-1], input.shape[-1] // 2))


@custom_op(
    namespace="ayaka",
    name="silu_and_mul",
    fake_impl=_act_and_mul_fake,
    reference=silu_and_mul_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
def silu_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``silu(input[..., :d]) * input[..., d:]``."""
    return _act_and_mul(input, _ACT_SILU, out)


@custom_op(
    namespace="ayaka",
    name="gelu_and_mul",
    fake_impl=_act_and_mul_fake,
    reference=gelu_and_mul_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
def gelu_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute exact-erf GELU on the first half and multiply by the second."""
    return _act_and_mul(input, _ACT_GELU, out)


@custom_op(
    namespace="ayaka",
    name="gelu_tanh_and_mul",
    fake_impl=_act_and_mul_fake,
    reference=gelu_tanh_and_mul_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute tanh-approximate GELU on the first half and multiply by the second."""
    return _act_and_mul(input, _ACT_GELU_TANH, out)


@custom_op(
    namespace="ayaka",
    name="gelu_quick",
    out_shape="input",
    reference=gelu_quick_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
def gelu_quick(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x * sigmoid(1.702 * x)`` without a multiply-gate input."""
    _validate_input(input)
    output = prepare_output(input, out, shape=tuple(input.shape))
    require_contiguous(output, "out")
    if input.numel() == 0:
        return output

    grid = (triton.cdiv(input.numel(), _BLOCK_SIZE),)
    with torch.cuda.device(input.device):
        cast(Any, _gelu_quick_kernel)[grid](
            input,
            output,
            input.numel(),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
    return output


@custom_op(
    namespace="ayaka",
    name="gelu",
    out_shape="input",
    reference=gelu_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
def gelu(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute exact-erf GELU without a multiply-gate input."""
    return _gelu_impl(input, _ACT_GELU, out)


@custom_op(
    namespace="ayaka",
    name="gelu_tanh",
    out_shape="input",
    reference=gelu_tanh_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
def gelu_tanh(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute tanh-approximate GELU without a multiply-gate input."""
    return _gelu_impl(input, _ACT_GELU_TANH, out)
