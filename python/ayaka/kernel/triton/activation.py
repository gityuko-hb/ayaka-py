from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh as _tanh

# Triton's libdevice stubs also describe interpreter-only None results.
tanh: Any = _tanh

_ACT_SILU = 0
_ACT_GELU = 1
_ACT_GELU_TANH = 2
_NUM_WARPS = 4
_BLOCK_SIZE = 256
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

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
        activated_f32 = x_f32 * (0.5 * (1.0 + tl.erf(x_f32 * 0.7071067811865476)))
    else:
        x_cubed = x_f32 * x_f32 * x_f32
        cdf = 0.5 * (1.0 + tanh(0.7978845608028654 * (x_f32 + 0.044715 * x_cubed)))
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
    activated_f32 = x_f32 / (1.0 + tl.exp(-1.702 * x_f32))
    tl.store(output_ptr + offsets, activated_f32.to(x.dtype), mask=mask)

def _validate_input(input: torch.Tensor) -> None:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if input.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"input dtype must be float16, bfloat16, or float32; got {input.dtype}")
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")


def _prepare_output(input: torch.Tensor,
                    shape: tuple[int, ...],
                    out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, device=input.device, dtype=input.dtype)
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out.device != input.device:
        raise ValueError("out must be on the same CUDA device as input")
    if out.dtype != input.dtype:
        raise TypeError("out must have the same dtype as input")
    if tuple(out.shape) != shape:
        raise ValueError(f"out has shape {tuple(out.shape)}, expected {shape}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out

def _act_and_mul(
    input: torch.Tensor,
    activation: int,
    out: torch.Tensor | None
) -> torch.Tensor:
    _validate_input(input=input)
    last_dim = input.shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(f"input.shape[-1] must be even; got {last_dim}")

    d = last_dim // 2
    output_shape = (*input.shape[:-1], d)
    output = _prepare_output(input=input, shape=output_shape, out=out)
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

def silu_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``silu(input[..., :d]) * input[..., d:]``."""

    return _act_and_mul(input, _ACT_SILU, out)


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute exact-erf GELU on the first half and multiply by the second."""

    return _act_and_mul(input, _ACT_GELU, out)


def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute tanh-approximate GELU on the first half and multiply by the second."""

    return _act_and_mul(input, _ACT_GELU_TANH, out)


def gelu_quick(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x * sigmoid(1.702 * x)`` without a multiply-gate input."""

    _validate_input(input)
    output = _prepare_output(input, tuple(input.shape), out)
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
