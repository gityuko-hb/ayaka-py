from __future__ import annotations
from typing import Any, cast

import torch
import triton
import triton.language as tl

_SUPPORTED_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_BLOCK_SIZE = 256
_NUM_WARPS = 8
_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    d,
    stride_input,
    stride_output,
    eps,
    WEIGHT_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq_lanes += x * x

    sum_sq = tl.sum(sum_sq_lanes, axis=0)
    rms_rcp = tl.rsqrt(sum_sq / d + eps)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = x * rms_rcp * (weight + WEIGHT_BIAS)
        tl.store(output_ptr + row * stride_output + offsets, output, mask=mask)


@triton.jit
def _rms_norm_quant_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    d,
    stride_input,
    stride_output,
    eps,
    WEIGHT_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq_lanes += x * x

    sum_sq = tl.sum(sum_sq_lanes, axis=0)
    rms_rcp = tl.rsqrt(sum_sq / d + eps)
    scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = x * rms_rcp * (weight + WEIGHT_BIAS) * scale_inv
        output = tl.maximum(-448.0, tl.minimum(output, 448.0))
        tl.store(output_ptr + row * stride_output + offsets, output.to(tl.float8e4nv), mask=mask)


@triton.jit
def _qk_rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    d,
    num_heads,
    stride_input_n,
    stride_input_h,
    stride_output_n,
    stride_output_h,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    job = tl.program_id(axis=0).to(tl.int64)
    batch_idx = job // num_heads
    head_idx = job - batch_idx * num_heads
    input_base = batch_idx * stride_input_n + head_idx * stride_input_h
    output_base = batch_idx * stride_output_n + head_idx * stride_output_h
    lanes = tl.arange(0, BLOCK_SIZE)
    sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq_lanes += x * x

    sum_sq = tl.sum(sum_sq_lanes, axis=0)
    rms_rcp = tl.rsqrt(sum_sq / d + eps)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + output_base + offsets, x * rms_rcp * weight, mask=mask)


@triton.jit
def _fused_add_prepare_kernel(
    input_ptr,
    residual_ptr,
    scratch_ptr,
    rstd_ptr,
    d,
    stride_input,
    stride_residual,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        input_value = tl.load(
            input_ptr + row * stride_input + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        residual_value = tl.load(
            residual_ptr + row * stride_residual + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = input_value + residual_value
        sum_sq_lanes += x * x
        # residual receives scalar_t rounding, scratch preserves the FP32 x.
        tl.store(residual_ptr + row * stride_residual + offsets, x, mask=mask)
        tl.store(scratch_ptr + row * d + offsets, x, mask=mask)

    sum_sq = tl.sum(sum_sq_lanes, axis=0)
    tl.store(rstd_ptr + row, tl.rsqrt(sum_sq / d + eps))


@triton.jit
def _fused_add_apply_kernel(
    scratch_ptr,
    rstd_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    d,
    stride_output,
    WEIGHT_BIAS: tl.constexpr,
    QUANTIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    tile = tl.program_id(axis=1).to(tl.int64)
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d
    x = tl.load(scratch_ptr + row * d + offsets, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + row)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = x * rstd * (weight + WEIGHT_BIAS)
    if QUANTIZE:
        scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
        output *= scale_inv
        output = tl.maximum(-448.0, tl.minimum(output, 448.0))
    tl.store(output_ptr + row * stride_output + offsets, output, mask=mask)


@triton.jit
def _layer_norm_kernel(
    input_ptr,
    weight_ptr,
    beta_ptr,
    output_ptr,
    d,
    stride_input,
    stride_output,
    eps,
    HAS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)
    sum_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_lanes += x
    mean = tl.sum(sum_lanes, axis=0) / d

    var_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var_lanes += diff * diff
    variance = tl.sum(var_lanes, axis=0) / d
    rstd = tl.rsqrt(variance + eps)

    for start in tl.range(0, d, BLOCK_SIZE):
        offsets = start + lanes
        mask = offsets < d
        x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = (x - mean) * rstd * weight
        if HAS_BETA:
            beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            output += beta
        tl.store(output_ptr + row * stride_output + offsets, output, mask=mask)


def _validate_input_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError(f"{name} dtype must be float16, bfloat16, or float32")
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if tensor.shape[-1] <= 0:
        raise ValueError(f"{name}'s normalized dimension must be non-empty")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name}'s normalized dimension must be contiguous")


def _row_layout(tensor: torch.Tensor, name: str) -> tuple[int, int, int]:
    _validate_input_tensor(tensor, name)
    d = int(tensor.shape[-1])
    rows = tensor.numel() // d
    if tensor.ndim == 2:
        return rows, d, int(tensor.stride(0))
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} with more than two dimensions must be contiguous; "
            "use a 2D view for explicit row strides"
        )
    return rows, d, d


def _validate_weight(
    weight: torch.Tensor,
    input: torch.Tensor,
    d: int,
    *,
    require_same_dtype: bool = True,
) -> None:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if not weight.is_cuda or weight.device != input.device:
        raise ValueError("weight and input must be on the same CUDA device")
    if weight.ndim != 1 or weight.numel() != d:
        raise ValueError(f"weight must have shape [{d}]")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if require_same_dtype and weight.dtype != input.dtype:
        raise TypeError("weight must have the same dtype as input")
    if not require_same_dtype and weight.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError("weight dtype must be float16, bfloat16, or float32")


def _prepare_output_like(
    input: torch.Tensor,
    out: torch.Tensor | None,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, int]:
    output_dtype = input.dtype if dtype is None else dtype
    if out is None:
        output = torch.empty(input.shape, device=input.device, dtype=output_dtype)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.device != input.device:
            raise ValueError("out and input must be on the same CUDA device")
        if out.dtype != output_dtype:
            raise TypeError(f"out must have dtype {output_dtype}")
        if tuple(out.shape) != tuple(input.shape):
            raise ValueError("out must have the same shape as input")
        if out.stride(-1) != 1:
            raise ValueError("out's final dimension must be contiguous")
        if out.ndim > 2 and not out.is_contiguous():
            raise ValueError("out with more than two dimensions must be contiguous")
        output = out
    stride_output = int(output.stride(0)) if output.ndim == 2 else int(output.shape[-1])
    return output, stride_output


def _validate_scale(scale: torch.Tensor, input: torch.Tensor) -> None:
    if not isinstance(scale, torch.Tensor):
        raise TypeError("scale must be a one-element CUDA tensor")
    if not scale.is_cuda or scale.device != input.device:
        raise ValueError("scale and input must be on the same CUDA device")
    if scale.dtype != torch.float32:
        raise TypeError("scale must have dtype torch.float32")
    if scale.numel() != 1:
        raise ValueError("scale must contain exactly one element")
    if not scale.is_contiguous():
        raise ValueError("scale must be contiguous")


def _quant_output(
    input: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    if _FP8_E4M3 is None:
        raise RuntimeError("this PyTorch build does not provide torch.float8_e4m3fn")
    return _prepare_output_like(input, out, dtype=_FP8_E4M3)


def _launch_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
    eps: float,
    weight_bias: float,
) -> torch.Tensor:
    rows, d, stride_input = _row_layout(input, "input")
    _validate_weight(weight, input, d)
    output, stride_output = _prepare_output_like(input, out)
    if rows == 0:
        return output
    with torch.cuda.device(input.device):
        cast(Any, _rms_norm_kernel)[(rows,)](
            input,
            weight,
            output,
            d,
            stride_input,
            stride_output,
            float(eps),
            WEIGHT_BIAS=float(weight_bias),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
    return output


def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute row-wise RMSNorm with CUDA-compatible FP32 accumulation."""

    return _launch_rms_norm(input, weight, out, eps, 0.0)


def gemma_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Gemma RMSNorm: multiply by ``1 + weight`` after normalization."""

    return _launch_rms_norm(input, weight, out, eps, 1.0)


def rms_norm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMSNorm followed by per-tensor FP8 E4M3 quantization.

    ``scale`` follows the CUDA kernel's dequantization-scale convention: output
    is ``clamp(normalized / scale[0], -448, 448)`` cast to FP8 E4M3.
    """

    rows, d, stride_input = _row_layout(input, "input")
    _validate_weight(weight, input, d)
    _validate_scale(scale, input)
    output, stride_output = _quant_output(input, out)
    if rows == 0:
        return output
    with torch.cuda.device(input.device):
        cast(Any, _rms_norm_quant_kernel)[(rows,)](
            input,
            weight,
            output,
            scale,
            d,
            stride_input,
            stride_output,
            float(eps),
            WEIGHT_BIAS=0.0,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
    return output


def qk_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply RMSNorm independently to ``[batch, heads, head_dim]`` rows."""

    _validate_input_tensor(input, "input")
    if input.ndim != 3:
        raise ValueError("qk_rms_norm input must have shape [batch, num_heads, head_dim]")
    batch_size, num_heads, d = map(int, input.shape)
    _validate_weight(weight, input, d)
    output, _ = _prepare_output_like(input, out)
    if output.ndim != 3:
        raise AssertionError("internal output rank mismatch")
    if batch_size == 0 or num_heads == 0:
        return output

    with torch.cuda.device(input.device):
        cast(Any, _qk_rms_norm_kernel)[(batch_size * num_heads,)](
            input,
            weight,
            output,
            d,
            num_heads,
            input.stride(0),
            input.stride(1),
            output.stride(0),
            output.stride(1),
            float(eps),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
    return output


def _validate_fused_inputs(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[int, int, int, int]:
    rows, d, stride_input = _row_layout(input, "input")
    residual_rows, residual_d, stride_residual = _row_layout(residual, "residual")
    if residual.device != input.device or residual.dtype != input.dtype:
        raise ValueError("residual must have the same device and dtype as input")
    if tuple(residual.shape) != tuple(input.shape):
        raise ValueError("residual must have the same shape as input")
    if residual_rows != rows or residual_d != d:
        raise ValueError("residual row layout must match input")
    if residual.data_ptr() == input.data_ptr():
        raise ValueError("input and residual must not alias")
    _validate_weight(weight, input, d)
    return rows, d, stride_input, stride_residual


def _fused_add_impl(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    weight_bias: float,
    quant_scale: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    rows, d, stride_input, stride_residual = _validate_fused_inputs(input, residual, weight)
    quantize = quant_scale is not None
    if quantize:
        assert quant_scale is not None
        _validate_scale(quant_scale, input)
        output, stride_output = _quant_output(input, out)
    else:
        if out is not None and out.data_ptr() != input.data_ptr():
            raise ValueError("non-quant fused add writes in place to input; out must alias input")
        output = input
        stride_output = stride_input

    if rows == 0:
        return output if quantize else None

    # Exact semantic bridge for CUDA shared-memory x_vec.
    scratch = torch.empty((rows, d), device=input.device, dtype=torch.float32)
    rstd = torch.empty((rows,), device=input.device, dtype=torch.float32)
    scale_ptr = quant_scale if quant_scale is not None else rstd

    with torch.cuda.device(input.device):
        cast(Any, _fused_add_prepare_kernel)[(rows,)](
            input,
            residual,
            scratch,
            rstd,
            d,
            stride_input,
            stride_residual,
            float(eps),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
        grid = (rows, triton.cdiv(d, _BLOCK_SIZE))
        cast(Any, _fused_add_apply_kernel)[grid](
            scratch,
            rstd,
            weight,
            output,
            scale_ptr,
            d,
            stride_output,
            WEIGHT_BIAS=float(weight_bias),
            QUANTIZE=quantize,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=4,
        )
    return output if quantize else None


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place ``residual = input + residual`` and ``input = RMSNorm(sum)``."""

    _fused_add_impl(
        input,
        residual,
        weight,
        eps=eps,
        weight_bias=0.0,
        quant_scale=None,
        out=input,
    )
    return input, residual


def gemma_fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma fused add RMSNorm using ``1 + weight`` and in-place outputs."""

    _fused_add_impl(
        input,
        residual,
        weight,
        eps=eps,
        weight_bias=1.0,
        quant_scale=None,
        out=input,
    )
    return input, residual


def fused_add_rms_norm_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Update residual in place and return FP8 RMSNorm output.

    As in CUDA, ``input`` itself is not overwritten by the quantized variant.
    """

    output = _fused_add_impl(
        input,
        residual,
        weight,
        eps=eps,
        weight_bias=0.0,
        quant_scale=scale,
        out=out,
    )
    assert output is not None
    return output


def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Two-pass LayerNorm matching ``LayerNorm`` in ``norm.cuh``."""

    rows, d, stride_input = _row_layout(input, "input")
    _validate_weight(weight, input, d, require_same_dtype=False)
    if beta is not None:
        _validate_weight(beta, input, d, require_same_dtype=False)
        if beta.dtype != weight.dtype:
            raise TypeError("beta and weight must have the same dtype")
    output, stride_output = _prepare_output_like(input, out)
    if rows == 0:
        return output
    beta_ptr = beta if beta is not None else weight

    with torch.cuda.device(input.device):
        cast(Any, _layer_norm_kernel)[(rows,)](
            input,
            weight,
            beta_ptr,
            output,
            d,
            stride_input,
            stride_output,
            float(eps),
            HAS_BETA=beta is not None,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
    return output
