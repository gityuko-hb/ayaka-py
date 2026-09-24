from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import (
    prepare_output,
    require_contiguous,
    require_cuda,
    require_dtype,
    require_last_dim_stride1,
    require_tensor,
)
from ayaka.kernel.triton.fp8_compat import (
    e4m3_f32_to_u8,
    e4m3_native_cx,
    fp8_kernel_view,
)
from ayaka.kernel.triton.reference.norm import (
    fused_add_rms_norm_ref,
    gemma_fused_add_rms_norm_ref,
    gemma_rms_norm_ref,
    layer_norm_ref,
    qk_rms_norm_ref,
    rms_norm_ref,
)
from ayaka.utils.math_utils import FP8_E4M3_MAX
from ayaka.utils.torch_utils import compute_torch_dtypes

_SUPPORTED_INPUT_DTYPES = compute_torch_dtypes()
_BLOCK_SIZE = 256
_NUM_WARPS = 8
_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)


def _norm_block_and_warps(d: int) -> tuple[int, int]:
    block_size = min(8192, triton.next_power_of_2(d))
    if block_size >= 4096:
        warps = 16
    elif block_size >= 1024:
        warps = 8
    else:
        warps = 4
    return block_size, warps


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

    if d <= BLOCK_SIZE:
        mask = lanes < d
        x = tl.load(input_ptr + row * stride_input + lanes, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)
        weight = tl.load(weight_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
        output = x * rms_rcp * (weight + WEIGHT_BIAS)
        tl.store(output_ptr + row * stride_output + lanes, output, mask=mask)
    else:
        sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            sum_sq_lanes += x * x

        sum_sq = tl.sum(sum_sq_lanes, axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)

        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
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

    if d <= BLOCK_SIZE:
        mask = lanes < d
        x = tl.load(input_ptr + row * stride_input + lanes, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)
        scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
        weight = tl.load(weight_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
        output = x * rms_rcp * (weight + WEIGHT_BIAS) * scale_inv
        output = tl.maximum(-FP8_E4M3_MAX, tl.minimum(output, FP8_E4M3_MAX))
        if e4m3_native_cx():
            tl.store(output_ptr + row * stride_output + lanes, output.to(tl.float8e4nv), mask=mask)
        else:
            tl.store(output_ptr + row * stride_output + lanes, e4m3_f32_to_u8(output), mask=mask)
    else:
        sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            sum_sq_lanes += x * x

        sum_sq = tl.sum(sum_sq_lanes, axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)
        scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)

        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            output = x * rms_rcp * (weight + WEIGHT_BIAS) * scale_inv
            output = tl.maximum(-FP8_E4M3_MAX, tl.minimum(output, FP8_E4M3_MAX))
            if e4m3_native_cx():
                tl.store(
                    output_ptr + row * stride_output + offsets,
                    output.to(tl.float8e4nv),
                    mask=mask,
                )
            else:
                tl.store(
                    output_ptr + row * stride_output + offsets,
                    e4m3_f32_to_u8(output),
                    mask=mask,
                )


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

    if d <= BLOCK_SIZE:
        mask = lanes < d
        x = tl.load(input_ptr + input_base + lanes, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)
        weight = tl.load(weight_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + output_base + lanes, x * rms_rcp * weight, mask=mask)
    else:
        sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_sq_lanes += x * x

        sum_sq = tl.sum(sum_sq_lanes, axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)

        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            tl.store(output_ptr + output_base + offsets, x * rms_rcp * weight, mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    d,
    stride_input,
    stride_residual,
    stride_output,
    eps,
    WEIGHT_BIAS: tl.constexpr,
    QUANTIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE)

    if d <= BLOCK_SIZE:
        mask = lanes < d
        in_val = tl.load(input_ptr + row * stride_input + lanes, mask=mask, other=0.0).to(
            tl.float32
        )
        res_val = tl.load(residual_ptr + row * stride_residual + lanes, mask=mask, other=0.0).to(
            tl.float32
        )
        x = in_val + res_val
        tl.store(residual_ptr + row * stride_residual + lanes, x, mask=mask)

        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)
        weight = tl.load(weight_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
        normed = x * rms_rcp * (weight + WEIGHT_BIAS)

        if QUANTIZE:
            scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
            normed *= scale_inv
            normed = tl.maximum(-FP8_E4M3_MAX, tl.minimum(normed, FP8_E4M3_MAX))
            if e4m3_native_cx():
                tl.store(
                    output_ptr + row * stride_output + lanes, normed.to(tl.float8e4nv), mask=mask
                )
            else:
                tl.store(
                    output_ptr + row * stride_output + lanes, e4m3_f32_to_u8(normed), mask=mask
                )
        else:
            tl.store(output_ptr + row * stride_output + lanes, normed, mask=mask)
    else:
        sum_sq_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            in_val = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            res_val = tl.load(
                residual_ptr + row * stride_residual + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            x = in_val + res_val
            sum_sq_lanes += x * x
            tl.store(residual_ptr + row * stride_residual + offsets, x, mask=mask)

        sum_sq = tl.sum(sum_sq_lanes, axis=0)
        rms_rcp = tl.rsqrt(sum_sq / d + eps)

        if QUANTIZE:
            scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
        else:
            scale_inv = 1.0

        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(residual_ptr + row * stride_residual + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            normed = x * rms_rcp * (weight + WEIGHT_BIAS)
            if QUANTIZE:
                normed *= scale_inv
                normed = tl.maximum(-FP8_E4M3_MAX, tl.minimum(normed, FP8_E4M3_MAX))
                if e4m3_native_cx():
                    tl.store(
                        output_ptr + row * stride_output + offsets,
                        normed.to(tl.float8e4nv),
                        mask=mask,
                    )
                else:
                    tl.store(
                        output_ptr + row * stride_output + offsets,
                        e4m3_f32_to_u8(normed),
                        mask=mask,
                    )
            else:
                tl.store(output_ptr + row * stride_output + offsets, normed, mask=mask)


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

    if d <= BLOCK_SIZE:
        mask = lanes < d
        x = tl.load(input_ptr + row * stride_input + lanes, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / d
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / d
        rstd = tl.rsqrt(var + eps)
        weight = tl.load(weight_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
        output = diff * rstd * weight
        if HAS_BETA:
            beta = tl.load(beta_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
            output += beta
        tl.store(output_ptr + row * stride_output + lanes, output, mask=mask)
    else:
        sum_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            sum_lanes += x
        mean = tl.sum(sum_lanes, axis=0) / d

        var_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            diff = tl.where(mask, x - mean, 0.0)
            var_lanes += diff * diff
        variance = tl.sum(var_lanes, axis=0) / d
        rstd = tl.rsqrt(variance + eps)

        for start in tl.range(0, d, BLOCK_SIZE):  # type: ignore
            offsets = start + lanes
            mask = offsets < d
            x = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            output = (x - mean) * rstd * weight
            if HAS_BETA:
                beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                output += beta
            tl.store(output_ptr + row * stride_output + offsets, output, mask=mask)


def _validate_input_tensor(tensor: torch.Tensor, name: str) -> None:
    require_tensor(tensor, name)
    require_cuda(tensor, name)
    require_dtype(tensor, name, _SUPPORTED_INPUT_DTYPES)
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if tensor.shape[-1] <= 0:
        raise ValueError(f"{name}'s normalized dimension must be non-empty")
    require_last_dim_stride1(tensor, name)


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
    require_tensor(weight, "weight")
    if not weight.is_cuda or weight.device != input.device:
        raise ValueError("weight and input must be on the same CUDA device")
    if weight.ndim != 1 or weight.numel() != d:
        raise ValueError(f"weight must have shape [{d}]")
    require_contiguous(weight, "weight")
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
    output = prepare_output(input, out, dtype=output_dtype)
    require_last_dim_stride1(output, "out")
    if output.ndim > 2:
        require_contiguous(output, "out")
    stride_output = int(output.stride(0)) if output.ndim == 2 else int(output.shape[-1])
    return output, stride_output


def _validate_scale(scale: torch.Tensor, input: torch.Tensor) -> None:
    require_tensor(scale, "scale")
    if not scale.is_cuda or scale.device != input.device:
        raise ValueError("scale and input must be on the same CUDA device")
    require_dtype(scale, "scale", (torch.float32,))
    if scale.numel() != 1:
        raise ValueError("scale must contain exactly one element")
    require_contiguous(scale, "scale")


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
    block_size, num_warps = _norm_block_and_warps(d)
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
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return output


@custom_op(
    namespace="ayaka",
    name="rms_norm",
    out_shape="input",
    reference=rms_norm_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute row-wise RMSNorm with CUDA-compatible FP32 accumulation."""
    return _launch_rms_norm(input, weight, out, eps, 0.0)


@custom_op(
    namespace="ayaka",
    name="gemma_rms_norm",
    out_shape="input",
    reference=gemma_rms_norm_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
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
    block_size, num_warps = _norm_block_and_warps(d)
    with torch.cuda.device(input.device):
        cast(Any, _rms_norm_quant_kernel)[(rows,)](
            input,
            weight,
            fp8_kernel_view(output),
            scale,
            d,
            stride_input,
            stride_output,
            float(eps),
            WEIGHT_BIAS=0.0,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return output


@custom_op(
    namespace="ayaka",
    name="qk_rms_norm",
    out_shape="input",
    reference=qk_rms_norm_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
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

    block_size, num_warps = _norm_block_and_warps(d)
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
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
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

    block_size, num_warps = _norm_block_and_warps(d)
    scale_ptr = quant_scale if quant_scale is not None else input

    with torch.cuda.device(input.device):
        cast(Any, _fused_add_rms_norm_kernel)[(rows,)](
            input,
            residual,
            weight,
            fp8_kernel_view(output) if quantize else output,
            scale_ptr,
            d,
            stride_input,
            stride_residual,
            stride_output,
            float(eps),
            WEIGHT_BIAS=float(weight_bias),
            QUANTIZE=quantize,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return output if quantize else None


def _fused_add_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    return input, residual


@custom_op(
    namespace="ayaka",
    name="fused_add_rms_norm",
    mutates_args=["input", "residual"],
    returns_aliases=("input", "residual"),
    fake_impl=_fused_add_fake,
    reference=fused_add_rms_norm_ref,
    dispatch_key="CUDA",
)
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


@custom_op(
    namespace="ayaka",
    name="gemma_fused_add_rms_norm",
    mutates_args=["input", "residual"],
    returns_aliases=("input", "residual"),
    fake_impl=_fused_add_fake,
    reference=gemma_fused_add_rms_norm_ref,
    dispatch_key="CUDA",
)
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


@custom_op(
    namespace="ayaka",
    name="layer_norm",
    out_shape="input",
    reference=layer_norm_ref,
    dispatch_key="CUDA",
    mutates_args=["out"],
)
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

    block_size, num_warps = _norm_block_and_warps(d)
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
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return output
