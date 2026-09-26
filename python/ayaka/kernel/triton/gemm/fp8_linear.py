"""Dense FP8 linear entry points shared by the FP8 quantization methods."""

from __future__ import annotations

import math

import torch

from ayaka.kernel.triton._host import require_cuda, require_dtype, require_tensor
from ayaka.kernel.triton.gemm._common import (
    require_fp8_dtype,
    validate_bias,
    validate_operand,
)
from ayaka.kernel.triton.gemm.fp8_gemm import fp8_blockwise_mm, fp8_scaled_mm
from ayaka.kernel.triton.gemm.fp8_gemv import fp8_weight_only_gemv
from ayaka.kernel.triton.gemm.fp8_w8a16_gemm import fp8_weight_only_gemm
from ayaka.kernel.triton.quant.fp8_quant import (
    per_token_group_quant_fp8,
    per_token_quant_fp8,
    static_quant_fp8,
)
from ayaka.utils.math_utils import div_ceil

__all__ = ["fp8_block_linear", "fp8_pertensor_linear", "fp8_rowwise_linear"]

#: K width of one block-FP8 scale in the checkpoint contract.
_BLOCK128 = 128


def _validate_activation(x: torch.Tensor) -> None:
    require_tensor(x, "x")
    require_dtype(x, "x", (torch.float16, torch.bfloat16))
    require_cuda(x, "x")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")


def _validate_dense_weight(weight: torch.Tensor, inner: int) -> None:
    validate_operand(weight, "weight", dtype=require_fp8_dtype(), allow_transposed=False)
    if weight.shape[1] != inner:
        raise ValueError(f"weight must be [N, K] with K={inner}, got {tuple(weight.shape)}")


def _flatten_activation(x: torch.Tensor) -> tuple[tuple[int, ...], int, int, torch.Tensor]:
    lead = tuple(x.shape[:-1])
    inner = int(x.shape[-1])
    rows = math.prod(lead) if lead else 1
    x2d = x.reshape(rows, inner)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    return lead, inner, rows, x2d


def _weight_only(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_size_y: int,
    block_size_x: int,
    scale_dtype: str,
) -> torch.Tensor:
    """W8A16 path: split-K GEMV at one row, tiled GEMM above it."""
    inner = int(x2d.shape[1])
    scale_row_stride = div_ceil(inner, block_size_x)
    if x2d.shape[0] == 1:
        return fp8_weight_only_gemv(
            x2d.reshape(inner),
            weight,
            weight_scale,
            scale_row_stride=scale_row_stride,
            block_size_y=block_size_y,
            block_size_x=block_size_x,
            scale_dtype=scale_dtype,
        )
    return fp8_weight_only_gemm(
        x2d,
        weight,
        weight_scale,
        scale_row_stride=scale_row_stride,
        block_size_y=block_size_y,
        block_size_x=block_size_x,
        scale_dtype=scale_dtype,
    )


def _output_dtype_columns(x: torch.Tensor, columns: int) -> torch.Tensor:
    return torch.empty((x.shape[0], columns), dtype=x.dtype, device=x.device)


def _finish(
    result: torch.Tensor,
    lead: tuple[int, ...],
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if bias is not None:
        result = result + bias
    return result.reshape(*lead, result.shape[-1])


def fp8_pertensor_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    uniform_scale: bool = False,
) -> torch.Tensor:
    """Compute ``x @ (weight_fp8 * weight_scale)^T`` for a per-row weight scale.

    Args:
        x: ``[..., K]`` fp16/bf16 CUDA activation.
        weight: ``[N, K]`` ``torch.float8_e4m3fn`` with ``stride(1) == 1``.
        weight_scale: Scalar or ``[N]`` fp32 dequantization scale. A genuine
            per-tensor checkpoint stores one scalar; a fused projection stores
            the concatenation of each part's scalar.
        bias: Optional ``[N]`` additive term.
        input_scale: Optional one-element static activation scale. When given,
            ``M > 1`` runs W8A8 (static activation quantization + scaled MM);
            ``M == 1`` always stays on the W8A16 GEMV.
        uniform_scale: Declare ``weight_scale`` piecewise-constant so the
            scaled-MM dispatch can take the tensor-wise path.

    Returns:
        ``[..., N]`` output in ``x.dtype``.
    """

    _validate_activation(x)
    lead, inner, rows, x2d = _flatten_activation(x)
    _validate_dense_weight(weight, inner)
    columns = int(weight.shape[0])
    validate_bias(bias, x2d, columns, reference_name="x")
    scale = weight_scale
    if uniform_scale and scale.numel() > 1:
        scale = scale.reshape(-1)[0]
    if rows == 1:
        result = _weight_only(
            x2d,
            weight,
            weight_scale,
            block_size_y=columns if weight_scale.numel() == 1 else 1,
            block_size_x=inner,
            scale_dtype="float32",
        )
    elif input_scale is not None:
        quantized = static_quant_fp8(x2d, input_scale)
        result = fp8_scaled_mm(
            quantized,
            weight.t(),
            input_scale,
            scale,
            out=_output_dtype_columns(x2d, columns),
            backend="auto",
        )
    else:
        result = _weight_only(
            x2d,
            weight,
            weight_scale,
            block_size_y=columns if weight_scale.numel() == 1 else 1,
            block_size_x=inner,
            scale_dtype="float32",
        )
    return _finish(result, lead, bias)


def fp8_rowwise_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x @ (weight_fp8 * weight_scale)^T`` with dynamic W8A8 activation.

    Args:
        x: ``[..., K]`` fp16/bf16 CUDA activation.
        weight: ``[N, K]`` ``torch.float8_e4m3fn`` with ``stride(1) == 1``.
        weight_scale: Scalar or ``[N]`` fp32 per-output-row dequantization scale.
        bias: Optional ``[N]`` additive term.

    Returns:
        ``[..., N]`` output in ``x.dtype``. ``M > 1`` quantizes the activation
        with one dynamic scale per token; ``M == 1`` uses the W8A16 GEMV.
    """

    _validate_activation(x)
    lead, inner, rows, x2d = _flatten_activation(x)
    _validate_dense_weight(weight, inner)
    columns = int(weight.shape[0])
    validate_bias(bias, x2d, columns, reference_name="x")
    if rows == 1:
        result = _weight_only(
            x2d,
            weight,
            weight_scale,
            block_size_y=columns if weight_scale.numel() == 1 else 1,
            block_size_x=inner,
            scale_dtype="float32",
        )
    else:
        quantized, activation_scale = per_token_quant_fp8(x2d)
        result = fp8_scaled_mm(
            quantized,
            weight.t(),
            activation_scale,
            weight_scale,
            out=_output_dtype_columns(x2d, columns),
            backend="auto",
        )
    return _finish(result, lead, bias)


def fp8_block_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x @ (weight_fp8 * weight_scale)^T`` with 128x128 block scales.

    Args:
        x: ``[..., K]`` fp16/bf16 CUDA activation.
        weight: ``[N, K]`` ``torch.float8_e4m3fn`` with ``stride(1) == 1``.
        weight_scale: ``[ceil(N / 128), ceil(K / 128)]`` fp32/bf16 block scale.
        bias: Optional ``[N]`` additive term.

    Returns:
        ``[..., N]`` output in ``x.dtype``. ``M > 1`` quantizes the activation
        dynamically per token and 128-K group (so ``K`` must divide by 128);
        ``M == 1`` stays on the block-scale W8A16 GEMV.
    """

    _validate_activation(x)
    lead, inner, rows, x2d = _flatten_activation(x)
    _validate_dense_weight(weight, inner)
    columns = int(weight.shape[0])
    validate_bias(bias, x2d, columns, reference_name="x")
    scale_dtype = "bfloat16" if weight_scale.dtype is torch.bfloat16 else "float32"
    if rows == 1:
        result = _weight_only(
            x2d,
            weight,
            weight_scale,
            block_size_y=_BLOCK128,
            block_size_x=_BLOCK128,
            scale_dtype=scale_dtype,
        )
    else:
        if inner % _BLOCK128:
            raise ValueError(f"K={inner} must be divisible by 128 for the blockwise W8A8 path")
        quantized, activation_scale = per_token_group_quant_fp8(x2d, _BLOCK128)
        result = fp8_blockwise_mm(
            quantized,
            weight,
            activation_scale,
            weight_scale,
            out=_output_dtype_columns(x2d, columns),
        )
    return _finish(result, lead, bias)
