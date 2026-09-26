"""Quantization kernels shared across models."""

from __future__ import annotations

from .fp8_quant import per_token_group_quant_fp8, per_token_quant_fp8, static_quant_fp8

__all__ = ["per_token_group_quant_fp8", "per_token_quant_fp8", "static_quant_fp8"]
