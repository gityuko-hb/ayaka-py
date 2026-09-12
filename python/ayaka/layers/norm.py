"""Dense normalization weights and residual contracts for inference models."""

from __future__ import annotations

from typing import Any, overload

import torch
import torch.nn.functional as F
from torch import nn

from ayaka.layers._common import (
    LayerBackend,
    check_dtype,
    check_out,
    check_rows,
    load_kernel,
    positive_float,
    write_output,
)
from ayaka.layers.base import BaseLayer
from ayaka.utils.validation import require_int


class _Norm(BaseLayer):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: LayerBackend = "triton",
        **runtime: Any,
    ) -> None:
        super().__init__(**runtime)
        if self.quant_config is not None:
            raise ValueError("dense norm layers do not support quant_config")
        require_int(hidden_size, "hidden_size", minimum=1)
        self.hidden_size = hidden_size
        self.variance_epsilon = positive_float(eps, "eps")
        self.backend: LayerBackend = backend
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        check_dtype(dtype)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        if not self.weight.is_meta:
            self.runtime_context(self.weight.device, dtype)

    def _validate(self, x: torch.Tensor, out: torch.Tensor | None) -> None:
        check_rows(x, "x", self.hidden_size, self.backend)
        check_dtype(self.weight.dtype)
        if self.weight.device != x.device or self.weight.shape != (self.hidden_size,):
            raise ValueError("weight must match x's device and the configured hidden_size")
        if not self.weight.is_contiguous():
            raise ValueError("weight must be contiguous")
        check_out(out, x, tuple(x.shape))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}, backend={self.backend!r}"
        )


class RMSNorm(_Norm):
    """RMSNorm with optional fused residual addition and frozen inference weight.

    ``layer(x)`` allocates output; ``layer(x, out=buffer)`` returns ``buffer``.
    ``layer(x, residual)`` writes the sum to residual, the normalized sum to x,
    and returns the exact ``(x, residual)`` objects. ``out`` is forbidden in that
    branch. Inputs/residual/weights must share dtype and device. Fused input and
    residual storage must not overlap. Normalization uses the FP32 sum before
    residual storage rounding, matching Ayaka's Triton kernel.
    """

    _weight_bias = 0.0
    _kernel_name = "rms_norm"
    _fused_kernel_name = "fused_add_rms_norm"

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: LayerBackend = "triton",
        **runtime: Any,
    ) -> None:
        super().__init__(hidden_size, eps, device=device, dtype=dtype, backend=backend, **runtime)
        if self._weight_bias:
            nn.init.zeros_(self.weight)
        self.op = load_kernel(backend, "norm", self._kernel_name)
        self.fused_op = load_kernel(backend, "norm", self._fused_kernel_name)

    @overload
    def forward(
        self, x: torch.Tensor, residual: None = None, *, out: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, *, out: None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate(x, out)
        if self.weight.dtype != x.dtype:
            raise TypeError("RMSNorm weight and x must have the same dtype")
        if residual is not None:
            if out is not None:
                raise ValueError("out is not supported with a fused residual")
            check_rows(residual, "residual", self.hidden_size, self.backend)
            if (
                residual.shape != x.shape
                or residual.device != x.device
                or residual.dtype != x.dtype
            ):
                raise ValueError("residual must match x's shape, device, and dtype")
            if residual is x:
                raise ValueError("x and residual must not alias")
            if self.fused_op is not None:
                return self.run_kernel(
                    self.fused_op, x, residual, self.weight, self.variance_epsilon
                )
        elif self.op is not None:
            return self.run_kernel(self.op, x, self.weight, out=out, eps=self.variance_epsilon)

        value = x.float() if residual is None else x.float() + residual.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        scale = self.weight.float() + self._weight_bias
        result = (value * torch.rsqrt(variance + self.variance_epsilon) * scale).to(x.dtype)
        if residual is None:
            return write_output(result, out)
        residual.copy_(value)
        x.copy_(result)
        return x, residual


class GemmaRMSNorm(RMSNorm):
    """RMSNorm using ``1 + weight``; checkpoint weight is initialized to zero."""

    _weight_bias = 1.0
    _kernel_name = "gemma_rms_norm"
    _fused_kernel_name = "gemma_fused_add_rms_norm"


class LayerNorm(_Norm):
    """Last-dimension LayerNorm with optional bias and FP32 accumulation.

    Weight/bias may use a different supported floating dtype from x. Inputs have
    rank >= 2; 2D row-strided views are accepted. ``out`` is contiguous and must
    not overlap weight/bias or partially overlap x. Parameters are initialized
    to weight=1, bias=0 and use checkpoint keys ``weight`` and ``bias``.
    """

    bias: nn.Parameter | None

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: LayerBackend = "triton",
        **runtime: Any,
    ) -> None:
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")
        super().__init__(hidden_size, eps, device=device, dtype=dtype, backend=backend, **runtime)
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros_like(self.weight), requires_grad=False) if bias else None,
        )
        self.op = load_kernel(backend, "norm", "layer_norm")

    def forward(self, x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        self._validate(x, out)
        if self.bias is not None and (
            self.bias.shape != self.weight.shape
            or self.bias.dtype != self.weight.dtype
            or self.bias.device != self.weight.device
            or not self.bias.is_contiguous()
        ):
            raise ValueError("bias must match weight shape, dtype, device, and contiguous layout")
        if self.op is not None:
            return self.run_kernel(
                self.op, x, self.weight, beta=self.bias, out=out, eps=self.variance_epsilon
            )
        result = F.layer_norm(
            x.float(),
            (self.hidden_size,),
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.variance_epsilon,
        ).to(x.dtype)
        return write_output(result, out)
