"""Inference activation modules with explicit Triton or PyTorch execution."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

from ayaka.layers._common import LayerBackend, check_input, check_out, load_kernel, write_output
from ayaka.layers.base import BaseLayer


class _Activation(BaseLayer):
    _kernel_name: str
    _gated = True

    def __init__(self, *, backend: LayerBackend = "triton", **runtime: Any) -> None:
        super().__init__(**runtime)
        if self.quant_config is not None:
            raise ValueError("activation layers do not support quant_config")
        self.backend: LayerBackend = backend
        self.op = load_kernel(backend, "activation", self._kernel_name)

    def forward(self, x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        """Return a new tensor or the supplied contiguous ``out`` buffer.

        Input is contiguous FP16/BF16/FP32 with at least one dimension. Gated
        variants require an even final width and halve it. Output storage must
        not overlap input storage. Invalid shapes/layouts fail before dispatch.
        """
        check_input(x, "x", self.backend)
        if x.ndim < 1 or not x.is_contiguous():
            raise ValueError("x must be contiguous with at least one dimension")
        if self._gated and x.shape[-1] % 2:
            raise ValueError("gated activation requires an even final dimension")
        shape = (*x.shape[:-1], x.shape[-1] // 2) if self._gated else tuple(x.shape)
        check_out(out, x, shape)
        if self.op is not None:
            return self.run_kernel(self.op, x, out=out)
        if not self._gated:
            value = x.float()
            result = (value * torch.sigmoid(1.702 * value)).to(x.dtype)
        else:
            gate, up = x.chunk(2, dim=-1)
            if self._kernel_name == "silu_and_mul":
                activated = F.silu(gate.float())
            else:
                approximate = "tanh" if self._kernel_name == "gelu_tanh_and_mul" else "none"
                activated = F.gelu(gate.float(), approximate=approximate)
            # The existing CUDA kernel rounds the activation before multiplication.
            result = activated.to(x.dtype) * up
        return write_output(result, out)

    def extra_repr(self) -> str:
        return f"backend={self.backend!r}"


class SiluAndMul(_Activation):
    """SwiGLU: ``silu(x[..., :d]) * x[..., d:]`` for packed [gate, up]."""

    _kernel_name = "silu_and_mul"


class GeluAndMul(_Activation):
    """GeGLU with exact-erf GELU (``none``) or the ``tanh`` approximation."""

    def __init__(
        self,
        approximate: Literal["none", "tanh"] = "none",
        *,
        backend: LayerBackend = "triton",
        **runtime: Any,
    ) -> None:
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")
        self.approximate = approximate
        self._kernel_name = "gelu_and_mul" if approximate == "none" else "gelu_tanh_and_mul"
        super().__init__(backend=backend, **runtime)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}, {super().extra_repr()}"


class QuickGELU(_Activation):
    """Ungated QuickGELU: ``x * sigmoid(1.702 * x)``; preserves input shape."""

    _kernel_name = "gelu_quick"
    _gated = False


def get_act_fn(name: str, *, backend: LayerBackend = "triton", **runtime: Any) -> BaseLayer:
    """Build a fresh layer for a semantic activation name; unknown names raise.

    Gated names are explicit so a model's plain ``silu``/``gelu`` configuration
    cannot silently halve its hidden dimension.
    """
    if name == "silu_and_mul":
        return SiluAndMul(backend=backend, **runtime)
    if name in ("gelu_and_mul", "gelu_tanh_and_mul"):
        approximate = "tanh" if name == "gelu_tanh_and_mul" else "none"
        return GeluAndMul(approximate, backend=backend, **runtime)
    if name in ("gelu_quick", "quick_gelu"):
        return QuickGELU(backend=backend, **runtime)
    raise ValueError(f"unsupported activation {name!r}")
