"""Plain-torch oracles for the activation kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Quick-GELU slope. Kept local rather than imported from the kernel module so
#: the reference has no dependency back on the code it verifies.
_QUICK_GELU_ALPHA = 1.702


def silu_and_mul_ref(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``silu(x) * gate`` with the FP32-then-cast rounding of the kernel."""
    d = input.shape[-1] // 2
    x = input[..., :d].float()
    gate = input[..., d:].float()
    res = (F.silu(x).to(input.dtype).float() * gate).to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def swigluoai_and_mul_ref(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
    alpha: float = 1.702,
    beta: float = 1.0,
    limit: float = 7.0,
) -> torch.Tensor:
    """Clamped, parameterized SwiGLU over packed [gate, up] halves."""
    gate, up = input.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    result = (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(input.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def relu2_ref(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Squared ReLU with one output cast."""
    result = torch.relu(input.float()).square().to(input.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def gelu_and_mul_ref(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``gelu(x, exact) * gate`` with the kernel's rounding."""
    d = input.shape[-1] // 2
    x = input[..., :d].float()
    gate = input[..., d:].float()
    res = (F.gelu(x, approximate="none").to(input.dtype).float() * gate).to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_tanh_and_mul_ref(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``gelu(x, tanh) * gate`` with the kernel's rounding."""
    d = input.shape[-1] // 2
    x = input[..., :d].float()
    gate = input[..., d:].float()
    res = (F.gelu(x, approximate="tanh").to(input.dtype).float() * gate).to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_quick_ref(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x * sigmoid(1.702 * x)``, the Quick-GELU approximation."""
    x = input.float()
    res = (x / (1.0 + torch.exp(-_QUICK_GELU_ALPHA * x))).to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_ref(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact erf-based GELU."""
    res = F.gelu(input.float(), approximate="none").to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_tanh_ref(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """tanh-approximate GELU."""
    res = F.gelu(input.float(), approximate="tanh").to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res
