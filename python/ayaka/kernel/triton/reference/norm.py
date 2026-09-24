"""Plain-torch oracles for the norm kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm_ref(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Row-wise RMSNorm with FP32 accumulation, mirroring the kernel."""
    variance = input.float().pow(2).mean(-1, keepdim=True)
    res = (input.float() * torch.rsqrt(variance + eps) * weight.float()).to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def gemma_rms_norm_ref(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Gemma RMSNorm: multiply by ``1 + weight`` after normalization."""
    variance = input.float().pow(2).mean(-1, keepdim=True)
    res = (input.float() * torch.rsqrt(variance + eps) * (weight.float() + 1.0)).to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


# qk_rms_norm normalizes ``[batch, heads, head_dim]`` rows the same way; the
# 3-D layout needs no separate oracle.
qk_rms_norm_ref = rms_norm_ref


def fused_add_rms_norm_ref(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place ``residual += input`` then ``input = RMSNorm(sum)``."""
    summed = input.float() + residual.float()
    variance = summed.pow(2).mean(-1, keepdim=True)
    normed = (summed * torch.rsqrt(variance + eps) * weight.float()).to(input.dtype)
    residual.copy_(summed)
    input.copy_(normed)
    return input, residual


def gemma_fused_add_rms_norm_ref(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma variant of :func:`fused_add_rms_norm_ref` (``1 + weight``)."""
    summed = input.float() + residual.float()
    variance = summed.pow(2).mean(-1, keepdim=True)
    scale = weight.float() + 1.0
    normed = (summed * torch.rsqrt(variance + eps) * scale).to(input.dtype)
    residual.copy_(summed)
    input.copy_(normed)
    return input, residual


def layer_norm_ref(
    input: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Two-pass LayerNorm matching ``LayerNorm`` in ``norm.cuh``."""
    res = F.layer_norm(
        input.float(),
        (input.shape[-1],),
        weight=weight.float() if weight is not None else None,
        bias=beta.float() if beta is not None else None,
        eps=eps,
    ).to(input.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res
