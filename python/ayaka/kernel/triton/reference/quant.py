"""Plain-torch oracles for the FP8 quantization kernels."""

from __future__ import annotations

import torch

from ayaka.types import DType

#: E4M3 finite max, derived from the dtype registry like the kernel constant.
E4M3_MAX = float(DType.FP8_E4M3.max_finite or 448.0)

#: K width of one dynamic activation scale group.
GROUP_SIZE = 128

#: Floor for a group's amax before the scale reciprocal is taken.
GROUP_FLOOR = 1e-10


def per_token_group_quant_ref(
    x: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token, per-K-group E4M3 quantization with fp32 scales."""
    *lead, k = x.shape
    x2d = x.reshape(-1, k).float()
    grouped = x2d.reshape(-1, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp_min(GROUP_FLOOR)
    scale = amax / E4M3_MAX
    quantized = (grouped / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX)
    return quantized.reshape(-1, k).to(torch.float8_e4m3fn), scale


#: K width of one MXFP8 UE8M0 scale group.
MX_GROUP = 32

#: Exponent bounds of a real (non-zero) UE8M0 scale.
MX_MIN_EXP = -126.0
MX_MAX_EXP = 127.0


def per_token_group_quant_mxfp8_ref(
    x: torch.Tensor, group_size: int = MX_GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token, per-K-group E4M3 quantization with UE8M0 power-of-two scales."""
    *lead, k = x.shape
    x2d = x.reshape(-1, k).float()
    grouped = x2d.reshape(-1, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / E4M3_MAX).clamp_min(1e-38))).clamp(
        MX_MIN_EXP, MX_MAX_EXP
    )
    zero = amax == 0
    code = torch.where(zero, torch.zeros_like(exponent), exponent + 127.0).to(torch.uint8)
    scale = torch.where(zero, torch.ones_like(amax), torch.exp2(exponent))
    quantized = (grouped / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX)
    quantized = torch.where(zero[..., None], torch.zeros_like(quantized), quantized)
    return quantized.reshape(-1, k).to(torch.float8_e4m3fn), code


def per_token_quant_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rowwise E4M3 quantization: one fp32 scale per token over the whole row."""
    *lead, k = x.shape
    x2d = x.reshape(-1, k).float()
    amax = x2d.abs().amax(dim=-1).clamp_min(GROUP_FLOOR)
    scale = amax / E4M3_MAX
    quantized = (x2d / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX)
    return quantized.to(torch.float8_e4m3fn), scale


def static_quant_ref(
    x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """One broadcast per-tensor scale, clamped to E4M3 range."""
    quantized = (
        (x.float() / scale.float().reshape(())).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    )
    if out is None:
        return quantized
    out.copy_(quantized)
    return out
