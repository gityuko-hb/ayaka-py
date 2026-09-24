"""Plain-torch oracles for the rotary-embedding kernel."""

from __future__ import annotations

import torch


def apply_rotary_ref(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    is_neox: bool,
) -> torch.Tensor:
    """Rotate ``tensor`` in place, mirroring the kernel's addressing."""
    rot_dim = cos_sin_cache.shape[1]
    embed_dim = rot_dim // 2
    flat_positions = positions.flatten()
    num_tokens = flat_positions.numel()
    num_heads = (
        tensor.shape[-2] if tensor.ndim == positions.ndim + 2 else tensor.shape[-1] // head_size
    )

    flat_tensor = tensor.view(num_tokens, num_heads, head_size)
    cache = cos_sin_cache[flat_positions]
    cos = cache[:, :embed_dim].unsqueeze(1).float()
    sin = cache[:, embed_dim:rot_dim].unsqueeze(1).float()

    if is_neox:
        x = flat_tensor[..., :embed_dim].float()
        y = flat_tensor[..., embed_dim:rot_dim].float()
        out_x = (x * cos - y * sin).to(tensor.dtype)
        out_y = (y * cos + x * sin).to(tensor.dtype)
        flat_tensor[..., :embed_dim] = out_x
        flat_tensor[..., embed_dim:rot_dim] = out_y
    else:
        x = flat_tensor[..., :rot_dim:2].float()
        y = flat_tensor[..., 1:rot_dim:2].float()
        out_x = (x * cos - y * sin).to(tensor.dtype)
        out_y = (y * cos + x * sin).to(tensor.dtype)
        flat_tensor[..., :rot_dim:2] = out_x
        flat_tensor[..., 1:rot_dim:2] = out_y
    return tensor


def rotary_embedding_inplace_ref(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Rotate ``query`` (and ``key`` when given) in place."""
    apply_rotary_ref(query, positions, cos_sin_cache, head_size, is_neox)
    if key is not None:
        apply_rotary_ref(key, positions, cos_sin_cache, head_size, is_neox)
