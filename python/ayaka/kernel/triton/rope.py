from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import (
    require_contiguous,
    require_cuda,
    require_dtype,
    require_last_dim_stride1,
    require_same_device,
    require_tensor,
)
from ayaka.kernel.triton.reference.rope import rotary_embedding_inplace_ref
from ayaka.utils.torch_utils import compute_torch_dtypes

_SUPPORTED_DTYPES = compute_torch_dtypes()
_BLOCK_SIZE = 256
_NUM_WARPS = 4


@triton.jit
def _rotary_embedding_kernel(
    positions_ptr,
    tensor_ptr,
    cache_ptr,
    token_stride,
    head_stride,
    cache_stride,
    num_heads,
    embed_dim,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0).to(tl.int64)
    tile_idx = tl.program_id(axis=1).to(tl.int64)
    linear = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_pairs = num_heads * embed_dim
    mask = linear < total_pairs

    head_idx = linear // embed_dim
    rot_offset = linear - head_idx * embed_dim
    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    cache_base = position * cache_stride

    cos = tl.load(cache_ptr + cache_base + rot_offset, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(
        cache_ptr + cache_base + embed_dim + rot_offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    if IS_NEOX:
        x_index = rot_offset
        y_index = embed_dim + rot_offset
    else:
        x_index = 2 * rot_offset
        y_index = x_index + 1

    head_base = token_idx * token_stride + head_idx.to(tl.int64) * head_stride
    x_ptr = tensor_ptr + head_base + x_index
    y_ptr = tensor_ptr + head_base + y_index
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr, mask=mask, other=0.0).to(tl.float32)

    # Assignment in the CUDA helper converts each result back to scalar_t.
    tl.store(x_ptr, x * cos - y * sin, mask=mask)
    tl.store(y_ptr, y * cos + x * sin, mask=mask)


def _validate_positions(positions: torch.Tensor) -> None:
    require_tensor(positions, "positions")
    require_cuda(positions, "positions")
    require_dtype(positions, "positions", (torch.int64,))
    if positions.ndim not in (1, 2):
        raise ValueError("positions must have shape [num_tokens] or [batch_size, seq_len]")
    require_contiguous(positions, "positions")


def _tensor_layout(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    name: str,
) -> tuple[int, int, int]:
    positions_ndim = positions.ndim
    if tensor.ndim not in (positions_ndim + 1, positions_ndim + 2):
        raise ValueError(
            f"{name} must have flattened heads or explicit [heads, head_size] dimensions"
        )
    if tuple(tensor.shape[:positions_ndim]) != tuple(positions.shape):
        raise ValueError(f"{name} and positions must have identical token dimensions")
    require_last_dim_stride1(tensor, name)

    if tensor.ndim == positions_ndim + 2:
        if tensor.shape[-1] != head_size:
            raise ValueError(
                f"{name}.shape[-1] must equal head_size={head_size}; got {tensor.shape[-1]}"
            )
        num_heads = int(tensor.shape[-2])
        head_stride = int(tensor.stride(-2))
    else:
        hidden_size = int(tensor.shape[-1])
        if hidden_size % head_size != 0:
            raise ValueError(f"{name} hidden size must be divisible by head_size")
        num_heads = hidden_size // head_size
        head_stride = head_size

    seq_dim_idx = positions_ndim - 1
    token_stride = int(tensor.stride(seq_dim_idx))
    if positions_ndim == 2:
        expected_batch_stride = int(positions.shape[1]) * token_stride
        if int(tensor.stride(0)) != expected_batch_stride:
            raise ValueError(
                f"{name} must be linearly addressable across [batch, seq] like the CUDA kernel"
            )
    if num_heads <= 0:
        raise ValueError(f"{name} must contain at least one attention head")
    return num_heads, token_stride, head_stride


def _validate_data_tensor(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    name: str,
) -> tuple[int, int, int]:
    require_tensor(tensor, name)
    require_cuda(tensor, name)
    require_same_device(tensor, name, positions, "positions")
    require_dtype(tensor, name, _SUPPORTED_DTYPES)
    return _tensor_layout(tensor, positions, head_size, name)


def _launch_rotary(
    positions: torch.Tensor,
    tensor: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    token_stride: int,
    head_stride: int,
    embed_dim: int,
    is_neox: bool,
) -> None:
    if positions.numel() == 0 or num_heads == 0 or embed_dim == 0:
        return
    grid = (
        positions.numel(),
        triton.cdiv(num_heads * embed_dim, _BLOCK_SIZE),
    )
    cast(Any, _rotary_embedding_kernel)[grid](
        positions,
        tensor,
        cos_sin_cache,
        token_stride,
        head_stride,
        cos_sin_cache.stride(0),
        num_heads,
        embed_dim,
        IS_NEOX=is_neox,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )


@custom_op(
    namespace="ayaka",
    name="rotary_embedding",
    mutates_args=["query", "key"],
    out_shape=None,
    reference=rotary_embedding_inplace_ref,
    dispatch_key="CUDA",
)
def _rotary_embedding_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    _validate_positions(positions)
    if not isinstance(head_size, int) or head_size <= 0:
        raise ValueError("head_size must be a positive integer")

    query_layout = _validate_data_tensor(query, positions, head_size, "query")

    if not isinstance(cos_sin_cache, torch.Tensor):
        raise TypeError("cos_sin_cache must be a torch.Tensor")
    require_cuda(cos_sin_cache, "cos_sin_cache")
    require_same_device(cos_sin_cache, "cos_sin_cache", query, "query")
    if cos_sin_cache.dtype != query.dtype:
        raise TypeError("cos_sin_cache must have the same dtype as query")
    if cos_sin_cache.ndim != 2:
        raise ValueError("cos_sin_cache must have shape [max_position, rot_dim]")
    require_last_dim_stride1(cos_sin_cache, "cos_sin_cache")

    rot_dim = int(cos_sin_cache.shape[1])
    if rot_dim <= 0 or rot_dim % 2 != 0:
        raise ValueError("cos_sin_cache.shape[1] must be a positive even rot_dim")
    if rot_dim > head_size:
        raise ValueError("rot_dim cannot exceed head_size")
    embed_dim = rot_dim // 2

    key_layout: tuple[int, int, int] | None = None
    if key is not None:
        key_layout = _validate_data_tensor(key, positions, head_size, "key")
        if key.dtype != query.dtype:
            raise TypeError("key must have the same dtype as query")
        if query_layout[0] % key_layout[0] != 0:
            raise ValueError("the number of query heads must be divisible by key heads")

    with torch.cuda.device(query.device):
        _launch_rotary(
            positions,
            query,
            cos_sin_cache,
            query_layout[0],
            query_layout[1],
            query_layout[2],
            embed_dim,
            bool(is_neox),
        )
        if key is not None and key_layout is not None:
            _launch_rotary(
                positions,
                key,
                cos_sin_cache,
                key_layout[0],
                key_layout[1],
                key_layout[2],
                embed_dim,
                bool(is_neox),
            )

    return None


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings in place.

    Args:
        positions: Contiguous int64 tensor shaped ``[tokens]`` or ``[batch, seq]``.
        query: Query tensor with flattened or explicit head dimensions.
        key: Optional key tensor using the same token dimensions.
        head_size: Full head dimension.  Only the first ``rot_dim`` elements rotate.
        cos_sin_cache: ``[max_position, rot_dim]`` with cosine then sine halves.
        is_neox: ``True`` for split-half NeoX layout, ``False`` for GPT-J pairs.

    Returns:
        ``query`` when ``key`` is ``None``; otherwise ``(query, key)``.  Returned
        tensors alias the inputs because the operation is in place.
    """
    _rotary_embedding_impl(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )
    if key is None:
        return query
    return query, key
