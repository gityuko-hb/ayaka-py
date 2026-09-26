from __future__ import annotations

import math
from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import (
    prepare_output,
    require_cuda,
    require_device,
    require_last_dim_stride1,
    require_ndim,
    require_tensor,
)
from ayaka.kernel.triton.fp8_compat import (
    e4m3_native_cx,
    e4m3_u8_to_f32,
    e5m2_u8_to_f32,
    fp8_kernel_view,
)
from ayaka.kernel.triton.reference.attention import (
    decode_paged_attention_op_ref,
    paged_attention_op_ref,
)
from ayaka.types import DType
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_int

_QUERY_DTYPES = (DType.FP16.torch_dtype, DType.BF16.torch_dtype)
_FP8_DTYPES = (DType.FP8_E4M3.torch_dtype, DType.FP8_E5M2.torch_dtype)


def _require_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device | None = None,
    ndim: int | tuple[int, ...] | None = None,
) -> None:
    require_tensor(tensor, name)
    require_cuda(tensor, name)
    if device is not None:
        require_device(tensor, name, device)
    if ndim is not None:
        require_ndim(tensor, name, ndim)


def _flatten_nhd_cache(cache: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    """Return ``[slots, kv_heads, head_dim]`` without copying an NHD cache.

    The repository cache port exposes ``[pages, page_size, kv_heads, head_dim]``.
    Callers that already hold a flat slot view may pass the equivalent 3-D form.
    """

    _require_cuda_tensor(cache, name, device=device, ndim=(3, 4))
    require_last_dim_stride1(cache, name)
    if cache.dim() == 3:
        return cache
    if cache.stride(0) != cache.shape[1] * cache.stride(1):
        raise ValueError(
            f"{name} page and page-slot dimensions must be flattenable without copying"
        )
    try:
        return cache.view(cache.shape[0] * cache.shape[1], cache.shape[2], cache.shape[3])
    except RuntimeError as exc:  # defensive: a future exotic layout may pass the stride check
        raise ValueError(f"{name} must use page-major NHD layout") from exc


def _validate_query_and_caches(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    _require_cuda_tensor(query, "query", ndim=3)
    require_last_dim_stride1(query, "query")
    if query.dtype not in _QUERY_DTYPES:
        raise TypeError(f"query dtype must be float16 or bfloat16, got {query.dtype}")

    key_cache = _flatten_nhd_cache(key_cache, "key_cache", query.device)
    value_cache = _flatten_nhd_cache(value_cache, "value_cache", query.device)
    if key_cache.dtype != value_cache.dtype:
        raise TypeError("key_cache and value_cache must have the same dtype")

    _, num_query_heads, head_dim = query.shape
    if num_query_heads < 1 or head_dim < 1:
        raise ValueError("query must have at least one head and one head-dimension element")
    if head_dim > 256:
        raise ValueError(f"head_dim must be <= 256, got {head_dim}")

    num_slots, num_kv_heads, key_head_dim = key_cache.shape
    if num_kv_heads < 1:
        raise ValueError("key_cache must have at least one KV head")
    if value_cache.shape != (num_slots, num_kv_heads, head_dim):
        raise ValueError(
            "value_cache must match key_cache slots/KV heads and query head_dim; "
            f"got {tuple(value_cache.shape)}"
        )
    if key_head_dim != head_dim:
        raise ValueError(
            f"key_cache head_dim must match query head_dim ({head_dim}), got {key_head_dim}"
        )
    if num_query_heads % num_kv_heads:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )
    return key_cache, value_cache, num_query_heads, num_kv_heads, head_dim


def _validate_index_tensor(
    tensor: torch.Tensor,
    name: str,
    device: torch.device,
    *,
    numel: int | None = None,
) -> None:
    _require_cuda_tensor(tensor, name, device=device, ndim=1)
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if tensor.stride(0) != 1:
        raise ValueError(f"{name} must be contiguous")
    if numel is not None and tensor.numel() != numel:
        raise ValueError(f"{name} must contain {numel} elements, got {tensor.numel()}")


def _validate_sm_scale(sm_scale: float) -> float:
    if isinstance(sm_scale, bool) or not isinstance(sm_scale, (int, float)):
        raise TypeError("sm_scale must be a real number")
    scale = float(sm_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("sm_scale must be finite and positive")
    return scale


def _validate_sliding_window(sliding_window: int | None) -> int:
    if sliding_window is None:
        return 0
    require_int(sliding_window, "sliding_window", minimum=1)
    return sliding_window


def _prepare_optional_sinks(
    sinks: torch.Tensor | None,
    query: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:
    if sinks is None:
        return query
    _require_cuda_tensor(sinks, "sinks", device=query.device, ndim=1)
    if sinks.numel() != num_query_heads:
        raise ValueError(
            f"sinks must contain one value per query head ({num_query_heads}), got {sinks.numel()}"
        )
    if sinks.dtype not in (*_QUERY_DTYPES, torch.float32):
        raise TypeError(f"sinks must be float16, bfloat16, or float32, got {sinks.dtype}")
    if not sinks.is_contiguous():
        raise ValueError("sinks must be contiguous")
    return sinks


def _prepare_kv_scales(
    key_cache: torch.Tensor,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    query: torch.Tensor,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    is_fp8_kv = key_cache.dtype in _FP8_DTYPES
    if not is_fp8_kv:
        if key_cache.dtype != query.dtype:
            raise TypeError(
                "unquantized key/value cache dtype must match query dtype; "
                f"got cache={key_cache.dtype}, query={query.dtype}"
            )
        if key_scale is not None or value_scale is not None:
            raise ValueError("key_scale and value_scale must be omitted for an unquantized cache")
        return False, query, query

    if key_scale is None or value_scale is None:
        raise ValueError("FP8 key/value caches require key_scale and value_scale")
    for scale, name in ((key_scale, "key_scale"), (value_scale, "value_scale")):
        _require_cuda_tensor(scale, name, device=query.device, ndim=(0, 1))
        if scale.numel() != 1:
            raise ValueError(f"{name} must be a per-layer device scalar")
        if scale.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32, got {scale.dtype}")
        if not scale.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return True, key_scale, value_scale


def _prepare_fp8_kernel_caches(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    is_fp8_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return the kernel-visible caches plus the E5M2 selector.

    Below sm_89 the launch hands Triton ``uint8`` byte views instead of FP8
    tensors, because the IR parser rejects FP8 dtypes for those targets; the
    kernel decodes the bytes itself. On sm_89+ the original tensors pass
    through untouched.
    """
    if not is_fp8_kv:
        return key_cache, value_cache, False
    is_e5m2 = key_cache.dtype == DType.FP8_E5M2.torch_dtype
    return fp8_kernel_view(key_cache), fp8_kernel_view(value_cache), is_e5m2


def _prepare_output(query: torch.Tensor, output: torch.Tensor | None) -> torch.Tensor:
    result = prepare_output(query, output, name="output", like_name="query")
    require_last_dim_stride1(result, "output")
    return result


@triton.jit
def _paged_attention_single_pass_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    output_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    query_to_request_ptr,
    query_position_ptr,
    softmax_scale,
    attention_sink_ptr,
    key_scale_ptr,
    value_scale_ptr,
    query_token_stride,
    query_head_stride,
    key_slot_stride,
    key_head_stride,
    value_slot_stride,
    value_head_stride,
    output_token_stride,
    output_head_stride,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    IS_FP8_KV: tl.constexpr,
    FP8_IS_E5M2: tl.constexpr,
):
    """Executes single-pass online softmax GEMV per (token, query head).

    Grid Mapping:
        - `program_id(0)`: Query token index (`query_token_index`).
        - `program_id(1)`: Query head index (`query_head_index`).
    """
    query_token_index = tl.program_id(0)
    query_head_index = tl.program_id(1)
    kv_head_index = query_head_index // GROUP

    request_index = tl.load(query_to_request_ptr + query_token_index)
    kv_start_offset = tl.load(kv_indptr_ptr + request_index)
    kv_end_offset = tl.load(kv_indptr_ptr + request_index + 1)
    kv_length = kv_end_offset - kv_start_offset
    query_position = tl.load(query_position_ptr + query_token_index)

    head_dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    head_dim_mask = head_dim_offsets < HEAD_DIM
    query = tl.load(
        query_ptr
        + query_token_index * query_token_stride
        + query_head_index * query_head_stride
        + head_dim_offsets,
        mask=head_dim_mask,
        other=0.0,
    ).to(tl.float32)

    if IS_FP8_KV:
        key_scale = tl.load(key_scale_ptr)
        value_scale = tl.load(value_scale_ptr)
    else:
        key_scale = 1.0
        value_scale = 1.0

    if HAS_SINKS:
        running_max = tl.load(attention_sink_ptr + query_head_index).to(tl.float32)
        running_sum = 1.0
    else:
        running_max = -float("inf")
        running_sum = 0.0
    accumulator = tl.zeros((BLOCK_HEAD_DIM,), dtype=tl.float32)

    for kv_block_start in range(0, kv_length, BLOCK_KV):
        kv_offsets = kv_block_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < kv_length
        kv_position = kv_offsets
        causal_mask = kv_position <= query_position
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & ((kv_position + SLIDING_WINDOW) > query_position)
        kv_mask = kv_mask & causal_mask

        skip_tile = tl.max(kv_mask.to(tl.int32), axis=0) == 0
        if not skip_tile:
            kv_slot_indices = tl.load(
                kv_indices_ptr + kv_start_offset + kv_offsets, mask=kv_offsets < kv_length, other=0
            )
            key_raw = tl.load(
                key_cache_ptr
                + kv_slot_indices[:, None] * key_slot_stride
                + kv_head_index * key_head_stride
                + head_dim_offsets[None, :],
                mask=(kv_offsets[:, None] < kv_length) & head_dim_mask[None, :],
                other=0.0,
            )
            if IS_FP8_KV:
                if e4m3_native_cx():
                    key = key_raw.to(tl.float32) * key_scale
                elif FP8_IS_E5M2:
                    key = e5m2_u8_to_f32(key_raw) * key_scale
                else:
                    key = e4m3_u8_to_f32(key_raw) * key_scale
            else:
                key = key_raw.to(tl.float32)
            scores = tl.sum(query[None, :] * key, axis=1) * softmax_scale
            scores = tl.where(kv_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=0)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            new_running_max = tl.maximum(row_max_fixed, running_max)
            rescale_factor = tl.exp(running_max - new_running_max)
            attention_weights = tl.exp(scores - new_running_max)

            value_raw = tl.load(
                value_cache_ptr
                + kv_slot_indices[:, None] * value_slot_stride
                + kv_head_index * value_head_stride
                + head_dim_offsets[None, :],
                mask=(kv_offsets[:, None] < kv_length) & head_dim_mask[None, :],
                other=0.0,
            )
            if IS_FP8_KV:
                if e4m3_native_cx():
                    value = value_raw.to(tl.float32) * value_scale
                elif FP8_IS_E5M2:
                    value = e5m2_u8_to_f32(value_raw) * value_scale
                else:
                    value = e4m3_u8_to_f32(value_raw) * value_scale
            else:
                value = value_raw.to(tl.float32)
            accumulator = accumulator * rescale_factor + tl.sum(
                attention_weights[:, None] * value, axis=0
            )
            running_sum = running_sum * rescale_factor + tl.sum(attention_weights, axis=0)
            running_max = new_running_max

    output_value = tl.where(running_sum == 0.0, 0.0, accumulator / running_sum)
    tl.store(
        output_ptr
        + query_token_index * output_token_stride
        + query_head_index * output_head_stride
        + head_dim_offsets,
        output_value.to(output_ptr.dtype.element_ty),
        mask=head_dim_mask,
    )


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    query_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    block_n: int = 32,
) -> torch.Tensor:
    """Run causal paged attention for packed query tokens.

    ``key_cache`` and ``value_cache`` accept either the repository's page-major
    ``[pages, page_size, kv_heads, head_dim]`` NHD layout or its zero-copy flattened
    ``[slots, kv_heads, head_dim]`` view. ``indices`` contains flat physical slot IDs;
    ``indptr`` partitions those IDs by request. Metadata contents and slot bounds are
    scheduler-owned and are therefore not copied to the host for validation here.

    FP8 caches use the repository's per-layer/per-plane scalar scale contract. All tensors
    must be on one CUDA device and dimensions addressed without an explicit inner stride
    must have ``stride(-1) == 1``.
    """

    key_cache, value_cache, num_query_heads, num_kv_heads, head_dim = _validate_query_and_caches(
        query, key_cache, value_cache
    )
    num_tokens = query.shape[0]
    _validate_index_tensor(indptr, "indptr", query.device)
    if indptr.numel() < 2:
        raise ValueError("indptr must contain at least one request interval")
    _validate_index_tensor(indices, "indices", query.device)
    _validate_index_tensor(
        query_to_request,
        "query_to_request",
        query.device,
        numel=num_tokens,
    )
    _validate_index_tensor(
        query_positions,
        "query_positions",
        query.device,
        numel=num_tokens,
    )
    require_int(block_n, "block_n", minimum=1)
    if block_n > 256 or block_n & (block_n - 1):
        raise ValueError("block_n must be a power of two no greater than 256")
    sliding_window_value = _validate_sliding_window(sliding_window)
    scale = _validate_sm_scale(sm_scale)
    sinks_arg = _prepare_optional_sinks(sinks, query, num_query_heads)
    is_fp8_kv, key_scale_arg, value_scale_arg = _prepare_kv_scales(
        key_cache,
        key_scale,
        value_scale,
        query,
    )
    key_cache_arg, value_cache_arg, fp8_is_e5m2 = _prepare_fp8_kernel_caches(
        key_cache,
        value_cache,
        is_fp8_kv,
    )
    o = _prepare_output(query, output)
    if num_tokens == 0:
        return o

    block_head_dim = triton.next_power_of_2(head_dim)
    grid = (num_tokens, num_query_heads)
    with torch.cuda.device(query.device):
        cast(Any, _paged_attention_single_pass_kernel)[grid](
            query,
            key_cache_arg,
            value_cache_arg,
            o,
            indptr,
            indices,
            query_to_request,
            query_positions,
            scale,
            sinks_arg,
            key_scale_arg,
            value_scale_arg,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            o.stride(0),
            o.stride(1),
            GROUP=num_query_heads // num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_HEAD_DIM=block_head_dim,
            BLOCK_KV=block_n,
            SLIDING_WINDOW=sliding_window_value,
            HAS_SINKS=sinks is not None,
            IS_FP8_KV=is_fp8_kv,
            FP8_IS_E5M2=fp8_is_e5m2,
            num_warps=8 if head_dim >= 256 else 4,
            num_stages=2,
        )
    return o


@triton.jit
def _paged_attention_split_kv_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    softmax_scale,
    kv_indptr_ptr,
    kv_indices_ptr,
    query_position_ptr,
    partial_output_ptr,
    partial_logsumexp_ptr,
    key_scale_ptr,
    value_scale_ptr,
    query_token_stride,
    query_head_stride,
    key_slot_stride,
    key_head_stride,
    value_slot_stride,
    value_head_stride,
    partial_output_batch_stride,
    partial_output_head_stride,
    partial_output_split_stride,
    partial_logsumexp_batch_stride,
    partial_logsumexp_head_stride,
    partial_logsumexp_split_stride,
    GROUP: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_VALUE_HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    VALID_HEAD_COUNT: tl.constexpr,
    BLOCKS_PER_KV_HEAD: tl.constexpr,
    KV_PARTITION_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_HEAD_DIM: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    IS_FP8_KV: tl.constexpr,
    FP8_IS_E5M2: tl.constexpr,
):
    batch_index = tl.program_id(0)
    head_block_index = tl.program_id(1)
    split_index = tl.program_id(2)

    kv_head_index = head_block_index // BLOCKS_PER_KV_HEAD
    head_subblock_index = head_block_index % BLOCKS_PER_KV_HEAD
    query_head_start = kv_head_index * GROUP + head_subblock_index * VALID_HEAD_COUNT
    query_head_indices = query_head_start + tl.arange(0, BLOCK_HEAD)
    query_head_mask = query_head_indices < (kv_head_index + 1) * GROUP
    query_head_mask = query_head_mask & (query_head_indices < NUM_QUERY_HEADS)

    head_dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    value_head_dim_offsets = tl.arange(0, BLOCK_VALUE_HEAD_DIM)
    head_dim_mask = head_dim_offsets < HEAD_DIM
    value_head_dim_mask = value_head_dim_offsets < VALUE_HEAD_DIM

    kv_start_offset = tl.load(kv_indptr_ptr + batch_index)
    kv_length = tl.load(kv_indptr_ptr + batch_index + 1) - kv_start_offset
    query_position = tl.load(query_position_ptr + batch_index)
    effective_kv_end = tl.minimum(kv_length, query_position + 1)
    effective_kv_start = 0
    if SLIDING_WINDOW > 0:
        effective_kv_start = tl.maximum(0, query_position - SLIDING_WINDOW + 1)
    effective_kv_length = tl.maximum(0, effective_kv_end - effective_kv_start)

    partition_start = KV_PARTITION_SIZE * split_index
    partition_end = tl.minimum(partition_start + KV_PARTITION_SIZE, effective_kv_length)

    if IS_FP8_KV:
        key_scale = tl.load(key_scale_ptr)
        value_scale = tl.load(value_scale_ptr)
    else:
        key_scale = 1.0
        value_scale = 1.0

    running_max = tl.zeros((BLOCK_HEAD,), dtype=tl.float32) - float("inf")
    running_sum = tl.zeros((BLOCK_HEAD,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_HEAD, BLOCK_VALUE_HEAD_DIM), dtype=tl.float32)

    query_offsets = (
        batch_index * query_token_stride
        + query_head_indices[:, None] * query_head_stride
        + head_dim_offsets[None, :]
    )
    key_base_offsets = kv_head_index * key_head_stride + head_dim_offsets[:, None]
    value_base_offsets = kv_head_index * value_head_stride + value_head_dim_offsets[None, :]

    if partition_end > partition_start:
        query = tl.load(
            query_ptr + query_offsets,
            mask=query_head_mask[:, None] & head_dim_mask[None, :],
            other=0.0,
        )

        for relative_kv_start in tl.range(partition_start, partition_end, BLOCK_KV):  # type: ignore
            relative_kv_offsets = relative_kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = relative_kv_offsets < partition_end
            logical_kv_offsets = effective_kv_start + relative_kv_offsets
            kv_slot_indices = tl.load(
                kv_indices_ptr + kv_start_offset + logical_kv_offsets, mask=kv_mask, other=0
            )

            key_raw = tl.load(
                key_cache_ptr + kv_slot_indices[None, :] * key_slot_stride + key_base_offsets,
                mask=kv_mask[None, :] & head_dim_mask[:, None],
                other=0.0,
            )
            if IS_FP8_KV:
                if e4m3_native_cx():
                    key = (key_raw.to(tl.float32) * key_scale).to(query.dtype)
                elif FP8_IS_E5M2:
                    key = (e5m2_u8_to_f32(key_raw) * key_scale).to(query.dtype)
                else:
                    key = (e4m3_u8_to_f32(key_raw) * key_scale).to(query.dtype)
            else:
                key = key_raw
            scores = tl.dot(query, key) * softmax_scale
            scores = tl.where(query_head_mask[:, None] & kv_mask[None, :], scores, -float("inf"))

            value_raw = tl.load(
                value_cache_ptr + kv_slot_indices[:, None] * value_slot_stride + value_base_offsets,
                mask=kv_mask[:, None] & value_head_dim_mask[None, :],
                other=0.0,
            )
            if IS_FP8_KV:
                if e4m3_native_cx():
                    value = (value_raw.to(tl.float32) * value_scale).to(query.dtype)
                elif FP8_IS_E5M2:
                    value = (e5m2_u8_to_f32(value_raw) * value_scale).to(query.dtype)
                else:
                    value = (e4m3_u8_to_f32(value_raw) * value_scale).to(query.dtype)
            else:
                value = value_raw

            new_running_max = tl.maximum(tl.max(scores, axis=1), running_max)
            rescale_factor = tl.exp(running_max - new_running_max)
            attention_weights = tl.exp(scores - new_running_max[:, None])
            rounded_weights = attention_weights.to(value.dtype)
            weighted_values = tl.dot(rounded_weights, value)
            if value.dtype == tl.bfloat16:
                # Keep the numerator consistent with the FP32 denominator.
                # BF16 probability rounding alone can bias every output channel;
                # carry its residual in a second tensor-core product.
                residual_weights = (attention_weights - rounded_weights.to(tl.float32)).to(
                    value.dtype
                )
                weighted_values += tl.dot(residual_weights, value)
            accumulator = accumulator * rescale_factor[:, None] + weighted_values
            running_sum = running_sum * rescale_factor + tl.sum(attention_weights, axis=1)
            running_max = new_running_max

        output_value = accumulator / running_sum[:, None]
        partial_output_offsets = (
            batch_index * partial_output_batch_stride
            + query_head_indices[:, None] * partial_output_head_stride
            + split_index * partial_output_split_stride
            + value_head_dim_offsets[None, :]
        )
        tl.store(
            partial_output_ptr + partial_output_offsets,
            output_value,
            mask=query_head_mask[:, None] & value_head_dim_mask[None, :],
        )

        partial_logsumexp_offsets = (
            batch_index * partial_logsumexp_batch_stride
            + query_head_indices * partial_logsumexp_head_stride
            + split_index * partial_logsumexp_split_stride
        )
        tl.store(
            partial_logsumexp_ptr + partial_logsumexp_offsets,
            running_max + tl.log(running_sum),
            mask=query_head_mask,
        )


@triton.jit
def _paged_attention_reduce_partitions_kernel(
    partial_output_ptr,
    partial_logsumexp_ptr,
    output_ptr,
    kv_indptr_ptr,
    query_position_ptr,
    attention_sink_ptr,
    partial_output_batch_stride,
    partial_output_head_stride,
    partial_output_split_stride,
    partial_logsumexp_batch_stride,
    partial_logsumexp_head_stride,
    partial_logsumexp_split_stride,
    output_token_stride,
    output_head_stride,
    MAX_KV_SPLITS: tl.constexpr,
    KV_PARTITION_SIZE: tl.constexpr,
    BLOCK_VALUE_HEAD_DIM: tl.constexpr,
    VALUE_HEAD_DIM: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_head_index = tl.program_id(1)

    kv_length = tl.load(kv_indptr_ptr + batch_index + 1) - tl.load(kv_indptr_ptr + batch_index)
    query_position = tl.load(query_position_ptr + batch_index)
    effective_kv_end = tl.minimum(kv_length, query_position + 1)
    effective_kv_start = 0
    if SLIDING_WINDOW > 0:
        effective_kv_start = tl.maximum(0, query_position - SLIDING_WINDOW + 1)
    effective_kv_length = tl.maximum(0, effective_kv_end - effective_kv_start)

    value_head_dim_offsets = tl.arange(0, BLOCK_VALUE_HEAD_DIM)
    value_head_dim_mask = value_head_dim_offsets < VALUE_HEAD_DIM
    if HAS_SINKS:
        running_max = tl.load(attention_sink_ptr + query_head_index).to(tl.float32)
        running_sum = 1.0
    else:
        running_max = -float("inf")
        running_sum = 0.0
    accumulator = tl.zeros((BLOCK_VALUE_HEAD_DIM,), dtype=tl.float32)

    partial_output_base = (
        batch_index * partial_output_batch_stride
        + query_head_index * partial_output_head_stride
        + value_head_dim_offsets
    )
    partial_logsumexp_base = (
        batch_index * partial_logsumexp_batch_stride
        + query_head_index * partial_logsumexp_head_stride
    )

    for split_index in tl.range(0, MAX_KV_SPLITS, num_stages=2):  # type: ignore
        partition_start = KV_PARTITION_SIZE * split_index
        partition_end = tl.minimum(partition_start + KV_PARTITION_SIZE, effective_kv_length)

        if partition_end > partition_start:
            partial_output = tl.load(
                partial_output_ptr
                + partial_output_base
                + split_index * partial_output_split_stride,
                mask=value_head_dim_mask,
                other=0.0,
            )
            partial_logsumexp = tl.load(
                partial_logsumexp_ptr
                + partial_logsumexp_base
                + split_index * partial_logsumexp_split_stride
            )
            new_running_max = tl.maximum(partial_logsumexp, running_max)
            rescale_factor = tl.exp(running_max - new_running_max)
            partial_rescale_factor = tl.exp(partial_logsumexp - new_running_max)
            accumulator = accumulator * rescale_factor + partial_output * partial_rescale_factor
            running_sum = running_sum * rescale_factor + partial_rescale_factor
            running_max = new_running_max

    output_value = tl.where(running_sum == 0.0, 0.0, accumulator / running_sum)
    tl.store(
        output_ptr
        + batch_index * output_token_stride
        + query_head_index * output_head_stride
        + value_head_dim_offsets,
        output_value.to(output_ptr.dtype.element_ty),
        mask=value_head_dim_mask,
    )


def compute_max_num_partitions(
    max_context_len_bucket: int,
    kv_partition_size: int = 512,
    sliding_window: int | None = None,
) -> int:
    """Return the split-KV workspace width required by a context bucket."""

    require_int(max_context_len_bucket, "max_context_len_bucket", minimum=1)
    require_int(kv_partition_size, "kv_partition_size", minimum=1)
    sliding_window_value = _validate_sliding_window(sliding_window)
    effective_bound = max_context_len_bucket
    if sliding_window_value:
        effective_bound = min(effective_bound, sliding_window_value)
    return div_ceil(effective_bound, kv_partition_size)


def validate_decode_workspace(
    query: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    max_context_len_bucket: int,
    kv_partition_size: int,
    sliding_window: int | None,
) -> None:
    """Validate split-KV scratch before cache mutation, without device reads."""
    batch, num_query_heads, head_dim = query.shape
    partitions = compute_max_num_partitions(
        max_context_len_bucket, kv_partition_size, sliding_window
    )
    for tensor, name, shape in (
        (attn_logits, "attn_logits", (batch, num_query_heads, partitions, head_dim)),
        (attn_lse, "attn_lse", (batch, num_query_heads, partitions)),
    ):
        _require_cuda_tensor(tensor, name, device=query.device, ndim=len(shape))
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
        if any(actual < required for actual, required in zip(tensor.shape, shape, strict=True)):
            raise ValueError(
                f"{name} workspace is too small; need at least {shape}, got {tuple(tensor.shape)}"
            )
    require_last_dim_stride1(attn_logits, "attn_logits")


def decode_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    query_positions: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    max_context_len_bucket: int,
    sm_scale: float,
    kv_partition_size: int = 512,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run split-KV causal attention for one query token per request.

    ``max_context_len_bucket`` is the scheduler-owned upper bound for every request in this
    launch. The function derives the exact partition count from that bound and refuses
    undersized partial-output workspaces. This keeps the hot path free of device-to-host
    synchronization; the caller must ensure admitted sequence lengths fit the bucket.

    Cache layout, FP8 scale, device, dtype, and inner-stride requirements match
    :func:`paged_attention`.
    """

    key_cache, value_cache, num_query_heads, num_kv_heads, head_dim = _validate_query_and_caches(
        query, key_cache, value_cache
    )
    batch = query.shape[0]
    _validate_index_tensor(indptr, "indptr", query.device, numel=batch + 1)
    _validate_index_tensor(indices, "indices", query.device)
    _validate_index_tensor(
        query_positions,
        "query_positions",
        query.device,
        numel=batch,
    )
    require_int(kv_partition_size, "kv_partition_size", minimum=32)
    if kv_partition_size % 32:
        raise ValueError("kv_partition_size must be a positive multiple of 32")
    sliding_window_value = _validate_sliding_window(sliding_window)
    max_num_partitions = compute_max_num_partitions(
        max_context_len_bucket,
        kv_partition_size,
        sliding_window,
    )
    scale = _validate_sm_scale(sm_scale)

    validate_decode_workspace(
        query, attn_logits, attn_lse, max_context_len_bucket, kv_partition_size, sliding_window
    )

    sinks_arg = _prepare_optional_sinks(sinks, query, num_query_heads)
    is_fp8_kv, key_scale_arg, value_scale_arg = _prepare_kv_scales(
        key_cache,
        key_scale,
        value_scale,
        query,
    )
    key_cache_arg, value_cache_arg, fp8_is_e5m2 = _prepare_fp8_kernel_caches(
        key_cache,
        value_cache,
        is_fp8_kv,
    )
    o = _prepare_output(query, out)
    if batch == 0:
        return o

    group = num_query_heads // num_kv_heads
    valid_head_count = min(16, group)
    block_head = triton.next_power_of_2(valid_head_count)
    block_head_dim = triton.next_power_of_2(head_dim)
    block_value_head_dim = triton.next_power_of_2(head_dim)
    blocks_per_kv_head = triton.cdiv(group, valid_head_count)

    # Launch Phase 1: Compute partial outputs per KV partition
    grid_split = (
        batch,
        num_kv_heads * blocks_per_kv_head,
        max_num_partitions,
    )
    with torch.cuda.device(query.device):
        cast(Any, _paged_attention_split_kv_kernel)[grid_split](
            query,
            key_cache_arg,
            value_cache_arg,
            scale,
            indptr,
            indices,
            query_positions,
            attn_logits,
            attn_lse,
            key_scale_arg,
            value_scale_arg,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            attn_lse.stride(0),
            attn_lse.stride(1),
            attn_lse.stride(2),
            GROUP=group,
            NUM_QUERY_HEADS=num_query_heads,
            BLOCK_HEAD_DIM=block_head_dim,
            BLOCK_VALUE_HEAD_DIM=block_value_head_dim,
            BLOCK_KV=32,
            BLOCK_HEAD=block_head,
            VALID_HEAD_COUNT=valid_head_count,
            BLOCKS_PER_KV_HEAD=blocks_per_kv_head,
            KV_PARTITION_SIZE=kv_partition_size,
            HEAD_DIM=head_dim,
            VALUE_HEAD_DIM=head_dim,
            SLIDING_WINDOW=sliding_window_value,
            IS_FP8_KV=is_fp8_kv,
            FP8_IS_E5M2=fp8_is_e5m2,
            num_warps=4,
            num_stages=2,
        )

        # Launch Phase 2: Reduce all partitions across the context.
        cast(Any, _paged_attention_reduce_partitions_kernel)[(batch, num_query_heads)](
            attn_logits,
            attn_lse,
            o,
            indptr,
            query_positions,
            sinks_arg,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            attn_lse.stride(0),
            attn_lse.stride(1),
            attn_lse.stride(2),
            o.stride(0),
            o.stride(1),
            MAX_KV_SPLITS=max_num_partitions,
            KV_PARTITION_SIZE=kv_partition_size,
            BLOCK_VALUE_HEAD_DIM=block_value_head_dim,
            VALUE_HEAD_DIM=head_dim,
            SLIDING_WINDOW=sliding_window_value,
            HAS_SINKS=sinks is not None,
            num_warps=4,
            num_stages=2,
        )
    return o


@custom_op(
    namespace="ayaka",
    name="triton_paged_attention",
    mutates_args=["out"],
    reference=paged_attention_op_ref,
    dispatch_key="CUDA",
)
def paged_attention_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    query_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int,
    sinks: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    """Compiler-visible single-pass paged-attention operation."""

    paged_attention(
        query,
        key_cache,
        value_cache,
        indptr,
        indices,
        query_to_request,
        query_positions,
        sm_scale,
        sliding_window=sliding_window or None,
        sinks=sinks,
        key_scale=key_scale,
        value_scale=value_scale,
        output=out,
    )


@custom_op(
    namespace="ayaka",
    name="triton_decode_paged_attention",
    mutates_args=["attn_logits", "attn_lse", "out"],
    reference=decode_paged_attention_op_ref,
    dispatch_key="CUDA",
)
def decode_paged_attention_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    query_positions: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    max_context_len_bucket: int,
    sm_scale: float,
    kv_partition_size: int,
    sliding_window: int,
    sinks: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    """Compiler-visible split-KV decode operation with caller-owned scratch."""

    decode_paged_attention(
        query,
        key_cache,
        value_cache,
        indptr,
        indices,
        query_positions,
        attn_logits,
        attn_lse,
        max_context_len_bucket,
        sm_scale,
        kv_partition_size=kv_partition_size,
        sliding_window=sliding_window or None,
        sinks=sinks,
        key_scale=key_scale,
        value_scale=value_scale,
        out=out,
    )
