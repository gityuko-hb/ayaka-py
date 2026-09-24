"""Plain-torch oracles for the paged-attention custom ops."""

from __future__ import annotations

import torch


def paged_attention_ref(
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
) -> torch.Tensor:
    """Single-token-per-row paged attention over the flattened cache."""
    key = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1]).float()
    value = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1]).float()
    if key_scale is not None:
        key = key * key_scale.float()
    if value_scale is not None:
        value = value * value_scale.float()

    result = torch.zeros_like(query)
    group = query.shape[1] // key.shape[1]
    for token_index in range(query.shape[0]):
        request_index = int(query_to_request[token_index].item())
        kv_start = int(indptr[request_index].item())
        kv_stop = int(indptr[request_index + 1].item())
        query_position = int(query_positions[token_index].item())
        visible_stop = min(kv_stop - kv_start, query_position + 1)
        visible_start = 0
        if sliding_window > 0:
            visible_start = max(0, query_position - sliding_window + 1)
        slots = indices[kv_start + visible_start : kv_start + visible_stop].long()

        for query_head in range(query.shape[1]):
            kv_head = query_head // group
            if slots.numel() == 0:
                continue
            scores = query[token_index, query_head].float() @ key[slots, kv_head].T
            scores = scores * sm_scale
            values = value[slots, kv_head]
            if sinks is not None:
                scores = torch.cat((sinks[query_head].float().reshape(1), scores))
                values = torch.cat((torch.zeros_like(values[:1]), values), dim=0)
            result[token_index, query_head] = (scores.softmax(dim=0) @ values).to(query.dtype)
    return result


def paged_attention_op_ref(
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
    """``out = paged_attention_ref(...)`` for the mutating custom op."""
    out.copy_(
        paged_attention_ref(
            query,
            key_cache,
            value_cache,
            indptr,
            indices,
            query_to_request,
            query_positions,
            sm_scale,
            sliding_window,
            sinks,
            key_scale,
            value_scale,
        )
    )


def decode_paged_attention_op_ref(
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
    """Decode path: fills the split-KV scratch and ``out``."""
    del max_context_len_bucket
    key = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1]).float()
    value = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1]).float()
    if key_scale is not None:
        key = key * key_scale.float()
    if value_scale is not None:
        value = value * value_scale.float()

    group = query.shape[1] // key.shape[1]
    for request_index in range(query.shape[0]):
        kv_start = int(indptr[request_index].item())
        kv_stop = int(indptr[request_index + 1].item())
        query_position = int(query_positions[request_index].item())
        visible_stop = min(kv_stop - kv_start, query_position + 1)
        visible_start = 0
        if sliding_window > 0:
            visible_start = max(0, query_position - sliding_window + 1)
        visible_slots = indices[kv_start + visible_start : kv_start + visible_stop].long()

        for query_head in range(query.shape[1]):
            kv_head = query_head // group
            for split_index, split_start in enumerate(
                range(0, visible_slots.numel(), kv_partition_size)
            ):
                slots = visible_slots[split_start : split_start + kv_partition_size]
                scores = query[request_index, query_head].float() @ key[slots, kv_head].T
                scores = scores * sm_scale
                attn_logits[request_index, query_head, split_index].copy_(
                    scores.softmax(dim=0) @ value[slots, kv_head]
                )
                attn_lse[request_index, query_head, split_index].copy_(scores.logsumexp(dim=0))

    request_ids = torch.arange(query.shape[0], dtype=torch.int32, device=query.device)
    out.copy_(
        paged_attention_ref(
            query,
            key_cache,
            value_cache,
            indptr,
            indices,
            request_ids,
            query_positions,
            sm_scale,
            sliding_window,
            sinks,
            key_scale,
            value_scale,
        )
    )
