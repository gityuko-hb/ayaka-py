from __future__ import annotations

import math
from typing import Any, Final

from ayaka.kvcache.retention.policy import RetentionPolicy
from ayaka.kvcache.retention.range import window_starts
from ayaka.utils.torch_utils import require_torch

#: Largest score tensor the reference will materialize. Generous for a
#: validation path, small enough to fail fast instead of swapping.
DEFAULT_SCORE_BUDGET_BYTES: Final[int] = 512 * 1024 * 1024
_FP32_BYTES: Final[int] = 4

def _validate(torch: Any, query: Any, key: Any, value: Any, query_positions: Any) -> Any:
    """Shared shape / device / range checks. Returns the positions tensor."""
    if any(not isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
        raise TypeError("query, key, and value must be tensors")
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("attention tensors must use token-head-dim layout")
    if query.shape[2] != key.shape[2] or query.shape[1] % key.shape[1]:
        raise ValueError("query and K/V head geometry is incompatible")
    if key.shape[0] == 0:
        raise ValueError("the retained attention cache must not be empty")
    if any(tensor.device != query.device for tensor in (key, value)):
        raise ValueError("attention tensors must share one device")

    positions = torch.as_tensor(query_positions, dtype=torch.long, device=query.device)
    if positions.shape != (query.shape[0],):
        raise ValueError("query_positions must contain one position per query")
    if positions.numel() and (
        # One fused predicate: one device synchronization instead of two.
        bool(((positions < 0) | (positions >= key.shape[0])).any().item())
    ):
        raise ValueError("query_positions address outside the attention cache")
    return positions

def _resolve_scale(query: Any, scale: float | None) -> float:
    """Resolve and validate the score scale.

    ``bool`` is rejected explicitly: it subclasses ``int``, so ``scale=True``
    would otherwise pass every numeric check and silently become 1.0.
    """
    if scale is None:
        return 1.0 / math.sqrt(query.shape[-1])
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a real number")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return float(scale)

def _start_tensor(torch: Any, policy: RetentionPolicy, positions: Any, layer_id: int) -> Any:
    """Per-query window starts, vectorized when the policy is a suffix window.

    ``suffix_span`` collapses the whole computation to one clamp. Only a policy
    that cannot be written as a suffix falls back to per-position calls, and
    that fallback is the only path that pays the ``tolist()`` device
    synchronization.
    """
    span = policy.suffix_span(layer_id)
    if span == math.inf:
        return torch.zeros_like(positions)
    if span is not None:
        return (positions + 1 - int(span)).clamp_(min=0)
    starts = window_starts(policy, tuple(positions.tolist()), layer_id=layer_id)
    return torch.as_tensor(starts, dtype=torch.long, device=positions.device)

def reference_retained_attention(
    query: Any,
    key: Any,
    value: Any,
    *,
    query_positions: Any,
    policy: RetentionPolicy,
    layer_id: int = 0,
    scale: float | None = None,
    score_budget_bytes: int = DEFAULT_SCORE_BUDGET_BYTES,
) -> Any:
    """Causal MHA/GQA attention restricted to a retention window.

    Args:
        query: ``(num_queries, num_heads, head_dim)``.
        key: ``(num_cache_tokens, num_kv_heads, head_dim)`` retained cache K.
        value: same shape as ``key``.
        query_positions: One logical position per query. A query at position
            ``p`` sees the interval the policy returns for length ``p + 1``.
        policy: Retention policy applied per query position.
        layer_id: Layer the policy applies to.
        scale: Score scale; defaults to ``1/sqrt(head_dim)``.
        score_budget_bytes: Refuse rather than materialize a larger score
            tensor. Use :func:`reference_retained_attention_chunked` instead of
            raising this.

    Returns:
        ``(num_queries, num_heads, head_dim)`` in the query dtype.

    Raises:
        TypeError: on a non-policy, non-tensor, or boolean scale.
        ValueError: on geometry, position, or budget violations.
    """
    if not isinstance(policy, RetentionPolicy):
        raise TypeError(f"policy must be a RetentionPolicy, got {type(policy).__name__}")
    torch = require_torch()
    positions = _validate(torch, query, key, value, query_positions)

    num_queries, num_heads = query.shape[0], query.shape[1]
    num_cached = key.shape[0]
    score_bytes = num_queries * num_heads * num_cached * _FP32_BYTES
    if score_bytes > score_budget_bytes:
        raise ValueError(
            f"score tensor would be {score_bytes / 2**20:.0f} MiB "
            f"({num_queries} queries x {num_heads} heads x {num_cached} cached tokens, fp32), "
            f"over the {score_budget_bytes / 2**20:.0f} MiB budget. "
            "Use reference_retained_attention_chunked() or raise score_budget_bytes."
    )

    attention_scale = _resolve_scale(query, scale)

    # GQA expansion: repeat K/V heads until they match the query head count.
    group_size = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(group_size, dim=1)
    expanded_value = value.repeat_interleave(group_size, dim=1)
    # fp32 accumulation regardless of storage precision: this is the oracle the
    # real kernels are diffed against, so its own rounding must be negligible.
    scores = torch.einsum("qhd,lhd->qhl", query.float(), expanded_key.float())

    starts = _start_tensor(torch, policy, positions, layer_id)
    cache_positions = torch.arange(num_cached, dtype=torch.long, device=query.device)
    # The mask enforces retention (>= start) and causality (<= position) at once.
    mask = (cache_positions[None, :] >= starts[:, None]) & (
        cache_positions[None, :] <= positions[:, None]
    )
    # Scale before masking so -inf stays -inf; scaling afterwards makes it NaN.
    scores = (scores * attention_scale).masked_fill(~mask[:, None, :], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("qhl,lhd->qhd", probabilities, expanded_value.float())
    return output.to(dtype=query.dtype)

def reference_retained_attention_chunked(
    query: Any,
    key: Any,
    value: Any,
    *,
    query_positions: Any,
    policy: RetentionPolicy,
    layer_id: int = 0,
    scale: float | None = None,
    query_chunk: int = 256,
) -> Any:
    """Same result as :func:`reference_retained_attention`, in bounded memory.

    Splits along the query axis and concatenates. Chunking queries rather than
    cache tokens keeps each chunk's softmax complete, so no online-softmax
    rescaling is needed and the result is bit-identical to the unchunked path.
    """
    torch = require_torch()
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")
    positions = torch.as_tensor(query_positions, dtype=torch.long, device=query.device)
    if query.shape[0] <= query_chunk:
        return reference_retained_attention(
            query,
            key,
            value,
            query_positions=positions,
            policy=policy,
            layer_id=layer_id,
            scale=scale,
            score_budget_bytes=1 << 62,
        )
    outputs = [
        reference_retained_attention(
            query[start : start + query_chunk],
            key,
            value,
            query_positions=positions[start : start + query_chunk],
            policy=policy,
            layer_id=layer_id,
            scale=scale,
            score_budget_bytes=1 << 62,
        )
        for start in range(0, query.shape[0], query_chunk)
    ]
    return torch.cat(outputs, dim=0)
