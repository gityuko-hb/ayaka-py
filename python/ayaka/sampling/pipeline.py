"""Reference sampling pipeline defining canonical stage execution order.

This module provides a specification implementation of the canonical sampling stage
order for parity verification against the optimized runtime path (`Sampler` in
`ayaka.sampling.plan`).

Canonical Pipeline Order:
    1. Penalties (`apply_penalties_`): Applied in place to raw model logits.
    2. Bitmask (`apply_allow_bitmask_`): Applied in place to disallow forbidden tokens (-inf).
    3. Temperature: Scaled exactly once prior to statistics collection and filtering.
    4. Statistics (`softmax_stats_scaled`): Computed on temperature-scaled logits.
    5. Filter and Sample (`topk_topp_sample` or `gumbel_sample`): Stochastic draw.

Notes:
    The runtime execution path in production is handled by `ayaka.sampling.plan.Sampler`.
    This independent reference implementation is retained for parity testing
    (`tests/test_pipeline_order.py`). Any semantic modification to the sampling pipeline
    must maintain parity between both implementations.
"""

from __future__ import annotations

import torch

from ayaka.sampling.metadata import SamplingMetadata
from ayaka.sampling.ops.penalties import PenaltyState, apply_penalties_
from ayaka.sampling.ops.sampling import (
    apply_allow_bitmask_,
    gumbel_sample,
    softmax_stats_scaled,
    topk_topp_sample,
)

STAGES = ("penalty", "bitmask", "temperature", "stats", "filter_sample")


def run_sampling_pipeline(
    logits: torch.Tensor,
    md: SamplingMetadata,
    penalty_state: PenaltyState,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    *,
    sampler: str = "inverse_cdf",  # "inverse_cdf" | "gumbel"
    compute_stats: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Execute the canonical sampling pipeline stages on logits in specification order.

    Applies penalties and grammar/allowed-token masks in place to `logits`. If the
    caller requires preservation of the original raw logits (e.g., for raw logprob
    computation), `logits` must be cloned prior to invoking this function.

    Args:
        logits: Float tensor of raw model logits of shape `[batch_size, vocab_size]`.
            Modified in place by penalties and bitmask stages.
        md: Sampling metadata container holding column parameters and active rows.
        penalty_state: State tracking per-slot token frequencies and penalty occurrences.
        temperature: Tensor of shape `[batch_size]` containing temperature scaling values.
        top_k: Tensor of shape `[batch_size]` specifying top-k filtering thresholds.
        top_p: Tensor of shape `[batch_size]` specifying top-p cumulative thresholds.
        min_p: Tensor of shape `[batch_size]` specifying min-p probability cutoffs.
        seed: Tensor of 64-bit random seeds per batch row.
        offset: Tensor of 64-bit RNG step offsets per batch row.
        mask: Optional packed 32-bit bitmask tensor defining allowed token sets.
        row_indices: Optional mapping of batch rows to corresponding mask rows.
        sampler: Sampling algorithm identifier, either `"inverse_cdf"` or `"gumbel"`.
        compute_stats: Whether to compute and return distribution statistics.

    Returns:
        A tuple `(token_ids, stats)` where `token_ids` is the sampled token tensor
        and `stats` is an optional tuple `(max_prob, entropy, exp_entropy)` or `None`.

    Raises:
        AssertionError: If `mask` is provided without `row_indices`.
        ValueError: If `sampler` is not recognized.
    """
    # Stage 1: Apply frequency, presence, and repetition penalties in place.
    apply_penalties_(logits, md, penalty_state)

    # Stage 2: Apply allowed-token bitmask constraints in place.
    if mask is not None:
        assert row_indices is not None, "mask không row_indices không rõ ràng buộc row nào"
        apply_allow_bitmask_(logits, mask, row_indices, logits.size(1))

    # Stage 3: Scale logits by temperature exactly once (clamped to prevent division by zero).
    scaled = logits / temperature.clamp_min(1e-6).unsqueeze(1)

    # Stage 4: Compute softmax statistics on temperature-scaled distribution if requested.
    stats = softmax_stats_scaled(scaled) if compute_stats else None

    # Stage 5: Stochastic filter and sampling draw.
    if sampler == "gumbel":
        tok = gumbel_sample(scaled, top_k, top_p, min_p, seed, offset)
    elif sampler == "inverse_cdf":
        tok = topk_topp_sample(scaled, top_k, top_p, min_p, seed, offset)
    else:
        raise ValueError(f"sampler không hợp lệ: {sampler!r}")

    return tok, stats
