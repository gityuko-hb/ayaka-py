"""Triton sampling kernels without sorting or softmax materialization.

Ported from FlashInfer C++/CUDA sampling operators:

- ``sampling_from_probs``
- ``min_p_sampling_from_probs``
- ``top_p_sampling_from_probs``

Plus a fused ``top-k + top-p + min-p`` path directly from logits that the
joint FlashInfer path lacks (it has no ``min_p``).

Conventions shared by every kernel in this module:

- One program per row (``grid=(batch,)``); the vocabulary dimension is
  tiled by ``BLOCK_SIZE = min(4096, next_power_of_2(V))``.
- Randomness comes from the Philox counter-based stream
  (``ayaka.kernel.triton.sampling.philox``) with ``flat = offset``, so each
  row is deterministic in ``(seed, offset)`` and independent of batch
  composition — no ``torch.Generator`` state, hence CUDA-graph safe.
- Launchers are registered via ``ayaka.kernel.ops.custom_op`` (``CUDA``
  dispatch) with a plain-torch reference each, so ``force_reference`` gives
  a CPU fallback and ``verify_against_reference`` compares kernel against
  reference on identical inputs.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton.reference.sampling import (
    fused_sampling_ref,
    min_p_sampling_ref,
    sampling_from_probs_ref,
    top_p_sampling_ref,
)
from ayaka.kernel.triton.sampling._common import (
    normalize_threshold,
    prepare_sampling_outputs,
    sampling_fake,
    validate_opt_seed_offset,
    validate_param_tensor,
    validate_probs,
)
from ayaka.kernel.triton.sampling.philox import philox_u01_f32


# Kernel: Sampling From Probs (Categorical CDF Scan)
@triton.jit
def _sampling_from_probs_kernel(
    probs_ptr,
    output_ptr,
    valid_ptr,
    seed_ptr,
    offset_ptr,
    vocab_size,
    probs_stride_b,
    seed_val,
    offset_val,
    HAS_SEED_TENSOR: tl.constexpr,
    HAS_OFFSET_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample one categorical row with a two-pass CDF scan.

    # ======================================================================
    # CATEGORICAL CDF SCAN (BLOCK_SIZE<=4096, num_warps=4, grid=(B,))
    #
    # Programs:
    #   1 program/row — row = tl.program_id(0); rows never communicate.
    # Tiles:
    #   vocab chunked by BLOCK_SIZE; OOB lanes read as 0.0.
    # Passes:
    #   1. total = sum(p) over the row.
    #   2. first index with cumsum(p) >= u * total (block prefix + running
    #      accum; earliest match wins via the sampled_idx<0 guard).
    # RNG:
    #   u = philox_u01_f32(seed, flat), flat = offset (NOT scaled by the
    #   program count, so the stream stays batch-invariant).
    # Memory:
    #   probs read twice (sum + scan); one int32 + one bool stored per row.
    # Precision/fallback:
    #   fp32 accumulation; if float rounding leaves no match, vocab_size-1.
    # ======================================================================

    Args:
        probs_ptr: Pointer to ``[batch, vocab]`` probabilities.
        output_ptr: Pointer to ``[batch]`` int32 sampled indices.
        valid_ptr: Pointer to ``[batch]`` bool validity flags.
        seed_ptr: Pointer to ``[batch]`` Philox seeds, or ignored when
            ``HAS_SEED_TENSOR`` is false.
        offset_ptr: Pointer to ``[batch]`` Philox offsets, or ignored when
            ``HAS_OFFSET_TENSOR`` is false.
        vocab_size: Number of vocabulary columns.
        probs_stride_b: Row stride (in elements) of ``probs_ptr``.
        seed_val: Scalar int64 seed used when no seed tensor is given.
        offset_val: Scalar int64 offset used when no offset tensor is given.
        HAS_SEED_TENSOR: Whether ``seed_ptr`` holds per-row seeds.
        HAS_OFFSET_TENSOR: Whether ``offset_ptr`` holds per-row offsets.
        BLOCK_SIZE: Tile width along the vocabulary dimension.

    Note:
        The RNG address is deliberately not scaled by the program count so
        the stream stays batch-invariant, matching ``counter_uniform`` in
        ``ayaka.sampling.rng``.
    """
    row_idx = tl.program_id(0).to(tl.int64)

    # 1. Draw u ~ U(0, 1) at flat = offset — do NOT scale by num_programs:
    # addressing must stay batch-invariant (matching counter_uniform in
    # rng.py; scaling by batch would tie a row's stream to batch membership).
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64) if HAS_SEED_TENSOR else seed_val
    offset = tl.load(offset_ptr + row_idx).to(tl.uint64) if HAS_OFFSET_TENSOR else offset_val
    u = philox_u01_f32(seed, offset.to(tl.int64))

    # 2. Pass 1: row-wide probability total
    row_offset = row_idx * probs_stride_b
    total_prob = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        total_prob += tl.sum(p, axis=0)

    # 3. Pass 2: first token crossing the CDF threshold: u * total_prob
    target = u * total_prob
    accum = 0.0
    sampled_idx = -1

    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)

        # In-block prefix sum
        block_cdf = accum + tl.cumsum(p, axis=0)
        matched = mask & (block_cdf >= target) & (sampled_idx < 0)

        if tl.sum(matched.to(tl.int32), axis=0) > 0:
            # Pick the first matching index
            first_match_offset = tl.min(tl.where(matched, cols, vocab_size), axis=0)
            if sampled_idx < 0:
                sampled_idx = first_match_offset

        accum += tl.sum(p, axis=0)

    # Float rounding may leave u unmatched — fall back to the last token
    if sampled_idx < 0:
        sampled_idx = vocab_size - 1

    tl.store(output_ptr + row_idx, sampled_idx)
    tl.store(valid_ptr + row_idx, True)


# Kernel: Min-P Sampling (Fused Max-prob + Filter + Renorm + Sample)
@triton.jit
def _min_p_sampling_kernel(
    probs_ptr,
    output_ptr,
    valid_ptr,
    min_p_ptr,
    seed_ptr,
    offset_ptr,
    vocab_size,
    probs_stride_b,
    min_p_val,
    seed_val,
    offset_val,
    HAS_MIN_P_TENSOR: tl.constexpr,
    HAS_SEED_TENSOR: tl.constexpr,
    HAS_OFFSET_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample one row with fused max-prob, filter, renorm, and CDF scan.

    # ======================================================================
    # MIN-P (BLOCK_SIZE<=4096, num_warps=4, grid=(B,))
    #
    # Programs:
    #   1 program/row — row = tl.program_id(0); rows never communicate.
    # Tiles:
    #   vocab chunked by BLOCK_SIZE; OOB lanes read as 0.0.
    # Passes:
    #   1. max_prob = max(p) over the row; cutoff = max_prob * min_p.
    #   2. sum_valid = sum(p where p >= cutoff).
    #   3. first index with filtered cumsum >= u * sum_valid, restricted
    #      to lanes with filtered_p > 0.
    # RNG:
    #   u = philox_u01_f32(seed, flat), flat = offset (batch-invariant).
    # Memory:
    #   probs read three times (max + sum + scan); one int32 + one bool
    #   stored per row.
    # Precision/fallback:
    #   fp32 accumulation; with no match, vocab_size-1.
    # ======================================================================

    Args:
        probs_ptr: Pointer to ``[batch, vocab]`` probabilities.
        output_ptr: Pointer to ``[batch]`` int32 sampled indices.
        valid_ptr: Pointer to ``[batch]`` bool validity flags.
        min_p_ptr: Pointer to ``[batch]`` per-row thresholds, or ignored
            when ``HAS_MIN_P_TENSOR`` is false.
        seed_ptr: Pointer to ``[batch]`` Philox seeds, or ignored when
            ``HAS_SEED_TENSOR`` is false.
        offset_ptr: Pointer to ``[batch]`` Philox offsets, or ignored when
            ``HAS_OFFSET_TENSOR`` is false.
        vocab_size: Number of vocabulary columns.
        probs_stride_b: Row stride (in elements) of ``probs_ptr``.
        min_p_val: Scalar threshold used when no ``min_p`` tensor is given.
        seed_val: Scalar int64 seed used when no seed tensor is given.
        offset_val: Scalar int64 offset used when no offset tensor is given.
        HAS_MIN_P_TENSOR: Whether ``min_p_ptr`` holds per-row thresholds.
        HAS_SEED_TENSOR: Whether ``seed_ptr`` holds per-row seeds.
        HAS_OFFSET_TENSOR: Whether ``offset_ptr`` holds per-row offsets.
        BLOCK_SIZE: Tile width along the vocabulary dimension.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_offset = row_idx * probs_stride_b

    min_p = tl.load(min_p_ptr + row_idx).to(tl.float32) if HAS_MIN_P_TENSOR else min_p_val
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64) if HAS_SEED_TENSOR else seed_val
    offset = tl.load(offset_ptr + row_idx).to(tl.uint64) if HAS_OFFSET_TENSOR else offset_val
    # flat = offset directly — batch-invariant (see _sampling_from_probs_kernel).
    u = philox_u01_f32(seed, offset.to(tl.int64))

    # Pass 1: row maximum probability (max_prob)
    max_prob = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        max_prob = tl.maximum(max_prob, tl.max(p, axis=0))

    # Min-P cutoff
    cutoff = max_prob * min_p

    # Pass 2: filtered probability total
    sum_valid_p = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        filtered_p = tl.where(p >= cutoff, p, 0.0)
        sum_valid_p += tl.sum(filtered_p, axis=0)

    # Pass 3: scan the filtered cumulative CDF for a token
    target = u * sum_valid_p
    accum = 0.0
    sampled_idx = -1

    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        filtered_p = tl.where(p >= cutoff, p, 0.0)

        block_cdf = accum + tl.cumsum(filtered_p, axis=0)
        matched = mask & (block_cdf >= target) & (filtered_p > 0.0) & (sampled_idx < 0)

        if tl.sum(matched.to(tl.int32), axis=0) > 0:
            first_match_offset = tl.min(tl.where(matched, cols, vocab_size), axis=0)
            if sampled_idx < 0:
                sampled_idx = first_match_offset

        accum += tl.sum(filtered_p, axis=0)

    if sampled_idx < 0:
        sampled_idx = vocab_size - 1

    tl.store(output_ptr + row_idx, sampled_idx)
    tl.store(valid_ptr + row_idx, True)


# Kernel: Top-P Renormalize & Sample (Two-Pass Histogram Bisection)
@triton.jit
def _top_p_sampling_kernel(
    probs_ptr,
    output_ptr,
    valid_ptr,
    top_p_ptr,
    seed_ptr,
    offset_ptr,
    vocab_size,
    probs_stride_b,
    top_p_val,
    seed_val: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
    offset_val: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
    HAS_TOP_P_TENSOR: tl.constexpr,
    HAS_SEED_TENSOR: tl.constexpr,
    HAS_OFFSET_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample one row with bisection threshold plus CDF scan.

    # ======================================================================
    # TOP-P (BLOCK_SIZE<=4096, num_warps=8, grid=(B,))
    #
    # Programs:
    #   1 program/row — row = tl.program_id(0); rows never communicate.
    # Tiles:
    #   vocab chunked by BLOCK_SIZE; OOB lanes read as 0.0.
    # Passes:
    #   1. 12 bisection iterations over [0, 1] for the largest cutoff with
    #      sum(p >= mid) >= top_p (~2e-4 precision).
    #   2. sum_valid over p >= threshold.
    #   3. first index with filtered cumsum >= u * sum_valid, restricted
    #      to lanes with filtered_p > 0.
    # RNG:
    #   u = philox_u01_f32(seed, flat), flat = offset (batch-invariant).
    # Memory:
    #   probs read 12x (bisection) + sum + scan; one int32 + one bool
    #   stored per row.
    # Precision/fallback:
    #   fp32 accumulation; with no match, vocab_size-1.
    # ======================================================================

    Args:
        probs_ptr: Pointer to ``[batch, vocab]`` probabilities.
        output_ptr: Pointer to ``[batch]`` int32 sampled indices.
        valid_ptr: Pointer to ``[batch]`` bool validity flags.
        top_p_ptr: Pointer to ``[batch]`` per-row mass targets, or ignored
            when ``HAS_TOP_P_TENSOR`` is false.
        seed_ptr: Pointer to ``[batch]`` Philox seeds, or ignored when
            ``HAS_SEED_TENSOR`` is false.
        offset_ptr: Pointer to ``[batch]`` Philox offsets, or ignored when
            ``HAS_OFFSET_TENSOR`` is false.
        vocab_size: Number of vocabulary columns.
        probs_stride_b: Row stride (in elements) of ``probs_ptr``.
        top_p_val: Scalar mass target used when no ``top_p`` tensor is
            given.
        seed_val: Scalar int64 seed used when no seed tensor is given.
        offset_val: Scalar int64 offset used when no offset tensor is given.
        HAS_TOP_P_TENSOR: Whether ``top_p_ptr`` holds per-row targets.
        HAS_SEED_TENSOR: Whether ``seed_ptr`` holds per-row seeds.
        HAS_OFFSET_TENSOR: Whether ``offset_ptr`` holds per-row offsets.
        BLOCK_SIZE: Tile width along the vocabulary dimension.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_offset = row_idx * probs_stride_b

    top_p = tl.load(top_p_ptr + row_idx).to(tl.float32) if HAS_TOP_P_TENSOR else top_p_val
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64) if HAS_SEED_TENSOR else seed_val
    offset = tl.load(offset_ptr + row_idx).to(tl.uint64) if HAS_OFFSET_TENSOR else offset_val
    # flat = offset directly — batch-invariant (see _sampling_from_probs_kernel).
    u = philox_u01_f32(seed, offset.to(tl.int64))

    # 1. Bisect down to the cutoff threshold (iteration count below):
    low = 0.0
    high = 1.0
    for _ in range(12):  # 12 binary iterations reach ~2e-4 precision
        mid = (low + high) * 0.5
        current_sum = 0.0
        for v_offset in range(0, vocab_size, BLOCK_SIZE):
            cols = v_offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
            current_sum += tl.sum(tl.where(p >= mid, p, 0.0), axis=0)

        if current_sum >= top_p:
            low = mid
        else:
            high = mid

    threshold = low

    # 2. Actual probability total over the top-p set
    sum_valid = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        sum_valid += tl.sum(tl.where(p >= threshold, p, 0.0), axis=0)

    # 3. CDF sampling
    target = u * sum_valid
    accum = 0.0
    sampled_idx = -1

    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        filtered_p = tl.where(p >= threshold, p, 0.0)

        block_cdf = accum + tl.cumsum(filtered_p, axis=0)
        matched = mask & (block_cdf >= target) & (filtered_p > 0.0) & (sampled_idx < 0)

        if tl.sum(matched.to(tl.int32), axis=0) > 0:
            first_match_offset = tl.min(tl.where(matched, cols, vocab_size), axis=0)
            if sampled_idx < 0:
                sampled_idx = first_match_offset

        accum += tl.sum(filtered_p, axis=0)

    if sampled_idx < 0:
        sampled_idx = vocab_size - 1

    tl.store(output_ptr + row_idx, sampled_idx)
    tl.store(valid_ptr + row_idx, True)


# Shared bisection + CDF core: used by the from-logits fallback kernel and by
# the radix sampler's overflow branch. All three filters reduce to value
# thresholds on logits, so one register-resident pass chain covers them.
@triton.jit
def _bisect_filtered_sample(
    logits_ptr,
    row_offset,
    vocab_size,
    top_k,
    top_p,
    min_p,
    seed,
    offset,
    HAS_TOP_K: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
    HAS_MIN_P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample one logits row through bisection thresholds plus a CDF scan.

    # ======================================================================
    # FUSED TOP-K(16) + TOP-P(12) + MIN-P BISECTION (BLOCK_SIZE<=4096)
    #
    # Tiles:
    #   vocab chunked by BLOCK_SIZE; OOB logits read as -inf so they add
    #   neither counts (mid can be negative) nor mass (exp gives 0).
    # Passes:
    #   A. x_max + x_min (x_min over finite values only: the bisection
    #      domain must stay finite).
    #   B. top-k: 16 count-bisection iterations for count(x >= mid) >= k.
    #   C. min-p: x >= x_max + log(min_p), renorm-invariant since p_i and
    #      p_max share the scale.
    #   D. top-p: S_k = sum exp(x - x_max) over the top-k set, then 12
    #      bisection iterations for the largest t with
    #      sum_{x >= t} exp(x - x_max) >= top_p * S_k.
    #   E. s_keep over x >= max(t_k, t_p, t_minp), then CDF scan on
    #      exp(x - x_max) restricted to lanes with w > 0.
    # RNG:
    #   u = philox_u01_f32(seed, flat), flat = offset (batch-invariant).
    # Memory:
    #   logits re-read per pass (1 + 16 + 12 + 1 + 1 sweeps); no sort, no
    #   softmax materialization.
    # Precision/fallback:
    #   fp32 accumulation; with no match, vocab_size-1.
    # ======================================================================

    Args:
        logits_ptr: Pointer to ``[batch, vocab]`` logits.
        row_offset: Element offset of the row inside ``logits_ptr``.
        vocab_size: Number of vocabulary columns.
        top_k: Per-row top-k limit (``<= 0`` or ``>= vocab_size`` disables).
        top_p: Per-row nucleus mass target (``>= 1`` disables).
        min_p: Per-row relative threshold (``<= 0`` disables).
        seed: Philox key for the row.
        offset: Philox flat stream address for the row.
        HAS_TOP_K: Compile-time gate for the top-k bisection pass.
        HAS_TOP_P: Compile-time gate for the top-p bisection pass.
        HAS_MIN_P: Compile-time gate for the min-p threshold pass.
        BLOCK_SIZE: Tile width along the vocabulary dimension.

    Returns:
        Int32 sampled token index (``vocab_size - 1`` when no lane matched).
    """
    u = philox_u01_f32(seed, offset.to(tl.int64))

    # Pass A: max + min (bisection domain for top-k).
    x_max = float("-inf")
    x_min = float("inf")
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        x = tl.load(logits_ptr + row_offset + cols, mask=mask, other=float("-inf")).to(tl.float32)
        x_max = tl.maximum(x_max, tl.max(x, axis=0))
        x_min = tl.minimum(x_min, tl.min(tl.where(x > float("-inf"), x, float("inf")), axis=0))

    # --- top-k threshold ---
    t_k = float("-inf")
    if HAS_TOP_K and top_k > 0 and top_k < vocab_size:
        lo = x_min
        hi = x_max
        for _ in range(16):
            mid = (lo + hi) * 0.5
            cnt = 0
            for v_offset in range(0, vocab_size, BLOCK_SIZE):
                cols = v_offset + tl.arange(0, BLOCK_SIZE)
                mask = cols < vocab_size
                x = tl.load(logits_ptr + row_offset + cols, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                cnt += tl.sum((x >= mid).to(tl.int32), axis=0)
            if cnt >= top_k:
                lo = mid
            else:
                hi = mid
        t_k = lo

    # --- min-p threshold (logit space, renorm-independent) ---
    t_minp = float("-inf")
    if HAS_MIN_P and min_p > 0.0:
        t_minp = x_max + tl.log(min_p)

    # --- top-p threshold over the top-k-renormalized dist ---
    t_p = float("-inf")
    if HAS_TOP_P and top_p < 1.0:
        # S_k = sum e^(x - x_max) over the top-k set (t_k = -inf => whole row).
        s_k = 0.0
        for v_offset in range(0, vocab_size, BLOCK_SIZE):
            cols = v_offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            x = tl.load(logits_ptr + row_offset + cols, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            s_k += tl.sum(tl.where(x >= t_k, tl.exp(x - x_max), 0.0), axis=0)
        # Bisect value t in [max(t_k, x_min), x_max] for the largest
        # threshold with sum_{x >= t} e^(x - x_max) >= top_p * S_k.
        lo = tl.where(t_k == float("-inf"), x_min, t_k)
        hi = x_max
        for _ in range(12):
            mid = (lo + hi) * 0.5
            partial = 0.0
            for v_offset in range(0, vocab_size, BLOCK_SIZE):
                cols = v_offset + tl.arange(0, BLOCK_SIZE)
                mask = cols < vocab_size
                x = tl.load(logits_ptr + row_offset + cols, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                partial += tl.sum(tl.where(x >= mid, tl.exp(x - x_max), 0.0), axis=0)
            if partial >= top_p * s_k:
                lo = mid
            else:
                hi = mid
        t_p = lo

    t_final = t_k
    if t_p > t_final:
        t_final = t_p
    if t_minp > t_final:
        t_final = t_minp

    # --- sum survivors, then CDF scan ---
    s_keep = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        x = tl.load(logits_ptr + row_offset + cols, mask=mask, other=float("-inf")).to(tl.float32)
        s_keep += tl.sum(tl.where(x >= t_final, tl.exp(x - x_max), 0.0), axis=0)

    target = u * s_keep
    accum = 0.0
    sampled_idx = -1
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        x = tl.load(logits_ptr + row_offset + cols, mask=mask, other=float("-inf")).to(tl.float32)
        w = tl.where(x >= t_final, tl.exp(x - x_max), 0.0)
        block_cdf = accum + tl.cumsum(w, axis=0)
        matched = mask & (block_cdf >= target) & (w > 0.0) & (sampled_idx < 0)
        if tl.sum(matched.to(tl.int32), axis=0) > 0:
            first_match_offset = tl.min(tl.where(matched, cols, vocab_size), axis=0)
            if sampled_idx < 0:
                sampled_idx = first_match_offset
        accum += tl.sum(w, axis=0)

    if sampled_idx < 0:
        sampled_idx = vocab_size - 1

    return sampled_idx


# Kernel: Fused top-k + top-p + min-p sampling FROM LOGITS (no sort, no
# softmax materialization — FlashInfer's joint path lacks min_p; this kernel
# covers ALL THREE filters in a single launch).
@triton.jit
def _fused_topk_topp_minp_kernel(
    logits_ptr,
    output_ptr,
    valid_ptr,
    top_k_ptr,
    top_p_ptr,
    min_p_ptr,
    seed_ptr,
    offset_ptr,
    vocab_size,
    logits_stride_b,
    HAS_TOP_K: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
    HAS_MIN_P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample one row from logits with fused top-k, top-p, and min-p.

    Follows HF filter order ``top-k -> renorm -> top-p -> min-p``. All
    three filters reduce to value thresholds on logits.

    # ======================================================================
    # FUSED TOP-K(16) + TOP-P(12) + MIN-P (BLOCK_SIZE<=4096, num_warps=4,
    # grid=(B,))
    #
    # Programs:
    #   1 program/row — row = tl.program_id(0); rows never communicate.
    # Tiles:
    #   vocab chunked by BLOCK_SIZE; OOB logits read as -inf so they add
    #   neither counts (mid can be negative) nor mass (exp gives 0).
    # Passes:
    #   A. x_max + x_min (x_min over finite values only: the bisection
    #      domain must stay finite).
    #   B. top-k: 16 count-bisection iterations for count(x >= mid) >= k.
    #   C. min-p: x >= x_max + log(min_p), renorm-invariant since p_i and
    #      p_max share the scale.
    #   D. top-p: S_k = sum exp(x - x_max) over the top-k set, then 12
    #      bisection iterations for the largest t with
    #      sum_{x >= t} exp(x - x_max) >= top_p * S_k.
    #   E. s_keep over x >= max(t_k, t_p, t_minp), then CDF scan on
    #      exp(x - x_max) restricted to lanes with w > 0.
    # RNG:
    #   u = philox_u01_f32(seed, flat), flat = offset (batch-invariant).
    # Memory:
    #   logits re-read per pass (1 + 16 + 12 + 1 + 1 sweeps); one int32 +
    #   one bool stored per row. No sort, no softmax materialization.
    # Precision/fallback:
    #   fp32 accumulation; with no match, vocab_size-1.
    # ======================================================================

    Args:
        logits_ptr: Pointer to ``[batch, vocab]`` logits.
        output_ptr: Pointer to ``[batch]`` int32 sampled indices.
        valid_ptr: Pointer to ``[batch]`` bool validity flags.
        top_k_ptr: Pointer to ``[batch]`` int32 top-k limits.
        top_p_ptr: Pointer to ``[batch]`` float32 nucleus targets.
        min_p_ptr: Pointer to ``[batch]`` float32 relative thresholds.
        seed_ptr: Pointer to ``[batch]`` Philox seeds.
        offset_ptr: Pointer to ``[batch]`` Philox offsets.
        vocab_size: Number of vocabulary columns.
        logits_stride_b: Row stride (in elements) of ``logits_ptr``.
        HAS_TOP_K: Whether the top-k filter is active for any row.
        HAS_TOP_P: Whether the top-p filter is active for any row.
        HAS_MIN_P: Whether the min-p filter is active for any row.
        BLOCK_SIZE: Tile width along the vocabulary dimension.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_offset = row_idx * logits_stride_b

    top_k = tl.load(top_k_ptr + row_idx).to(tl.int32)
    top_p = tl.load(top_p_ptr + row_idx).to(tl.float32)
    min_p = tl.load(min_p_ptr + row_idx).to(tl.float32)
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64)
    offset = tl.load(offset_ptr + row_idx).to(tl.uint64)

    sampled_idx = _bisect_filtered_sample(
        logits_ptr,
        row_offset,
        vocab_size,
        top_k,
        top_p,
        min_p,
        seed,
        offset,
        HAS_TOP_K,
        HAS_TOP_P,
        HAS_MIN_P,
        BLOCK_SIZE,
    )

    tl.store(output_ptr + row_idx, sampled_idx)
    tl.store(valid_ptr + row_idx, True)


# ===========================================================================
# Radix-select fused sampling (no torch.topk, no host sync)
#
# The previous fast path spent ~97% of its GPU time inside PyTorch's
# multi-block radix select and paid four host synchronizations per call
# (``.item()``, ``.cpu()``, ``.any()``), which dominate on Windows/WDDM.
# This pipeline replaces both with a 4-launch, fully device-side sequence:
#
#   K1  histogram the top 8 bits of the monotonic key + chunk maxima
#   K2  per-row reduce: x_max, boundary bin b* holding the k-th largest value
#   K3  collect every value with bin >= b* (bounded by CAP) + token ids
#   K4  sort the candidates in registers, apply top-k/top-p/min-p, Philox CDF
#
# The dispatch decision needs only ``dtype``/``shape``, so the op never reads
# a device scalar back to the host.
#
# Correctness: candidates contain every element of bins >= b*, and the k-th
# largest value always lives in b*, so the top-k set is exactly the first
# ``k`` sorted candidates. Every survivor of the later filters (top-p cuts
# strictly less mass than top-k, min-p is a value threshold) therefore sits in
# the candidate buffer, and the in-register sort makes the result exact. Rows
# whose candidate count exceeds ``CAP`` — and rows with top-k disabled, whose
# survivor set is unbounded — rerun the bisection core inside K4.
# ===========================================================================

_RADIX_BINS = 256
_RADIX_SPLIT = 16
_RADIX_CAP = 1024
_RADIX_MIN_VOCAB = 4096
_RADIX_BLOCK = 2048

_RADIX_BUFFERS: dict[tuple[int, int], dict[str, torch.Tensor]] = {}


@triton.jit
def _radix_key16(bits):
    """Map a 16-bit fp16/bf16 bit pattern to a monotonic unsigned sort key.

    Positive patterns get their sign bit set, negative patterns are bitwise
    inverted, so unsigned key order equals float order (``-inf`` lowest,
    ``+inf`` highest). ``-0.0`` must be normalized to ``+0.0`` before the
    bitcast, otherwise it would map above every positive value.

    Args:
        bits: Int32 lanes holding the raw 16-bit pattern.

    Returns:
        Int32 lanes in ``[0, 65535]`` ordered like the source floats.
    """
    sign = bits >> 15
    return tl.where(sign == 1, 0xFFFF - bits, bits + 0x8000)


@triton.jit
def _radix_bins(logits_ptr, mask, other):
    """Load a tile and return ``(float32 values, 8-bit bin ids)``.

    Values are binned through their float16 rounding: rounding is monotonic,
    so bin order equals value order for every input dtype, and the bin is
    ~19% wide in value near the top of the vocabulary — narrow enough that
    the bin holding the k-th largest value usually holds few candidates.
    Exact ordering inside a bin is recovered later from the untouched
    float32 values, so binning never changes the sampled distribution.

    Args:
        logits_ptr: Pointer block to the tile.
        mask: In-vocabulary lane mask.
        other: Fill value for masked lanes.

    Returns:
        Tuple ``(x, bins)`` where ``x`` is float32 and ``bins`` is the top
        8 bits of the monotonic key (0 = most negative, 255 = largest).
    """
    xn = tl.load(logits_ptr, mask=mask, other=other)
    x = xn.to(tl.float32)
    bits = x.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.int32)
    bits = tl.where(x == 0.0, 0, bits)
    return x, _radix_key16(bits) >> 8


@triton.jit
def _radix_hist_kernel(
    logits_ptr,
    hist_ptr,
    chunk_max_ptr,
    vocab_size,
    logits_stride_b,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
):
    """Pass 1: per-chunk count histogram + maximum.

    # ======================================================================
    # RADIX HISTOGRAM (BLOCK_SIZE<=2048, grid=(B, NUM_BUCKETS))
    #
    # Each program owns a contiguous vocabulary chunk, so the histogram is
    # written without atomics (deterministic) and reduced by the threshold
    # kernel in a fixed order. A chunk past the vocabulary stores zeros.
    # ======================================================================

    Args:
        logits_ptr: Pointer to ``[batch, vocab]`` logits.
        hist_ptr: Pointer to ``[batch, NUM_BUCKETS, NUM_BINS]`` int32 counts.
        chunk_max_ptr: Pointer to ``[batch, NUM_BUCKETS]`` float32 maxima.
        vocab_size: Number of vocabulary columns.
        logits_stride_b: Row stride (in elements) of ``logits_ptr``.
        chunk_size: Elements per bucket.
        BLOCK_SIZE: Tile width along the vocabulary dimension.
        NUM_BINS: Histogram bins (256).
        NUM_BUCKETS: Vocabulary chunks per row.
    """
    row = tl.program_id(0).to(tl.int64)
    bucket = tl.program_id(1)
    start = bucket * chunk_size
    limit = tl.minimum(vocab_size, start + chunk_size)
    row_offset = row * logits_stride_b

    hist = tl.zeros([NUM_BINS], dtype=tl.int32)
    local_max = float("-inf")
    for off in range(0, chunk_size, BLOCK_SIZE):
        cols = start + off + tl.arange(0, BLOCK_SIZE)
        mask = cols < limit
        x, bins = _radix_bins(logits_ptr + row_offset + cols, mask, float("-inf"))
        local_max = tl.maximum(local_max, tl.max(x, axis=0))
        hist += tl.histogram(bins, NUM_BINS, mask=mask)

    tl.store(hist_ptr + (row * NUM_BUCKETS + bucket) * NUM_BINS + tl.arange(0, NUM_BINS), hist)
    tl.store(chunk_max_ptr + row * NUM_BUCKETS + bucket, local_max)


@triton.jit
def _radix_threshold_kernel(
    hist_ptr,
    chunk_max_ptr,
    top_k_ptr,
    x_max_ptr,
    b_star_ptr,
    counter_ptr,
    NUM_BUCKETS: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    """Pass 2: reduce histograms, pick the boundary bin, reset the counter.

    # ======================================================================
    # RADIX THRESHOLD (grid=(B,), one program per row)
    #
    # ``b*`` is the smallest bin whose count of strictly larger values is
    # below ``k``: the k-th largest value is guaranteed to live there. The
    # candidate counter is reset here so the collect pass needs no extra
    # memset launch.
    # ======================================================================

    Args:
        hist_ptr: Pointer to ``[batch, NUM_BUCKETS, NUM_BINS]`` int32 counts.
        chunk_max_ptr: Pointer to ``[batch, NUM_BUCKETS]`` float32 maxima.
        top_k_ptr: Pointer to ``[batch]`` int32 top-k limits.
        x_max_ptr: Pointer to ``[batch]`` float32 row maxima (output).
        b_star_ptr: Pointer to ``[batch]`` int32 boundary bins (output).
        counter_ptr: Pointer to ``[batch]`` int32 candidate counters (reset).
        NUM_BUCKETS: Vocabulary chunks per row.
        NUM_BINS: Histogram bins (256).
    """
    row = tl.program_id(0).to(tl.int64)
    bins = tl.arange(0, NUM_BINS)
    counts = tl.zeros([NUM_BINS], dtype=tl.int32)
    x_max = float("-inf")
    for bucket in range(NUM_BUCKETS):
        counts += tl.load(hist_ptr + (row * NUM_BUCKETS + bucket) * NUM_BINS + bins)
        x_max = tl.maximum(x_max, tl.load(chunk_max_ptr + row * NUM_BUCKETS + bucket))

    total = tl.sum(counts, axis=0)
    above = total - tl.cumsum(counts, axis=0)
    k = tl.load(top_k_ptr + row).to(tl.int32)
    k_eff = tl.where((k > 0) & (k < total), k, total)
    ok = above < k_eff
    b_star = tl.min(tl.where(ok, bins, NUM_BINS), axis=0)
    b_star = tl.minimum(b_star, NUM_BINS - 1)

    tl.store(x_max_ptr + row, x_max)
    tl.store(b_star_ptr + row, b_star)
    tl.store(counter_ptr + row, 0)


@triton.jit
def _radix_collect_kernel(
    logits_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    counter_ptr,
    b_star_ptr,
    vocab_size,
    logits_stride_b,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
    CAP: tl.constexpr,
):
    """Pass 3: append every value with ``bin >= b*`` to the row's buffer.

    # ======================================================================
    # RADIX COLLECT (BLOCK_SIZE<=2048, grid=(B, NUM_BUCKETS))
    #
    # One atomic reservation per tile (not per element) plus a register
    # prefix sum keeps the append cheap; lanes past ``CAP`` are dropped and
    # detected by the sampling pass. Candidate order is unspecified on
    # purpose: the sampler sorts by value with a token-id tie-break, so the
    # result does not depend on the atomic schedule.
    # ======================================================================

    Args:
        logits_ptr: Pointer to ``[batch, vocab]`` logits.
        cand_val_ptr: Pointer to ``[batch, CAP]`` float32 candidate values.
        cand_idx_ptr: Pointer to ``[batch, CAP]`` int32 candidate token ids.
        counter_ptr: Pointer to ``[batch]`` int32 candidate counters.
        b_star_ptr: Pointer to ``[batch]`` int32 boundary bins.
        vocab_size: Number of vocabulary columns.
        logits_stride_b: Row stride (in elements) of ``logits_ptr``.
        chunk_size: Elements per bucket.
        BLOCK_SIZE: Tile width along the vocabulary dimension.
        CAP: Candidate buffer capacity per row.
    """
    row = tl.program_id(0).to(tl.int64)
    bucket = tl.program_id(1)
    b_star = tl.load(b_star_ptr + row)
    start = bucket * chunk_size
    limit = tl.minimum(vocab_size, start + chunk_size)
    row_offset = row * logits_stride_b

    for off in range(0, chunk_size, BLOCK_SIZE):
        cols = start + off + tl.arange(0, BLOCK_SIZE)
        mask = cols < limit
        x, bins = _radix_bins(logits_ptr + row_offset + cols, mask, float("-inf"))
        take = mask & (bins >= b_star)
        n = tl.sum(take.to(tl.int32), axis=0)
        base = tl.atomic_add(counter_ptr + row, n)
        pos = base + tl.cumsum(take.to(tl.int32), axis=0) - 1
        keep = take & (pos < CAP)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)
        tl.store(cand_idx_ptr + row * CAP + pos, cols, mask=keep)


@triton.jit
def _radix_sample_kernel(
    logits_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    counter_ptr,
    x_max_ptr,
    output_ptr,
    valid_ptr,
    top_k_ptr,
    top_p_ptr,
    min_p_ptr,
    seed_ptr,
    offset_ptr,
    vocab_size,
    logits_stride_b,
    CAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Pass 4: exact filter + Philox CDF draw over the candidate set.

    # ======================================================================
    # RADIX SAMPLE (grid=(B,), one program per row)
    #
    # Candidates are sorted in registers by ``(value, token id)`` descending,
    # which reconstructs the exact global order for every element at or above
    # ``b*``. Top-k keeps the first ``k`` candidates; top-p then cuts the
    # prefix whose leading mass is still below ``top_p``; min-p drops
    # candidates below ``min_p * p_max``. A row whose candidate count exceeds
    # ``CAP`` (or whose top-k is disabled) reruns the bisection core.
    # ======================================================================

    Args:
        logits_ptr: Pointer to ``[batch, vocab]`` logits.
        cand_val_ptr: Pointer to ``[batch, CAP]`` float32 candidate values.
        cand_idx_ptr: Pointer to ``[batch, CAP]`` int32 candidate token ids.
        counter_ptr: Pointer to ``[batch]`` int32 candidate counters.
        x_max_ptr: Pointer to ``[batch]`` float32 row maxima.
        output_ptr: Pointer to ``[batch]`` int32 sampled indices.
        valid_ptr: Pointer to ``[batch]`` bool validity flags.
        top_k_ptr: Pointer to ``[batch]`` int32 top-k limits.
        top_p_ptr: Pointer to ``[batch]`` float32 nucleus targets.
        min_p_ptr: Pointer to ``[batch]`` float32 relative thresholds.
        seed_ptr: Pointer to ``[batch]`` Philox seeds.
        offset_ptr: Pointer to ``[batch]`` Philox offsets.
        vocab_size: Number of vocabulary columns.
        logits_stride_b: Row stride (in elements) of ``logits_ptr``.
        CAP: Candidate buffer capacity per row.
        BLOCK_SIZE: Tile width used by the bisection fallback.
    """
    row = tl.program_id(0).to(tl.int64)
    top_k = tl.load(top_k_ptr + row).to(tl.int32)
    top_p = tl.load(top_p_ptr + row).to(tl.float32)
    min_p = tl.load(min_p_ptr + row).to(tl.float32)
    seed = tl.load(seed_ptr + row).to(tl.uint64)
    offset = tl.load(offset_ptr + row).to(tl.uint64)
    count = tl.load(counter_ptr + row)

    if count > CAP:
        token = _bisect_filtered_sample(
            logits_ptr,
            row * logits_stride_b,
            vocab_size,
            top_k,
            top_p,
            min_p,
            seed,
            offset,
            tl.constexpr(True),
            tl.constexpr(True),
            tl.constexpr(True),
            BLOCK_SIZE,
        )
        tl.store(output_ptr + row, token)
    else:
        lanes = tl.arange(0, CAP)
        lane_ok = lanes < count
        v = tl.load(cand_val_ptr + row * CAP + lanes, mask=lane_ok, other=0.0)
        idx = tl.load(cand_idx_ptr + row * CAP + lanes, mask=lane_ok, other=0)

        # Sort key: monotonic fp32 key (high 32 bits) | token id (low 32).
        vz = tl.where(v == 0.0, 0.0, v)
        vb = vz.to(tl.uint32, bitcast=True).to(tl.int64)
        sign = vb >> 31
        key = tl.where(sign == 1, 0xFFFFFFFF - vb, vb + 0x80000000)
        # ``key - 0x80000000`` maps the unsigned 32-bit key into the signed
        # range, so the packed 64-bit key orders like the float. Equal values
        # break ties on the LOWER token id, matching the stable descending
        # sort of the torch oracle. Lanes past ``count`` collapse to the
        # minimum int64 and sort to the tail.
        key64 = ((key - 0x80000000) << 32) | (0x7FFFFFFF - idx.to(tl.int64))
        key64 = tl.where(lane_ok, key64, -9223372036854775808)
        ordered = tl.sort(key64, descending=tl.constexpr(True))

        rank_ok = lanes < count
        skey = (ordered >> 32) + 0x80000000
        sidx = 0x7FFFFFFF - (ordered & 0xFFFFFFFF)
        sbits = tl.where(skey >= 0x80000000, skey - 0x80000000, 0xFFFFFFFF - skey)
        vals = sbits.to(tl.int32).to(tl.float32, bitcast=True)

        x_max = tl.load(x_max_ptr + row)
        k_eff = tl.where((top_k > 0) & (top_k < vocab_size), top_k, vocab_size)
        e = tl.where(rank_ok & (lanes < k_eff), tl.exp(vals - x_max), 0.0)
        s_k = tl.sum(e, axis=0).to(tl.float32)
        probs = tl.where(s_k > 0.0, e / s_k, 0.0)

        prev = tl.where(s_k > 0.0, (tl.cumsum(e, axis=0) - e) / s_k, 0.0)
        keep_top_p = (lanes == 0) | (top_p >= 1.0) | (prev <= top_p)
        probs = tl.where(keep_top_p, probs, 0.0)

        p_max = tl.sum(tl.where(lanes == 0, probs, 0.0), axis=0)
        keep_min_p = (min_p <= 0.0) | (probs >= min_p * p_max)
        probs = tl.where(keep_min_p, probs, 0.0)

        total_p = tl.sum(probs, axis=0)
        u = philox_u01_f32(seed, offset.to(tl.int64))
        target = u * total_p
        filtered_cum = tl.cumsum(probs, axis=0)
        matched = rank_ok & (probs > 0.0) & (filtered_cum >= target)
        matched_idx = tl.min(tl.where(matched, lanes, CAP), axis=0)
        final_idx = tl.where(matched_idx < CAP, matched_idx, 0)
        token = tl.sum(tl.where(lanes == final_idx, sidx, 0), axis=0).to(tl.int32)
        tl.store(output_ptr + row, token)

    tl.store(valid_ptr + row, True)


def _radix_buffers(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Return the reusable intermediate buffers for one ``(batch, device)``.

    Intermediates are scratch only: every element the sampler reads is
    written by the current launch, so reuse is safe on a single stream and
    keeps the op free of per-step allocations (and CUDA-graph friendly).

    Args:
        batch_size: Leading logits dimension.
        device: CUDA device of the logits.

    Returns:
        Mapping with ``hist``, ``chunk_max``, ``x_max``, ``b_star``,
        ``counter``, ``cand_val`` and ``cand_idx`` tensors.
    """
    key = (batch_size, device.index or 0)
    cached = _RADIX_BUFFERS.get(key)
    if cached is None:
        cached = {
            "hist": torch.empty(
                (batch_size, _RADIX_SPLIT, _RADIX_BINS), dtype=torch.int32, device=device
            ),
            "chunk_max": torch.empty(
                (batch_size, _RADIX_SPLIT), dtype=torch.float32, device=device
            ),
            "x_max": torch.empty((batch_size,), dtype=torch.float32, device=device),
            "b_star": torch.empty((batch_size,), dtype=torch.int32, device=device),
            "counter": torch.empty((batch_size,), dtype=torch.int32, device=device),
            "cand_val": torch.empty((batch_size, _RADIX_CAP), dtype=torch.float32, device=device),
            "cand_idx": torch.empty((batch_size, _RADIX_CAP), dtype=torch.int32, device=device),
        }
        if len(_RADIX_BUFFERS) >= 8:
            _RADIX_BUFFERS.clear()
        _RADIX_BUFFERS[key] = cached
    return cached


def _radix_sampling_from_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the 4-launch radix-select sampler over ``[B, V]`` logits.

    Caller contract: ``logits`` is a CUDA ``[B, V]`` float tensor with
    ``V >= _RADIX_MIN_VOCAB``; every filter parameter is read per row on the
    device, so any mix of active/inactive top-k, top-p and min-p is legal.
    No host synchronization is performed.

    Args:
        logits: ``[B, V]`` float logits.
        top_k: ``[B]`` int32 top-k limits.
        top_p: ``[B]`` float32 nucleus targets.
        min_p: ``[B]`` float32 relative thresholds.
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 Philox offsets.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` ids and bool ``[B]``
        flags.
    """
    batch_size, vocab_size = logits.shape
    device = logits.device
    buffers = _radix_buffers(batch_size, device)
    output = torch.empty((batch_size,), dtype=torch.int32, device=device)
    valid = torch.empty((batch_size,), dtype=torch.bool, device=device)
    chunk_size = triton.cdiv(vocab_size, _RADIX_SPLIT)
    block = min(_RADIX_BLOCK, triton.next_power_of_2(chunk_size))

    grid = (batch_size, _RADIX_SPLIT)
    cast(Any, _radix_hist_kernel)[grid](
        logits,
        buffers["hist"],
        buffers["chunk_max"],
        vocab_size=vocab_size,
        logits_stride_b=logits.stride(0),
        chunk_size=chunk_size,
        BLOCK_SIZE=block,
        NUM_BINS=_RADIX_BINS,
        NUM_BUCKETS=_RADIX_SPLIT,
        num_warps=4,
    )
    cast(Any, _radix_threshold_kernel)[(batch_size,)](
        buffers["hist"],
        buffers["chunk_max"],
        top_k,
        buffers["x_max"],
        buffers["b_star"],
        buffers["counter"],
        NUM_BUCKETS=_RADIX_SPLIT,
        NUM_BINS=_RADIX_BINS,
        num_warps=4,
    )
    cast(Any, _radix_collect_kernel)[grid](
        logits,
        buffers["cand_val"],
        buffers["cand_idx"],
        buffers["counter"],
        buffers["b_star"],
        vocab_size=vocab_size,
        logits_stride_b=logits.stride(0),
        chunk_size=chunk_size,
        BLOCK_SIZE=block,
        CAP=_RADIX_CAP,
        num_warps=4,
    )
    cast(Any, _radix_sample_kernel)[(batch_size,)](
        logits,
        buffers["cand_val"],
        buffers["cand_idx"],
        buffers["counter"],
        buffers["x_max"],
        output,
        valid,
        top_k,
        top_p,
        min_p,
        seed,
        offset,
        vocab_size=vocab_size,
        logits_stride_b=logits.stride(0),
        CAP=_RADIX_CAP,
        BLOCK_SIZE=min(4096, triton.next_power_of_2(vocab_size)),
        num_warps=4,
    )
    return output, valid


@custom_op(
    namespace="ayaka",
    reference=fused_sampling_ref,
    fake_impl=sampling_fake,
    dispatch_key="CUDA",
)
def fused_topk_topp_minp_sampling_from_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from temperature-scaled logits with all filters in one launch.

    Args:
        logits: ``[B, V]`` float logits, already temperature-scaled.
        top_k: ``[B]`` int top-k limits; ``<= 0`` or ``>= V`` disables.
        top_p: ``[B]`` float nucleus targets; ``>= 1`` disables.
        min_p: ``[B]`` float relative thresholds; ``<= 0`` disables.
        seed: ``[B]`` int64 counter-based Philox seeds.
        offset: ``[B]`` int64 base offsets with ``flat = offset``.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` sampled ids and
        bool ``[B]`` validity flags on the same device as ``logits``.

    Raises:
        TypeError: If any input is not a tensor, has a non-float logits
            dtype, or ``seed``/``offset`` is not int64.
        ValueError: If any tensor is not CUDA, ``logits`` is not 2-D, or a
            parameter shape/device mismatches the batch.

    Note:
        Pure function: inputs are never mutated and both outputs are freshly
        allocated. Deterministic in ``(seed, offset)`` with no generator
        state, hence CUDA-graph safe. ``V >= 4096`` dispatches the 4-launch
        radix pipeline (device-side histogram, boundary-bin selection,
        candidate collection, in-register filter + Philox draw); smaller
        vocabularies use the single-launch bisection kernel. Neither path
        reads a device value back to the host, so no step synchronizes.
        Reference: torch top-k/top-p/min-p filtering plus a CDF draw, usable
        with ``verify_against_reference``.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be a float dtype; got {logits.dtype}")
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [B, V]; got {tuple(logits.shape)}")
    batch_size, vocab_size = logits.shape
    for tensor, tensor_name in (
        (top_k, "top_k"),
        (top_p, "top_p"),
        (min_p, "min_p"),
        (seed, "seed"),
        (offset, "offset"),
    ):
        validate_param_tensor(tensor, tensor_name, batch_size, logits.device)
    if seed.dtype != torch.int64:
        raise TypeError(f"seed must have dtype torch.int64; got {seed.dtype}")
    if offset.dtype != torch.int64:
        raise TypeError(f"offset must have dtype torch.int64; got {offset.dtype}")

    # Radix path for production vocabularies: no torch.topk, no host sync,
    # every filter decision taken per row on the device. Smaller vocabularies
    # keep the single-launch bisection kernel, whose fixed 30-sweep cost is
    # cheaper than four launches at that size.
    if vocab_size >= _RADIX_MIN_VOCAB:
        return _radix_sampling_from_logits(logits, top_k, top_p, min_p, seed, offset)

    output = torch.empty((batch_size,), dtype=torch.int32, device=logits.device)
    valid = torch.empty((batch_size,), dtype=torch.bool, device=logits.device)

    grid = (batch_size,)
    block_size = min(4096, triton.next_power_of_2(vocab_size))
    num_warps = 8 if vocab_size >= 32768 else 4

    cast(Any, _fused_topk_topp_minp_kernel)[grid](
        logits,
        output,
        valid,
        top_k,
        top_p,
        min_p,
        seed,
        offset,
        vocab_size=vocab_size,
        logits_stride_b=logits.stride(0),
        HAS_TOP_K=True,
        HAS_TOP_P=True,
        HAS_MIN_P=True,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output, valid


@custom_op(
    namespace="ayaka",
    reference=sampling_from_probs_ref,
    fake_impl=sampling_fake,
    dispatch_key="CUDA",
)
def sampling_from_probs(
    probs: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw categorical samples from per-row probabilities.

    Args:
        probs: ``[B, V]`` float probabilities, need not be normalized.
        seed_arr: Optional ``[B]`` int64 Philox seeds; ``seed_val`` is used
            when ``None``.
        seed_val: Scalar seed fallback when ``seed_arr`` is ``None``.
        offset_arr: Optional ``[B]`` int64 Philox offsets; ``offset_val`` is
            used when ``None``.
        offset_val: Scalar offset fallback when ``offset_arr`` is ``None``.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` sampled ids and
        bool ``[B]`` validity flags on the same device as ``probs``.

    Raises:
        TypeError: If ``probs`` is not a floating tensor or a seed/offset
            tensor is not int64.
        ValueError: If ``probs`` is not a CUDA ``[B, V]`` tensor or a
            seed/offset shape/device mismatches the batch.

    Note:
        Pure function: ``probs`` is never mutated. Tensor seeds keep each
        row deterministic in ``(seed, offset)`` and batch-invariant; scalar
        fallbacks share one stream. Uses ``BLOCK_SIZE`` of
        ``min(4096, next_power_of_2(V))``, ``grid=(B,)``, ``num_warps=4``.
    """
    batch_size, vocab_size = validate_probs(probs)
    validate_opt_seed_offset(seed_arr, "seed_arr", batch_size, probs.device)
    validate_opt_seed_offset(offset_arr, "offset_arr", batch_size, probs.device)
    output, valid = prepare_sampling_outputs(probs)

    grid = (batch_size,)
    block_size = min(4096, triton.next_power_of_2(vocab_size))
    num_warps = 8 if vocab_size >= 32768 else 4

    cast(Any, _sampling_from_probs_kernel)[grid](
        probs,
        output,
        valid,
        seed_arr if seed_arr is not None else probs,
        offset_arr if offset_arr is not None else probs,
        vocab_size=vocab_size,
        probs_stride_b=probs.stride(0),
        seed_val=seed_val,
        offset_val=offset_val,
        HAS_SEED_TENSOR=seed_arr is not None,
        HAS_OFFSET_TENSOR=offset_arr is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output, valid


@custom_op(
    namespace="ayaka",
    name="min_p_sampling_from_probs",
    reference=min_p_sampling_ref,
    fake_impl=sampling_fake,
    dispatch_key="CUDA",
)
def _min_p_sampling_op(
    probs: torch.Tensor,
    min_p: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU min-p sampling with a tensor-only threshold (registered op).

    Scalar thresholds are normalized to ``[B]`` by the public
    ``min_p_sampling_from_probs`` wrapper because ``torch.library`` schemas
    cannot express ``Tensor | float`` unions.

    Raises:
        TypeError: If ``probs``/``min_p`` fail dtype validation.
        ValueError: If ``probs`` is not a CUDA ``[B, V]`` tensor or a
            parameter shape/device mismatches the batch.
    """
    batch_size, vocab_size = validate_probs(probs)
    validate_param_tensor(min_p, "min_p", batch_size, probs.device)
    validate_opt_seed_offset(seed_arr, "seed_arr", batch_size, probs.device)
    validate_opt_seed_offset(offset_arr, "offset_arr", batch_size, probs.device)
    output, valid = prepare_sampling_outputs(probs)

    has_tensor = True
    min_p_val = 0.0
    min_p_ptr = min_p

    grid = (batch_size,)
    block_size = min(4096, triton.next_power_of_2(vocab_size))
    num_warps = 8 if vocab_size >= 32768 else 4

    cast(Any, _min_p_sampling_kernel)[grid](
        probs,
        output,
        valid,
        min_p_ptr,
        seed_arr if seed_arr is not None else probs,
        offset_arr if offset_arr is not None else probs,
        vocab_size=vocab_size,
        probs_stride_b=probs.stride(0),
        min_p_val=min_p_val,
        seed_val=seed_val,
        offset_val=offset_val,
        HAS_MIN_P_TENSOR=has_tensor,
        HAS_SEED_TENSOR=seed_arr is not None,
        HAS_OFFSET_TENSOR=offset_arr is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output, valid


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: torch.Tensor | float = 0.05,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw min-p samples, compatible with FlashInfer ``min_p`` sampling.

    Args:
        probs: ``[B, V]`` float probabilities.
        min_p: Per-row ``[B]`` threshold tensor or scalar; entries below
            ``max_prob * min_p`` are filtered out.
        seed_arr: Optional ``[B]`` int64 Philox seeds; ``seed_val`` is used
            when ``None``.
        seed_val: Scalar seed fallback when ``seed_arr`` is ``None``.
        offset_arr: Optional ``[B]`` int64 Philox offsets; ``offset_val`` is
            used when ``None``.
        offset_val: Scalar offset fallback when ``offset_arr`` is ``None``.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` sampled ids and
        bool ``[B]`` validity flags on the same device as ``probs``.

    Raises:
        TypeError: If ``min_p`` is neither a tensor nor a scalar.
        ValueError: If ``probs`` is not 2-D or a tensor threshold has the
            wrong shape/device. CUDA/device errors from the registered op
            propagate unchanged.

    Note:
        Pure function: inputs are never mutated. Uses ``BLOCK_SIZE`` of
        ``min(4096, next_power_of_2(V))``, ``grid=(B,)``, ``num_warps=4``.
        Reference: torch min-p filtering plus a CDF draw, suitable for
        ``verify_against_reference``.
    """
    if probs.dim() != 2:
        raise ValueError(f"probs must have shape [B, V]; got {tuple(probs.shape)}")
    min_p_t = normalize_threshold(min_p, "min_p", probs.size(0), probs.device)
    return _min_p_sampling_op(probs, min_p_t, seed_arr, seed_val, offset_arr, offset_val)


@custom_op(
    namespace="ayaka",
    name="top_p_sampling_from_probs",
    reference=top_p_sampling_ref,
    fake_impl=sampling_fake,
    dispatch_key="CUDA",
)
def _top_p_sampling_op(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU nucleus sampling with a tensor-only mass target (registered op).

    Scalar targets are normalized to ``[B]`` by the public
    ``top_p_sampling_from_probs`` wrapper because ``torch.library`` schemas
    cannot express ``Tensor | float`` unions.

    Raises:
        TypeError: If ``probs``/``top_p`` fail dtype validation.
        ValueError: If ``probs`` is not a CUDA ``[B, V]`` tensor or a
            parameter shape/device mismatches the batch.
    """
    batch_size, vocab_size = validate_probs(probs)
    validate_param_tensor(top_p, "top_p", batch_size, probs.device)
    validate_opt_seed_offset(seed_arr, "seed_arr", batch_size, probs.device)
    validate_opt_seed_offset(offset_arr, "offset_arr", batch_size, probs.device)
    output, valid = prepare_sampling_outputs(probs)

    has_tensor = True
    top_p_val = 0.0
    top_p_ptr = top_p

    grid = (batch_size,)
    block_size = min(4096, triton.next_power_of_2(vocab_size))

    cast(Any, _top_p_sampling_kernel)[grid](
        probs,
        output,
        valid,
        top_p_ptr,
        seed_arr if seed_arr is not None else probs,
        offset_arr if offset_arr is not None else probs,
        vocab_size=vocab_size,
        probs_stride_b=probs.stride(0),
        top_p_val=top_p_val,
        seed_val=seed_val,
        offset_val=offset_val,
        HAS_TOP_P_TENSOR=has_tensor,
        HAS_SEED_TENSOR=seed_arr is not None,
        HAS_OFFSET_TENSOR=offset_arr is not None,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output, valid


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: torch.Tensor | float = 0.9,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw nucleus (top-p) samples, compatible with FlashInfer top-p.

    Args:
        probs: ``[B, V]`` float probabilities.
        top_p: Per-row ``[B]`` cumulative-mass target tensor or scalar.
        seed_arr: Optional ``[B]`` int64 Philox seeds; ``seed_val`` is used
            when ``None``.
        seed_val: Scalar seed fallback when ``seed_arr`` is ``None``.
        offset_arr: Optional ``[B]`` int64 Philox offsets; ``offset_val`` is
            used when ``None``.
        offset_val: Scalar offset fallback when ``offset_arr`` is ``None``.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` sampled ids and
        bool ``[B]`` validity flags on the same device as ``probs``.

    Raises:
        TypeError: If ``top_p`` is neither a tensor nor a scalar.
        ValueError: If ``probs`` is not 2-D or a tensor target has the
            wrong shape/device. CUDA/device errors from the registered op
            propagate unchanged.

    Note:
        Pure function: inputs are never mutated. Uses ``BLOCK_SIZE`` of
        ``min(4096, next_power_of_2(V))``, ``grid=(B,)``, ``num_warps=8``.
        Reference: torch nucleus filtering plus a CDF draw, suitable for
        ``verify_against_reference``.
    """
    if probs.dim() != 2:
        raise ValueError(f"probs must have shape [B, V]; got {tuple(probs.shape)}")
    top_p_t = normalize_threshold(top_p, "top_p", probs.size(0), probs.device)
    return _top_p_sampling_op(probs, top_p_t, seed_arr, seed_val, offset_arr, offset_val)
