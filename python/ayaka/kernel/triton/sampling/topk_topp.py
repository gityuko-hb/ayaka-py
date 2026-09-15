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
from ayaka.kernel.triton.sampling.philox import philox_u01_f32
from ayaka.sampling.rng import philox4x32_10


def _lsr_ref(x: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-extend a right shift on an int64 tensor.

    Args:
        x: Int64 input tensor holding unsigned bit patterns.
        k: Shift distance with ``0 < k < 64``.

    Returns:
        ``x`` shifted right by ``k`` with zero fill (Triton ``>>`` on an
        int64 lane behaves the same once masked).
    """
    return (x >> k) & ((1 << (64 - k)) - 1)


def _philox_u01_f32_ref(seed: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Torch mirror of Triton ``philox_u01_f32`` (high 24 bits as float32).

    Args:
        seed: ``[B]`` int64 Philox keys, one stream per row.
        flat: ``[B]`` int64 flat stream addresses.

    Returns:
        Float32 ``[B]`` uniforms in ``[0, 1)`` with 24 bits of precision.

    Note:
        Exact for non-negative ``seed``/``flat``, which is the
        counter-based sampling contract (seeds derive via ``derive_seed``,
        offsets count steps); negative inputs may diverge in sign handling.
        Uses the public ``philox4x32_10`` from ``ayaka.sampling.rng`` so the
        two implementations cannot drift apart unnoticed.
    """
    counter = flat >> 1
    odd = flat & 1
    r0, r1, r2, r3 = philox4x32_10(
        counter & 0xFFFFFFFF,
        _lsr_ref(counter, 32),
        torch.zeros_like(counter),
        torch.zeros_like(counter),
        seed & 0xFFFFFFFF,
        _lsr_ref(seed, 32),
    )
    v = torch.where(odd.bool(), (r2 << 32) | r3, (r0 << 32) | r1)
    return (_lsr_ref(v, 29) & 0x00FFFFFF).to(torch.float32) / 16777216.0


def _validate_probs(probs: torch.Tensor, name: str = "probs") -> tuple[int, int]:
    """Validate a ``[B, V]`` CUDA float batch.

    Args:
        probs: Candidate probability/logit batch.
        name: Argument name used in error messages.

    Returns:
        Tuple ``(batch, vocab)`` of Python ints.

    Raises:
        TypeError: If ``probs`` is not a tensor or not floating-point.
        ValueError: If ``probs`` is not CUDA or not 2-D.
    """
    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not probs.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not probs.is_floating_point():
        raise TypeError(f"{name} must be a float dtype; got {probs.dtype}")
    if probs.dim() != 2:
        raise ValueError(f"{name} must have shape [B, V]; got {tuple(probs.shape)}")
    return probs.size(0), probs.size(1)


def _validate_param_tensor(
    tensor: torch.Tensor, name: str, batch: int, device: torch.device
) -> None:
    """Validate a ``[B]`` per-row parameter tensor on the batch device.

    Args:
        tensor: Candidate per-row parameter (thresholds, seeds, offsets).
        name: Argument name used in error messages.
        batch: Expected leading dimension.
        device: Expected device (the batch device).

    Raises:
        TypeError: If ``tensor`` is not a tensor.
        ValueError: If the shape is not ``[B]`` or the device mismatches.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 1 or tensor.size(0) != batch:
        raise ValueError(f"{name} must have shape [B] matching the batch")
    if tensor.device != device:
        raise ValueError(f"{name} and probs must be on the same device")


def _validate_opt_seed_offset(
    arr: torch.Tensor | None, name: str, batch: int, device: torch.device
) -> None:
    """Validate an optional ``[B]`` int64 seed/offset tensor.

    Args:
        arr: Seed/offset tensor, or ``None`` for the scalar fallback.
        name: Argument name used in error messages.
        batch: Expected leading dimension.
        device: Expected device (the batch device).

    Raises:
        TypeError: If ``arr`` is a tensor with dtype other than int64.
        ValueError: If the shape is not ``[B]`` or the device mismatches.
    """
    if arr is None:
        return
    _validate_param_tensor(arr, name, batch, device)
    if arr.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64; got {arr.dtype}")


def _resolve_seed_offset(
    probs: torch.Tensor,
    arr: torch.Tensor | None,
    val: int,
) -> torch.Tensor:
    """Materialize a ``[B]`` int64 seed/offset stream.

    Args:
        probs: ``[B, V]`` batch providing the batch size and device.
        arr: Per-row tensor stream, or ``None`` to broadcast ``val``.
        val: Scalar fallback used when ``arr`` is ``None``.

    Returns:
        Int64 ``[B]`` tensor on the batch device.
    """
    if arr is not None:
        return arr
    return torch.full((probs.size(0),), val, dtype=torch.int64, device=probs.device)


def _sampling_fake(
    probs: torch.Tensor, *args: object, **kwargs: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta kernel shared by all sampling ops in this module.

    Args:
        probs: ``[B, V]`` batch providing the batch size and device.
        *args: Ignored positional launcher arguments.
        **kwargs: Ignored launcher keyword arguments.

    Returns:
        Tuple ``(token_ids, valid)`` with uninitialized int32 ``[B]`` and
        bool ``[B]`` tensors, for shape inference under ``torch.compile``.
    """
    batch = probs.shape[0]
    return (
        torch.empty(batch, dtype=torch.int32, device=probs.device),
        torch.empty(batch, dtype=torch.bool, device=probs.device),
    )


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
    u = philox_u01_f32(seed, offset.to(tl.int64))

    # Pass A: max + min (bisection domain for top-k). OOB loads are -inf so
    # they are NOT miscounted in count-bisection (mid can be negative) and
    # exp() maps them to 0; x_min spans finite values only (the bisection
    # domain must stay finite).
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

    tl.store(output_ptr + row_idx, sampled_idx)
    tl.store(valid_ptr + row_idx, True)


def _fused_sampling_ref(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch mirror of the fused threshold + CDF-scan.

    Replicates every kernel pass (top-k count bisection, min-p logit
    threshold, top-p bisection over the top-k-renormalized mass, survivor
    sum, CDF scan) with the same iteration counts, so kernel and reference
    agree up to fp32 summation order and libdevice-vs-torch transcendental
    rounding.

    Args:
        logits: ``[B, V]`` float logits.
        top_k: ``[B]`` integer top-k limits.
        top_p: ``[B]`` float nucleus targets.
        min_p: ``[B]`` float relative thresholds.
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 base offsets.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` ids and bool ``[B]``
        flags on the same device as ``logits``.
    """
    x = logits.to(torch.float32)
    batch, vocab = x.shape
    neg_inf = torch.full((batch,), float("-inf"), dtype=torch.float32, device=x.device)
    x_max = x.amax(dim=-1)
    x_min = torch.where(torch.isfinite(x), x, torch.full_like(x, float("inf"))).amin(dim=-1)

    has_top_k = bool(((top_k > 0) & (top_k < vocab)).any())
    if has_top_k:
        active = (top_k > 0) & (top_k < vocab)
        lo = x_min.clone()
        hi = x_max.clone()
        for _ in range(16):
            mid = (lo + hi) * 0.5
            cnt = (x >= mid.unsqueeze(1)).sum(dim=-1)
            go_lo = active & (cnt >= top_k)
            go_hi = active & ~(cnt >= top_k)
            lo = torch.where(go_lo, mid, lo)
            hi = torch.where(go_hi, mid, hi)
        t_k = torch.where(active, lo, neg_inf)
    else:
        t_k = neg_inf

    has_min_p = bool((min_p > 0.0).any())
    if has_min_p:
        t_minp = torch.where(
            min_p.to(torch.float32) > 0.0,
            x_max + torch.log(min_p.to(torch.float32)),
            neg_inf,
        )
    else:
        t_minp = neg_inf

    has_top_p = bool((top_p < 1.0).any())
    if has_top_p:
        active_p = top_p.to(torch.float32) < 1.0
        mass_p = top_p.to(torch.float32)
        centered = x - x_max.unsqueeze(1)
        s_k = torch.where(
            x >= t_k.unsqueeze(1), torch.exp(centered), torch.zeros_like(centered)
        ).sum(dim=-1)
        lo = torch.where(t_k == float("-inf"), x_min, t_k)
        hi = x_max.clone()
        for _ in range(12):
            mid = (lo + hi) * 0.5
            partial = torch.where(
                x >= mid.unsqueeze(1), torch.exp(centered), torch.zeros_like(centered)
            ).sum(dim=-1)
            go_lo = active_p & (partial >= mass_p * s_k)
            go_hi = active_p & ~(partial >= mass_p * s_k)
            lo = torch.where(go_lo, mid, lo)
            hi = torch.where(go_hi, mid, hi)
        t_p = torch.where(active_p, lo, neg_inf)
    else:
        t_p = neg_inf

    t_final = torch.maximum(torch.maximum(t_k, t_p), t_minp)
    centered = x - x_max.unsqueeze(1)
    w = torch.where(x >= t_final.unsqueeze(1), torch.exp(centered), torch.zeros_like(centered))
    u = _philox_u01_f32_ref(seed, offset)
    pos = _cdf_first_match(w, u * w.sum(dim=-1), vocab)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=x.device)


@custom_op(
    namespace="ayaka",
    reference=_fused_sampling_ref,
    fake_impl=_sampling_fake,
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
        state, hence CUDA-graph safe. Uses ``BLOCK_SIZE`` of
        ``min(4096, next_power_of_2(V))``, ``grid=(B,)``, and ``num_warps=4``.
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
        _validate_param_tensor(tensor, tensor_name, batch_size, logits.device)
    if seed.dtype != torch.int64:
        raise TypeError(f"seed must have dtype torch.int64; got {seed.dtype}")
    if offset.dtype != torch.int64:
        raise TypeError(f"offset must have dtype torch.int64; got {offset.dtype}")
    output = torch.empty((batch_size,), dtype=torch.int32, device=logits.device)
    valid = torch.empty((batch_size,), dtype=torch.bool, device=logits.device)

    grid = (batch_size,)
    block_size = min(4096, triton.next_power_of_2(vocab_size))

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
        HAS_TOP_K=bool(((top_k > 0) & (top_k < vocab_size)).any()),
        HAS_TOP_P=bool((top_p < 1.0).any()),
        HAS_MIN_P=bool((min_p > 0.0).any()),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output, valid


def _prepare_sampling_outputs(probs: torch.Tensor):
    """Allocate fresh ``(output, valid)`` buffers for a probs batch.

    Args:
        probs: ``[batch, vocab]`` tensor providing device and batch size.

    Returns:
        Tuple ``(output, valid)`` with int32 ``[batch]`` indices and bool
        ``[batch]`` flags on the same device as ``probs``.
    """
    batch_size = probs.shape[0]
    output = torch.empty((batch_size,), device=probs.device, dtype=torch.int32)
    valid = torch.empty((batch_size,), device=probs.device, dtype=torch.bool)
    return output, valid


def _sampling_from_probs_ref(
    probs: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch mirror of the two-pass CDF scan.

    Args:
        probs: ``[B, V]`` float probabilities.
        seed_arr: Optional ``[B]`` int64 seeds; ``seed_val`` is used when
            ``None``.
        seed_val: Scalar seed fallback.
        offset_arr: Optional ``[B]`` int64 offsets; ``offset_val`` is used
            when ``None``.
        offset_val: Scalar offset fallback.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` ids and bool ``[B]``
        flags on the same device as ``probs``.
    """
    batch, vocab = probs.shape
    seed = _resolve_seed_offset(probs, seed_arr, seed_val)
    offset = _resolve_seed_offset(probs, offset_arr, offset_val)
    p = probs.to(torch.float32)
    u = _philox_u01_f32_ref(seed, offset)
    target = u * p.sum(dim=-1)
    pos = (p.cumsum(dim=-1) < target.unsqueeze(1)).sum(dim=-1).clamp(max=vocab - 1)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=probs.device)


@custom_op(
    namespace="ayaka",
    reference=_sampling_from_probs_ref,
    fake_impl=_sampling_fake,
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
    batch_size, vocab_size = _validate_probs(probs)
    _validate_opt_seed_offset(seed_arr, "seed_arr", batch_size, probs.device)
    _validate_opt_seed_offset(offset_arr, "offset_arr", batch_size, probs.device)
    output, valid = _prepare_sampling_outputs(probs)

    grid = (batch_size,)
    block_size = min(4096, triton.next_power_of_2(vocab_size))

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
        num_warps=4,
    )
    return output, valid


def _normalize_threshold(
    value: torch.Tensor | float, name: str, batch: int, device: torch.device
) -> torch.Tensor:
    """Broadcast a scalar threshold to ``[B]`` float32.

    Args:
        value: Per-row ``[B]`` tensor (validated, cast to float32) or a
            Python scalar broadcast to the batch.
        name: Argument name used in error messages.
        batch: Expected leading dimension.
        device: Expected device.

    Returns:
        Float32 ``[B]`` tensor on ``device``.

    Raises:
        TypeError: If a tensor ``value`` fails validation.
        ValueError: If a tensor ``value`` has the wrong shape/device.
    """
    if isinstance(value, torch.Tensor):
        _validate_param_tensor(value, name, batch, device)
        return value.to(torch.float32)
    return torch.full((batch,), float(value), dtype=torch.float32, device=device)


def _cdf_first_match(weights: torch.Tensor, target: torch.Tensor, vocab: int) -> torch.Tensor:
    """First index with ``w > 0`` and ``cumsum(w) >= target``.

    Args:
        weights: ``[B, V]`` non-negative filtered weights.
        target: ``[B]`` CDF thresholds (already scaled by the valid mass).
        vocab: Vocabulary size, used as the no-match fallback.

    Returns:
        Int64 ``[B]`` positions; rows with no match yield ``vocab - 1``,
        mirroring the kernel fallback.
    """
    cdf = weights.cumsum(dim=-1)
    cand = (weights > 0) & (cdf >= target.unsqueeze(1))
    pos = cand.float().argmax(dim=-1)
    return torch.where(cand.any(dim=-1), pos, torch.full_like(pos, vocab - 1))


def _min_p_sampling_ref(
    probs: torch.Tensor,
    min_p: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch mirror of the fused max/filter/scan.

    Args:
        probs: ``[B, V]`` float probabilities.
        min_p: ``[B]`` threshold tensor (scalars are normalized by the
            caller before reaching the registered op).
        seed_arr: Optional ``[B]`` int64 seeds; ``seed_val`` is used when
            ``None``.
        seed_val: Scalar seed fallback.
        offset_arr: Optional ``[B]`` int64 offsets; ``offset_val`` is used
            when ``None``.
        offset_val: Scalar offset fallback.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` ids and bool ``[B]``
        flags on the same device as ``probs``.
    """
    batch, vocab = probs.shape
    min_p_t = _normalize_threshold(min_p, "min_p", batch, probs.device)
    p = probs.to(torch.float32)
    cutoff = p.amax(dim=-1) * min_p_t
    filt = torch.where(p >= cutoff.unsqueeze(1), p, torch.zeros_like(p))
    seed = _resolve_seed_offset(probs, seed_arr, seed_val)
    offset = _resolve_seed_offset(probs, offset_arr, offset_val)
    u = _philox_u01_f32_ref(seed, offset)
    pos = _cdf_first_match(filt, u * filt.sum(dim=-1), vocab)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=probs.device)


@custom_op(
    namespace="ayaka",
    name="min_p_sampling_from_probs",
    reference=_min_p_sampling_ref,
    fake_impl=_sampling_fake,
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
    batch_size, vocab_size = _validate_probs(probs)
    _validate_param_tensor(min_p, "min_p", batch_size, probs.device)
    _validate_opt_seed_offset(seed_arr, "seed_arr", batch_size, probs.device)
    _validate_opt_seed_offset(offset_arr, "offset_arr", batch_size, probs.device)
    output, valid = _prepare_sampling_outputs(probs)

    has_tensor = True
    min_p_val = 0.0
    min_p_ptr = min_p

    grid = (batch_size,)
    block_size = min(4096, triton.next_power_of_2(vocab_size))

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
        num_warps=4,
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
    min_p_t = _normalize_threshold(min_p, "min_p", probs.size(0), probs.device)
    return _min_p_sampling_op(probs, min_p_t, seed_arr, seed_val, offset_arr, offset_val)


def _top_p_sampling_ref(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch mirror of bisection threshold plus CDF scan.

    Replicates the 12 bisection iterations over ``[0, 1]`` and the
    filtered CDF scan with the same arithmetic, so kernel and reference
    agree up to fp32 summation order.

    Args:
        probs: ``[B, V]`` float probabilities.
        top_p: ``[B]`` mass-target tensor (scalars are normalized by the
            caller before reaching the registered op).
        seed_arr: Optional ``[B]`` int64 seeds; ``seed_val`` is used when
            ``None``.
        seed_val: Scalar seed fallback.
        offset_arr: Optional ``[B]`` int64 offsets; ``offset_val`` is used
            when ``None``.
        offset_val: Scalar offset fallback.

    Returns:
        Tuple ``(token_ids, valid)`` with int32 ``[B]`` ids and bool ``[B]``
        flags on the same device as ``probs``.
    """
    batch, vocab = probs.shape
    target_mass = _normalize_threshold(top_p, "top_p", batch, probs.device)
    p = probs.to(torch.float32)
    low = torch.zeros(batch, dtype=torch.float32, device=probs.device)
    high = torch.ones(batch, dtype=torch.float32, device=probs.device)
    for _ in range(12):
        mid = (low + high) * 0.5
        mass = torch.where(p >= mid.unsqueeze(1), p, torch.zeros_like(p)).sum(dim=-1)
        take_low = mass >= target_mass
        low = torch.where(take_low, mid, low)
        high = torch.where(take_low, high, mid)
    filt = torch.where(p >= low.unsqueeze(1), p, torch.zeros_like(p))
    seed = _resolve_seed_offset(probs, seed_arr, seed_val)
    offset = _resolve_seed_offset(probs, offset_arr, offset_val)
    u = _philox_u01_f32_ref(seed, offset)
    pos = _cdf_first_match(filt, u * filt.sum(dim=-1), vocab)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=probs.device)


@custom_op(
    namespace="ayaka",
    name="top_p_sampling_from_probs",
    reference=_top_p_sampling_ref,
    fake_impl=_sampling_fake,
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
    batch_size, vocab_size = _validate_probs(probs)
    _validate_param_tensor(top_p, "top_p", batch_size, probs.device)
    _validate_opt_seed_offset(seed_arr, "seed_arr", batch_size, probs.device)
    _validate_opt_seed_offset(offset_arr, "offset_arr", batch_size, probs.device)
    output, valid = _prepare_sampling_outputs(probs)

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
    top_p_t = _normalize_threshold(top_p, "top_p", probs.size(0), probs.device)
    return _top_p_sampling_op(probs, top_p_t, seed_arr, seed_val, offset_arr, offset_val)
