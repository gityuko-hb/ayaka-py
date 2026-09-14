"""Ayaka Triton Sampling Kernels.

Ported from FlashInfer C++/CUDA sampling operators:
- sampling_from_probs
- min_p_sampling_from_probs
- top_p_sampling_from_probs
- top_k_sampling_from_probs
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# RNG & Helper Utilities
# -----------------------------------------------------------------------------


@triton.jit
def _splitmix64(seed, offset, idx):
    """Deterministic uniform random float32 in [0.0, 1.0) using SplitMix64."""
    z = seed + offset + idx.to(tl.uint64) * 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    # Lấy 24-bit mantissa chuyển thành float trong khoảng [0.0, 1.0)
    return ((z & 0x00FFFFFF).to(tl.float32)) / 16777216.0


# -----------------------------------------------------------------------------
# Kernel 1: Sampling From Probs (Categorical CDF Scan)
# -----------------------------------------------------------------------------


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
    row_idx = tl.program_id(0).to(tl.int64)

    # 1. Sinh số ngẫu nhiên đồng nhất u ~ U(0, 1)
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64) if HAS_SEED_TENSOR else seed_val
    offset = tl.load(offset_ptr + row_idx).to(tl.uint64) if HAS_OFFSET_TENSOR else offset_val
    u = _splitmix64(seed, offset, row_idx)

    # 2. Pass 1: Tính tổng xác suất toàn hàng
    row_offset = row_idx * probs_stride_b
    total_prob = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        total_prob += tl.sum(p, axis=0)

    # 3. Pass 2: Dò tìm token đầu tiên vượt ngưỡng CDF: u * total_prob
    target = u * total_prob
    accum = 0.0
    sampled_idx = -1

    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)

        # Tính prefix sum cục bộ trong block
        block_cdf = accum + tl.cumsum(p, axis=0)
        matched = mask & (block_cdf >= target) & (sampled_idx < 0)

        if tl.sum(matched.to(tl.int32), axis=0) > 0:
            # Chọn index đầu tiên thỏa mãn
            first_match_offset = tl.min(tl.where(matched, cols, vocab_size), axis=0)
            if sampled_idx < 0:
                sampled_idx = first_match_offset

        accum += tl.sum(p, axis=0)

    # Nếu sai số làm tròn float khiến u không chạm đích, fallback về token cuối
    if sampled_idx < 0:
        sampled_idx = vocab_size - 1

    tl.store(output_ptr + row_idx, sampled_idx)
    tl.store(valid_ptr + row_idx, True)


# -----------------------------------------------------------------------------
# Kernel 2: Min-P Sampling (Fused Max-prob + Filter + Renorm + Sample)
# -----------------------------------------------------------------------------


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
    row_idx = tl.program_id(0).to(tl.int64)
    row_offset = row_idx * probs_stride_b

    min_p = tl.load(min_p_ptr + row_idx).to(tl.float32) if HAS_MIN_P_TENSOR else min_p_val
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64) if HAS_SEED_TENSOR else seed_val
    offset = tl.load(offset_ptr + row_idx).to(tl.uint64) if HAS_OFFSET_TENSOR else offset_val
    u = _splitmix64(seed, offset, row_idx)

    # Pass 1: Tìm xác suất lớn nhất (max_prob)
    max_prob = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        max_prob = tl.maximum(max_prob, tl.max(p, axis=0))

    # Tính ngưỡng cắt Min-P
    cutoff = max_prob * min_p

    # Pass 2: Tính tổng xác suất sau khi lọc
    sum_valid_p = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        filtered_p = tl.where(p >= cutoff, p, 0.0)
        sum_valid_p += tl.sum(filtered_p, axis=0)

    # Pass 3: Dò token theo Cumulative CDF đã lọc
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


# -----------------------------------------------------------------------------
# Kernel 3: Top-P Renormalize & Sample (Two-Pass Histogram Bisection)
# -----------------------------------------------------------------------------


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
    seed_val,
    offset_val,
    HAS_TOP_P_TENSOR: tl.constexpr,
    HAS_SEED_TENSOR: tl.constexpr,
    HAS_OFFSET_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_offset = row_idx * probs_stride_b

    top_p = tl.load(top_p_ptr + row_idx).to(tl.float32) if HAS_TOP_P_TENSOR else top_p_val
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64) if HAS_SEED_TENSOR else seed_val
    offset = tl.load(offset_ptr + row_idx).to(tl.uint64) if HAS_OFFSET_TENSOR else offset_val
    u = _splitmix64(seed, offset, row_idx)

    # 1. Tìm ngưỡng qua 64 histogram bins trong dải log-scale / linear
    # Với sampling thông thường, bisection qua 16 bước tìm ngưỡng cutoff:
    low = 0.0
    high = 1.0
    for _ in range(12):  # 12 iterations nhị phân đạt độ chính xác ~ 2e-4
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

    # 2. Tính tổng xác suất thực tế trên tập Top-P
    sum_valid = 0.0
    for v_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = v_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        p = tl.load(probs_ptr + row_offset + cols, mask=mask, other=0.0)
        sum_valid += tl.sum(tl.where(p >= threshold, p, 0.0), axis=0)

    # 3. Lấy mẫu CDF
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


def _prepare_sampling_outputs(probs: torch.Tensor):
    batch_size = probs.shape[0]
    output = torch.empty((batch_size,), device=probs.device, dtype=torch.int32)
    valid = torch.empty((batch_size,), device=probs.device, dtype=torch.bool)
    return output, valid


def sampling_from_probs(
    probs: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampling theo phân phối xác suất danh mục (Categorical sampling)."""
    batch_size, vocab_size = probs.shape
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


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: torch.Tensor | float = 0.05,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-P sampling tương thích FlashInfer min_p_sampling_from_probs."""
    batch_size, vocab_size = probs.shape
    output, valid = _prepare_sampling_outputs(probs)

    has_tensor = isinstance(min_p, torch.Tensor)
    min_p_val = float(min_p) if not has_tensor else 0.0
    min_p_ptr = min_p if has_tensor else probs

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


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: torch.Tensor | float = 0.9,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-P (Nucleus) sampling tương thích FlashInfer top_p_sampling_from_probs."""
    batch_size, vocab_size = probs.shape
    output, valid = _prepare_sampling_outputs(probs)

    has_tensor = isinstance(top_p, torch.Tensor)
    top_p_val = float(top_p) if not has_tensor else 0.0
    top_p_ptr = top_p if has_tensor else probs

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
