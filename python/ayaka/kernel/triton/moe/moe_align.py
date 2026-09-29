"""MoE block alignment / expert histogram / token sorting (Triton port of ``moe_align_kernel.cu``).

Contract (identical to the CUDA extension)
------------------------------------------
``topk_ids`` holds expert ids in ``[-1, num_experts - 2]``.  Internally every id is shifted by
``+1`` so that ``-1`` (a token owned by another expert-parallel rank) falls into *bin 0*; therefore
``num_experts`` is the number of **bins** (= real experts + 1).  Bin ``b`` is padded to a multiple
of ``block_size``, bins are laid out back to back (exclusive prefix sum of the padded counts) and

* ``sorted_token_ids``   token slot ids (``0 .. numel-1``) grouped by bin, padded with ``numel``
                         (the padding value is only written when ``pad_sorted_token_ids`` is set),
* ``experts_ids[blk]``   ``bin_of(blk) - 1``  (``-1`` for the blocks that hold foreign tokens),
* ``num_tokens_post_pad`` total padded length,
* ``cumsum_buffer``      ``num_experts + 1`` int32 slots; on exit slot ``b`` holds
                         ``prefix[b] + count[b]`` and slot ``num_experts`` holds the padded total
                         (exactly the state the CUDA kernels leave behind).

Two code paths, same dispatch rule as CUDA (``numel < 1024 and num_experts <= 64`` -> small batch):

* small batch : ONE Triton program (fill, histogram, padded exclusive scan, expert ids and scatter).
  The rank of a token inside its bin follows the CUDA thread order (thread ``i % S`` first, then
  its loop iteration), so the output is bit-identical to ``moe_align_block_size_small_batch_
  expert_kernel`` including the order of the tokens inside each bin.
* general     : fill -> multi-program histogram -> scan -> expert-id search -> atomic scatter.
  Like the CUDA ``count_and_sort_expert_tokens_kernel`` the order of the tokens *inside* one bin is
  decided by ``atomicAdd`` arrival order and is therefore not deterministic; every consumer
  (grouped MoE GEMM) is insensitive to it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import cdiv, check, check_tensor, device_guard, next_pow2, same_device

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

SMALL_BATCH_MAX_NUMEL = 1024  # exclusive upper bound, as in the CUDA dispatcher
SMALL_BATCH_MAX_EXPERTS = 64


# ======================================================================================
# shared: expert histogram  (used by moe_align (SHIFT=1) and prepare_moe_input (SHIFT=0))
# ======================================================================================
@triton.jit
def expert_histogram_kernel(
    ids_ptr,
    counts_ptr,
    numel,
    num_bins,
    SHIFT: tl.constexpr,
    E_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """counts[b] += #{i : ids[i] + SHIFT == b}, b in [0, num_bins).  Out-of-range ids are ignored.

    ``E_PAD`` must be a power of two >= num_bins + 1: the last bin (E_PAD - 1) is a trash bin that
    receives masked / out-of-range lanes so ``tl.histogram`` never sees an out-of-range value.
    """
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < numel
    e = tl.load(ids_ptr + offs, mask=valid, other=0).to(tl.int32) + SHIFT
    ok = valid & (e >= 0) & (e < num_bins)
    e = tl.where(ok, e, E_PAD - 1)
    h = tl.histogram(e, E_PAD)
    bins = tl.arange(0, E_PAD)
    tl.atomic_add(counts_ptr + bins, h, mask=(bins < num_bins) & (h > 0), sem="relaxed")


# ======================================================================================
# general path kernels
# ======================================================================================
@triton.jit
def moe_align_fill_kernel(
    sorted_token_ids_ptr,
    cumsum_ptr,
    fill_value,
    max_num_tokens_padded,
    num_counters,
    PAD_SORTED: tl.constexpr,
    BLOCK: tl.constexpr,
    ZBLOCK: tl.constexpr,
):
    """(CUDA: block 1 of moe_align_block_size_kernel) fill ``sorted_token_ids`` with ``numel``
    and let program 0 clear the histogram counters that live in ``cumsum_buffer``."""
    pid = tl.program_id(0)
    if PAD_SORTED:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        vals = tl.zeros([BLOCK], dtype=tl.int32) + fill_value
        tl.store(sorted_token_ids_ptr + offs, vals, mask=offs < max_num_tokens_padded)
    if pid == 0:
        zoffs = tl.arange(0, ZBLOCK)
        for z in range(0, num_counters, ZBLOCK):
            idx = z + zoffs
            tl.store(cumsum_ptr + idx, tl.zeros([ZBLOCK], dtype=tl.int32), mask=idx < num_counters)


@triton.jit
def moe_align_block_size_kernel(
    cumsum_ptr,
    total_tokens_post_pad_ptr,
    num_experts,
    block_size,
    E_PAD: tl.constexpr,
):
    """(CUDA: block 0 of moe_align_block_size_kernel) padded counts -> exclusive prefix sum.

    Input : cumsum_ptr[0:num_experts] = per-bin token counts.
    Output: cumsum_ptr[0:num_experts] = exclusive prefix of the padded counts,
            cumsum_ptr[num_experts] = total_tokens_post_pad = *total_tokens_post_pad_ptr.
    The CUDA Blelloch / warp scan is replaced by ``tl.cumsum`` (Triton's associative scan lowers to
    a warp-shuffle + shared-memory scan); integer arithmetic makes the result exact either way.
    """
    bins = tl.arange(0, E_PAD)
    m = bins < num_experts
    counts = tl.load(cumsum_ptr + bins, mask=m, other=0)
    padded = ((counts + block_size - 1) // block_size) * block_size
    incl = tl.cumsum(padded, axis=0)
    excl = incl - padded
    total = tl.sum(padded, axis=0)
    tl.store(cumsum_ptr + bins, excl, mask=m)
    tl.store(cumsum_ptr + num_experts, total)
    tl.store(total_tokens_post_pad_ptr, total)


@triton.jit
def moe_align_expert_ids_kernel(
    cumsum_ptr,
    total_tokens_post_pad_ptr,
    expert_ids_ptr,
    num_experts,
    block_size,
    max_blocks,
    NUM_ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """expert_ids[blk] = upper_bound(prefix[0:num_experts], blk * block_size) - 2.

    Same binary search as the CUDA kernel, vectorised over ``BLOCK`` blocks per program.
    """
    pid = tl.program_id(0)
    total = tl.load(total_tokens_post_pad_ptr)
    num_blocks = total // block_size
    b = pid * BLOCK + tl.arange(0, BLOCK)
    valid = (b < num_blocks) & (b < max_blocks)
    block_start = b * block_size
    left = tl.zeros([BLOCK], dtype=tl.int32)
    right = tl.zeros([BLOCK], dtype=tl.int32) + num_experts
    for it in tl.static_range(NUM_ITERS):
        active = left < right
        mid = (left + right) >> 1
        pv = tl.load(cumsum_ptr + mid, mask=valid & active, other=0)
        go_right = pv <= block_start
        left = tl.where(active & go_right, mid + 1, left)
        right = tl.where(active & (pv > block_start), mid, right)
    tl.store(expert_ids_ptr + b, left - 2, mask=valid)


@triton.jit
def count_and_sort_expert_tokens_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    cumsum_ptr,
    numel,
    num_experts,
    BLOCK: tl.constexpr,
):
    """rank = atomicAdd(cumsum[bin], 1); sorted_token_ids[rank] = i   (bin = topk_ids[i] + 1)."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=valid, other=0).to(tl.int32) + 1
    ok = valid & (e >= 0) & (e < num_experts)
    rank = tl.atomic_add(cumsum_ptr + e, 1, mask=ok, sem="relaxed")
    tl.store(sorted_token_ids_ptr + rank, offs.to(tl.int32), mask=ok)


# ======================================================================================
# small-batch path: a single program does everything
# ======================================================================================
@triton.jit
def moe_align_block_size_small_batch_expert_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_pad_ptr,
    num_experts,
    block_size,
    numel,
    max_num_tokens_padded,
    max_blocks,
    stride_threads,
    iters_per_thread,
    PAD_SORTED: tl.constexpr,
    E_PAD: tl.constexpr,
    CHUNK: tl.constexpr,
    FILL_BLOCK: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """One-program port of ``moe_align_block_size_small_batch_expert_kernel``.

    CUDA gives token ``i`` to thread ``t = i % S`` (iteration ``j = i // S``, ``S = stride_threads``)
    and ranks tokens of one bin by (thread, iteration).  We enumerate tokens in exactly that order,
    ``p = t * J + j`` (``J = iters_per_thread``), and obtain the per-bin rank with a one-hot cumsum.
    """
    bins = tl.arange(0, E_PAD)
    p_local = tl.arange(0, CHUNK)
    total_p = stride_threads * iters_per_thread

    # ---- (1) sentinel fill (CUDA: the extra "fill" threads) -----------------------------
    if PAD_SORTED:
        foffs = tl.arange(0, FILL_BLOCK)
        for s in range(0, max_num_tokens_padded, FILL_BLOCK):
            o = s + foffs
            tl.store(
                sorted_token_ids_ptr + o,
                tl.zeros([FILL_BLOCK], dtype=tl.int32) + numel,
                mask=o < max_num_tokens_padded,
            )
        # the scatter below overwrites some of these slots: order the two sets of stores
        tl.debug_barrier()

    # ---- (2) per-bin token counts ---------------------------------------------------------
    counts = tl.zeros([E_PAD], dtype=tl.int32)
    for p0 in range(0, total_p, CHUNK):
        p = p0 + p_local
        t = p // iters_per_thread
        j = p % iters_per_thread
        i = j * stride_threads + t
        valid = (p < total_p) & (i < numel)
        e = tl.load(topk_ids_ptr + i, mask=valid, other=0).to(tl.int32) + 1
        ok = valid & (e >= 0) & (e < num_experts)
        oh = ((e[:, None] == bins[None, :]) & ok[:, None]).to(tl.int32)
        counts += tl.sum(oh, axis=0)

    # ---- (3) padded exclusive prefix sum -----------------------------------------------------
    padded = ((counts + block_size - 1) // block_size) * block_size
    incl = tl.cumsum(padded, axis=0)
    cum_start = incl - padded  # bins >= num_experts have padded == 0 -> cum_start == total
    total = tl.sum(padded, axis=0)
    tl.store(total_tokens_post_pad_ptr, total)

    # ---- (4) expert ids: expert_ids[blk] = (#bins with start <= blk*block_size) - 2 ------------
    num_blocks = total // block_size
    bl = tl.arange(0, BLOCK_B)
    for b0 in range(0, num_blocks, BLOCK_B):
        b = b0 + bl
        bv = (b < num_blocks) & (b < max_blocks)
        bstart = b * block_size
        le = tl.sum((cum_start[None, :] <= bstart[:, None]).to(tl.int32), axis=1)
        tl.store(expert_ids_ptr + b, le - 2, mask=bv)

    # ---- (5) scatter token ids to their padded slots -------------------------------------------
    run = tl.zeros([E_PAD], dtype=tl.int32)
    for p0 in range(0, total_p, CHUNK):
        p = p0 + p_local
        t = p // iters_per_thread
        j = p % iters_per_thread
        i = j * stride_threads + t
        valid = (p < total_p) & (i < numel)
        e = tl.load(topk_ids_ptr + i, mask=valid, other=0).to(tl.int32) + 1
        ok = valid & (e >= 0) & (e < num_experts)
        oh = ((e[:, None] == bins[None, :]) & ok[:, None]).to(tl.int32)
        csum = tl.cumsum(oh, axis=0)  # inclusive per-bin rank along the CUDA thread order
        own_incl = tl.sum(csum * oh, axis=1)
        own_run = tl.sum(oh * run[None, :], axis=1)
        own_base = tl.sum(oh * cum_start[None, :], axis=1)
        pos = own_base + own_run + own_incl - 1
        tl.store(sorted_token_ids_ptr + pos, i, mask=ok)
        run += tl.sum(oh, axis=0)


# ======================================================================================
# prepare_moe_input.cu :: compute_arg_sorts
# ======================================================================================
@triton.jit
def compute_arg_sorts_kernel(
    topk_ids_ptr,
    atomic_buffer_ptr,
    input_permutation_ptr,
    output_permutation_ptr,
    numel,
    topk,
    num_experts,
    BLOCK: tl.constexpr,
):
    """start = atomicAdd(cursor[e], 1); input_permutation[start] = i // topk; output_permutation[i] = start.

    ``atomic_buffer`` must hold the exclusive prefix sum of the per-expert counts on entry.
    """
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=valid, other=-1)
    ok = valid & (e >= 0) & (e < num_experts)
    start = tl.atomic_add(atomic_buffer_ptr + e, 1, mask=ok, sem="relaxed")
    tl.store(input_permutation_ptr + start, (offs // topk).to(tl.int32), mask=ok)
    tl.store(output_permutation_ptr + offs, start, mask=ok)


# ======================================================================================
# host wrappers
# ======================================================================================
_HIST_BLOCK = 2048
_SORT_BLOCK = 1024
_FILL_BLOCK = 1024
_EXPERT_ID_BLOCK = 128


def launch_expert_histogram(
    ids: torch.Tensor, counts: torch.Tensor, num_bins: int, shift: int
) -> None:
    numel = ids.numel()
    if numel == 0:
        return
    e_pad = max(2, next_pow2(num_bins + 1))
    expert_histogram_kernel[(cdiv(numel, _HIST_BLOCK),)](
        ids, counts, numel, num_bins, SHIFT=shift, E_PAD=e_pad, BLOCK=_HIST_BLOCK, num_warps=8
    )


def launch_compute_arg_sorts(
    topk_ids: torch.Tensor,
    atomic_buffer: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    topk: int,
    num_experts: int,
) -> None:
    numel = topk_ids.numel()
    if numel == 0:
        return
    compute_arg_sorts_kernel[(cdiv(numel, _SORT_BLOCK),)](
        topk_ids,
        atomic_buffer,
        input_permutation,
        output_permutation,
        numel,
        topk,
        num_experts,
        BLOCK=_SORT_BLOCK,
        num_warps=4,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    """Drop-in replacement for the CUDA ``moe_align_block_size`` op (all results written in place).

    Args mirror the C++ signature.  ``num_experts`` counts *bins*, i.e. real experts + 1 (see module
    docstring).  ``cumsum_buffer`` needs at least ``num_experts + 1`` int32 slots; it is only used
    (and only modified) on the general path, exactly like the CUDA implementation.
    """
    check_tensor(topk_ids, "topk_ids", dtypes=_INT_DTYPES, contiguous=True)
    check_tensor(
        sorted_token_ids, "sorted_token_ids", dtypes=(torch.int32,), ndim=1, contiguous=True
    )
    check_tensor(experts_ids, "experts_ids", dtypes=(torch.int32,), ndim=1, contiguous=True)
    check_tensor(num_tokens_post_pad, "num_tokens_post_pad", dtypes=(torch.int32,), contiguous=True)
    check_tensor(cumsum_buffer, "cumsum_buffer", dtypes=(torch.int32,), contiguous=True)
    same_device(topk_ids, sorted_token_ids, experts_ids, num_tokens_post_pad, cumsum_buffer)
    check(int(num_experts) > 0, "num_experts must be positive")
    check(int(block_size) > 0, "block_size must be positive")
    check(num_tokens_post_pad.numel() >= 1, "num_tokens_post_pad must hold at least one element")

    num_experts, block_size = int(num_experts), int(block_size)
    numel = topk_ids.numel()
    max_num_tokens_padded = sorted_token_ids.size(0)
    max_blocks = experts_ids.size(0)
    pad = bool(pad_sorted_token_ids)
    max_padded_upper_bound = numel + min(numel, num_experts) * (block_size - 1)
    max_blocks_upper_bound = cdiv(max_padded_upper_bound, block_size)
    check(
        max_num_tokens_padded >= max_padded_upper_bound,
        "sorted_token_ids must have capacity for the worst-case padded token count "
        f"({max_padded_upper_bound})",
    )
    check(
        max_blocks >= max_blocks_upper_bound,
        "experts_ids must have capacity for the worst-case padded block count "
        f"({max_blocks_upper_bound})",
    )

    with device_guard(topk_ids):
        if numel < SMALL_BATCH_MAX_NUMEL and num_experts <= SMALL_BATCH_MAX_EXPERTS:
            stride_threads = max(num_experts, 32)  # CUDA: number of "work" threads
            iters = max(cdiv(numel, stride_threads), 1)
            e_pad = max(2, next_pow2(num_experts))
            chunk = min(1024, max(16, 4096 // e_pad))
            moe_align_block_size_small_batch_expert_kernel[(1,)](
                topk_ids,
                sorted_token_ids,
                experts_ids,
                num_tokens_post_pad,
                num_experts,
                block_size,
                numel,
                max_num_tokens_padded,
                max_blocks,
                stride_threads,
                iters,
                PAD_SORTED=pad,
                E_PAD=e_pad,
                CHUNK=chunk,
                FILL_BLOCK=_FILL_BLOCK,
                BLOCK_B=64,
                num_warps=4,
            )
            return

        check(
            cumsum_buffer.numel() >= num_experts + 1,
            f"cumsum_buffer needs >= num_experts + 1 = {num_experts + 1} slots, got {cumsum_buffer.numel()}",
        )
        e_pad_scan = max(2, next_pow2(num_experts))
        n_fill = max(1, cdiv(max_num_tokens_padded, _FILL_BLOCK)) if pad else 1
        moe_align_fill_kernel[(n_fill,)](
            sorted_token_ids,
            cumsum_buffer,
            numel,
            max_num_tokens_padded,
            num_experts + 1,
            PAD_SORTED=pad,
            BLOCK=_FILL_BLOCK,
            ZBLOCK=1024,
            num_warps=4,
        )
        launch_expert_histogram(topk_ids, cumsum_buffer, num_experts, shift=1)
        moe_align_block_size_kernel[(1,)](
            cumsum_buffer,
            num_tokens_post_pad,
            num_experts,
            block_size,
            E_PAD=e_pad_scan,
            num_warps=4,
        )
        max_id_blocks = min(max_blocks, cdiv(max_num_tokens_padded, block_size))
        if max_id_blocks > 0:
            moe_align_expert_ids_kernel[(cdiv(max_id_blocks, _EXPERT_ID_BLOCK),)](
                cumsum_buffer,
                num_tokens_post_pad,
                experts_ids,
                num_experts,
                block_size,
                max_id_blocks,
                NUM_ITERS=num_experts.bit_length(),
                BLOCK=_EXPERT_ID_BLOCK,
                num_warps=4,
            )
        if numel > 0:
            count_and_sort_expert_tokens_kernel[(cdiv(numel, _SORT_BLOCK),)](
                topk_ids,
                sorted_token_ids,
                cumsum_buffer,
                numel,
                num_experts,
                BLOCK=_SORT_BLOCK,
                num_warps=4,
            )
