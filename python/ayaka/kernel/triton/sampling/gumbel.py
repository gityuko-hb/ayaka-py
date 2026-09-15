"""Seeded Gumbel-argmax kernel over pre-filtered top-k candidates.

Generates Gumbel noise only for the first ``TOPK_BOUND`` positions after
``filter_probs`` has sorted and filtered them, never for the full ``V``.
Inputs ``(sp, si)`` are the ``filter_probs`` outputs in descending order.
Filtering stays on the torch side on purpose: it is multi-pass, already
correct, and covered by ``topk_topp`` tests, so this kernel implements only
the new part (noise plus argmax) and reuses the filter via dispatch instead
of reimplementing it.

Follows the direct-``triton`` import convention of
``ayaka.kernel.triton.paged_attention``: ``triton`` lives in the ``cuda``
extra, so this module imports only with Triton present while the caller in
``ayaka.sampling.ops.sampling`` catches ``ImportError`` and falls back to the
torch oracle.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton.sampling.philox import philox_u01
from ayaka.sampling.rng import counter_uniform_cols

HAS_TRITON_GUMBEL = True


@triton.jit
def _gumbel_argmax_kernel(
    sp_ptr,
    si_ptr,
    out_ptr,
    seed_ptr,
    offset_ptr,
    stride_row,
    TOPK_BOUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gumbel-argmax over one filtered row per program.

    Args:
        sp_ptr: Pointer to ``[B, TOPK_BOUND]`` filtered probabilities.
        si_ptr: Pointer to ``[B, TOPK_BOUND]`` token ids aligned with
            ``sp_ptr``.
        out_ptr: Pointer to ``[B]`` int64 sampled token ids.
        seed_ptr: Pointer to ``[B]`` per-row Philox seeds.
        offset_ptr: Pointer to ``[B]`` per-row base offsets.
        stride_row: Row stride (in elements) of ``sp_ptr``/``si_ptr``.
        TOPK_BOUND: Number of valid candidates per row.
        BLOCK: Tile width covering ``TOPK_BOUND``.

    Note:
        The per-column address is ``flat = offset_base * TOPK_BOUND + col``,
        matching ``counter_uniform_cols`` linear addressing. This keeps the
        Gumbel stream disjoint from one-dimensional ``counter_uniform``
        calls. Uniforms are clamped to ``[1e-12, 1 - 1e-7]`` before
        ``-log(-log(u))``; masked lanes score ``-inf`` so ``argmax`` only
        selects valid candidates. A filtered-out position with ``sp == 0``
        maps to ``log(sp) == -inf`` and can never win regardless of noise,
        preserving the "never pick a filtered token" oracle invariant.
    """
    row = tl.program_id(0)
    seed = tl.load(seed_ptr + row)
    offset_base = tl.load(offset_ptr + row)

    col = tl.arange(0, BLOCK)
    mask = col < TOPK_BOUND
    sp = tl.load(sp_ptr + row * stride_row + col, mask=mask, other=0.0)

    logp = tl.where(sp > 0, tl.log(sp), float("-inf"))

    # Dedicated address space for gumbel, mirroring counter_uniform_cols in
    # gumbel.py (never offset+col directly, to avoid colliding with other
    # one-dimensional counter_uniform call sites).
    offset = offset_base * TOPK_BOUND + col
    u = philox_u01(seed, offset)
    u = tl.minimum(tl.maximum(u, 1e-12), 1.0 - 1e-7)
    gumbel = -tl.log(-tl.log(u))

    score = tl.where(mask, logp + gumbel.to(tl.float32), float("-inf"))
    best_local = tl.argmax(score, axis=0)
    winner_id = tl.load(si_ptr + row * stride_row + best_local)
    tl.store(out_ptr + row, winner_id)


def _fused_gumbel_ref(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    topk_bound: int = 512,
) -> torch.Tensor:
    """Plain-torch reference mirroring the Triton truncation and RNG stream."""
    from ayaka.sampling.ops.sampling import filter_probs

    sp, si = filter_probs(logits, top_k, top_p, min_p)
    bound = min(topk_bound, sp.size(1))
    sp = sp[:, :bound]
    si = si[:, :bound]
    u = counter_uniform_cols(seed, offset, bound)
    u = u.clamp(min=1e-12, max=1.0 - 1e-7)
    gumbel = (-torch.log(-torch.log(u))).to(sp.dtype)
    log_sp = torch.where(
        sp > 0,
        torch.log(sp.clamp_min(torch.finfo(sp.dtype).tiny)),
        torch.full_like(sp, float("-inf")),
    )
    winner = torch.argmax(log_sp + gumbel, dim=-1)
    return si.gather(1, winner.unsqueeze(1)).squeeze(1)


def _fused_gumbel_fake(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    topk_bound: int = 512,
) -> torch.Tensor:
    """Meta kernel: fresh int64 ``[B]`` without touching memory."""
    return torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)


@custom_op(
    namespace="ayaka",
    reference=_fused_gumbel_ref,
    fake_impl=_fused_gumbel_fake,
    dispatch_key="CUDA",
)
def fused_gumbel_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    *,
    topk_bound: int = 512,
) -> torch.Tensor:
    """Sample token ids with fused filter plus Gumbel-argmax.

    Args:
        logits: ``[B, V]`` unnormalized logits.
        top_k: ``[B]`` per-row top-k limits applied by ``filter_probs``.
        top_p: ``[B]`` per-row nucleus thresholds applied by
            ``filter_probs``.
        min_p: ``[B]`` per-row relative thresholds applied by
            ``filter_probs``.
        seed: ``[B]`` int64 Philox seeds, one stream per row.
        offset: ``[B]`` int64 base offsets, batch-invariant across rows.
        topk_bound: Maximum candidates kept per row by ``filter_probs``.
            Must cover the largest effective ``top_k`` in the batch;
            candidates beyond it are silently truncated.

    Returns:
        Fresh int64 ``[B]`` tensor of sampled token ids on the same device
        as ``logits``.

    Note:
        Pure function: ``logits`` and filter tensors are never mutated.
        Deterministic in ``(seed, offset)`` with no generator state, hence
        CUDA-graph safe. Reference: ``filter_probs`` followed by a torch
        Gumbel-argmax, suitable for ``verify_against_reference``.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be a float dtype; got {logits.dtype}")
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [B, V]; got {tuple(logits.shape)}")
    batch = logits.size(0)
    for tensor, tensor_name in (
        (top_k, "top_k"),
        (top_p, "top_p"),
        (min_p, "min_p"),
        (seed, "seed"),
        (offset, "offset"),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{tensor_name} must be a torch.Tensor")
        if tensor.dim() != 1 or tensor.size(0) != batch:
            raise ValueError(f"{tensor_name} must have shape [B] matching logits batch")
        if tensor.device != logits.device:
            raise ValueError(f"{tensor_name} and logits must be on the same device")
    if not isinstance(topk_bound, int) or topk_bound <= 0:
        raise ValueError("topk_bound must be a positive integer")
    from ayaka.sampling.ops.sampling import filter_probs

    sp, si = filter_probs(logits, top_k, top_p, min_p)
    b, v = sp.shape
    block = triton.next_power_of_2(min(topk_bound, v))
    out = torch.empty(b, dtype=torch.int64, device=logits.device)

    cast(Any, _gumbel_argmax_kernel)[(b,)](
        sp,
        si,
        out,
        seed,
        offset,
        sp.stride(0),
        TOPK_BOUND=min(topk_bound, v),
        BLOCK=block,
    )
    return out
