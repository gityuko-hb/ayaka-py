"""Plain-torch oracles for the sampling kernels.

The two threshold references deliberately mirror the kernel bisection loops
iteration for iteration, and ``philox_u01_f32_ref`` reuses the public
``philox4x32_10`` from ``ayaka.sampling.rng`` so the Triton and torch streams
cannot drift apart unnoticed.
"""

from __future__ import annotations

import torch

from ayaka.kernel.triton.sampling._common import (
    normalize_threshold,
    resolve_seed_offset,
)
from ayaka.sampling.rng import counter_uniform_cols, philox4x32_10


def _lsr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-extend a right shift on an int64 tensor.

    Triton ``>>`` on an int64 lane behaves the same once masked, so this
    mirrors the kernel's logical shift for unsigned bit patterns.
    """
    return (x >> k) & ((1 << (64 - k)) - 1)


def philox_u01_f32_ref(seed: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Torch mirror of Triton ``philox_u01_f32`` (high 24 bits as float32).

    Args:
        seed: ``[B]`` int64 Philox keys, one stream per row.
        flat: ``[B]`` int64 flat stream addresses.

    Returns:
        Float32 ``[B]`` uniforms in ``[0, 1)`` with 24 bits of precision.

    Note:
        Exact for non-negative ``seed``/``flat``, which is the counter-based
        sampling contract (seeds derive via ``derive_seed``, offsets count
        steps); negative inputs may diverge in sign handling.
    """
    counter = flat >> 1
    odd = flat & 1
    r0, r1, r2, r3 = philox4x32_10(
        counter & 0xFFFFFFFF,
        _lsr(counter, 32),
        torch.zeros_like(counter),
        torch.zeros_like(counter),
        seed & 0xFFFFFFFF,
        _lsr(seed, 32),
    )
    v = torch.where(odd.bool(), (r2 << 32) | r3, (r0 << 32) | r1)
    return (_lsr(v, 29) & 0x00FFFFFF).to(torch.float32) / 16777216.0


def _cdf_first_match(weights: torch.Tensor, target: torch.Tensor, vocab: int) -> torch.Tensor:
    """First index with ``w > 0`` and ``cumsum(w) >= target``.

    Rows with no match yield ``vocab - 1``, mirroring the kernel fallback.
    """
    cdf = weights.cumsum(dim=-1)
    cand = (weights > 0) & (cdf >= target.unsqueeze(1))
    pos = cand.float().argmax(dim=-1)
    return torch.where(cand.any(dim=-1), pos, torch.full_like(pos, vocab - 1))


def sampling_from_probs_ref(
    probs: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch mirror of the two-pass CDF scan."""
    batch, vocab = probs.shape
    seed = resolve_seed_offset(probs, seed_arr, seed_val)
    offset = resolve_seed_offset(probs, offset_arr, offset_val)
    p = probs.to(torch.float32)
    u = philox_u01_f32_ref(seed, offset)
    target = u * p.sum(dim=-1)
    pos = (p.cumsum(dim=-1) < target.unsqueeze(1)).sum(dim=-1).clamp(max=vocab - 1)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=probs.device)


def min_p_sampling_ref(
    probs: torch.Tensor,
    min_p: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch mirror of the fused max/filter/scan."""
    batch, vocab = probs.shape
    min_p_t = normalize_threshold(min_p, "min_p", batch, probs.device)
    p = probs.to(torch.float32)
    cutoff = p.amax(dim=-1) * min_p_t
    filt = torch.where(p >= cutoff.unsqueeze(1), p, torch.zeros_like(p))
    seed = resolve_seed_offset(probs, seed_arr, seed_val)
    offset = resolve_seed_offset(probs, offset_arr, offset_val)
    u = philox_u01_f32_ref(seed, offset)
    pos = _cdf_first_match(filt, u * filt.sum(dim=-1), vocab)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=probs.device)


def top_p_sampling_ref(
    probs: torch.Tensor,
    top_p: torch.Tensor,
    seed_arr: torch.Tensor | None = None,
    seed_val: int = 42,
    offset_arr: torch.Tensor | None = None,
    offset_val: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-torch mirror of bisection threshold plus CDF scan.

    Replicates the 12 bisection iterations over ``[0, 1]`` and the filtered
    CDF scan with the same arithmetic, so kernel and reference agree up to
    fp32 summation order.
    """
    batch, vocab = probs.shape
    target_mass = normalize_threshold(top_p, "top_p", batch, probs.device)
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
    seed = resolve_seed_offset(probs, seed_arr, seed_val)
    offset = resolve_seed_offset(probs, offset_arr, offset_val)
    u = philox_u01_f32_ref(seed, offset)
    pos = _cdf_first_match(filt, u * filt.sum(dim=-1), vocab)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=probs.device)


def fused_sampling_ref(
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
    u = philox_u01_f32_ref(seed, offset)
    pos = _cdf_first_match(w, u * w.sum(dim=-1), vocab)
    return pos.to(torch.int32), torch.ones(batch, dtype=torch.bool, device=x.device)


def fused_gumbel_ref(
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


def row_logsumexp_ref(logits: torch.Tensor, row_gate: torch.Tensor | None) -> torch.Tensor:
    """Plain-torch reference with identical semantics (CPU-runnable).

    Gated rows are the ones with ``row_gate > 0``; the rest get ``0.0``.
    """
    n = logits.size(0)
    if row_gate is None:
        return torch.logsumexp(logits.to(torch.float32), dim=-1)
    gate = row_gate > 0
    lse = torch.zeros(n, dtype=torch.float32, device=logits.device)
    if bool(gate.any()):
        lse[gate] = torch.logsumexp(logits[gate].to(torch.float32), dim=-1)
    return lse
