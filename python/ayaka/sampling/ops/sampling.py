"""ayaka/sampling/ops — mask → temperature → filter/sample → report.

Merged five legacy files (bitmask.py, topk_topp.py, gumbel.py, mirostat.py,
support_capture.py) into ONE module: the second half of the sampling
pipeline, running after bias/penalties/DRY (see ops/penalties.py, the
matching merged module) have finished adjusting the logits:

    ... -> penalties -> DRY -> mask -> temperature -> filter/sample -> report

The five sections below follow exactly that order:

    BITMASK          — mask: grammar-constrained decoding (allow bitmask).
    TOPK_TOPP        — main filter/sample path: top_k/top_p/min_p +
                       multinomial (filter_probs/softmax_stats are the
                       shared foundation used by the next two sections).
    GUMBEL           — alternative filter/sample path: Gumbel-max over the
                       SAME TOPK_TOPP filter_probs (not a new filter, only
                       a different final sampling step).
    MIROSTAT         — another alternative filter/sample path: directly
                       controls perplexity, does NOT use filter_probs.
    SUPPORT_CAPTURE  — report: packs the distribution ALREADY used for
                       sampling, runs AFTER sampling (consumes no RNG,
                       changes no token).

No code was lost relative to the five original files — only the shared
imports (torch, enum, dataclasses, counter_uniform/counter_uniform_cols,
trace_sampler) and the three optional-dependency try/excepts (xgrammar,
flashinfer, the Triton kernels) were hoisted to the top of the file, and
one now-redundant internal cross-import was dropped because it lives in
the same module:

  * old gumbel.py: `from ayaka.sampling.ops.topk_topp import filter_probs`
    (filter_probs is now a same-module function, defined in section
    TOPK_TOPP, placed BEFORE section GUMBEL in this file).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from ayaka.sampling.rng import counter_uniform, counter_uniform_cols
from ayaka.sampling.trace import trace_sampler

NEG_INF = float("-inf")

try:  # pragma: no cover - environment dependent
    import xgrammar as _xgr  # type: ignore[import-not-found]

    _HAS_XGRAMMAR = True
except Exception:  # pragma: no cover
    _xgr = None
    _HAS_XGRAMMAR = False

try:  # pragma: no cover
    import flashinfer.sampling as _fi  # type: ignore[import-not-found]

    _HAS_FLASHINFER = True
except Exception:  # pragma: no cover
    _fi = None
    _HAS_FLASHINFER = False

try:  # pragma: no cover
    from ayaka.kernel.triton.sampling.topk_topp import (
        fused_topk_topp_minp_sampling_from_logits as _triton_fused_sample,
    )
    from ayaka.kernel.triton.sampling.topk_topp import (
        sampling_from_probs as _triton_cdf_sample,
    )

    _HAS_TRITON_SAMPLING = True
except Exception:  # pragma: no cover
    _triton_fused_sample = None
    _triton_cdf_sample = None
    _HAS_TRITON_SAMPLING = False

try:  # pragma: no cover
    from ayaka.kernel.triton.sampling.gumbel import HAS_TRITON_GUMBEL, fused_gumbel_sample
except Exception:  # pragma: no cover
    HAS_TRITON_GUMBEL = False
    fused_gumbel_sample = None

# Fast-path thresholds ported from FlashInfer `_top_k_first_fast_path`: only
# wins when vocab is large enough (avoids full-vocab work) and k is small
# enough (cheap survivors).
_TOPK_FIRST_MAX_K = 256
_TOPK_FIRST_MIN_VOCAB = 65536

__all__ = [
    "NEG_INF",
    "has_xgrammar_kernel",
    "rows_fully_masked",
    "apply_allow_bitmask_",
    "bitmask_to_allowed_ids",
    "has_flashinfer",
    "has_triton_sampling",
    "topk_first_hint",
    "topk_topp_sample",
    "filter_probs",
    "filter_logits",
    "softmax_stats",
    "softmax_stats_scaled",
    "gumbel_sample",
    "mirostat_v2_step",
    "mirostat_v2_init",
    "SamplingSupportStatus",
    "SamplingSupportTensors",
    "build_support_output",
    "greedy_support_output",
]


# ======================================================================
# BITMASK  (was: ops/bitmask.py)
# ======================================================================

# has_xgrammar_kernel / rows_fully_masked / apply_allow_bitmask_ /
# bitmask_to_allowed_ids — grammar-constrained decoding (allow bitmask).
# (The original bitmask.py had no module docstring — this comment restates
# exactly the per-function docstrings below, adding no new information.)
#
# `row_indices` is a compact-row gather: mask row k constrains logits row
# row_indices[k]. It turns 64 grammar streams into ONE apply instead of
# 64 sequential applies — exactly what rtp-llm cannot do because it hard
# codes batch_size==1.
#
# `_apply_reference_` is the pure-torch reference path — it must always
# exist and must always be tested IN PARALLEL with the kernel (xgrammar)
# path — it is the oracle. Two independent sampling backends with no shared
# oracle is how bugs hide (see: TensorRT-LLM has a TRT backend and a
# PyTorch backend doing the same job).


def has_xgrammar_kernel() -> bool:
    """Return whether the xgrammar kernel backend is available.

    Returns:
        True when the optional ``xgrammar`` import succeeded, False
        otherwise.
    """
    return _HAS_XGRAMMAR


def rows_fully_masked(mask: torch.Tensor) -> torch.Tensor:
    """Return which mask rows forbid every token.

    Args:
        mask: ``[n_rows, W]`` packed allow-bitmask (bit=1 allows).

    Returns:
        ``[n_rows]`` bool; True marks a fully-masked row. Costs
        ``O(n_rows x W)``, not ``O(n_rows x V)``.
    """
    return ~(mask != 0).any(dim=1)


def apply_allow_bitmask_(
    logits: torch.Tensor,
    mask: torch.Tensor,
    row_indices: torch.Tensor,
    vocab_size: int,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Apply an allow-bitmask to logits in place.

    A set bit keeps the logit, a cleared bit becomes ``NEG_INF``.

    ``row_indices`` is a compact-row gather: mask row ``k`` constrains
    logits row ``row_indices[k]``. It turns 64 grammar streams into ONE
    apply instead of 64 sequential applies — exactly what rtp-llm cannot
    do because it hard-codes ``batch_size==1``.

    Args:
        logits: ``[B, V]`` logits, modified in place.
        mask: ``[n_rows, W]`` packed allow-bitmask.
        row_indices: ``[n_rows]`` int64 mapping mask rows to logits rows.
        vocab_size: Vocabulary size used to decode the packed bits.
        validate: When True, reject fully-masked rows before applying.

    Returns:
        The same ``logits`` tensor, modified in place.

    Raises:
        ValueError: If ``logits`` is not 2-D, if ``mask`` is not
            ``[n_rows, W]`` matching ``row_indices``, or if a mask row
            allows no token.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D, got {logits.dim()}")
    if mask.dim() != 2 or mask.size(0) != row_indices.size(0):
        raise ValueError("mask must be [n_rows, W] and match row_indices")
    if validate:
        bad = rows_fully_masked(mask)
        if bool(bad.any()):
            idx = int(torch.nonzero(bad)[0])
            raise ValueError(
                f"mask row {idx} allows no token. The producer must fail closed "
                "to EOS; leaving it would yield NaN after softmax."
            )

    if _HAS_XGRAMMAR and _xgr is not None and logits.is_cuda:  # pragma: no cover
        _xgr.apply_token_bitmask_inplace(
            logits, mask, vocab_size=vocab_size, indices=row_indices.tolist()
        )
        return logits

    return _apply_reference_(logits, mask, row_indices, vocab_size)


def _apply_reference_(
    logits: torch.Tensor,
    mask: torch.Tensor,
    row_indices: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Pure-torch reference path for bitmask application.

    This path must always exist and must always be tested IN PARALLEL with
    the kernel path — it is the oracle. Two independent sampling backends
    with no shared oracle is how bugs hide (see: TensorRT-LLM has a TRT
    backend and a PyTorch backend doing the same job).

    Args:
        logits: ``[B, V]`` logits, modified in place.
        mask: ``[n_rows, W]`` packed allow-bitmask.
        row_indices: ``[n_rows]`` int64 mapping mask rows to logits rows.
        vocab_size: Vocabulary size used to decode the packed bits.

    Returns:
        The same ``logits`` tensor, modified in place.
    """
    device = logits.device
    tok = torch.arange(vocab_size, device=device)
    word_idx = torch.div(tok, 32, rounding_mode="floor")  # [V]
    bit_idx = (tok % 32).to(torch.int32)  # [V]
    sel = mask.index_select(1, word_idx)  # [n_rows, V]
    allowed = ((sel >> bit_idx) & 1).to(torch.bool)  # [n_rows, V]

    rows = row_indices.to(torch.long)
    target = logits.index_select(0, rows)
    target = torch.where(allowed, target, torch.full_like(target, NEG_INF))
    logits.index_copy_(0, rows, target)
    return logits


def bitmask_to_allowed_ids(mask_row: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Unpack one mask row into a list of allowed token ids.

    Used by tests and by the sparse path (not yet wired into the sampler —
    see the README).

    Args:
        mask_row: ``[W]`` packed allow-bitmask for a single row.
        vocab_size: Vocabulary size bounding the decoded token ids.

    Returns:
        1-D int64 tensor of allowed token ids.
    """
    tok = torch.arange(vocab_size, device=mask_row.device)
    word = mask_row.index_select(0, torch.div(tok, 32, rounding_mode="floor"))
    bit = (tok % 32).to(torch.int32)
    return tok[((word >> bit) & 1).to(torch.bool)]


# ======================================================================
# TOPK_TOPP  (was: ops/topk_topp.py)
# ======================================================================

# Top-k / top-p / min-p + multinomial.
#
# The RNG IS COUNTER-BASED, NOT torch.Generator. This is a design decision,
# not a detail:
#
#   * torch.Generator carries hidden state, so the result of row i depends
#     on how many other rows share the batch and how many random numbers
#     they consume. Rejection sampling consumes a variable count, so it is
#     not batch-invariant.
#   * A Generator cannot be captured into a CUDA graph.
#
# Philox4x32-10 (ayaka.sampling.rng) gives each (seed, offset) an
# independent, pure-function, reproducible stream, and (seed, offset) are
# tensors so FlashInfer can graph-capture them — which is why FlashInfer
# requires tensor-form seed/offset instead of ints.
#
# UPDATE: the tensor RNG is Philox4x32-10 from ayaka.sampling.rng (the
# canonical source, with a Triton twin in
# ayaka.kernel.triton.sampling.philox); SplitMix64 remains only for
# host-side derive_seed.


def has_flashinfer() -> bool:
    """Return whether the FlashInfer sampling backend is available.

    Returns:
        True when the optional ``flashinfer.sampling`` import succeeded.
    """
    return _HAS_FLASHINFER


def has_triton_sampling() -> bool:
    """Return whether the Triton top-k/top-p sampling kernels are available.

    Returns:
        True when the optional Triton sampling kernel imports succeeded.
    """
    return _HAS_TRITON_SAMPLING


def topk_first_hint(top_k: torch.Tensor, vocab_size: int) -> bool:
    """Report whether the top_k_first fast path can trigger.

    Host-side check with ZERO device sync.

    Called from the sampler with staging columns (CPU tensors).

    Args:
        top_k: ``[B]`` per-row top-k values; must hold one uniform scalar
            across the whole batch.
        vocab_size: Vocabulary size of the current run.

    Returns:
        True when ``top_k`` is a uniform scalar in ``(1, 256]`` and the
        vocabulary is large enough.
    """
    if top_k.numel() == 0 or vocab_size < _TOPK_FIRST_MIN_VOCAB:
        return False
    k0 = int(top_k[0].item())
    if not 1 < k0 <= _TOPK_FIRST_MAX_K:
        return False
    return bool(torch.all(top_k == k0).item())


def _topk_first_fast_path(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """FlashInfer `_top_k_first_fast_path` ported to torch + Triton.

    Precondition (the caller must check via :func:`topk_first_hint`):
    uniform scalar ``top_k`` with ``k`` in ``(1, 256]`` and
    ``vocab >= 65536``. Runs ``torch.topk`` on the logits, then the WHOLE
    filter (top-k renorm → top-p → min-p) on ``[B, k]`` — then a local CDF
    sample mapped back to token ids through the gathered indices.

    Args:
        logits: ``[B, V]`` logits.
        top_k: ``[B]`` per-row top-k; only element 0 is read.
        top_p: ``[B]`` per-row nucleus thresholds.
        min_p: ``[B]`` per-row min-p thresholds.
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 Philox offsets.

    Returns:
        ``[B]`` int32 sampled token ids.
    """
    k = int(top_k[0].item())
    values, gathered = torch.topk(logits.to(torch.float32), k=k, dim=-1, sorted=True)
    sp = torch.softmax(values, dim=-1)  # renorm over the k survivors = top-k renorm

    cum = sp.cumsum(dim=-1)
    drop = (cum - sp) > top_p.unsqueeze(1)
    drop &= (top_p < 1.0).unsqueeze(1)  # top_p == 1 must not drop anything (fp cumsum)
    drop[:, 0] = False  # min_tokens_to_keep=1, matches HF
    sp = sp.masked_fill(drop, 0.0)

    if bool((min_p > 0.0).any()):
        sp = sp.masked_fill(sp < min_p.unsqueeze(1) * sp[:, :1], 0.0)

    # caller must hold the _HAS_TRITON_SAMPLING guard
    assert _triton_cdf_sample is not None
    local, _ = _triton_cdf_sample(sp, seed_arr=seed, offset_arr=offset)
    return gathered.gather(1, local.to(torch.long).unsqueeze(1)).squeeze(1).to(torch.int32)


def topk_topp_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    *,
    force_reference: bool = False,
    allow_topk_first: bool = False,
) -> torch.Tensor:
    """Sample one token per row through the 5-level dispatch ladder.

    GPU backends come first, the torch oracle last:

      1. FlashInfer fused `top_k_top_p_sampling_from_logits` — unsorted
         rejection sampling, the fastest; does NOT support min_p.
      2. Triton top_k_first fast path for host-proven scalar-k plus a large
         vocab — torch.topk + filter on ``[B, k]`` + local CDF sample
         (covers min_p).
      3. Triton fused `top_k_top_p_min_p_sampling_from_logits` — every
         filter combo in ONE kernel, no sort, no prob materialization.
      4. FlashInfer renorm chain + `min_p_sampling_from_probs` (tensor
         seed/offset) — only when Triton is missing.
      5. Torch oracle `_reference_sample` — CPU, force_reference, or no
         backend available.

    FlashInfer 0.6 fused has NO min_p: taking it for a row with min_p > 0
    would silently swallow the min_p filter — a wrong distribution with no
    error. The ladder above guarantees min_p is never swallowed.

    RNG streams: one draw per (seed, offset) per row per step; Triton uses
    24-bit f32 u01, the oracle uses 53-bit f64, FlashInfer uses its internal
    philox. The three paths are NOT bit-exact with each other (they only
    match in distribution — chi-square).

    Args:
        logits: ``[B, V]`` logits.
        top_k: ``[B]`` per-row top-k limits (``<= 0`` disables).
        top_p: ``[B]`` per-row nucleus thresholds.
        min_p: ``[B]`` per-row min-p thresholds.
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 Philox offsets.
        force_reference: When True, skip all GPU backends and use the
            torch oracle.
        allow_topk_first: When True, allow level 2 (the caller must still
            gate on :func:`topk_first_hint`).

    Returns:
        ``[B]`` sampled token ids.
    """
    any_min_p = bool((min_p > 0.0).any())
    if logits.is_cuda and not force_reference:
        if _HAS_FLASHINFER and _fi is not None and not any_min_p:
            # FlashInfer: unsorted rejection sampling, many rounds fused into
            # ONE kernel. seed/offset must be int64 tensors to CUDA-graph.
            trace_sampler("backend_choice", backend="flashinfer")
            sampled: torch.Tensor = _fi.top_k_top_p_sampling_from_logits(
                logits,
                top_k,
                top_p,
                filter_apply_order="top_k_first",
                seed=seed,
                offset=offset,
            )
            return sampled
        if _HAS_TRITON_SAMPLING and _triton_fused_sample is not None:
            if allow_topk_first:
                trace_sampler("backend_choice", backend="triton_topk_first")
                return _topk_first_fast_path(logits, top_k, top_p, min_p, seed, offset)
            trace_sampler("backend_choice", backend="triton_fused")
            return _triton_fused_sample(logits, top_k, top_p, min_p, seed, offset)[0]
        if _HAS_FLASHINFER and _fi is not None:
            # SGLang renorm chain: top_k_renorm -> top_p_renorm -> min_p,
            # tensor seed/offset for determinism/graph capture.
            trace_sampler("backend_choice", backend="flashinfer_min_p")
            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            if bool((top_k > 0).any()):
                probs = _fi.top_k_renorm_probs(probs, top_k)
            if bool((top_p < 1.0).any()):
                probs = _fi.top_p_renorm_probs(probs, top_p)
            return _fi.min_p_sampling_from_probs(probs, min_p, seed=seed, offset=offset)
    trace_sampler("backend_choice", backend="reference")
    return _reference_sample(logits, top_k, top_p, min_p, seed, offset)


def filter_probs(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter logits into normalized sorted probs plus sorted indices.

    Splitting filter from sample is deliberate: filtering is the
    DETERMINISTIC part, comparable bit-exact against HuggingFace; sampling
    is the random part, comparable only statistically. Folding both into one
    function would leave no strict oracle for the former.

    Follows the exact HUGGINGFACE order: top_k → RENORM → top_p → min_p.

    The middle RENORM step MUST NOT be dropped. HF implements each warper
    as a separate LogitsProcessor writing -inf into the scores, and the
    next warper calls softmax(scores) AGAIN FROM SCRATCH, so probs are
    renormalized over the post-top_k survivors before top_p cuts. Dropping
    the renorm runs the top_p cumsum over a sum < 1, so the nucleus cuts
    wider than HF (keeps more tokens than it should). min_p is renorm
    invariant (both p_i and p_max scale together), so no second renorm is
    needed.

    Args:
        logits: ``[B, V]`` logits.
        top_k: ``[B]`` per-row top-k limits (``<= 0`` disables).
        top_p: ``[B]`` per-row nucleus thresholds.
        min_p: ``[B]`` per-row min-p thresholds.

    Returns:
        Tuple ``(sorted_probs, sorted_indices)``; ``sorted_probs`` is
        normalized with filtered positions set to 0.
    """
    v = logits.size(1)
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)

    if bool((top_k > 0).any()):
        rank = torch.arange(v, device=logits.device).unsqueeze(0)
        k = top_k.to(torch.long).unsqueeze(1)
        sp = sp.masked_fill((k > 0) & (rank >= k), 0.0)
        sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(sp.dtype).tiny)

    if bool((top_p < 1.0).any()):
        cum = sp.cumsum(dim=-1)
        drop = (cum - sp) > top_p.unsqueeze(1)
        # FIXED BUG (caught by the HF oracle, 2026-08-20):
        # `(top_p < 1.0).any()` tests the WHOLE BATCH, so this branch runs
        # for EVERY row — including rows with top_p == 1.0 meaning "filter
        # nothing". With top_p == 1.0, fp32 cumsum error makes (cum - sp) at
        # the tail come out as 1.0000001 > 1.0 and spuriously drops a few
        # trailing tokens.
        #   Symptom: a top_p=1.0 row silently lost ~9/256 of its smallest
        #   tokens. It only shows with a NON-uniform batch (another row has
        #   top_p<1) — a uniform batch never enters this branch, so the old
        #   tests always stayed green.
        drop &= (top_p < 1.0).unsqueeze(1)
        drop[:, 0] = False  # min_tokens_to_keep=1, matches HF
        sp = sp.masked_fill(drop, 0.0)

    if bool((min_p > 0.0).any()):
        sp = sp.masked_fill(sp < min_p.unsqueeze(1) * sp[:, :1], 0.0)

    return sp, si


def filter_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """Unsorted ``[B, V]`` view of :func:`filter_probs` — oracle use only.

    Args:
        logits: ``[B, V]`` logits.
        top_k: ``[B]`` per-row top-k limits (``<= 0`` disables).
        top_p: ``[B]`` per-row nucleus thresholds.
        min_p: ``[B]`` per-row min-p thresholds.

    Returns:
        ``[B, V]`` filtered probs scattered back to token-id order.
    """
    sp, si = filter_probs(logits, top_k, top_p, min_p)
    out = torch.zeros_like(sp)
    out.scatter_(1, si, sp)
    return out


def _reference_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Sort-based reference path. Slow, but exactly correct — it is the oracle.

    Verifies the FlashInfer path via chi-square.

    Args:
        logits: ``[B, V]`` logits.
        top_k: ``[B]`` per-row top-k limits (``<= 0`` disables).
        top_p: ``[B]`` per-row nucleus thresholds.
        min_p: ``[B]`` per-row min-p thresholds.
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 Philox offsets.

    Returns:
        ``[B]`` sampled token ids.
    """
    v = logits.size(1)
    sp, si = filter_probs(logits, top_k, top_p, min_p)

    total = sp.sum(dim=-1, keepdim=True)
    degenerate = (total <= 0).squeeze(1)
    sp = sp / total.clamp_min(torch.finfo(sp.dtype).tiny)

    u = counter_uniform(seed, offset).to(sp.dtype).unsqueeze(1)
    pos = (sp.cumsum(dim=-1) < u).sum(dim=-1).clamp_(max=v - 1)
    tok = si.gather(1, pos.unsqueeze(1)).squeeze(1)

    if bool(degenerate.any()):
        # Only happens when every token was filtered out — an upper-layer
        # contract violation. Fall back to argmax instead of silently
        # returning NaN.
        tok = torch.where(degenerate, logits.argmax(dim=-1), tok)
    return tok


def softmax_stats(
    logits: torch.Tensor, temperature: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Divide by temperature, then compute stats — legacy reference contract.

    Args:
        logits: ``[B, V]`` logits.
        temperature: ``[B]`` per-row temperatures.

    Returns:
        Tuple ``(max, sum_exp, entropy)``; see :func:`softmax_stats_scaled`.
    """
    scaled = logits.to(torch.float32) / temperature.clamp_min(1e-6).unsqueeze(1)
    return softmax_stats_scaled(scaled)


def softmax_stats_scaled(
    scaled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stats over ALREADY temperature-scaled logits — the canonical caller form.

    Returns (max, sum_exp, entropy) in ONE pass.

    These three numbers are a deliberate extension point: min-p, dynamic
    min-p, typical sampling, mirostat v2, and entropy-based sampling are ALL
    pure functions of them. Computing them here means every future strategy
    costs ~0 extra and needs NO host round-trip to keep state — exactly
    where rtp-llm would break if it wanted mirostat, because all of its
    state lives on the host.

    Args:
        scaled: ``[B, V]`` temperature-scaled logits (float32).

    Returns:
        Tuple ``(max, sum_exp, entropy)``, each ``[B]``.
    """
    x = scaled.to(torch.float32)
    m = x.max(dim=-1, keepdim=True).values
    d = x - m
    finite = torch.isfinite(d)
    e = torch.where(finite, torch.exp(d), torch.zeros_like(d))
    s = e.sum(dim=-1, keepdim=True)
    ent = torch.log(s) - (e * torch.where(finite, d, torch.zeros_like(d))).sum(
        dim=-1, keepdim=True
    ) / s.clamp_min(torch.finfo(torch.float32).tiny)
    return m.squeeze(1), s.squeeze(1), ent.squeeze(1)


# ======================================================================
# GUMBEL  (was: ops/gumbel.py)
# ======================================================================

# seeded_gumbel — sampling via the Gumbel-max trick instead of inverse-CDF.
#
# Gumbel-max: argmax_i(log p_i + g_i) with independent g_i ~ Gumbel(0,1)
# yields exactly the softmax(logit) distribution. Since argmax is invariant
# to adding a constant, and log p_i = x_i - LSE(x) (x already temperature
# scaled), NORMALIZING p_i into a true distribution (dividing by the sum)
# is not required for the sampling decision — only the relative order
# matters. Differs from `_reference_sample` in the TOPK_TOPP section above in
# exactly one place: the final sampling step (cumsum+threshold → noise+argmax).
# The filter_probs chain (top_k→renorm→top_p→min_p) is KEPT AS IS and reused
# verbatim — this is not a new filter, only a different pick within the
# filtered set.
#
# WHY a dedicated kernel is worth it instead of reusing FlashInfer's
# rejection sampler:
#   - Rejection sampling takes a variable number of rounds, which is harder
#     to CUDA-graph. Gumbel-max has FIXED cost per row (one argmax over a
#     known-size set).
#   - The Philox4x32-10 at hand is a PURE function of the address, so it
#     generates independent (row, candidate) noise statelessly with perfect
#     parallelism — a fit for gumbel-max (which needs MANY random numbers
#     per row), unlike inverse-CDF with a single number per row.
#   - No FlashInfer dependency for the core sampling primitive (consistent
#     with the "zero external deps" already applied to
#     FlashAttention/Marlin).
#
# MOST IMPORTANT PERF DECISION (for the Triton build, see
# ayaka/kernel/triton/sampling/gumbel.py): generate noise ONLY for the
# post-filter survivors (bounded by the maximum allowed top_k, usually a
# few hundred), NOT for the whole V (~150K) — the difference can exceed
# 1000x in RNG draws. The oracle below deliberately generates noise for ALL
# of V (simpler; not a hot path, in the spirit of "the reference may be
# slow").


def _reference_gumbel_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """CPU/torch oracle. Same shape as the topk_topp `_reference_sample`.

    Shares filter_probs, only swaps the final sampling step.

    Normalizing sp into a true distribution is NOT needed here (unlike
    `_reference_sample`): unnormalized sp (summing to < 1 after filtering)
    still argmaxes correctly, because the constant -log(sum) added evenly
    to every candidate in the row does not change the order.

    Args:
        logits: ``[B, V]`` logits.
        top_k: ``[B]`` per-row top-k limits (``<= 0`` disables).
        top_p: ``[B]`` per-row nucleus thresholds.
        min_p: ``[B]`` per-row min-p thresholds.
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 Philox offsets.

    Returns:
        ``[B]`` sampled token ids.
    """
    v = logits.size(1)
    sp, si = filter_probs(logits, top_k, top_p, min_p)

    degenerate = sp.sum(dim=-1) <= 0

    u = counter_uniform_cols(seed, offset, v).to(sp.dtype)
    u = u.clamp(min=torch.finfo(sp.dtype).tiny, max=1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(u))

    log_sp = torch.where(
        sp > 0,
        torch.log(sp.clamp_min(torch.finfo(sp.dtype).tiny)),
        torch.full_like(sp, float("-inf")),
    )
    winner = torch.argmax(log_sp + gumbel, dim=-1)
    tok = si.gather(1, winner.unsqueeze(1)).squeeze(1)

    if bool(degenerate.any()):
        tok = torch.where(degenerate, logits.argmax(dim=-1), tok)
    return tok


def gumbel_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    *,
    force_reference: bool = False,
) -> torch.Tensor:
    """Dispatch: Triton kernel when present, torch oracle otherwise.

    Same dispatch pattern as :func:`topk_topp_sample` (see
    ``ayaka/kernel/triton/sampling/gumbel.py`` for the kernel side).

    Args:
        logits: ``[B, V]`` logits.
        top_k: ``[B]`` per-row top-k limits (``<= 0`` disables).
        top_p: ``[B]`` per-row nucleus thresholds.
        min_p: ``[B]`` per-row min-p thresholds.
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 Philox offsets.
        force_reference: When True, skip the Triton kernel and use the
            torch oracle.

    Returns:
        ``[B]`` sampled token ids.
    """
    if (
        HAS_TRITON_GUMBEL
        and fused_gumbel_sample is not None
        and logits.is_cuda
        and not force_reference
    ):  # pragma: no cover
        result: torch.Tensor = fused_gumbel_sample(logits, top_k, top_p, min_p, seed, offset)
        return result
    return _reference_gumbel_sample(logits, top_k, top_p, min_p, seed, offset)


# ======================================================================
# MIROSTAT  (was: ops/mirostat.py)
# ======================================================================

# Mirostat v2 — directly controls perplexity instead of static top_k/top_p.
#
# Ported from `ayaka-sampling` (retired Rust crate, had Mirostat v2 +
# choose_gumbel). NOT yet re-checked against the original Rust — if the mu
# update formula or the surprisal unit (bits vs nats) differs, the Rust
# version is the standard, not this file.
#
# Principle (Basu et al., "Mirostat: A Neural Text Decoding Algorithm that
# Directly Controls Perplexity"):
#   1. Truncate the distribution at the first position whose surprisal
#      (-log2 p) exceeds mu.
#   2. Sample within the truncated set (renormalize, reusing
#      counter_uniform).
#   3. Measure the TRUE surprisal of the chosen token, update mu from the
#      error against tau.
#
# Does NOT use filter_probs (top_k/top_p/min_p) — Mirostat decides
# truncation itself from per-step surprisal, not from a fixed rank or
# cumulative-prob cutoff. The two filtering mechanisms are NOT combined in
# one call (pick one).
#
# The `mu` state MUST persist per slot — like PenaltyState, not a value
# recomputed from scratch each step. The caller (core.plan or equivalent)
# is responsible for carrying the new mu across steps, the same way
# PenaltyState is carried by the caller through record()/move()/reset().


def mirostat_v2_step(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    tau: torch.Tensor,
    mu: torch.Tensor,
    eta: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one Mirostat v2 step and return the updated control state.

    Truncates by surprisal, samples inside the truncated set, then adapts
    ``mu`` from the observed surprisal error against ``tau``.

    Args:
        logits: ``[B, V]`` logits.
        temperature: ``[B]`` per-row temperatures.
        tau: ``[B]`` target surprisal in bits (typically 3.0-5.0, i.e. a
            target perplexity of ~2**tau).
        mu: ``[B]`` current control state; the caller must persist the
            returned value per slot, like ``PenaltyState``.
        eta: ``[B]`` learning rate for the ``mu`` update (typically ~0.1).
        seed: ``[B]`` int64 Philox seeds.
        offset: ``[B]`` int64 Philox offsets.

    Returns:
        Tuple ``(token_ids, mu_new)`` with ``[B]`` sampled token ids and
        the ``[B]`` control state for the next step.
    """
    v = logits.size(1)
    x = logits.to(torch.float32) / temperature.clamp_min(1e-6).unsqueeze(1)
    sorted_x, sorted_idx = x.sort(dim=-1, descending=True)

    m = sorted_x[:, :1]
    e = torch.exp(sorted_x - m)
    cum_e = e.cumsum(dim=-1)
    z = cum_e[:, -1:].clamp_min(torch.finfo(torch.float32).tiny)
    p = e / z  # true distribution, sorted descending by rank
    surprisal_bits = -torch.log2(p.clamp_min(torch.finfo(p.dtype).tiny))

    # Cut position: the first token with surprisal > mu is dropped, ALWAYS
    # keeping >= 1 token (even when the top-1 token already exceeds mu — the
    # cut must never go empty).
    keep = surprisal_bits <= mu.unsqueeze(1)
    keep[:, 0] = True
    k = keep.to(torch.int64).cumprod(dim=-1).sum(dim=-1, keepdim=True).clamp_(min=1)
    # cumprod instead of .sum(keep) directly: guarantees the cut at the FIRST
    # violation, not a count of scattered satisfying tokens (surprisal is not
    # guaranteed perfectly monotonic due to fp rounding on the very flat tail).

    rank = torch.arange(v, device=logits.device).unsqueeze(0)
    trunc_p = torch.where(rank < k, p, torch.zeros_like(p))
    trunc_p = trunc_p / trunc_p.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)

    u = counter_uniform(seed, offset).to(trunc_p.dtype).unsqueeze(1)
    pos = (trunc_p.cumsum(dim=-1) < u).sum(dim=-1).clamp_(max=v - 1)
    tok = sorted_idx.gather(1, pos.unsqueeze(1)).squeeze(1)

    observed_surprisal = surprisal_bits.gather(1, pos.unsqueeze(1)).squeeze(1)
    mu_new = mu - eta * (observed_surprisal - tau)
    return tok, mu_new


def mirostat_v2_init(tau: torch.Tensor) -> torch.Tensor:
    """Return the initial mu state ``2 * tau``, per the original paper.

    Call once when a request starts; the caller persists the result like
    ``PenaltyState``.

    Args:
        tau: ``[B]`` target surprisal in bits.

    Returns:
        ``[B]`` initial ``mu`` state.
    """
    return 2.0 * tau


# ======================================================================
# SUPPORT_CAPTURE  (was: ops/support_capture.py)
# ======================================================================

# Sampling support capture — reporting the distribution ALREADY used for
# sampling.
#
# Mirrors SGLang's ``_build_sampling_mask_output``: for opt-in rows, packs
# the positive support of the post-filter distribution into a fixed
# device-resident result:
#
#   * ``token_ids`` — top-k token ids by weight (packed, sorted desc, capped
#     by ``sampling_support_max_tokens``);
#   * ``lengths`` — actual support token counts, clamped to the packed size;
#   * ``selected_logprobs`` — ``log(w_sampled / support_mass)`` — the exact
#     ``LogprobMode.SAMPLING`` semantics (renorm over the surviving set; an
#     off-support token has weight 0 → INVALID status);
#   * ``statuses`` — OK / INVALID / OVERFLOW.
#
# Capture is pure REPORTING: computed AFTER sampling, consumes no RNG,
# changes no token. weights is the filtered distribution in token-id space
# (per-row sum ≤ 1) — ``selected_logprobs`` applies one final renorm through
# ``support_mass``.
#
# Path-agnostic: weights are always recomputed with the torch filter (HF
# oracle) on exactly the temperature-scaled logits the sampler used — the
# backends differ only in near-threshold tie semantics, so the described
# action space stays correct (same "correctness invariant" note SGLang uses
# for its FlashInfer path).


class SamplingSupportStatus(enum.IntEnum):
    """Status of one captured support row (mirrors SamplingMaskStatus).

    Attributes:
        OK: The sampled token is inside the support and nothing overflowed.
        INVALID: The sampled token has zero weight or the logprob is not
            finite (off-support sample).
        OVERFLOW: The realized support is larger than the packed capacity.
    """

    OK = 0
    INVALID = 1
    OVERFLOW = 2


@dataclass(frozen=True, slots=True)
class SamplingSupportTensors:
    """Device-resident support capture for the opt-in rows of one step.

    Row ``j`` corresponds to sampling row ``row_indices[j]`` (packed
    sampling order). ``token_ids`` padding: when the support is shorter
    than K, the spare slots hold zero-weight tokens (topk over a mostly
    zero row) — consumers stop at ``lengths``.

    Attributes:
        token_ids: ``[R, K]`` int32, sorted desc by weight.
        lengths: ``[R]`` int32, clamped to K.
        selected_logprobs: ``[R]`` float32.
        statuses: ``[R]`` int32 (:class:`SamplingSupportStatus`).
        row_indices: ``[R]`` int64.
    """

    token_ids: torch.Tensor  # [R, K] int32, sorted desc by weight
    lengths: torch.Tensor  # [R] int32, clamped to K
    selected_logprobs: torch.Tensor  # [R] float32
    statuses: torch.Tensor  # [R] int32 (SamplingSupportStatus)
    row_indices: torch.Tensor  # [R] int64


def _validate_inputs(
    weights_by_id: torch.Tensor,
    sampled_tokens: torch.Tensor,
    row_indices: torch.Tensor,
    max_tokens: int,
) -> None:
    """Validate shapes and dtypes for support capture.

    Args:
        weights_by_id: ``[R, V]`` float tensor of filtered weights.
        sampled_tokens: ``[R]`` sampled token ids.
        row_indices: ``[R]`` packed sampling rows being captured.
        max_tokens: Packed support capacity; must be ``>= 1``.

    Raises:
        ValueError: If any shape, dtype, or capacity contract is violated.
    """
    if weights_by_id.dim() != 2 or not weights_by_id.is_floating_point():
        raise ValueError(
            f"weights_by_id must be a 2-D float tensor, got {tuple(weights_by_id.shape)}"
        )
    rows = weights_by_id.shape[0]
    if sampled_tokens.dim() != 1 or sampled_tokens.shape[0] != rows:
        raise ValueError(f"sampled_tokens must be [{rows}], got {tuple(sampled_tokens.shape)}")
    if row_indices.dim() != 1 or row_indices.shape[0] != rows:
        raise ValueError(f"row_indices must be [{rows}], got {tuple(row_indices.shape)}")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")


def build_support_output(
    weights_by_id: torch.Tensor,
    sampled_tokens: torch.Tensor,
    *,
    max_tokens: int,
    row_indices: torch.Tensor,
) -> SamplingSupportTensors:
    """Pack a filtered distribution (token-id space) into a capture result.

    Args:
        weights_by_id: ``[R, V]`` float32 — filtered probs scattered back
            to token-id space (per-row sum ≤ 1; 0 = off-support).
        sampled_tokens: ``[R]`` int sampled token ids.
        max_tokens: Packed support cap (``sampling_support_max_tokens``).
        row_indices: ``[R]`` int64 — packed sampling rows being captured.

    Returns:
        The device-resident :class:`SamplingSupportTensors` capture.
    """
    _validate_inputs(weights_by_id, sampled_tokens, row_indices, max_tokens)
    sampled = sampled_tokens.to(torch.long)
    support = weights_by_id > 0
    support_mass = weights_by_id.sum(dim=-1, dtype=torch.float32)
    selected_weight = weights_by_id.gather(1, sampled.view(-1, 1)).squeeze(1).to(torch.float32)
    selected_logprobs = torch.log(selected_weight / support_mass)
    invalid = ~((selected_weight > 0) & (support_mass > 0) & torch.isfinite(selected_logprobs))
    realized = support.sum(dim=-1, dtype=torch.int32)
    overflow = realized > max_tokens
    statuses = (
        torch.where(
            invalid,
            torch.full_like(realized, int(SamplingSupportStatus.INVALID)),
            torch.where(
                overflow,
                torch.full_like(realized, int(SamplingSupportStatus.OVERFLOW)),
                torch.full_like(realized, int(SamplingSupportStatus.OK)),
            ),
        )
    ).to(torch.int32)

    packed_size = min(max_tokens, weights_by_id.shape[-1])
    _, packed = torch.topk(weights_by_id, k=packed_size, dim=-1, largest=True, sorted=True)
    return SamplingSupportTensors(
        token_ids=packed.to(torch.int32),
        lengths=realized.clamp(max=packed_size),
        selected_logprobs=selected_logprobs,
        statuses=statuses,
        row_indices=row_indices,
    )


def greedy_support_output(
    sampled_tokens: torch.Tensor,
    row_indices: torch.Tensor,
) -> SamplingSupportTensors:
    """Support for greedy rows: the action space holds only the winner.

    Single-token support. Mirrors SGLang's
    ``_build_greedy_sampling_mask_output``: token_ids = winner, length = 1,
    selected_logprob = 0, status OK.

    Args:
        sampled_tokens: ``[R]`` sampled (argmax) token ids.
        row_indices: ``[R]`` packed sampling rows being captured.

    Returns:
        The device-resident :class:`SamplingSupportTensors` capture.

    Raises:
        ValueError: If the inputs are not 1-D or their lengths differ.
    """
    if sampled_tokens.dim() != 1 or row_indices.dim() != 1:
        raise ValueError("sampled_tokens/row_indices must be 1-D")
    if sampled_tokens.shape[0] != row_indices.shape[0]:
        raise ValueError("sampled_tokens and row_indices must have the same length")
    device = sampled_tokens.device
    num = sampled_tokens.shape[0]
    return SamplingSupportTensors(
        token_ids=sampled_tokens.to(torch.int32).view(-1, 1),
        lengths=torch.ones(num, dtype=torch.int32, device=device),
        selected_logprobs=torch.zeros(num, dtype=torch.float32, device=device),
        statuses=torch.full(
            (num,),
            int(SamplingSupportStatus.OK),
            dtype=torch.int32,
            device=device,
        ),
        row_indices=row_indices,
    )
