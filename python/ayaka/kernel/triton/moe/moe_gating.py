"""MoE routing: fused score function + top-k + optional re-normalisation (Triton port of
``moe_topk_softmax_kernels.cu`` and ``moe_topk_sigmoid_kernels.cu``).

One kernel replaces the CUDA families ``topkGatingSoftmax`` / ``moeSoftmax + moeTopK / moeTopKFast`` and
``topkGatingSigmoid`` / ``moeSigmoid + moeTopK``: the CUDA code needs different kernels only because a row has
to be spread over threads (power-of-two expert counts) or over a block + a workspace (other expert counts).
Here one program owns ``ROWS`` complete rows in registers, so any expert count works and no workspace exists.

Semantics reproduced from the CUDA sources
------------------------------------------
softmax   x -> [tanh(x / cap) * cap  if cap != 0] -> [+ correction_bias] -> p = exp(x - max) * (1 / sum)
          (the bias is added *before* the softmax; the returned weights are the probabilities p)
sigmoid   x -> s = 1 / (1 + exp(-x)) -> [+ correction_bias] ; the bias only steers the selection: the returned
          weight of a chosen expert is ``(s + bias[e]) - bias[e]`` (bias subtracted back, in fp32)
top-k     k rounds of arg-max; on exact ties the **lowest available expert index wins**; selected experts are
          removed from the candidate set, so extreme correction biases cannot make an expert appear twice
renorm    weights *= 1 / (sum of the k selected weights)   (sum accumulated in selection order, fp32)
indices   ``expert - start_expert`` if the row is unfinished and ``start_expert <= expert < end_expert``,
          else ``num_experts`` (sentinel), exactly like the CUDA kernels

Numerics: everything is fp32.  ``expf`` / ``tanhf`` / IEEE division are the accurate libdevice versions (the same
routines CUDA's math library uses), *not* Triton's fast ``tl.exp`` / ``/``.  What can still differ from CUDA in the
last fp32 bit is the summation order of the softmax denominator (thread-tree in CUDA, ``tl.sum`` tree here).

For softmax with ``num_experts`` outside ``{1,2,4,...,512}`` and ``topk >= 2``, CUDA's own general path uses
``moeTopKFast`` (a top-2-per-iteration variant) instead of ``moeTopK`` â€” a common combination in practice (e.g.
160 experts, top-8). Its pairwise reducer keeps whichever operand it calls "candidate2" when two max values tie,
which is a function of CUB's internal reduction-tree/thread-pairing order, not the expert index, so it is *not*
a lowest-index rule and is not fully pinned down by the visible source. Every other path here â€” ``topkGatingSoftmax``,
``moeTopK`` in both files, and ``topkGatingSigmoid`` â€” implements the same explicit "lowest index wins" rule via
``cub::ArgMax`` or an equivalent hand-written comparison, which this port reproduces exactly. This port therefore
applies lowest-index tie-breaking uniformly, including in place of ``moeTopKFast``. Because softmax scores are
continuous (``exp(...)/sum(...)`` over arbitrary activations), the two rules disagree only on an exact bit-for-bit
tie between two experts' scores in the same row and iteration â€” for real (non-adversarial, non-degenerate) model
activations this essentially never occurs, so the practical outputs match; a synthetic input engineered to tie
(e.g. two identical logits) is the only case where results could differ from CUDA's ``moeTopKFast`` specifically.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import div_rn, exp, tanh

from ._common import autotune, cdiv, check, check_tensor, device_guard, f32, next_pow2, same_device

SCORE_SOFTMAX = 0
SCORE_SIGMOID = 1

_GATING_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_MAX_EXPERTS = 32768  # one row is held in a single register tile
_TILE_ELEMS = 4096  # target ROWS * BLOCK_E per program

# num_warps-only search, kept separate from the shared row/GEMM banks because
# it tunes a different kernel shape (expert tiles, not hidden-dim tiles).
_GATING_CONFIGS = [triton.Config({}, num_warps=w) for w in (1, 2, 4, 8)]


@autotune(configs=_GATING_CONFIGS, key=["BLOCK_E", "TOPK", "ROWS", "SCORE_FN"])
@triton.jit
def moe_topk_gating_kernel(
    gating_ptr,
    bias_ptr,
    finished_ptr,
    weights_ptr,
    indices_ptr,
    num_rows,
    num_experts,
    start_expert,
    end_expert,
    softcap,
    SCORE_FN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SUB_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    HAS_FINISHED: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_ok = rows < num_rows
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, BLOCK_E)
    col_ok = cols < num_experts

    # ---- load the logits of ROWS full rows, convert to fp32 (convert_to_float) -----------------
    ld_mask = row_ok[:, None] & col_ok[None, :]
    x = tl.load(gating_ptr + rows64[:, None] * num_experts + cols[None, :], mask=ld_mask, other=0.0)
    x = x.to(tl.float32)
    ones2d = tl.zeros([ROWS, BLOCK_E], dtype=tl.float32) + 1.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)

    # ---- score function --------------------------------------------------------------------------
    if SCORE_FN == 0:
        # topkGatingSoftmax / moeSoftmax
        if USE_SOFTCAP:
            capv = tl.zeros([ROWS, BLOCK_E], dtype=tl.float32) + softcap
            x = tanh(div_rn(x, capv)) * softcap  # tanhf(val / cap) * cap
        if HAS_BIAS:
            x = x + bias[None, :]  # bias enters BEFORE the softmax
        row_max = tl.max(tl.where(col_ok[None, :], x, float("-inf")), axis=1)
        e = exp(x - row_max[:, None])
        e = tl.where(col_ok[None, :], e, 0.0)
        row_sum = tl.sum(e, axis=1)
        inv_sum = div_rn(tl.zeros([ROWS], dtype=tl.float32) + 1.0, row_sum)
        score = e * inv_sum[:, None]
    else:
        # topkGatingSigmoid / moeSigmoid:  1 / (1 + expf(-x))  (+ bias)
        score = div_rn(ones2d, ones2d + exp(-x))
        if HAS_BIAS:
            score = score + bias[None, :]

    # ---- iterative arg-max top-k (lowest index wins ties) -----------------------------------------
    sel = tl.where(col_ok[None, :], score, float("-inf"))
    available = tl.full([ROWS, BLOCK_E], True, dtype=tl.int1) & col_ok[None, :]
    kk = tl.arange(0, BLOCK_K)
    out_w = tl.zeros([ROWS, BLOCK_K], dtype=tl.float32)
    out_i = tl.zeros([ROWS, BLOCK_K], dtype=tl.int32)
    sum_w = tl.zeros([ROWS], dtype=tl.float32)
    if HAS_FINISHED:
        fin = tl.load(finished_ptr + rows, mask=row_ok, other=0)
        active = fin == 0
    else:
        active = row_ok

    for ki in tl.static_range(TOPK):
        candidate_scores = tl.where(available, sel, float("-inf"))
        mx = tl.max(candidate_scores, axis=1)
        cand = tl.where(available & (candidate_scores == mx[:, None]), cols[None, :], BLOCK_E)
        ex = tl.min(cand, axis=1)
        has_candidate = ex < num_experts
        ex = tl.where(has_candidate, ex, 0)  # keep masked pointer arithmetic in range
        val = tl.where(has_candidate, mx, 0.0)
        if SUB_BIAS:
            val = mx - tl.load(bias_ptr + ex, mask=row_ok, other=0.0).to(tl.float32)
            val = tl.where(has_candidate, val, 0.0)
        should = active & has_candidate & (ex >= start_expert) & (ex < end_expert)
        idx_out = tl.where(should, ex - start_expert, num_experts)
        is_k = kk[None, :] == ki
        out_w = tl.where(is_k, val[:, None], out_w)
        out_i = tl.where(is_k, idx_out[:, None], out_i)
        sum_w += val
        available = available & (cols[None, :] != ex[:, None])

    if RENORMALIZE:
        inv = div_rn(tl.zeros([ROWS], dtype=tl.float32) + 1.0, sum_w)
        out_w = out_w * inv[:, None]

    st_mask = row_ok[:, None] & (kk[None, :] < TOPK)
    st_off = rows64[:, None] * TOPK + kk[None, :]
    tl.store(weights_ptr + st_off, out_w, mask=st_mask)
    tl.store(indices_ptr + st_off, out_i, mask=st_mask)


def moe_topk_gating(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    *,
    score_fn: int,
    renormalize: bool,
    moe_softcapping: float = 0.0,
    correction_bias: torch.Tensor | None = None,
    finished: torch.Tensor | None = None,
    start_expert: int = 0,
    end_expert: int | None = None,
) -> None:
    """Low-level entry point: ``moeTopK``-style parameters (``finished`` / ``start_expert`` / ``end_expert``)
    are exposed here; ``topk_softmax`` / ``topk_sigmoid`` call it with the same defaults as the CUDA launchers
    (``finished=None, start_expert=0, end_expert=num_experts``)."""
    check(score_fn in (SCORE_SOFTMAX, SCORE_SIGMOID), "unknown score function")
    check_tensor(gating_output, "gating_output", dtypes=_GATING_DTYPES, ndim=2, contiguous=True)
    check_tensor(topk_weights, "topk_weights", dtypes=(torch.float32,), ndim=2, contiguous=True)
    check_tensor(topk_indices, "topk_indices", dtypes=(torch.int32,), ndim=2, contiguous=True)
    same_device(gating_output, topk_weights, topk_indices)
    check(
        gating_output.size(0) == topk_weights.size(0),
        "First dimension of topk_weights must match num_tokens in gating_output",
    )
    check(
        gating_output.size(0) == topk_indices.size(0),
        "First dimension of topk_indices must match num_tokens in gating_output",
    )
    check(
        topk_weights.size(-1) == topk_indices.size(-1),
        "Second dimension of topk_indices must match topk in topk_weights",
    )
    check(
        topk_weights.size(-1) <= gating_output.size(-1),
        "topk must be less than or equal to num_experts",
    )
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    if num_tokens == 0:
        return
    check(num_experts > 0, "num_experts must be greater than 0")
    check(topk > 0, "topk must be greater than 0")
    check(
        num_experts <= _MAX_EXPERTS,
        f"num_experts > {_MAX_EXPERTS} is not supported by the Triton port",
    )
    if correction_bias is not None:
        check_tensor(
            correction_bias, "correction_bias", dtypes=(torch.float32,), ndim=1, contiguous=True
        )
        same_device(gating_output, correction_bias)
        check(correction_bias.size(0) == num_experts, "correction_bias size must match num_experts")
    if finished is not None:
        check_tensor(
            finished, "finished", dtypes=(torch.bool, torch.uint8), ndim=1, contiguous=True
        )
        check(finished.size(0) == num_tokens, "finished must have one flag per row")
        same_device(gating_output, finished)
        finished = finished.view(torch.uint8) if finished.dtype == torch.bool else finished
    if end_expert is None:
        end_expert = num_experts
    start_expert, end_expert = int(start_expert), int(end_expert)
    check(
        0 <= start_expert <= end_expert <= num_experts,
        "expert range must satisfy 0 <= start_expert <= end_expert <= num_experts",
    )
    cap = f32(moe_softcapping)
    use_softcap = score_fn == SCORE_SOFTMAX and cap != 0.0

    block_e = max(2, next_pow2(num_experts))
    rows = max(1, min(16, _TILE_ELEMS // block_e))
    has_bias = correction_bias is not None
    with device_guard(gating_output):
        moe_topk_gating_kernel[(cdiv(num_tokens, rows),)](
            gating_output,
            correction_bias
            if has_bias
            else gating_output,  # dummy pointers when the feature is off
            finished if finished is not None else gating_output,
            topk_weights,
            topk_indices,
            num_tokens,
            num_experts,
            start_expert,
            end_expert,
            cap,
            SCORE_FN=score_fn,
            HAS_BIAS=has_bias,
            SUB_BIAS=has_bias and score_fn == SCORE_SIGMOID,
            USE_SOFTCAP=use_softcap,
            HAS_FINISHED=finished is not None,
            RENORMALIZE=bool(renormalize),
            TOPK=topk,
            BLOCK_K=max(2, next_pow2(topk)),
            BLOCK_E=block_e,
            ROWS=rows,
        )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    moe_softcapping: float = 0.0,
    correction_bias: torch.Tensor | None = None,
) -> None:
    """Signature-compatible replacement for the CUDA ``topk_softmax`` op.

    ``gating_output: [num_tokens, num_experts]`` (fp32 / fp16 / bf16); ``topk_weights: [num_tokens, topk]`` fp32
    and ``topk_indices: [num_tokens, topk]`` int32 are written in place.
    """
    moe_topk_gating(
        topk_weights,
        topk_indices,
        gating_output,
        score_fn=SCORE_SOFTMAX,
        renormalize=renormalize,
        moe_softcapping=moe_softcapping,
        correction_bias=correction_bias,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
) -> None:
    """Signature-compatible replacement for the CUDA ``topk_sigmoid`` op (see module docstring)."""
    moe_topk_gating(
        topk_weights,
        topk_indices,
        gating_output,
        score_fn=SCORE_SIGMOID,
        renormalize=renormalize,
        correction_bias=correction_bias,
    )
