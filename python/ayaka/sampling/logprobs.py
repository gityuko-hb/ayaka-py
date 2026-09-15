"""Logprob semantics — RAW first, typed results, no silent degradation.

Two modes, decided up front so the public contract never has to break:

  * ``RAW``: log P_raw(t) = z_t - logsumexp(z) computed on FP32 model logits
    AFTER model-owned normalization (logit scale / softcap), BEFORE penalties,
    mask, temperature and any stochastic filtering. Raw logprobs describe the
    model, so they are stable for scoring and comparable across sampling
    configurations. Enabling them must not change sampled tokens.
  * ``SAMPLING``: log P_sampling(t) on the distribution ACTUALLY used to
    select the token — penalties and mask applied, temperature scaled, then
    top-k/top-p/min-p filtered and renormalized over the surviving support.
    A token outside the surviving support has probability zero -> ``-inf``.
    The selected token is always drawn from that support, so its logprob is
    finite; a fully-filtered row (numerical underflow only) falls back to the
    processed-distribution logprob. The mode governs GENERATION logprobs;
    prompt logprobs have no selection step and are always raw.

Contract for ``logprobs=k``: the selected token's logprob is reported as
``token_logprob``; ``top_logprobs`` lists the k highest-probability tokens of
the same distribution, each token at most once. The selected token may also
appear inside ``top_logprobs`` when it ranks there -- it is never duplicated
within the list, and its logprob is identical in both places. Greedy rows
report the real logprob of the argmax token under the chosen mode, never 0
(for SAMPLING mode a greedy row's distribution is the temperature->0 limit,
so its logprob legitimately approaches 0).

Device residency: token ids and logprobs stay tensors until the completion
boundary materializes host values (``SampleOutputs`` -> ``LogprobResult``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import torch

from ayaka.sampling.ops.sampling import filter_probs
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.validation import require_int

__all__ = [
    "LOGPROBS_DISABLED",
    "LogprobEntry",
    "LogprobMode",
    "LogprobPlan",
    "LogprobResult",
    "MODE_ORDINALS",
    "PromptLogprobSliceReport",
    "PromptLogprobTensors",
    "SampleLogprobTensors",
    "TokenLogprob",
    "compute_prompt_logprobs",
    "compute_raw_logprobs",
    "compute_sampling_logprobs",
]


class LogprobMode(StrEnum):
    """Which distribution a reported logprob describes (see module docstring)."""

    RAW = "raw"
    SAMPLING = "sampling"


#: Device-column encoding for ``logprob_mode`` (I32).
MODE_ORDINALS: Final[dict[LogprobMode, int]] = {
    LogprobMode.RAW: 0,
    LogprobMode.SAMPLING: 1,
}

#: Device-column encoding for ``logprobs_k`` / ``prompt_logprobs_k``: disabled.
LOGPROBS_DISABLED: Final[int] = -1


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token_id: int
    logprob: float

    def __post_init__(self) -> None:
        require_int(self.token_id, "token_id")
        if isinstance(self.logprob, bool) or not isinstance(self.logprob, float):
            raise TypeError("logprob must be a float")


@dataclass(frozen=True, slots=True)
class LogprobResult:
    """Host-side materialization of one reported logprob row."""

    token_logprob: float
    top_logprobs: tuple[TokenLogprob, ...]
    token_ids_logprobs: tuple[TokenLogprob, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.token_logprob, bool) or not isinstance(self.token_logprob, float):
            raise TypeError("token_logprob must be a float")
        if type(self.top_logprobs) is not tuple:
            raise TypeError("top_logprobs must be a tuple")
        if type(self.token_ids_logprobs) is not tuple:
            raise TypeError("token_ids_logprobs must be a tuple")


@dataclass(frozen=True, slots=True)
class LogprobEntry:
    """One report request: sampling row ``sampling_row`` needs top-``k``."""

    sampling_row: int
    k: int

    def __post_init__(self) -> None:
        require_int(self.sampling_row, "logprob sampling_row")
        require_int(self.k, "logprob k", minimum=0)


@dataclass(frozen=True, slots=True)
class LogprobPlan:
    """Immutable reporting description for one step; no tensors, no live state.

    Built by the runner from scheduler-owned request parameters. Entries are
    sorted by ``sampling_row`` (packed sampling order) and reference rows
    inside the step's sampling-row prefix of the logits.
    """

    entries: tuple[LogprobEntry, ...] = ()
    mode: LogprobMode = LogprobMode.RAW

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise TypeError("entries must be a tuple")
        if not isinstance(self.mode, LogprobMode):
            raise TypeError("mode must be LogprobMode")
        previous = -1
        for entry in self.entries:
            if not isinstance(entry, LogprobEntry):
                raise TypeError("entries must contain LogprobEntry")
            if entry.sampling_row <= previous:
                raise ValueError("logprob entries must be sorted by sampling_row")
            previous = entry.sampling_row

    def __bool__(self) -> bool:
        return bool(self.entries)


@dataclass(frozen=True, slots=True)
class SampleLogprobTensors:
    """Device-resident logprob results for one step's generation report rows.

    Rows correspond 1:1 with ``SampleOutputs.logprob_rows`` (sampling-row
    indices). ``top_token_ids`` uses -1 as padding so per-row k can differ
    under a shared top width; materialization stops at the first -1.
    """

    token_logprob: torch.Tensor  # [R] FP32
    top_token_ids: torch.Tensor  # [R, K] int64, -1 = padding
    top_logprobs: torch.Tensor  # [R, K] FP32


@dataclass(frozen=True, slots=True)
class PromptLogprobTensors:
    """Device-resident prompt-logprob results for one step, packed by scored row.

    Rows are ordered by (slice_index, position) across the step; the host
    descriptors in ``SampleOutputs.prompt_logprob_slices`` map rows back to
    absolute prompt positions. Targets (the prompt token at position p) are
    known before sampling, so prompt scoring never depends on the sample.
    """

    token_logprob: torch.Tensor  # [P] FP32
    top_token_ids: torch.Tensor  # [P, K] int64, -1 = padding
    top_logprobs: torch.Tensor  # [P, K] FP32


@dataclass(frozen=True, slots=True)
class PromptLogprobSliceReport:
    """Host descriptor for one slice's prompt-scored rows (no tensors).

    The slice covers absolute prompt positions ``[start, end)``; positions
    inside that range without an entry in ``scored_positions`` are no-value
    markers (first token of the prompt, positions whose predecessor lies in
    the prefix-cached region). ``scored_positions`` is ascending and unique.
    """

    slice_index: int
    start: int
    end: int
    scored_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        require_int(self.slice_index, "prompt logprob slice_index")
        require_int(self.start, "prompt logprob start")
        require_int(self.end, "prompt logprob end")
        if self.start >= self.end:
            raise ValueError("prompt logprob slice must cover a non-empty range")
        if type(self.scored_positions) is not tuple:
            raise TypeError("scored_positions must be a tuple")
        previous = -1
        for position in self.scored_positions:
            require_int(position, "prompt logprob position")
            if position <= previous:
                raise ValueError("scored_positions must be ascending")
            if not self.start <= position < self.end:
                raise IndexError(
                    f"scored position {position} outside the slice range [{self.start}, {self.end})"
                )
            previous = position


def _logprob_rows(
    z: torch.Tensor,
    selected: torch.Tensor,
    ks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared raw math: one normalization value per row, no log-softmax matrix.

    z [R, V] FP32 (one row per report entry, caller-aligned); selected [R]
    int64 targets; ks [R] top-k per row (0 = selected only). Returns
    (token_logprob, top_ids, top_logprobs) with -1/-inf padding for rows whose
    k is below the batch's max k. Top-k tokens are unique by construction and
    the selected token's logprob is identical in token_logprob and in its top
    entry when it ranks within k.
    """
    lse = torch.logsumexp(z, dim=-1)
    token_logprob = z.gather(1, selected.unsqueeze(1)).squeeze(1) - lse
    k_max = int(ks.max().item()) if ks.numel() else 0
    if k_max > 0:
        top_vals, top_ids = torch.topk(z, k_max, dim=-1)
        top_logprobs = top_vals - lse.unsqueeze(1)
        padding = torch.arange(k_max, device=z.device).unsqueeze(0) >= ks.unsqueeze(1)
        top_ids = top_ids.masked_fill(padding, -1)
        top_logprobs = top_logprobs.masked_fill(padding, float("-inf"))
    else:
        top_ids = torch.empty((z.size(0), 0), dtype=torch.long, device=z.device)
        top_logprobs = torch.empty((z.size(0), 0), dtype=torch.float32, device=z.device)
    return token_logprob, top_ids, top_logprobs


def compute_raw_logprobs(
    raw_logits: torch.Tensor,
    selected_tokens: torch.Tensor,
    plan: LogprobPlan,
) -> SampleLogprobTensors:
    """Raw logprobs for the plan's report rows: log P_raw(t) = z_t - logsumexp(z).

    ``raw_logits`` must be the FP32 raw snapshot taken BEFORE any sampling
    transform mutates them (penalties/mask/temperature are in-place).
    ``selected_tokens`` holds one token id per SAMPLING row (packed order);
    rows are gathered per entry. FP32 throughout; one normalization value per
    row, no materialized log-softmax matrix.
    """
    if plan.mode is not LogprobMode.RAW:
        raise CapabilityError(
            "logprob_mode",
            detail=(
                f"logprob_mode={plan.mode.value!r} is not implemented; only "
                "'raw' distribution logprobs are supported"
            ),
            remedy="use logprob_mode=LogprobMode.RAW, or wait for sampling logprobs",
        )
    if not plan.entries:
        raise ValueError("logprob plan has no entries")
    if raw_logits.dim() != 2:
        raise ValueError(f"raw_logits must be 2-D, got {tuple(raw_logits.shape)}")
    if selected_tokens.dim() != 1:
        raise ValueError(f"selected_tokens must be 1-D, got {tuple(selected_tokens.shape)}")

    device = raw_logits.device
    rows = torch.tensor(
        [entry.sampling_row for entry in plan.entries], dtype=torch.long, device=device
    )
    if int(rows.max()) >= raw_logits.size(0):
        raise IndexError("logprob entry references a row outside raw_logits")
    z = raw_logits.to(torch.float32)
    selected = selected_tokens.index_select(0, rows).to(torch.long)
    ks = torch.tensor([entry.k for entry in plan.entries], dtype=torch.long, device=device)
    token_logprob, top_ids, top_logprobs = _logprob_rows(z, selected, ks)

    return SampleLogprobTensors(
        token_logprob=token_logprob, top_token_ids=top_ids, top_logprobs=top_logprobs
    )


@dataclass(frozen=True, slots=True)
class TokenIdsLogprobTensors:
    """Device-resident log probabilities for explicit token ids (`token_ids_logprobs`).

    Row `j` corresponds to `SampleOutputs.ids_logprob_rows[j]`.

    Attributes:
        logprobs: Tensor of shape `[batch_size, max_tokens]` in FP32 containing log
            probabilities for each requested token id. Unused padded elements are `-inf`.
        token_logprob: Tensor of shape `[batch_size]` in FP32 containing the log probability
            of the sampled token evaluated under identical distribution semantics.
    """

    logprobs: torch.Tensor  # [R, K] FP32
    token_logprob: torch.Tensor  # [R] FP32


def _token_ids_tensor(token_lists: Sequence[Sequence[int]], device) -> torch.Tensor:
    counts = [len(tokens) for tokens in token_lists]
    width = max(counts) if counts else 0
    if width == 0:
        return torch.empty((len(token_lists), 0), dtype=torch.long, device=device)
    ids = torch.full((len(token_lists), width), -1, dtype=torch.long, device=device)
    for row, tokens in enumerate(token_lists):
        if tokens:
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    return ids


def compute_token_ids_raw_logprobs(
    raw_logits: torch.Tensor,
    token_lists: Sequence[Sequence[int]],
    selected_tokens: torch.Tensor,
) -> TokenIdsLogprobTensors:
    """Compute raw log probabilities for explicit token id lists.

    Computes `z_t - logsumexp(z)` per row for each requested token id without
    materializing a full `[batch_size, vocab_size]` log-softmax matrix.

    Args:
        raw_logits: Float32 tensor of pre-transform model logits of shape
            `[batch_size, vocab_size]`.
        token_lists: Sequence of target token id sequences to score per batch row.
        selected_tokens: Tensor of shape `[batch_size]` containing sampled tokens.

    Returns:
        TokenIdsLogprobTensors holding explicit token logprobs and selected token logprobs.

    Raises:
        ValueError: If `raw_logits` is not 2-D or row counts mismatch `token_lists`.
    """
    if raw_logits.dim() != 2:
        raise ValueError(f"raw_logits must be 2-D, got {tuple(raw_logits.shape)}")
    if len(token_lists) != raw_logits.size(0):
        raise ValueError(f"token_lists must have {raw_logits.size(0)} rows, got {len(token_lists)}")
    z = raw_logits.to(torch.float32)
    lse = torch.logsumexp(z, dim=-1)
    ids = _token_ids_tensor(token_lists, z.device)
    if ids.numel():
        safe = ids.clamp_min(0)
        logprobs = z.gather(1, safe) - lse.unsqueeze(1)
        logprobs = torch.where(ids >= 0, logprobs, torch.full_like(logprobs, float("-inf")))
    else:
        logprobs = torch.empty_like(ids, dtype=torch.float32)
    token_logprob = z.gather(1, selected_tokens.to(torch.long).unsqueeze(1)).squeeze(1) - lse
    return TokenIdsLogprobTensors(logprobs=logprobs, token_logprob=token_logprob)


def compute_token_ids_sampling_logprobs(
    processed_logits: torch.Tensor,
    token_lists: Sequence[Sequence[int]],
    selected_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> TokenIdsLogprobTensors:
    """Compute sampling-distribution log probabilities for explicit token id lists.

    Evaluates `log(p_renorm(t))` over the surviving positive support. Tokens outside
    the filtered support evaluate to `-inf`. Follows the exact distribution semantics
    of `compute_sampling_logprobs` without consuming RNG state.

    Args:
        processed_logits: Post-penalty, post-mask logits of shape `[batch_size, vocab_size]`.
        token_lists: Sequence of target token id sequences to score per batch row.
        selected_tokens: Tensor of shape `[batch_size]` containing sampled tokens.
        temperature: Tensor of shape `[batch_size]` containing temperature scaling values.
        top_k: Tensor of shape `[batch_size]` specifying top-k filtering thresholds.
        top_p: Tensor of shape `[batch_size]` specifying top-p cumulative thresholds.
        min_p: Tensor of shape `[batch_size]` specifying min-p probability cutoffs.

    Returns:
        TokenIdsLogprobTensors holding explicit token logprobs and selected token logprobs.

    Raises:
        ValueError: If `processed_logits` is not 2-D or row counts mismatch `token_lists`.
    """
    if processed_logits.dim() != 2:
        raise ValueError(f"processed_logits must be 2-D, got {tuple(processed_logits.shape)}")
    if len(token_lists) != processed_logits.size(0):
        raise ValueError(
            f"token_lists must have {processed_logits.size(0)} rows, got {len(token_lists)}"
        )

    selected = selected_tokens.to(torch.long)
    scaled = processed_logits.to(torch.float32) / temperature.clamp_min(1e-6).unsqueeze(1)
    sp, si = filter_probs(scaled, top_k, top_p, min_p)
    total = sp.sum(dim=-1, keepdim=True)
    by_id = torch.zeros_like(sp).scatter_(1, si, sp)
    log_denom = torch.log(total.clamp_min(torch.finfo(sp.dtype).tiny)).squeeze(1)

    ids = _token_ids_tensor(token_lists, scaled.device)
    if ids.numel():
        safe = ids.clamp_min(0)
        weights = by_id.gather(1, safe)
        logprobs = torch.log(weights) - log_denom.unsqueeze(1)
        logprobs = torch.where(ids >= 0, logprobs, torch.full_like(logprobs, float("-inf")))
    else:
        logprobs = torch.empty_like(ids, dtype=torch.float32)

    token_logprob = compute_token_ids_sampling_selected(
        processed_logits, selected, temperature, top_k, top_p, min_p
    )
    return TokenIdsLogprobTensors(logprobs=logprobs, token_logprob=token_logprob)


def compute_token_ids_sampling_selected(
    processed_logits: torch.Tensor,
    selected_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """Compute sampling-distribution log probability for the selected tokens.

    Isolates selected token logprob computation for token-id scoring rows, matching
    the `token_logprob` semantics in `compute_sampling_logprobs`.

    Args:
        processed_logits: Post-penalty, post-mask logits of shape `[batch_size, vocab_size]`.
        selected_tokens: Tensor of sampled token ids of shape `[batch_size]`.
        temperature: Tensor of shape `[batch_size]` containing temperature scaling values.
        top_k: Tensor of shape `[batch_size]` specifying top-k filtering thresholds.
        top_p: Tensor of shape `[batch_size]` specifying top-p cumulative thresholds.
        min_p: Tensor of shape `[batch_size]` specifying min-p probability cutoffs.

    Returns:
        Tensor of shape `[batch_size]` containing selected token log probabilities.
    """
    selected = selected_tokens.to(torch.long)
    scaled = processed_logits.to(torch.float32) / temperature.clamp_min(1e-6).unsqueeze(1)
    sp, si = filter_probs(scaled, top_k, top_p, min_p)
    total = sp.sum(dim=-1, keepdim=True)
    degenerate = (total <= 0).squeeze(1)
    p_renorm = sp / total.clamp_min(torch.finfo(sp.dtype).tiny)
    position = (si == selected.unsqueeze(1)).int().argmax(dim=-1)
    token_logprob = torch.log(p_renorm.gather(1, position.unsqueeze(1)).squeeze(1))
    if bool(degenerate.any()):
        fallback_lse = torch.logsumexp(scaled, dim=-1)
        fallback = scaled.gather(1, selected.unsqueeze(1)).squeeze(1) - fallback_lse
        token_logprob = torch.where(degenerate, fallback, token_logprob)
    return token_logprob


def compute_prompt_logprobs(
    raw_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    ks: Sequence[int],
) -> PromptLogprobTensors:
    """Raw prompt logprobs for one step's scored rows (causal shift applied upstream).

    ``raw_logits`` rows are the FP32 raw snapshots of the hidden row at
    position ``p - 1`` (packed in (slice, position) order); ``target_tokens``
    are the prompt tokens at position ``p`` — known before sampling, so
    scoring never depends on the sample. ``ks`` holds the per-row top-k (0 =
    token logprob only); all rows share the raw semantics.
    """
    if raw_logits.dim() != 2:
        raise ValueError(f"raw_logits must be 2-D, got {tuple(raw_logits.shape)}")
    if target_tokens.dim() != 1 or target_tokens.size(0) != raw_logits.size(0):
        raise ValueError(
            f"target_tokens must be [{raw_logits.size(0)}], got {tuple(target_tokens.shape)}"
        )
    if len(ks) != raw_logits.size(0):
        raise ValueError(f"ks must have {raw_logits.size(0)} entries, got {len(ks)}")

    z = raw_logits.to(torch.float32)
    selected = target_tokens.to(torch.long)
    ks_tensor = torch.tensor(list(ks), dtype=torch.long, device=z.device)
    token_logprob, top_ids, top_logprobs = _logprob_rows(z, selected, ks_tensor)
    return PromptLogprobTensors(
        token_logprob=token_logprob, top_token_ids=top_ids, top_logprobs=top_logprobs
    )


def compute_sampling_logprobs(
    processed_logits: torch.Tensor,
    selected_tokens: torch.Tensor,
    ks: Sequence[int],
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> SampleLogprobTensors:
    """Compute sampling-distribution log probabilities after penalties, masking, and filtering.

    The `processed_logits` rows represent post-penalty, post-mask logits (the sampler
    having already applied penalties and bitmasks in place). Temperature scaling is
    applied here identically to the sampler (`clamp_min(1e-6)`). The filtered
    distribution is constructed via `filter_probs` (top_k -> renorm -> top_p -> min_p),
    and final renormalization over the surviving support yields the sampling distribution:

        log P_sampling(t) = log(p_renorm(t)),  -inf when t lies outside support

    The selected token is drawn from this support, so its logprob is finite. Similarly,
    top-k lists only surviving tokens. Degenerate rows (where surviving probability mass
    underflows to zero due to numerical limits) fall back to the processed-distribution
    logprob to avoid returning NaN values.

    Args:
        processed_logits: Post-penalty, post-mask logits of shape `[batch_size, vocab_size]`.
        selected_tokens: Tensor of sampled token ids of shape `[batch_size]`.
        ks: Sequence specifying the number of top logprobs to return per row.
        temperature: Tensor of shape `[batch_size]` containing temperature scaling values.
        top_k: Tensor of shape `[batch_size]` specifying top-k filtering thresholds.
        top_p: Tensor of shape `[batch_size]` specifying top-p cumulative thresholds.
        min_p: Tensor of shape `[batch_size]` specifying min-p probability cutoffs.

    Returns:
        SampleLogprobTensors holding token logprobs, top token ids, and top logprobs.

    Raises:
        ValueError: If `processed_logits` or `selected_tokens` shapes mismatch `ks`.

    Notes:
        This helper recomputes probabilities post-hoc without consuming randomness,
        guaranteeing that enabling logprob generation never alters sampled tokens.
    """
    if processed_logits.dim() != 2:
        raise ValueError(f"processed_logits must be 2-D, got {tuple(processed_logits.shape)}")
    if selected_tokens.dim() != 1 or selected_tokens.size(0) != processed_logits.size(0):
        raise ValueError(
            f"selected_tokens must be [{processed_logits.size(0)}], got "
            f"{tuple(selected_tokens.shape)}"
        )
    if len(ks) != processed_logits.size(0):
        raise ValueError(f"ks must have {processed_logits.size(0)} entries, got {len(ks)}")

    selected = selected_tokens.to(torch.long)
    scaled = processed_logits.to(torch.float32) / temperature.clamp_min(1e-6).unsqueeze(1)
    sp, si = filter_probs(scaled, top_k, top_p, min_p)
    total = sp.sum(dim=-1, keepdim=True)
    degenerate = (total <= 0).squeeze(1)
    p_renorm = sp / total.clamp_min(torch.finfo(sp.dtype).tiny)

    # Locate selected token in the sorted permutation order (unique match).
    position = (si == selected.unsqueeze(1)).int().argmax(dim=-1)
    token_logprob = torch.log(p_renorm.gather(1, position.unsqueeze(1)).squeeze(1))
    # Fallback to processed-distribution logprob if surviving support underflows to zero.
    if bool(degenerate.any()):
        fallback_lse = torch.logsumexp(scaled, dim=-1)
        fallback = scaled.gather(1, selected.unsqueeze(1)).squeeze(1) - fallback_lse
        token_logprob = torch.where(degenerate, fallback, token_logprob)

    k_max = max(ks) if ks else 0
    if k_max > 0:
        top_ids = si[:, :k_max].contiguous()
        # Outside surviving positive support: p_renorm = 0 -> logprob = -inf.
        top_logprobs = torch.log(p_renorm[:, :k_max])
        ks_tensor = torch.tensor(list(ks), dtype=torch.long, device=scaled.device)
        padding = torch.arange(k_max, device=scaled.device).unsqueeze(0) >= ks_tensor.unsqueeze(1)
        top_ids = top_ids.masked_fill(padding, -1)
        top_logprobs = top_logprobs.masked_fill(padding, float("-inf"))
    else:
        top_ids = torch.empty((processed_logits.size(0), 0), dtype=torch.long, device=scaled.device)
        top_logprobs = torch.empty(
            (processed_logits.size(0), 0), dtype=torch.float32, device=scaled.device
        )
    return SampleLogprobTensors(
        token_logprob=token_logprob, top_token_ids=top_ids, top_logprobs=top_logprobs
    )
