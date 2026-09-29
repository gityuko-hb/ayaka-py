"""Per-request eligibility and draft-length bounds for speculative planning.

Pure functions over immutable snapshots: they read a request's parameters and
its committed ``RequestStepInput`` and never mutate either. The scheduler budget,
KV reservation and projection-row ceiling are applied by the coordinator on top
of these per-request limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.request.schema import Request
from ayaka.sched.plan import RequestStepInput, ScheduledSlice
from ayaka.speculative.config import AcceptanceMethod
from ayaka.speculative.plan import SpecDisableReason

__all__ = ["DraftBound", "draft_bound", "request_disable_reason"]


def request_disable_reason(
    request: Request, *, acceptance: AcceptanceMethod, custom_sampling_ops: bool = False
) -> SpecDisableReason | None:
    """Why a request cannot speculate at all, or ``None`` when it may.

    Greedy verification is exact only when every extension row sees the same
    logit transformations target-only decoding would apply. Rows whose
    transformations depend on tentative history (penalties), need per-row
    reporting (logprobs), advance external state (grammars) or need decoded
    text (stop strings) are excluded until verification models them.
    """
    params = request.sampling
    if custom_sampling_ops:
        return SpecDisableReason.CUSTOM_OPS
    if not params.is_greedy and acceptance is AcceptanceMethod.GREEDY:
        return SpecDisableReason.SAMPLING
    if (
        params.repetition_penalty != 1.0
        or params.frequency_penalty != 0.0
        or params.presence_penalty != 0.0
    ):
        return SpecDisableReason.PENALTIES
    if params.logit_bias:
        return SpecDisableReason.LOGIT_BIAS
    if (
        params.logprobs is not None
        or params.prompt_logprobs is not None
        or params.token_ids_logprobs is not None
        or params.return_sampling_support
    ):
        return SpecDisableReason.LOGPROBS
    if request.constraint is not None:
        return SpecDisableReason.GRAMMAR
    if request.stop.stop_strings:
        return SpecDisableReason.STOP_STRINGS
    return None


@dataclass(frozen=True, slots=True)
class DraftBound:
    """Largest draft length one decode slice may verify, and what bound it."""

    limit: int
    reason: SpecDisableReason | None = None


def draft_bound(
    scheduled: ScheduledSlice,
    value: RequestStepInput,
    *,
    max_draft_tokens: int,
    max_model_len: int,
    draft_context_limit: int | None = None,
) -> DraftBound:
    """Per-request ceiling on draft rows for one decode slice.

    ``k`` drafts can publish ``k + 1`` tokens and occupy positions
    ``L .. L + k - 1`` for ``L`` known tokens, so ``k`` is limited by the
    remaining output budget minus one and by the served (and draft model)
    context. A slice that is not at the known-token boundary (recompute
    replay) never speculates.
    """
    if not scheduled.sample_last_query:
        return DraftBound(0, SpecDisableReason.NOT_AT_BOUNDARY)
    known = len(value.known_tokens)
    generated = known - value.prompt_tokens
    output_room = value.max_output_tokens - generated - 1
    context_room = max_model_len - known
    if draft_context_limit is not None:
        context_room = min(context_room, draft_context_limit - known)
    limit = min(max_draft_tokens, output_room, context_room)
    if limit > 0:
        return DraftBound(limit)
    if output_room <= 0:
        return DraftBound(0, SpecDisableReason.OUTPUT_BUDGET)
    return DraftBound(0, SpecDisableReason.CONTEXT_LIMIT)
