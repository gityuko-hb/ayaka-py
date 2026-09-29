"""Immutable scheduler contract for one speculative decode slice.

A plan never changes request state. Draft tokens are tentative and live only
here; ``RequestStepInput.known_tokens`` keeps committed prompt/output tokens.
The owning ``ScheduledSlice`` still describes the committed base row, while this
plan adds ``effective_k`` extension rows that verification may or may not
commit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from ayaka.speculative.config import MAX_DRAFT_TOKENS
from ayaka.speculative.mode import SpeculativeMode
from ayaka.utils.validation import require_int, require_text

__all__ = ["SpecDisableReason", "SpeculativeSlicePlan"]


class SpecDisableReason(StrEnum):
    """Finite, request-free labels for a decode row that did not speculate."""

    POLICY = "policy"
    NO_PROPOSAL = "no_proposal"
    TOKEN_BUDGET = "token_budget"
    STEP_DRAFT_CAP = "step_draft_cap"
    OUTPUT_BUDGET = "output_budget"
    CONTEXT_LIMIT = "context_limit"
    NOT_AT_BOUNDARY = "not_at_boundary"
    SAMPLING = "sampling_unsupported"
    PENALTIES = "penalties"
    LOGIT_BIAS = "logit_bias"
    LOGPROBS = "logprobs"
    GRAMMAR = "grammar"
    STOP_STRINGS = "stop_strings"
    CUSTOM_OPS = "custom_sampling_ops"
    MEMORY_PRESSURE = "memory_pressure"


@dataclass(frozen=True, slots=True)
class SpeculativeSlicePlan:
    """Speculative extension of one decode slice in packed order.

    Attributes:
        slice_index: Index of the owning ``ScheduledSlice`` in the step.
        request_id: Owning request; must equal the slice's request.
        mode: Method that produced the drafts.
        requested_k: Draft length the policy asked for.
        effective_k: Draft rows actually verified, after proposal length,
            output/context limits and budget. Always at least one: a slice
            that does not speculate has no plan.
        draft_token_ids: Tentative tokens for positions ``L .. L+k-1``.
        reserve_tokens: KV rows reserved for the slice (base + drafts).
        verify_tokens: Target query rows executed for the slice.
        graph_bucket: Captured verification bucket, or ``None`` for eager.
    """

    slice_index: int
    request_id: str
    mode: SpeculativeMode
    requested_k: int
    effective_k: int
    draft_token_ids: tuple[int, ...]
    reserve_tokens: int
    verify_tokens: int
    graph_bucket: int | None = None

    def __post_init__(self) -> None:
        if __debug__:
            self.validate()

    def validate(self) -> None:
        """Validate explicitly, including under optimized Python."""
        require_int(self.slice_index, "speculative slice_index")
        require_text(self.request_id, "speculative request_id")
        if not isinstance(self.mode, SpeculativeMode) or self.mode in (
            SpeculativeMode.NONE,
            SpeculativeMode.AUTO,
        ):
            raise ValueError("speculative plan needs a concrete speculative mode")
        require_int(self.requested_k, "requested_k", minimum=1)
        require_int(self.effective_k, "effective_k", minimum=1)
        if self.effective_k > self.requested_k or self.requested_k > MAX_DRAFT_TOKENS:
            raise ValueError("effective_k must be in [1, requested_k <= MAX_DRAFT_TOKENS]")
        if type(self.draft_token_ids) is not tuple:
            raise TypeError("draft_token_ids must be a tuple")
        if len(self.draft_token_ids) != self.effective_k:
            raise ValueError("one draft token is required per effective draft row")
        for token in self.draft_token_ids:
            require_int(token, "draft token id")
        require_int(self.reserve_tokens, "reserve_tokens", minimum=2)
        require_int(self.verify_tokens, "verify_tokens", minimum=2)
        if self.reserve_tokens != 1 + self.effective_k:
            raise ValueError("a decode slice reserves its base row plus every draft row")
        if self.verify_tokens != self.reserve_tokens:
            raise ValueError("linear verification executes exactly the reserved rows")
        if self.graph_bucket is not None:
            require_int(self.graph_bucket, "graph_bucket", minimum=1)

    def truncated(self, effective_k: int) -> SpeculativeSlicePlan:
        """The same plan keeping only the first ``effective_k`` drafts."""
        if not 1 <= effective_k <= self.effective_k:
            raise ValueError("a truncated plan keeps between one and all of its drafts")
        committed = self.reserve_tokens - self.effective_k
        return replace(
            self,
            effective_k=effective_k,
            draft_token_ids=self.draft_token_ids[:effective_k],
            reserve_tokens=committed + effective_k,
            verify_tokens=committed + effective_k,
        )

    def with_slice_index(self, slice_index: int) -> SpeculativeSlicePlan:
        return replace(self, slice_index=slice_index)
