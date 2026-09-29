"""Host control plane for speculative decoding.

The coordinator owns the proposer, the draft-length policy and the metrics. The
scheduler asks it for speculative extensions of decode slices it has already
admitted; the coordinator returns immutable :class:`SpeculativeSlicePlan`
values and never touches request lifecycles, KV or sampling state. After
completion the scheduler hands back settled verifications so the policy and
proposer learn from committed outcomes only.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ayaka.request.schema import Request
from ayaka.sched.plan import Phase, RequestStepInput, ScheduledSlice
from ayaka.speculative.config import SpeculativeDecodingConfig
from ayaka.speculative.interface import ProposalRequest, ProposerRuntimeState, SpeculativeProposer
from ayaka.speculative.metrics import SpeculativeMetrics
from ayaka.speculative.mode import SpeculativeMode, resolve_mode
from ayaka.speculative.plan import SpecDisableReason, SpeculativeSlicePlan
from ayaka.speculative.policy import PolicyContext, StepObservation, build_policy
from ayaka.speculative.resources import draft_bound, request_disable_reason

if TYPE_CHECKING:
    from ayaka.executor.completion import CompletionResult
    from ayaka.sched.plan import BatchStepPlan

__all__ = ["DecodeCandidate", "SpeculationProposal", "SpeculativeCoordinator"]


@dataclass(frozen=True, slots=True)
class DecodeCandidate:
    """One admitted decode slice the scheduler may extend with drafts."""

    slice_index: int
    scheduled: ScheduledSlice
    value: RequestStepInput
    request: Request


@dataclass(frozen=True, slots=True)
class SpeculationProposal:
    """Plans for one candidate step, before the scheduler applies its budget."""

    plans: tuple[SpeculativeSlicePlan, ...]
    disabled: Counter[SpecDisableReason] = field(default_factory=Counter)
    policy_k: int = 0
    draft_latency_ns: int = 0


class SpeculativeCoordinator:
    """Resolve draft plans for decode slices and learn from settled outcomes.

    Args:
        config: Validated subsystem configuration.
        max_model_len: Served context ceiling; drafts never pass it.
        proposer: Draft source; defaults to the n-gram proposer for ``NGRAM``.
        stop_token_ids: Per-request token ids that end generation, used to
            end proposals early (nothing after a stop can publish).
        custom_sampling_ops: Whether the sampler applies custom logit ops,
            which verification does not replicate.
        draft_context_limit: Context ceiling of a neural drafter, if any.
    """

    def __init__(
        self,
        config: SpeculativeDecodingConfig,
        *,
        max_model_len: int,
        proposer: SpeculativeProposer | None = None,
        stop_token_ids: Callable[[str], frozenset[int]] | None = None,
        custom_sampling_ops: bool = False,
        draft_context_limit: int | None = None,
    ) -> None:
        if not isinstance(config, SpeculativeDecodingConfig):
            raise TypeError("config must be SpeculativeDecodingConfig")
        mode = resolve_mode(config.mode)
        if mode is SpeculativeMode.NONE:
            raise ValueError("no production speculative mode is available for AUTO")
        config.require_runnable(mode)
        resolved: SpeculativeProposer
        if proposer is not None:
            resolved = proposer
        elif mode is SpeculativeMode.NGRAM:
            from ayaka.speculative.proposer.ngram import NGramProposer

            resolved = NGramProposer(config.ngram, config.max_draft_tokens)
        else:
            raise ValueError(f"speculative mode {mode.value!r} needs an explicit proposer")
        if resolved.capabilities.mode is not mode:
            raise ValueError("proposer mode disagrees with the configured mode")
        if not resolved.capabilities.host_drafts:
            raise ValueError("this build verifies host drafts only")
        self.config = config
        self.mode = mode
        self.proposer: SpeculativeProposer = resolved
        self.policy = build_policy(config.policy, config.max_draft_tokens)
        self.metrics = SpeculativeMetrics(config.max_draft_tokens)
        self.max_model_len = max_model_len
        self.draft_context_limit = draft_context_limit
        self.custom_sampling_ops = custom_sampling_ops
        self._stop_token_ids = stop_token_ids
        self.closed = False

    # -- planning ---------------------------------------------------------

    def propose(
        self, candidates: Sequence[DecodeCandidate], *, step_hint: int = 0
    ) -> SpeculationProposal:
        """Draft plans for admitted decode slices, before budget clamping.

        Pure with respect to requests: only proposer caches of committed tokens
        and policy counters change.
        """
        if self.closed:
            raise RuntimeError("speculative coordinator is closed")
        disabled: Counter[SpecDisableReason] = Counter()
        decodes = [c for c in candidates if c.scheduled.phase is Phase.DECODE]
        if not decodes:
            return SpeculationProposal((), disabled)
        decision = self.policy.choose_k(
            PolicyContext(
                batch_size=len(decodes),
                max_draft_tokens=self.config.max_draft_tokens,
                step_id=step_hint,
            )
        )
        if decision.k == 0:
            assert decision.reason is not None
            disabled[decision.reason] += len(decodes)
            return SpeculationProposal((), disabled)
        eligible: list[tuple[DecodeCandidate, int]] = []
        requests: list[ProposalRequest] = []
        for candidate in decodes:
            reason = request_disable_reason(
                candidate.request,
                acceptance=self.config.acceptance,
                custom_sampling_ops=self.custom_sampling_ops,
            )
            if reason is not None:
                disabled[reason] += 1
                continue
            bound = draft_bound(
                candidate.scheduled,
                candidate.value,
                max_draft_tokens=decision.k,
                max_model_len=self.max_model_len,
                draft_context_limit=self.draft_context_limit,
            )
            if bound.reason is not None:
                disabled[bound.reason] += 1
                continue
            request = candidate.request
            eligible.append((candidate, bound.limit))
            requests.append(
                ProposalRequest(
                    request_id=candidate.scheduled.request_id,
                    sequence_epoch=candidate.scheduled.sequence_epoch,
                    token_ids=candidate.value.known_tokens,
                    max_draft_tokens=bound.limit,
                    # Shared proposer state never crosses tenant or salt.
                    namespace=f"{request.tenant_id}\x00{request.cache.cache_salt or ''}",
                    stop_token_ids=(
                        frozenset()
                        if self._stop_token_ids is None
                        else self._stop_token_ids(candidate.scheduled.request_id)
                    ),
                )
            )
        if not requests:
            return SpeculationProposal((), disabled, decision.k)
        started = time.perf_counter_ns()
        batch = self.proposer.propose(
            requests,
            max_draft_tokens=decision.k,
            runtime_state=ProposerRuntimeState(batch_size=len(decodes), step_id=step_hint),
        )
        elapsed = time.perf_counter_ns() - started
        step_cap = self.config.max_draft_tokens_per_step
        used = 0
        plans: list[SpeculativeSlicePlan] = []
        for candidate, limit in eligible:
            proposal = batch.get(candidate.scheduled.request_id)
            if proposal is None or not proposal.length:
                disabled[SpecDisableReason.NO_PROPOSAL] += 1
                continue
            k = min(proposal.length, limit)
            if step_cap is not None:
                k = min(k, step_cap - used)
                if k <= 0:
                    disabled[SpecDisableReason.STEP_DRAFT_CAP] += 1
                    continue
            used += k
            plans.append(
                SpeculativeSlicePlan(
                    slice_index=candidate.slice_index,
                    request_id=candidate.scheduled.request_id,
                    mode=self.mode,
                    requested_k=decision.k,
                    effective_k=k,
                    draft_token_ids=proposal.token_ids[:k],
                    reserve_tokens=candidate.scheduled.query_count + k,
                    verify_tokens=candidate.scheduled.query_count + k,
                )
            )
        return SpeculationProposal(tuple(plans), disabled, decision.k, elapsed)

    def record_adopted(
        self,
        step: BatchStepPlan,
        disabled: Counter[SpecDisableReason],
        *,
        draft_latency_ns: int = 0,
    ) -> None:
        """Account one adopted step (planning attempts that never ran are not counted)."""
        if not any(s.phase is Phase.DECODE for s in step.slices):
            return
        self.metrics.record_plan(
            mode=self.mode,
            request_ids=tuple(plan.request_id for plan in step.speculative),
            draft_counts=tuple(plan.effective_k for plan in step.speculative),
            disabled=disabled,
            draft_latency_ns=draft_latency_ns,
        )

    # -- feedback ---------------------------------------------------------

    def observe(self, step: BatchStepPlan, result: CompletionResult) -> None:
        """Learn from one settled step; only committed outcomes are used."""
        decode_rows = sum(1 for s in step.slices if s.phase is Phase.DECODE)
        if not decode_rows:
            return
        from ayaka.executor.ticket import TerminalStatus

        if result.status is not TerminalStatus.SUCCEEDED:
            return
        elapsed = (
            None
            if result.submitted_ns is None or result.terminal_ns is None
            else max(0, result.terminal_ns - result.submitted_ns)
        )
        verifications = result.speculative
        self.metrics.record_step(
            batch_size=decode_rows,
            verifications=verifications,
            decode_rows=decode_rows,
            elapsed_ns=elapsed,
            settlement_ns=result.settlement_ns,
        )
        self.policy.observe(
            StepObservation(
                batch_size=decode_rows,
                k=max((plan.requested_k for plan in step.speculative), default=0),
                verifications=verifications,
                published_tokens=len(result.published),
                decode_rows=decode_rows,
                elapsed_ns=elapsed,
            )
        )
        for verification in verifications:
            self.proposer.update_after_verify(verification)

    def release_request(self, request_id: str) -> None:
        """Drop per-request proposer state; idempotent."""
        self.proposer.release_request(request_id)
        self.metrics.release(request_id)

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "certification": self.mode.capabilities.certification.value,
            "max_draft_tokens": self.config.max_draft_tokens,
            "acceptance": self.config.acceptance.value,
            "policy": self.policy.snapshot(),
            "proposer": self.proposer.stats(),
            **self.metrics.snapshot(),
        }

    def close(self) -> None:
        if not self.closed:
            self.proposer.close()
            self.closed = True
