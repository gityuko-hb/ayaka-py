"""Speculative decoding counters and a speedup estimate.

Acceptance rate alone cannot say whether speculation helps: a high rate with
expensive verification can still lose to plain decode. Besides acceptance,
these metrics keep KV reservation/commit/reclaim totals, wasted verification
rows, tokens per target forward and host-observed step latency for decode steps
with and without speculation per batch-size bucket. ``estimated_speedup``
compares tokens per second per request in those two populations; it is only as
good as the host latency it is built from (see ``adaptive``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ayaka.speculative.adaptive import StepCostModel
from ayaka.speculative.metadata import SpeculativeVerification
from ayaka.speculative.mode import SpeculativeMode
from ayaka.speculative.plan import SpecDisableReason

__all__ = ["SpeculativeMetrics"]


@dataclass(slots=True)
class _Ewma:
    value: float | None = None
    samples: int = 0

    def add(self, sample: float, alpha: float = 0.2) -> None:
        self.value = sample if self.value is None else (1 - alpha) * self.value + alpha * sample
        self.samples += 1


class SpeculativeMetrics:
    """Engine-lifetime counters; every label is bounded and request-free."""

    def __init__(self, max_draft_tokens: int) -> None:
        self.max_draft_tokens = max_draft_tokens
        self.spec_requests = 0
        self.spec_iterations = 0
        self.decode_iterations = 0
        self.spec_disabled_iterations = 0
        self.disable_reasons: Counter[str] = Counter()
        self.effective_k: Counter[int] = Counter()
        self.per_mode: Counter[str] = Counter()
        self.draft_tokens_proposed = 0
        self.draft_tokens_accepted = 0
        self.verifications = 0
        self.discarded_verifications = 0
        self.published_tokens = 0
        self.rejected_verify_tokens = 0
        self.wasted_verify_tokens = 0
        self.kv_tokens_reserved = 0
        self.kv_tokens_committed = 0
        self.kv_tokens_reclaimed = 0
        self.target_forwards = 0
        self.draft_forwards = 0
        self.draft_graph_hits = 0
        self.verify_graph_hits = 0
        self.accepted_per_position = [0] * (max_draft_tokens + 1)
        self.proposed_per_position = [0] * (max_draft_tokens + 1)
        self.draft_latency_ms = _Ewma()
        self.acceptance_latency_ms = _Ewma()
        self._spec_latency: dict[int, _Ewma] = {}
        self._plain_latency: dict[int, _Ewma] = {}
        self._spec_tokens: dict[int, _Ewma] = {}
        self._live: set[str] = set()

    # -- planning ---------------------------------------------------------

    def record_plan(
        self,
        *,
        mode: SpeculativeMode,
        request_ids: tuple[str, ...],
        draft_counts: tuple[int, ...],
        disabled: Counter[SpecDisableReason],
        draft_latency_ns: int,
    ) -> None:
        """One planned step with ``len(request_ids)`` speculative slices."""
        self.decode_iterations += 1
        for reason, count in disabled.items():
            self.disable_reasons[reason.value] += count
        if not request_ids:
            self.spec_disabled_iterations += 1
            return
        self.spec_iterations += 1
        self.per_mode[mode.value] += len(request_ids)
        for request_id in request_ids:
            if request_id not in self._live:
                self._live.add(request_id)
                self.spec_requests += 1
        for k in draft_counts:
            self.effective_k[k] += 1
        self.draft_latency_ms.add(draft_latency_ns / 1e6)

    def release(self, request_id: str) -> None:
        self._live.discard(request_id)

    # -- settlement -------------------------------------------------------

    def record_step(
        self,
        *,
        batch_size: int,
        verifications: tuple[SpeculativeVerification, ...],
        decode_rows: int,
        elapsed_ns: int | None,
        settlement_ns: int | None = None,
    ) -> None:
        """One settled step that contained decode rows."""
        if decode_rows:
            self.target_forwards += 1
        bucket = StepCostModel.bucket_of(max(1, batch_size))
        for verification in verifications:
            self.verifications += 1
            self.draft_tokens_proposed += verification.proposed
            self.kv_tokens_reserved += verification.reserved_kv
            self.kv_tokens_committed += verification.committed_kv
            self.kv_tokens_reclaimed += verification.reclaimed_kv
            if verification.discarded:
                self.discarded_verifications += 1
                self.wasted_verify_tokens += verification.reserved_kv
                continue
            self.draft_tokens_accepted += verification.accepted
            self.published_tokens += verification.published
            self.rejected_verify_tokens += verification.rejected
            # Rows computed but not published: rejected drafts plus accepted
            # tokens dropped by a stop or the output budget.
            self.wasted_verify_tokens += verification.reserved_kv - verification.published
            for position in range(1, min(verification.proposed, self.max_draft_tokens) + 1):
                self.proposed_per_position[position] += 1
                if verification.accepted >= position:
                    self.accepted_per_position[position] += 1
        live = [v for v in verifications if not v.discarded]
        if elapsed_ns is not None and decode_rows:
            table = self._spec_latency if verifications else self._plain_latency
            table.setdefault(bucket, _Ewma()).add(elapsed_ns / 1e6)
        if live:
            tokens = sum(v.published for v in live) / len(live)
            self._spec_tokens.setdefault(bucket, _Ewma()).add(tokens)
        if settlement_ns is not None and verifications:
            self.acceptance_latency_ms.add(settlement_ns / 1e6)

    # -- reporting --------------------------------------------------------

    def estimated_speedup(self) -> dict[int, float]:
        """Per batch bucket: speculative / plain tokens-per-second per request.

        A bucket appears only once both populations were measured. Values
        below 1.0 mean speculation is slower there.
        """
        result: dict[int, float] = {}
        for bucket, spec in self._spec_latency.items():
            plain = self._plain_latency.get(bucket)
            tokens = self._spec_tokens.get(bucket)
            if (
                plain is None
                or tokens is None
                or spec.value is None
                or plain.value is None
                or tokens.value is None
                or spec.value <= 0
            ):
                continue
            result[bucket] = round((tokens.value / spec.value) / (1.0 / plain.value), 4)
        return result

    def snapshot(self) -> dict[str, object]:
        live = self.verifications - self.discarded_verifications
        acceptance = (
            self.draft_tokens_accepted / self.draft_tokens_proposed
            if self.draft_tokens_proposed
            else None
        )
        return {
            "spec_requests": self.spec_requests,
            "spec_iterations": self.spec_iterations,
            "decode_iterations": self.decode_iterations,
            "spec_disabled_iterations": self.spec_disabled_iterations,
            "spec_disable_reasons": dict(self.disable_reasons),
            "draft_tokens_proposed": self.draft_tokens_proposed,
            "draft_tokens_accepted": self.draft_tokens_accepted,
            "acceptance_rate": acceptance,
            "mean_accepted_length": (1 + self.draft_tokens_accepted / live) if live else None,
            "acceptance_rate_per_position": [
                round(accepted / proposed, 4) if proposed else None
                for accepted, proposed in zip(
                    self.accepted_per_position[1:], self.proposed_per_position[1:], strict=True
                )
            ],
            "effective_k": dict(sorted(self.effective_k.items())),
            "per_mode_slices": dict(self.per_mode),
            "verifications": self.verifications,
            "discarded_verifications": self.discarded_verifications,
            "draft_latency_ms": self.draft_latency_ms.value,
            "verify_latency_ms": {b: e.value for b, e in sorted(self._spec_latency.items())},
            "plain_decode_latency_ms": {b: e.value for b, e in sorted(self._plain_latency.items())},
            "acceptance_latency_ms": self.acceptance_latency_ms.value,
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "rejected_verify_tokens": self.rejected_verify_tokens,
            "wasted_verify_tokens": self.wasted_verify_tokens,
            "kv_tokens_reserved": self.kv_tokens_reserved,
            "kv_tokens_committed": self.kv_tokens_committed,
            "kv_tokens_reclaimed": self.kv_tokens_reclaimed,
            "draft_graph_hits": self.draft_graph_hits,
            "verify_graph_hits": self.verify_graph_hits,
            "tokens_per_target_forward": (self.published_tokens / live) if live else None,
            "estimated_speedup": self.estimated_speedup(),
        }
