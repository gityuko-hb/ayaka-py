"""Feedback-driven draft length (Stages B and C).

Stage B tracks acceptance per draft position with exponential decay, which
yields the expected accepted length for any candidate ``k``. Stage C measures
step cost per batch-size bucket and candidate ``k`` and picks the candidate with
the best estimated tokens per second:

    throughput(k) = (1 + E[accepted | k]) / latency(bucket, k)

A switch needs a relative gain above ``hysteresis``. Every ``probe_interval``
steps one other candidate (including ``k=0`` and larger ``k``) runs once, so a
window that made speculation look unprofitable can never disable it for the
rest of the process — the opposite of a permanent gate.

Latency is the host-observed launch-to-terminal time of a ticket: an upper
bound on device time that includes polling delay. It is adequate for relative
comparison under a stable engine loop, not for absolute performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ayaka.speculative.config import PolicyConfig, PolicyKind
from ayaka.speculative.plan import SpecDisableReason
from ayaka.speculative.policy import (
    BatchSizeSchedulePolicy,
    KDecision,
    PolicyContext,
    StepObservation,
)
from ayaka.utils.validation import require_int

__all__ = ["AcceptanceTracker", "AdaptiveSpeculationPolicy", "StepCostModel"]


class AcceptanceTracker:
    """Decayed per-position draft/accept counts.

    ``rate(i)`` estimates P(accepted >= i | position i was proposed). Positions
    never observed extrapolate the last observed conditional acceptance, or a
    neutral 0.5 before any observation.
    """

    def __init__(self, max_k: int, *, alpha: float) -> None:
        require_int(max_k, "tracker max_k", minimum=1)
        self.max_k = max_k
        self.alpha = float(alpha)
        self.drafted = [0.0] * (max_k + 1)
        self.accepted = [0.0] * (max_k + 1)
        self.samples = 0

    def observe(self, proposed_accepted: list[tuple[int, int]]) -> None:
        if not proposed_accepted:
            return
        keep = 1.0 - self.alpha
        for index in range(1, self.max_k + 1):
            self.drafted[index] *= keep
            self.accepted[index] *= keep
        for proposed, accepted in proposed_accepted:
            for position in range(1, min(proposed, self.max_k) + 1):
                self.drafted[position] += 1.0
                if accepted >= position:
                    self.accepted[position] += 1.0
        self.samples += len(proposed_accepted)

    def rate(self, position: int) -> float:
        rates = self.rates(position)
        return rates[-1] if rates else 1.0

    def rates(self, k: int) -> list[float]:
        """Estimated ``rate(1..k)``, monotonically non-increasing."""
        out: list[float] = []
        previous = 1.0
        conditional = 0.5
        for position in range(1, k + 1):
            drafted = self.drafted[position] if position <= self.max_k else 0.0
            if drafted >= 1e-6:
                current = min(previous, self.accepted[position] / drafted)
                if previous > 0:
                    conditional = current / previous
            else:
                current = previous * conditional
            out.append(current)
            previous = current
        return out

    def expected_accepted(self, k: int) -> float:
        return sum(self.rates(k))


@dataclass(slots=True)
class _Cost:
    latency_ns: float
    samples: int = 1


@dataclass(slots=True)
class _Bucket:
    current_k: int
    steps: int = 0
    probe_cursor: int = 0
    costs: dict[int, _Cost] = field(default_factory=dict)


class StepCostModel:
    """EWMA step latency per ``(batch bucket, k)``."""

    def __init__(self, *, alpha: float) -> None:
        self.alpha = float(alpha)
        self.costs: dict[tuple[int, int], _Cost] = {}

    @staticmethod
    def bucket_of(batch_size: int) -> int:
        """Power-of-two bucket: 1, 2, 4, 8, ..."""
        bucket = 1
        while bucket < batch_size:
            bucket *= 2
        return bucket

    def observe(self, batch_size: int, k: int, elapsed_ns: int) -> None:
        key = (self.bucket_of(batch_size), k)
        cost = self.costs.get(key)
        if cost is None:
            self.costs[key] = _Cost(float(elapsed_ns))
        else:
            cost.latency_ns = (1 - self.alpha) * cost.latency_ns + self.alpha * elapsed_ns
            cost.samples += 1

    def latency(self, batch_size: int, k: int) -> float | None:
        cost = self.costs.get((self.bucket_of(batch_size), k))
        return None if cost is None else cost.latency_ns


class AdaptiveSpeculationPolicy:
    """Choose among ``candidate_k`` per batch bucket with hysteresis and probing."""

    def __init__(self, config: PolicyConfig, max_draft_tokens: int) -> None:
        if config.kind is not PolicyKind.ADAPTIVE:
            raise ValueError("adaptive policy needs PolicyKind.ADAPTIVE")
        config.validate_against(max_draft_tokens)
        self.config = config
        self.candidates = config.candidate_k
        self.acceptance = AcceptanceTracker(max(max(self.candidates), 1), alpha=config.ewma_alpha)
        self.cost = StepCostModel(alpha=config.ewma_alpha)
        self._schedule = (
            BatchSizeSchedulePolicy(config.batch_schedule) if config.batch_schedule else None
        )
        self._initial = max_draft_tokens if config.fixed_k is None else config.fixed_k
        self._buckets: dict[int, _Bucket] = {}
        self.switches = 0
        self.probes = 0

    def _nearest(self, k: int) -> int:
        return min(self.candidates, key=lambda candidate: (abs(candidate - k), candidate))

    def _bucket(self, batch_size: int) -> _Bucket:
        key = StepCostModel.bucket_of(max(1, batch_size))
        bucket = self._buckets.get(key)
        if bucket is None:
            start = self._schedule.k_for(key) if self._schedule is not None else self._initial
            bucket = self._buckets[key] = _Bucket(current_k=self._nearest(start))
        return bucket

    def estimated_throughput(self, batch_size: int, k: int) -> float | None:
        """Tokens per request per second for candidate ``k``, if measured."""
        latency = self.cost.latency(batch_size, k)
        if latency is None or latency <= 0:
            return None
        tokens = 1.0 + (self.acceptance.expected_accepted(k) if k else 0.0)
        return tokens / latency * 1e9

    def choose_k(self, context: PolicyContext) -> KDecision:
        bucket = self._bucket(context.batch_size)
        bucket.steps += 1
        config = self.config
        if bucket.steps > config.warmup_steps:
            if bucket.steps % config.probe_interval == 0 and len(self.candidates) > 1:
                others = [k for k in self.candidates if k != bucket.current_k]
                probe = others[bucket.probe_cursor % len(others)]
                bucket.probe_cursor += 1
                self.probes += 1
                return self._decision(min(probe, context.max_draft_tokens), probe=True)
            if bucket.steps % config.update_interval == 0:
                self._maybe_switch(bucket, context.batch_size)
        return self._decision(min(bucket.current_k, context.max_draft_tokens))

    def _maybe_switch(self, bucket: _Bucket, batch_size: int) -> None:
        current = self.estimated_throughput(batch_size, bucket.current_k)
        best_k, best = bucket.current_k, current
        for k in self.candidates:
            value = self.estimated_throughput(batch_size, k)
            if value is not None and (best is None or value > best):
                best_k, best = k, value
        if best_k == bucket.current_k or best is None:
            return
        if current is None or best > current * (1.0 + self.config.hysteresis):
            bucket.current_k = best_k
            self.switches += 1

    @staticmethod
    def _decision(k: int, *, probe: bool = False) -> KDecision:
        if k:
            return KDecision(k, probe=probe)
        return KDecision(0, SpecDisableReason.POLICY, probe=probe)

    def observe(self, observation: StepObservation) -> None:
        self.acceptance.observe(
            [(v.proposed, v.accepted) for v in observation.verifications if not v.discarded]
        )
        if observation.elapsed_ns is not None and observation.decode_rows:
            self.cost.observe(observation.batch_size, observation.k, observation.elapsed_ns)

    def snapshot(self) -> dict[str, object]:
        return {
            "kind": PolicyKind.ADAPTIVE.value,
            "candidates": self.candidates,
            "current_k": {bucket: state.current_k for bucket, state in self._buckets.items()},
            "expected_accepted": {
                k: round(self.acceptance.expected_accepted(k), 4) for k in self.candidates
            },
            "switches": self.switches,
            "probes": self.probes,
        }
