"""Draft-length policies: how many draft tokens a step may request.

A policy only proposes an upper bound; the coordinator still clamps it per
request by proposal length, output budget, context, step caps and token budget.
Policies are host-only, deterministic given their observations, and never
mutate request state.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ayaka.speculative.config import PolicyConfig, PolicyKind
from ayaka.speculative.metadata import SpeculativeVerification
from ayaka.speculative.plan import SpecDisableReason
from ayaka.utils.validation import require_int

__all__ = [
    "BatchSizeSchedulePolicy",
    "FixedKPolicy",
    "KDecision",
    "PolicyContext",
    "SpeculationPolicy",
    "StepObservation",
    "build_policy",
]


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Facts available before a step's speculative rows are planned.

    Attributes:
        batch_size: Decode requests in the step being planned.
        max_draft_tokens: Configured ceiling.
        step_id: Scheduler step identity (for probe cadence).
    """

    batch_size: int
    max_draft_tokens: int
    step_id: int = 0

    def __post_init__(self) -> None:
        require_int(self.batch_size, "policy batch_size")
        require_int(self.max_draft_tokens, "policy max_draft_tokens", minimum=1)
        require_int(self.step_id, "policy step_id")


@dataclass(frozen=True, slots=True)
class KDecision:
    """Chosen draft length; ``reason`` explains a zero."""

    k: int
    reason: SpecDisableReason | None = None
    probe: bool = False

    def __post_init__(self) -> None:
        require_int(self.k, "decided k")
        if (self.k == 0) != (self.reason is not None):
            raise ValueError("a zero draft length needs a reason, and only a zero has one")


@dataclass(frozen=True, slots=True)
class StepObservation:
    """One settled decode step as seen by a policy.

    ``k`` is the step-level draft length the policy chose (0 for a step planned
    without speculation). ``elapsed_ns`` is the host-observed launch-to-terminal
    time, an upper bound on device time; it is ``None`` when unavailable.
    """

    batch_size: int
    k: int
    verifications: tuple[SpeculativeVerification, ...]
    published_tokens: int
    decode_rows: int
    elapsed_ns: int | None = None

    def __post_init__(self) -> None:
        require_int(self.batch_size, "observation batch_size")
        require_int(self.k, "observation k")
        require_int(self.published_tokens, "observation published_tokens")
        require_int(self.decode_rows, "observation decode_rows")
        if self.elapsed_ns is not None:
            require_int(self.elapsed_ns, "observation elapsed_ns")


@runtime_checkable
class SpeculationPolicy(Protocol):
    def choose_k(self, context: PolicyContext) -> KDecision: ...

    def observe(self, observation: StepObservation) -> None: ...

    def snapshot(self) -> dict[str, object]: ...


class FixedKPolicy:
    """Always the same draft length (the baseline, not a recommended default)."""

    def __init__(self, k: int) -> None:
        require_int(k, "fixed k")
        self.k = k

    def choose_k(self, context: PolicyContext) -> KDecision:
        k = min(self.k, context.max_draft_tokens)
        return KDecision(k) if k else KDecision(0, SpecDisableReason.POLICY)

    def observe(self, observation: StepObservation) -> None:
        del observation

    def snapshot(self) -> dict[str, object]:
        return {"kind": PolicyKind.FIXED.value, "k": self.k}


class BatchSizeSchedulePolicy:
    """Stage A: draft length by decode batch size through inclusive ranges.

    Batch sizes beyond the last range speculate with ``k=0``. The schedule is
    operator data derived from benchmarks; no universal default is shipped.
    """

    def __init__(self, schedule: tuple[tuple[int, int, int], ...]) -> None:
        if not schedule:
            raise ValueError("a batch-size schedule needs at least one range")
        PolicyConfig(kind=PolicyKind.BATCH_SCHEDULE, batch_schedule=schedule)
        self.schedule = schedule
        self._starts = [start for start, _, _ in schedule]

    def k_for(self, batch_size: int) -> int:
        index = bisect.bisect_right(self._starts, batch_size) - 1
        if index < 0:
            return 0
        _, end, k = self.schedule[index]
        return k if batch_size <= end else 0

    def choose_k(self, context: PolicyContext) -> KDecision:
        k = min(self.k_for(context.batch_size), context.max_draft_tokens)
        return KDecision(k) if k else KDecision(0, SpecDisableReason.POLICY)

    def observe(self, observation: StepObservation) -> None:
        del observation

    def snapshot(self) -> dict[str, object]:
        return {"kind": PolicyKind.BATCH_SCHEDULE.value, "schedule": self.schedule}


def build_policy(config: PolicyConfig, max_draft_tokens: int) -> SpeculationPolicy:
    """Instantiate the configured policy."""
    if config.kind is PolicyKind.FIXED:
        return FixedKPolicy(max_draft_tokens if config.fixed_k is None else config.fixed_k)
    if config.kind is PolicyKind.BATCH_SCHEDULE:
        return BatchSizeSchedulePolicy(config.batch_schedule)
    from ayaka.speculative.adaptive import AdaptiveSpeculationPolicy

    return AdaptiveSpeculationPolicy(config, max_draft_tokens)
