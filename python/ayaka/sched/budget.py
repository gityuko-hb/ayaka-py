"""Bounded scheduler budgets and a shape-sensitive, online latency estimate."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from ayaka.configs.scheduler import (
    PreemptionMode,
    ResolvedSchedulerPlan,
    SchedulerCapabilities,
    SchedulerConfig,
    SchedulingPolicy,
    scheduler_config_from_dict,
)
from ayaka.sched.plan import BatchStepPlan
from ayaka.utils.validation import require_int

__all__ = [
    "PreemptionMode",
    "ResolvedSchedulerPlan",
    "SchedulerCapabilities",
    "SchedulerConfig",
    "SchedulingPolicy",
    "TimeEstimator",
    "scheduler_config_from_dict",
]


@dataclass(slots=True)
class BatchBudget:
    """Mutable, per-iteration compute/sequence budget.

    The budget is expressed in *logical* query tokens but every admission check
    is validated through ``physical_token_slots`` so backend padding/alignment is
    never accidentally ignored.
    """

    max_physical_tokens: int
    max_sequences: int
    capabilities: SchedulerCapabilities
    logical_tokens: int = 0
    sequences: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    @classmethod
    def from_plan(cls, plan: ResolvedSchedulerPlan) -> BatchBudget:
        cap = plan.capabilities
        max_tokens = min(
            plan.max_num_scheduled_tokens,
            plan.execution.compute.max_num_batched_tokens,
        )
        max_sequences = min(plan.max_num_seqs, cap.max_num_seqs)
        return cls(max_tokens, max_sequences, cap)

    @property
    def physical_tokens(self) -> int:
        return self.capabilities.physical_token_slots(self.logical_tokens)

    @property
    def remaining_sequences(self) -> int:
        return max(0, self.max_sequences - self.sequences)

    def can_add(self, tokens: int, *, new_sequence: bool = True) -> bool:
        require_int(tokens, "tokens", minimum=1)
        seqs = self.sequences + int(new_sequence)
        if seqs > self.max_sequences:
            return False
        physical = self.capabilities.physical_token_slots(self.logical_tokens + tokens)
        return physical <= self.max_physical_tokens

    def largest_fittable(self, requested: int, *, new_sequence: bool = True) -> int:
        """Largest positive logical token count that still fits, or zero."""
        require_int(requested, "requested", minimum=1)
        if self.remaining_sequences <= 0 and new_sequence:
            return 0
        lo, hi = 0, requested
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.can_add(mid, new_sequence=new_sequence):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def consume(self, tokens: int, *, phase: str, new_sequence: bool = True) -> None:
        if not self.can_add(tokens, new_sequence=new_sequence):
            raise ValueError("scheduler budget exceeded")
        self.logical_tokens += tokens
        self.sequences += int(new_sequence)
        if phase == "prefill":
            self.prefill_tokens += tokens
        elif phase == "decode":
            self.decode_tokens += tokens
        else:
            raise ValueError(f"unknown phase: {phase}")


class TimeEstimator:
    """EWMA nanoseconds per model work unit, calibrated per runner instance.

    Work includes dense projections, attention over actual context lengths,
    sampling-vocabulary rows, page boundaries and KV transfer bytes. The runner
    fixes geometry, dtype, layout and backend; estimates must not be shared
    between different runners.
    """

    def __init__(self, runner, *, ns_per_unit: float = 0.01):
        if not math.isfinite(ns_per_unit) or ns_per_unit <= 0:
            raise ValueError("ns_per_unit must be finite and positive")
        self.runner = runner
        self.ns_per_unit = ns_per_unit
        self.errors: deque[int] = deque(maxlen=256)
        self.observations = 0

    def units(self, step: BatchStepPlan, *, transfer_bytes: int = 0) -> int:
        c = self.runner.weights.config
        h = c.hidden_size
        layers = getattr(c, "num_hidden_layers", getattr(c, "num_layers", 1))
        intermediate_size = getattr(
            c, "intermediate_size", getattr(c, "moe_intermediate_size", 4 * h)
        )
        num_attention_heads = getattr(c, "num_attention_heads", 1)
        head_dim = getattr(c, "head_dim", None) or (h // num_attention_heads)
        dense = step.num_tokens * layers * (4 * h * h + 3 * h * intermediate_size)
        attention = (
            sum(
                s.query_count * s.query_start + s.query_count * (s.query_count + 1) // 2
                for s in step.slices
            )
            * layers
            * num_attention_heads
            * head_dim
            * 2
        )
        pages = sum(
            (s.query_end + g.page_size - 1) // g.page_size
            for s in step.slices
            for g in self.runner.attention.execution.attention_groups
        )
        logits = len(step.sampling_rows) * h * getattr(c, "vocab_size", 0)
        return max(1, dense + attention + logits + pages * h + transfer_bytes)

    def predict(self, step: BatchStepPlan, *, transfer_bytes: int = 0) -> int:
        return max(1, math.ceil(self.units(step, transfer_bytes=transfer_bytes) * self.ns_per_unit))

    def observe(
        self,
        step: BatchStepPlan,
        elapsed_ns: int,
        predicted_ns: int,
        *,
        transfer_bytes: int = 0,
    ) -> None:
        require_int(elapsed_ns, "elapsed_ns")
        require_int(predicted_ns, "predicted_ns")
        require_int(transfer_bytes, "transfer_bytes")
        self.errors.append(elapsed_ns - predicted_ns)
        ratio = max(1, elapsed_ns) / self.units(step, transfer_bytes=transfer_bytes)
        self.ns_per_unit = 0.8 * self.ns_per_unit + 0.2 * ratio
        self.observations += 1
