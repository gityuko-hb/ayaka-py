"""Scheduler token/sequence budgets and online latency estimation."""

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
from ayaka.sched.plan import BatchStepPlan, Phase
from ayaka.utils.validation import require_int

__all__ = [
    "BatchBudget",
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
    """Per-iteration logical budget checked through physical padding rules."""

    max_physical_tokens: int
    max_sequences: int
    capabilities: SchedulerCapabilities
    logical_tokens: int = 0
    sequences: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    @classmethod
    def from_plan(cls, plan: ResolvedSchedulerPlan) -> "BatchBudget":
        return cls(
            max_physical_tokens=min(
                plan.max_num_scheduled_tokens,
                plan.execution.compute.max_num_batched_tokens,
            ),
            max_sequences=min(plan.max_num_seqs, plan.capabilities.max_num_seqs),
            capabilities=plan.capabilities,
        )

    @property
    def physical_tokens(self) -> int:
        return self.capabilities.physical_token_slots(self.logical_tokens)

    @property
    def remaining_sequences(self) -> int:
        return max(0, self.max_sequences - self.sequences)

    def can_add(self, tokens: int, *, new_sequence: bool = True) -> bool:
        require_int(tokens, "tokens", minimum=1)
        if self.sequences + int(new_sequence) > self.max_sequences:
            return False
        slots = self.capabilities.physical_token_slots(self.logical_tokens + tokens)
        return slots <= self.max_physical_tokens

    def largest_fittable(self, requested: int, *, new_sequence: bool = True) -> int:
        """Largest positive logical token count that fits, otherwise zero.

        ``physical_token_slots`` pads to ``token_padding_multiple``, so the
        largest increment is computed directly instead of binary-searching
        ``can_add``.  The remainder term credits the unused tail of the current
        padded block, which a plain ``remaining = max - physical`` subtraction
        would undercount.
        """
        require_int(requested, "requested", minimum=1)
        if new_sequence and self.remaining_sequences <= 0:
            return 0
        multiple = self.capabilities.token_padding_multiple
        capacity = (self.max_physical_tokens // multiple) * multiple
        headroom = capacity - self.physical_tokens + ((-self.logical_tokens) % multiple)
        return min(requested, max(0, headroom))

    def try_consume(
        self,
        tokens: int,
        *,
        phase: Phase | str,
        new_sequence: bool = True,
    ) -> bool:
        """Consume when the budget allows it, reporting refusal without raising."""
        require_int(tokens, "tokens", minimum=1)
        if self.sequences + int(new_sequence) > self.max_sequences:
            return False
        slots = self.capabilities.physical_token_slots(self.logical_tokens + tokens)
        if slots > self.max_physical_tokens:
            return False
        self._record(tokens, phase=phase, new_sequence=new_sequence)
        return True

    def consume(
        self,
        tokens: int,
        *,
        phase: Phase | str,
        new_sequence: bool = True,
    ) -> None:
        if not self.try_consume(tokens, phase=phase, new_sequence=new_sequence):
            raise ValueError("scheduler budget exceeded")

    def _record(self, tokens: int, *, phase: Phase | str, new_sequence: bool) -> None:
        name = getattr(phase, "value", phase)
        if name not in (Phase.PREFILL.value, Phase.DECODE.value):
            raise ValueError(f"unknown scheduling phase: {phase}")
        self.logical_tokens += tokens
        self.sequences += int(new_sequence)
        if name == Phase.PREFILL.value:
            self.prefill_tokens += tokens
        else:
            self.decode_tokens += tokens


class TimeEstimator:
    """EWMA nanoseconds per model-work unit, calibrated per runner instance."""

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
