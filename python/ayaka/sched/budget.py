"""Bounded scheduler budgets and a shape-sensitive, online latency estimate."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from ayaka.sched.plan import BatchStepPlan
from ayaka.types import MemoryTier
from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Engine-thread policy limits. Times are absolute monotonic nanoseconds.

    ``iteration_ns`` limits predicted execution time, not a promised deadline.
    Memory is charged by the shared ledger during reversible preparation; an
    optional lower tier limit is checked before adoption. Transfer credits cover
    metadata, model input/output copies and KV copies through ticket retirement.
    ``max_bypass`` bounds successful scheduling opportunities before an older
    runnable request takes precedence over ordinary decode priority.
    """

    max_tokens: int = 128
    prefill_chunk: int = 64
    max_batch_requests: int = 16
    max_requests: int = 256
    max_inflight: int = 1
    max_bypass: int = 8
    iteration_ns: int = 1_000_000_000
    retry_ns: int = 1_000_000
    transfer_bytes: int = 64 << 20
    tier_limits: tuple[tuple[MemoryTier, int], ...] = ()

    def __post_init__(self):
        for name in (
            "max_tokens",
            "prefill_chunk",
            "max_batch_requests",
            "max_requests",
            "max_inflight",
            "max_bypass",
            "iteration_ns",
            "retry_ns",
        ):
            require_int(getattr(self, name), name, minimum=1)
        require_int(self.transfer_bytes, "transfer_bytes")
        if type(self.tier_limits) is not tuple:
            raise TypeError("tier_limits must be an immutable tuple")
        if len({tier for tier, _ in self.tier_limits}) != len(self.tier_limits):
            raise ValueError("duplicate memory tier limit")
        for tier, limit in self.tier_limits:
            if not isinstance(tier, MemoryTier) or not tier.allocatable:
                raise ValueError("invalid memory tier")
            require_int(limit, "tier limit")


class TimeEstimator:
    """EWMA nanoseconds per model work unit, calibrated per runner instance.

    Work includes dense projections, attention over actual context lengths,
    sampling-vocabulary rows, page boundaries and KV transfer bytes. The runner
    fixes geometry, dtype, layout and backend; estimates must not be shared
    between different runners. Observations include host-observed completion
    delay. They are deliberately conservative when an engine polls infrequently.
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
