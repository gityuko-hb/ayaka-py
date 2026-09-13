"""Immutable configuration contracts for continuous-batching schedulers.

The scheduler has two token limits. ``max_num_batched_tokens`` is the executor
capacity, while ``max_num_scheduled_tokens`` is the smaller or equal budget the
scheduler may issue. Keeping them separate leaves room for executors that append
tokens after scheduling, such as speculative decoding.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.types import MemoryTier
from ayaka.utils.validation import require_int

__all__ = [
    "PreemptionMode",
    "ResolvedSchedulerPlan",
    "SchedulerConfig",
    "SchedulingPolicy",
]


class SchedulingPolicy(enum.StrEnum):
    """Ordering policy applied within decode and prefill work queues."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    LONGEST_PREFIX_MATCH = "longest_prefix_match"
    SHORTEST_REMAINING = "shortest_remaining"


class PreemptionMode(enum.StrEnum):
    """How a scheduler releases KV state when it preempts a request."""

    RECOMPUTE = "recompute"
    SWAP = "swap"


@dataclass(frozen=True, slots=True)
class SchedulerConfig(ConfigMixin):
    """Policy, admission, and execution budgets for one scheduler instance.

    Token and sequence limits bound a single engine iteration. Queue and
    in-flight limits bound host-side pressure. Timing values use monotonic
    nanoseconds. Transfer and tier limits are byte budgets charged through
    ticket retirement.

    ``max_num_partial_prefills`` caps partially processed prompts that may stay
    active together. Scheduler implementations must enforce the resolved value;
    it is valid only with chunked prefill and cannot exceed ``max_num_seqs``.
    """

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_num_scheduled_tokens: int | None = None
    max_queued_requests: int | None = 256
    enable_chunked_prefill: bool = True
    max_num_partial_prefills: int = 1
    long_prefill_token_threshold: int | None = None
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FCFS
    preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE
    priority_preemption: bool = False
    max_decode_steps_per_schedule: int = 1

    max_inflight: int = 1
    max_bypass: int = 8
    iteration_ns: int = 1_000_000_000
    retry_ns: int = 1_000_000
    transfer_bytes: int = 64 << 20
    tier_limits: tuple[tuple[MemoryTier, int], ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid values and contradictory scheduler policies."""
        for name in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_num_partial_prefills",
            "max_decode_steps_per_schedule",
            "max_inflight",
            "max_bypass",
            "iteration_ns",
            "retry_ns",
        ):
            require_int(getattr(self, name), name, minimum=1)

        if self.max_num_scheduled_tokens is not None:
            require_int(self.max_num_scheduled_tokens, "max_num_scheduled_tokens", minimum=1)
            if self.max_num_scheduled_tokens > self.max_num_batched_tokens:
                raise ConfigError(
                    "scheduler.max_num_scheduled_tokens",
                    "SCHEDULED_TOKEN_BUDGET_TOO_LARGE",
                    "scheduled-token budget cannot exceed the executable token budget",
                )
        if self.max_queued_requests is not None:
            require_int(self.max_queued_requests, "max_queued_requests", minimum=1)
        if self.long_prefill_token_threshold is not None:
            require_int(
                self.long_prefill_token_threshold,
                "long_prefill_token_threshold",
                minimum=1,
            )
        if type(self.enable_chunked_prefill) is not bool:
            raise TypeError("enable_chunked_prefill must be bool")
        if type(self.priority_preemption) is not bool:
            raise TypeError("priority_preemption must be bool")
        if not isinstance(self.scheduling_policy, SchedulingPolicy):
            raise TypeError("scheduling_policy must be SchedulingPolicy")
        if not isinstance(self.preemption_mode, PreemptionMode):
            raise TypeError("preemption_mode must be PreemptionMode")
        if self.max_num_partial_prefills > self.max_num_seqs:
            raise ConfigError(
                "scheduler.max_num_partial_prefills",
                "PARTIAL_PREFILL_LIMIT_TOO_LARGE",
                "partial-prefill concurrency cannot exceed the sequence budget",
            )
        if not self.enable_chunked_prefill and self.max_num_partial_prefills != 1:
            raise ConfigError(
                "scheduler.max_num_partial_prefills",
                "PARTIAL_PREFILL_WITHOUT_CHUNKING",
                "partial-prefill concurrency requires chunked prefill",
            )
        if self.priority_preemption and self.scheduling_policy is not SchedulingPolicy.PRIORITY:
            raise ConfigError(
                "scheduler.priority_preemption",
                "PRIORITY_PREEMPTION_WITHOUT_PRIORITY",
                "priority preemption requires priority scheduling",
            )

        require_int(self.transfer_bytes, "transfer_bytes")
        if type(self.tier_limits) is not tuple:
            raise TypeError("tier_limits must be an immutable tuple")
        if len({tier for tier, _ in self.tier_limits}) != len(self.tier_limits):
            raise ValueError("duplicate memory tier limit")
        for tier, limit in self.tier_limits:
            if not isinstance(tier, MemoryTier) or not tier.allocatable:
                raise ValueError("invalid memory tier")
            require_int(limit, "tier limit")

    @property
    def effective_scheduled_token_budget(self) -> int:
        """Return the scheduler issue budget after applying its default."""
        if self.max_num_scheduled_tokens is None:
            return self.max_num_batched_tokens
        return self.max_num_scheduled_tokens

    def effective_long_prefill_threshold(self, max_model_len: int) -> int:
        """Resolve the explicit threshold or four percent of model context."""
        if type(max_model_len) is not int or max_model_len < 1:
            raise ConfigError(
                "model.max_model_len",
                "MODEL_LENGTH_INVALID",
                "max model length must be a positive integer",
            )
        if self.long_prefill_token_threshold is not None:
            return self.long_prefill_token_threshold
        return max(1, int(max_model_len * 0.04))

    def resolve(self, max_model_len: int) -> ResolvedSchedulerPlan:
        """Freeze all derived defaults for runtime and fingerprinting."""
        return ResolvedSchedulerPlan(
            max_model_len=max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_scheduled_tokens=self.effective_scheduled_token_budget,
            max_queued_requests=self.max_queued_requests,
            chunked_prefill=self.enable_chunked_prefill,
            max_num_partial_prefills=self.max_num_partial_prefills,
            long_prefill_token_threshold=self.effective_long_prefill_threshold(max_model_len),
            scheduling_policy=self.scheduling_policy,
            preemption_mode=self.preemption_mode,
            priority_preemption=self.priority_preemption,
            max_decode_steps_per_schedule=self.max_decode_steps_per_schedule,
            max_inflight=self.max_inflight,
            max_bypass=self.max_bypass,
            iteration_ns=self.iteration_ns,
            retry_ns=self.retry_ns,
            transfer_bytes=self.transfer_bytes,
            tier_limits=self.tier_limits,
        )


@dataclass(frozen=True, slots=True)
class ResolvedSchedulerPlan(ConfigMixin):
    """Scheduler settings with every model-dependent default resolved."""

    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_num_scheduled_tokens: int
    max_queued_requests: int | None
    chunked_prefill: bool
    max_num_partial_prefills: int
    long_prefill_token_threshold: int
    scheduling_policy: SchedulingPolicy
    preemption_mode: PreemptionMode
    priority_preemption: bool
    max_decode_steps_per_schedule: int
    max_inflight: int
    max_bypass: int
    iteration_ns: int
    retry_ns: int
    transfer_bytes: int
    tier_limits: tuple[tuple[MemoryTier, int], ...]

    def __post_init__(self) -> None:
        """Defend the public resolved contract against direct invalid construction."""
        for name in (
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_num_scheduled_tokens",
            "max_num_partial_prefills",
            "long_prefill_token_threshold",
            "max_decode_steps_per_schedule",
            "max_inflight",
            "max_bypass",
            "iteration_ns",
            "retry_ns",
        ):
            require_int(getattr(self, name), name, minimum=1)
        if self.max_num_scheduled_tokens > self.max_num_batched_tokens:
            raise ValueError("scheduled-token budget exceeds executable token budget")
        if self.max_queued_requests is not None:
            require_int(self.max_queued_requests, "max_queued_requests", minimum=1)
        if type(self.chunked_prefill) is not bool:
            raise TypeError("chunked_prefill must be bool")
        if type(self.priority_preemption) is not bool:
            raise TypeError("priority_preemption must be bool")
        if not isinstance(self.scheduling_policy, SchedulingPolicy):
            raise TypeError("scheduling_policy must be SchedulingPolicy")
        if not isinstance(self.preemption_mode, PreemptionMode):
            raise TypeError("preemption_mode must be PreemptionMode")
        if self.max_num_partial_prefills > self.max_num_seqs:
            raise ValueError("partial-prefill concurrency exceeds sequence budget")
        if not self.chunked_prefill and self.max_num_partial_prefills != 1:
            raise ValueError("partial-prefill concurrency requires chunked prefill")
        if self.priority_preemption and self.scheduling_policy is not SchedulingPolicy.PRIORITY:
            raise ValueError("priority preemption requires priority scheduling")
        require_int(self.transfer_bytes, "transfer_bytes")
        if type(self.tier_limits) is not tuple:
            raise TypeError("tier_limits must be an immutable tuple")
        if len({tier for tier, _ in self.tier_limits}) != len(self.tier_limits):
            raise ValueError("duplicate memory tier limit")
        for tier, limit in self.tier_limits:
            if not isinstance(tier, MemoryTier) or not tier.allocatable:
                raise ValueError("invalid memory tier")
            require_int(limit, "tier limit")
