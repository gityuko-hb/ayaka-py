"""Scheduler configuration and execution budget policies for continuous batching.

This module defines:

* :class:`SchedulerConfig` — immutable limits governing token budgets, prefill
  chunking, step iteration budgets, aging priority bypass rules, and hierarchical
  memory tier allocations.

Data flow::

    Incoming Requests
        │
        ▼  RequestLifecycle
    Continuous Batching Scheduler (guided by SchedulerConfig)
        │
        ├─ max_tokens / prefill_chunk ──→ BatchStepPlan token slice sizing
        ├─ max_batch_requests / max_inflight ──→ Concurrency control
        ├─ max_bypass                 ──→ Starvation prevention
        ├─ iteration_ns / retry_ns    ──→ Time budgets and backoff
        └─ tier_limits / transfer_bytes ──→ Ledger memory allocation & transfers
        │
        ▼
    Executor Step Forward Execution
"""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.configs.base import ConfigMixin
from ayaka.types import MemoryTier
from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class SchedulerConfig(ConfigMixin):
    """Engine-thread policy limits and execution budgets for continuous batching.

    Times are absolute monotonic nanoseconds. ``iteration_ns`` limits predicted
    execution time, not a promised hard deadline. Memory is charged by the shared
    ledger during reversible preparation; an optional lower tier limit is checked
    before adoption. Transfer credits cover metadata, model input/output copies,
    and KV-cache copies through ticket retirement. ``max_bypass`` bounds successful
    scheduling opportunities before an older runnable request takes precedence over
    ordinary decode priority.

    Attributes:
        max_tokens: Maximum number of active tokens (prefill chunk tokens plus
            decode tokens) scheduled in a single engine step forward execution.
            Default is ``128``.
        prefill_chunk: Maximum prompt tokens processed per chunk for long requests,
            enabling chunked prefill co-scheduled alongside decode tokens.
            Default is ``64``.
        max_batch_requests: Maximum number of concurrent active requests permitted
            in a single batched step. Default is ``16``.
        max_requests: Global maximum request queue capacity before incoming requests
            are rejected with backpressure. Default is ``256``.
        max_inflight: Maximum number of pending unretired step execution tickets
            dispatched to the execution worker. Default is ``1``.
        max_bypass: Aging threshold. If a runnable request has been bypassed by
            higher-priority decode requests for ``max_bypass`` consecutive rounds,
            it is granted immediate service to prevent starvation. Default is ``8``.
        iteration_ns: Target execution time budget in monotonic nanoseconds for
            a single engine iteration (default 1 second = ``1_000_000_000`` ns).
            Limits predicted execution duration to guarantee predictable step cadence.
        retry_ns: Monotonic nanoseconds to backoff before retrying scheduling when
            resources or memory budgets are temporarily exhausted (default 1 ms =
            ``1_000_000`` ns).
        transfer_bytes: Maximum byte budget allocated for asynchronous KV-cache and
            tensor transfers per step (default 64 MiB = ``64 << 20``).
        tier_limits: Explicit per-tier memory allocation caps represented as
            immutable pairs of ``(MemoryTier, byte_limit)``. Only allocatable
            tiers (:term:`DEVICE` through :term:`DISK`) may be configured.
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
        """Validate scheduler parameters and memory tier invariants."""
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
