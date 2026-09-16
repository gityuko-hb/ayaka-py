"""Scheduler implementation selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.request.lifecycle import LifecycleManager
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sched.continuous import ContinuousScheduler
from ayaka.sched.eager import EagerScheduler
from ayaka.sched.interfaces import (
    AdmissionAdvisor,
    PreemptionController,
    PrefixHintProvider,
    RequestPreparer,
    SequenceAllocator,
    StepRuntime,
)

__all__ = ["create_scheduler"]


def create_scheduler(
    kind: str,
    plan: ResolvedSchedulerPlan,
    requests: LifecycleManager,
    runtime: StepRuntime,
    allocator: SequenceAllocator,
    *,
    text_stops: bool = False,
    sampling: SamplingCoordinator | None = None,
    request_preparer: RequestPreparer | None = None,
    clock: Callable[[], int] | None = None,
    prefix_hints: PrefixHintProvider | None = None,
    admission: AdmissionAdvisor | None = None,
    preemption: PreemptionController | None = None,
    decode_first: bool = True,
    allow_mixed_batches: bool = True,
    prefill_chunk_size: int | None = None,
    max_bypass: int = 64,
) -> EagerScheduler | ContinuousScheduler:
    """Construct the selected scheduler without aliasing implementations."""

    name = kind.strip().lower().replace("-", "_")
    common: dict[str, Any] = {
        "text_stops": text_stops,
        "sampling": sampling,
        "request_preparer": request_preparer,
        "clock": clock,
    }
    if name in {"eager", "reference", "debug"}:
        return EagerScheduler(plan, requests, runtime, allocator, **common)
    if name in {"continuous", "production"}:
        return ContinuousScheduler(
            plan,
            requests,
            runtime,
            allocator,
            **common,
            prefix_hints=prefix_hints,
            admission=admission,
            preemption=preemption,
            decode_first=decode_first,
            allow_mixed_batches=allow_mixed_batches,
            prefill_chunk_size=prefill_chunk_size,
            max_bypass=max_bypass,
        )
    raise ValueError(f"unknown scheduler implementation: {kind!r}")
