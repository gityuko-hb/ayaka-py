"""Scheduler implementation selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ayaka.configs.role import EngineRole
from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.request.lifecycle import LifecycleManager
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sched.continuous import ContinuousScheduler
from ayaka.sched.eager import EagerScheduler
from ayaka.sched.interfaces import (
    AdmissionAdvisor,
    ChunkCapAdvisor,
    PreemptionController,
    PrefixHintProvider,
    RequestPreparer,
    SequenceAllocator,
    StepRuntime,
)
from ayaka.sched.role import DecodeScheduler, PrefillScheduler

__all__ = ["create_role_scheduler", "create_scheduler"]


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
    capacity_hint: ChunkCapAdvisor | None = None,
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
            capacity_hint=capacity_hint,
            preemption=preemption,
            decode_first=decode_first,
            allow_mixed_batches=allow_mixed_batches,
            prefill_chunk_size=prefill_chunk_size,
            max_bypass=max_bypass,
        )
    raise ValueError(f"unknown scheduler implementation: {kind!r}")


def create_role_scheduler(
    role: EngineRole | str,
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
    capacity_hint: ChunkCapAdvisor | None = None,
    preemption: PreemptionController | None = None,
    prefill_chunk_size: int | None = None,
    max_bypass: int = 64,
) -> EagerScheduler | ContinuousScheduler:
    """Build the scheduler matching a disaggregation role.

    Policy and preemption machinery are shared with the continuous scheduler;
    only the batch formation differs per role.
    """
    if isinstance(role, EngineRole):
        name = role.value
    else:
        name = str(role).strip().lower().replace("-", "_")
    common: dict[str, Any] = {
        "text_stops": text_stops,
        "sampling": sampling,
        "request_preparer": request_preparer,
        "clock": clock,
    }
    if name in {"auto", ""}:
        return create_scheduler(
            "continuous",
            plan,
            requests,
            runtime,
            allocator,
            **common,
            prefix_hints=prefix_hints,
            admission=admission,
            capacity_hint=capacity_hint,
            preemption=preemption,
            prefill_chunk_size=prefill_chunk_size,
            max_bypass=max_bypass,
        )
    if name == "prefill":
        return PrefillScheduler(
            plan,
            requests,
            runtime,
            allocator,
            **common,
            prefix_hints=prefix_hints,
            admission=admission,
            capacity_hint=capacity_hint,
            preemption=preemption,
            decode_first=False,
            allow_mixed_batches=False,
            prefill_chunk_size=prefill_chunk_size,
            max_bypass=max_bypass,
        )
    if name == "decode":
        return DecodeScheduler(
            plan,
            requests,
            runtime,
            allocator,
            **common,
            prefix_hints=prefix_hints,
            admission=admission,
            capacity_hint=capacity_hint,
            preemption=preemption,
            decode_first=True,
            allow_mixed_batches=False,
            prefill_chunk_size=prefill_chunk_size,
            max_bypass=max_bypass,
        )
    raise ValueError(f"unknown engine role: {role!r}")
