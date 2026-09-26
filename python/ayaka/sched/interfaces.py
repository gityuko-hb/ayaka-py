"""Shared scheduler/runtime contracts.

The scheduler owns logical ordering and immutable per-step decisions. Physical
KV/page ownership remains authoritative in ``StepRuntime.prepare``.  Keeping
these protocols policy-free prevents eager/reference and continuous/production
schedulers from accidentally acquiring memory-manager responsibilities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ayaka.executor.ticket import ExecutionTicket
from ayaka.handles import SequenceHandle
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.sched.plan import BatchStepPlan, PreparedStep

__all__ = [
    "AdmissionAdvisor",
    "AdmissionCallback",
    "ChunkCapAdvisor",
    "DeadlineExceededError",
    "OverloadedError",
    "PrefixHintProvider",
    "PreemptionCallback",
    "PreemptionController",
    "SequenceAllocator",
    "RequestPreparer",
    "StepPrepareError",
    "StepRuntime",
]


class OverloadedError(RuntimeError):
    """Admission backpressure: a request or queue ceiling is already reached."""


class DeadlineExceededError(RuntimeError):
    """Admission refused because the request deadline already elapsed."""


class StepPrepareError(RuntimeError):
    """Physical step preparation failed.

    ``transient=True`` means another scheduling choice may succeed without
    declaring the request invalid.  ``request_id`` may identify a request whose
    reservation can never satisfy configured engine limits.
    """

    def __init__(
        self,
        message: str,
        *,
        transient: bool,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.transient = transient
        self.request_id = request_id


class SequenceAllocator(Protocol):
    """Sequence-arena ownership used at admission and terminal release."""

    def create(self, request_id: str) -> SequenceHandle: ...

    def release(self, sequence: SequenceHandle) -> None: ...

    def advance_epoch(self, completed_steps: int) -> int: ...


class StepRuntime(Protocol):
    """Authoritative physical prepare/adopt/cancel boundary.

    ``prepare`` must validate the immutable logical plan against the *current*
    KV and physical-memory state.  A prefix hint, token budget, or admission
    precheck never proves ownership.
    """

    def prepare(self, step: BatchStepPlan) -> PreparedStep: ...

    def adopt(self, prepared: PreparedStep) -> ExecutionTicket: ...

    def cancel(self, ticket: ExecutionTicket, reason: str) -> None: ...


class RequestPreparer(Protocol):
    """Acquire reusable state or rebind evicted state before taking a snapshot.

    Return False while the request is not ready. Implementations must leave
    lifecycle and physical state consistent even when the batch is not adopted.
    """

    def prepare_request(self, lifecycle: RequestLifecycle) -> bool: ...


class PrefixHintProvider(Protocol):
    """Cheap non-owning prefix/cache locality estimate.

    The returned count is ranking information only.  It must not advance
    ``computed_tokens`` or mutate a page table.
    """

    def estimate_cached_tokens(self, lifecycle: RequestLifecycle) -> int: ...


class AdmissionAdvisor(Protocol):
    """Optional physical-capacity precheck.

    Implementations may account for full-ISL reservation, watermarks,
    grouped/hybrid KV, COW, host restore, SWA/Mamba state, or other physical
    constraints.  ``StepRuntime.prepare`` remains authoritative.
    """

    def can_admit(
        self,
        lifecycle: RequestLifecycle,
        *,
        full_prompt_remaining: int,
        scheduled_prompt_tokens: int,
    ) -> bool: ...


class ChunkCapAdvisor(Protocol):
    """Advisory prefill-chunk ceiling from the current capacity reading.

    Returned tokens are a bound, never a reservation, and must not exceed the
    resolved prefill chunk cap.  ``None`` means no opinion.  The scheduler
    applies its own hysteresis and ``StepRuntime.prepare`` stays authoritative.
    """

    def prefill_chunk_hint(self) -> int | None: ...


class PreemptionController(Protocol):
    """Bridge to authoritative KV/runtime preemption.

    On success, the controller must leave request lifecycle and physical state
    self-consistent for a later ``snapshot()``.  For recompute mode that usually
    means dropping unretained KV and resetting the lifecycle's computed prefix;
    for swap mode it means publishing a restorable host-backed state.
    """

    def preempt(
        self,
        lifecycle: RequestLifecycle,
        sequence: SequenceHandle,
        *,
        mode: str,
    ) -> bool: ...


AdmissionCallback = Callable[[RequestLifecycle, int, int], bool]
PreemptionCallback = Callable[[RequestLifecycle, SequenceHandle, str], bool]
