"""Shared scheduler lifecycle machinery.

``SchedulerCore`` deliberately does *not* decide which request runs next.  It
centralizes request admission, sequence ownership, abort settlement, reports,
and common inspection so EagerScheduler and ContinuousScheduler do not drift in
lifecycle semantics.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from ayaka.configs.base import ConfigError
from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.executor.ticket import ExecutionTicket
from ayaka.handles import SequenceHandle
from ayaka.plan import SamplingPlan
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.request.schema import Request
from ayaka.sampling.logprobs import LogprobMode
from ayaka.sched.base import BaseScheduler
from ayaka.sched.interfaces import OverloadedError, SequenceAllocator, StepRuntime
from ayaka.sched.outcome import FinishReason, RequestOutcome, RequestReport, SchedulerReport
from ayaka.sched.plan import PromptLogprobSlicePlan, RequestStepInput
from ayaka.sched.policy import QueueEntry

if TYPE_CHECKING:
    from ayaka.sampling.engine import SamplingCoordinator
    from ayaka.sched.plan import ScheduledSlice

__all__ = ["SchedulerCore"]


class SchedulerCore(BaseScheduler):
    """Policy-free scheduler infrastructure shared by all implementations."""

    def __init__(
        self,
        plan: ResolvedSchedulerPlan,
        requests: LifecycleManager,
        runtime: StepRuntime,
        allocator: SequenceAllocator,
        *,
        text_stops: bool = False,
        clock: Callable[[], int] | None = None,
        sampling: SamplingCoordinator | None = None,
    ) -> None:
        if not isinstance(plan, ResolvedSchedulerPlan):
            raise TypeError("plan must be ResolvedSchedulerPlan")
        if not isinstance(requests, LifecycleManager):
            raise TypeError("requests must be LifecycleManager")
        if sampling is not None and plan.max_num_requests is not None:
            if sampling.max_batch_size < plan.max_num_requests:
                raise ValueError(
                    f"sampling capacity {sampling.max_batch_size} < max_num_requests "
                    f"{plan.max_num_requests}; slots are held for the whole admitted "
                    "lifetime, not just the running set"
                )

        self._plan = plan
        self._requests = requests
        self._runtime = runtime
        self._allocator = allocator
        self._text_stops = text_stops
        self._clock = clock or time.monotonic_ns
        self._sampling = sampling

        self._waiting: list[QueueEntry] = []
        self._prefilling: dict[str, RequestLifecycle] = {}
        self._running: dict[str, RequestLifecycle] = {}
        self._sequences: dict[str, SequenceHandle] = {}
        self._pending: list[RequestReport] = []

        self._inflight: ExecutionTicket | None = None
        self._inflight_request_ids: frozenset[str] = frozenset()
        self._abort_pending: set[str] = set()
        self._resource_blocked = False

        self._next_ordinal = 0
        self._ordinals: dict[str, int] = {}
        self._round = 0
        self._step_id = -1
        self._report_step_id = -1

    # ------------------------------------------------------------------
    # Base lifecycle operations
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> RequestLifecycle:
        """Validate, bind a sequence, and enqueue the request."""
        self._plan.validate_request(request)
        self._validate_request_features(request)
        self._check_capacity()

        # Claim the sampling slot before the lifecycle so a capacity failure
        # leaves no half-registered request behind. ``create`` repeats the
        # duplicate check; doing it first keeps the slot accounting clean.
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        if self._sampling is not None:
            self._sampling.add(
                request.request_id,
                request.sampling,
                request.prompt_token_ids,
                request_index=self._next_ordinal,
            )

        lifecycle = self._requests.create(request)
        self._requests.advance_to_queue(lifecycle.request_id)
        sequence = self._allocator.create(lifecycle.request_id)
        lifecycle.bind_sequence(sequence, state_version=0, computed_tokens=0)
        self._sequences[lifecycle.request_id] = sequence
        self._waiting.append(self._new_queue_entry(lifecycle, sequence))
        self._queue_report(
            RequestReport(
                lifecycle.request_id,
                RequestOutcome.ADMITTED,
                lifecycle.sequence_epoch,
            )
        )
        return lifecycle

    def _validate_request_features(self, request: Request) -> None:
        params = request.sampling
        if params.n != 1:
            raise ConfigError(
                "request.sampling",
                "UNSUPPORTED_SAMPLING",
                "scheduler core supports n == 1 only; parallel sampling is not ported",
            )
        if not params.is_greedy and self._sampling is None:
            raise ConfigError(
                "request.sampling",
                "UNSUPPORTED_SAMPLING",
                "non-greedy sampling requires a wired SamplingCoordinator",
            )
        if params.prompt_logprobs is not None and self._sampling is None:
            raise ConfigError(
                "request.sampling.prompt_logprobs",
                "UNSUPPORTED_PROMPT_LOGPROBS",
                "prompt logprobs require a wired SamplingCoordinator",
            )
        if params.logprob_mode is LogprobMode.SAMPLING and params.logprobs is None:
            raise ConfigError(
                "request.sampling.logprob_mode",
                "INVALID_LOGPROB_MODE",
                "logprob_mode=sampling needs logprobs to be enabled",
            )
        if params.logprobs is not None and self._sampling is None:
            raise ConfigError(
                "request.sampling.logprobs",
                "UNSUPPORTED_LOGPROBS",
                "generation logprobs require a wired SamplingCoordinator",
            )
        if request.stop.stop_strings and not self._text_stops:
            raise ConfigError(
                "request.stop.stop_strings",
                "UNSUPPORTED_STOP",
                "text stop strings require a tokenizer-backed output owner",
            )

    def abort(self, request_id: str) -> bool:
        """Cancel one request without poisoning unrelated co-batched requests.

        If the request is already in an adopted mixed batch, only its cancellation
        token is marked.  The whole execution ticket is cancelled only when every
        request in that ticket is pending abort.  Sequence release/reporting is
        deferred until the ticket settles, preserving lease and state-version
        safety.
        """
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or lifecycle.is_terminal:
            return False
        if request_id in self._abort_pending:
            return False

        lifecycle.token.cancel("aborted by scheduler")
        self._abort_pending.add(request_id)

        if self._owns_inflight(request_id):
            if (
                self._inflight is not None
                and self._inflight_request_ids
                and self._inflight_request_ids.issubset(self._abort_pending)
            ):
                self._runtime.cancel(self._inflight, "all requests in ticket aborted")
            return True

        self._finalize_abort(request_id)
        return True

    @property
    def has_unfinished(self) -> bool:
        return self._requests.num_active > 0

    # schedule() and update_from_output() remain algorithm-specific.
    @abstractmethod
    def schedule(self) -> ExecutionTicket | None: ...

    @abstractmethod
    def update_from_output(self, output) -> None: ...

    # ------------------------------------------------------------------
    # Terminal/reporting operations shared by schedulers
    # ------------------------------------------------------------------

    def finish_request(
        self,
        request_id: str,
        reason: FinishReason,
        detail: str = "",
    ) -> SequenceHandle | None:
        lifecycle = self._requests.find(request_id)
        if (
            lifecycle is None
            or lifecycle.is_terminal
            or lifecycle.inflight_slice is not None
            or self._owns_inflight(request_id)
        ):
            return None
        self._drop(request_id)
        self._abort_pending.discard(request_id)
        if self._sampling is not None:
            self._sampling.release(request_id)
        self._queue_report(
            RequestReport(
                request_id,
                RequestOutcome.FINISHED,
                lifecycle.sequence_epoch,
                finish_reason=reason,
                detail=detail,
            )
        )
        return self._sequences.pop(request_id, None)

    def fail_request(self, request_id: str, detail: str) -> SequenceHandle | None:
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or lifecycle.is_terminal:
            return None
        self._drop(request_id)
        self._abort_pending.discard(request_id)
        if self._sampling is not None:
            self._sampling.release(request_id)
        self._queue_report(
            RequestReport(
                request_id,
                RequestOutcome.FAILED,
                lifecycle.sequence_epoch,
                detail=detail,
            )
        )
        return self._sequences.pop(request_id, None)

    def flush_reports(self) -> bool:
        """Apply accumulated scheduler observations to LifecycleManager."""
        if not self._pending:
            return False
        self._report_step_id += 1
        pending = tuple(self._pending)
        report = SchedulerReport(
            step_id=self._report_step_id,
            reports=pending,
            num_running=len(self._running) + len(self._prefilling),
            num_waiting=len(self._waiting),
            num_preempted_this_step=sum(
                item.outcome in (RequestOutcome.PREEMPTED_RECOMPUTE, RequestOutcome.PREEMPTED_SWAP)
                for item in pending
            ),
        )
        self._pending.clear()
        self._requests.apply(report)
        return True

    # ------------------------------------------------------------------
    # Inflight bookkeeping
    # ------------------------------------------------------------------

    def _set_inflight(self, ticket: ExecutionTicket, request_ids: Iterable[str]) -> None:
        ids = frozenset(request_ids)
        if not ids:
            raise ValueError("an adopted ticket must own at least one request")
        self._inflight = ticket
        self._inflight_request_ids = ids

    def _clear_inflight(self, ticket_id: object) -> frozenset[str] | None:
        if self._inflight is None or self._inflight.id != ticket_id:
            return None
        ids = self._inflight_request_ids
        self._inflight = None
        self._inflight_request_ids = frozenset()
        return ids

    def _owns_inflight(self, request_id: str) -> bool:
        return request_id in self._inflight_request_ids

    def _finalize_settled_aborts(self, request_ids: Iterable[str]) -> None:
        for request_id in tuple(request_ids):
            if request_id in self._abort_pending:
                self._finalize_abort(request_id)

    def _finalize_abort(self, request_id: str) -> None:
        lifecycle = self._requests.find(request_id)
        self._drop(request_id)
        if self._sampling is not None:
            self._sampling.release(request_id)
        if lifecycle is not None and not lifecycle.is_terminal:
            self._queue_report(
                RequestReport(
                    request_id,
                    RequestOutcome.ABORTED,
                    lifecycle.sequence_epoch,
                    detail="aborted by scheduler",
                )
            )
        sequence = self._sequences.pop(request_id, None)
        if sequence is not None:
            self._allocator.release(sequence)
        self._abort_pending.discard(request_id)

    # ------------------------------------------------------------------
    # Shared queue/state helpers
    # ------------------------------------------------------------------

    def _new_queue_entry(
        self,
        lifecycle: RequestLifecycle,
        sequence: SequenceHandle | None,
    ) -> QueueEntry:
        ordinal = self._ordinals.get(lifecycle.request_id)
        if ordinal is None:
            ordinal = self._next_ordinal
            self._ordinals[lifecycle.request_id] = ordinal
            self._next_ordinal += 1
        return QueueEntry(
            lifecycle=lifecycle,
            ordinal=ordinal,
            ready_round=self._round,
            ready_ns=self._clock(),
            sequence=sequence,
        )

    def _requeue(self, lifecycle: RequestLifecycle) -> None:
        if lifecycle.is_terminal or lifecycle.token.is_cancelled:
            return
        if any(entry.lifecycle.request_id == lifecycle.request_id for entry in self._waiting):
            return
        self._waiting.append(
            self._new_queue_entry(lifecycle, self._sequences.get(lifecycle.request_id))
        )

    def _queue_report(self, report: RequestReport) -> None:
        """Keep at most one pending observation per request/report step."""
        self._pending = [item for item in self._pending if item.request_id != report.request_id]
        self._pending.append(report)

    def _check_capacity(self) -> None:
        cap = self._plan.max_num_requests
        if cap is not None and self._requests.num_active >= cap:
            raise OverloadedError(f"max_num_requests={cap} reached")
        queued = self._plan.max_queued_requests
        if queued is not None and len(self._waiting) >= queued:
            raise OverloadedError(f"max_queued_requests={queued} reached")

    def _drop_waiting(self, request_id: str) -> None:
        if self._waiting:
            self._waiting = [
                entry for entry in self._waiting if entry.lifecycle.request_id != request_id
            ]

    def _drop(self, request_id: str) -> None:
        self._prefilling.pop(request_id, None)
        self._running.pop(request_id, None)
        self._drop_waiting(request_id)

    def _seq_cap(self) -> int:
        return min(self._plan.max_num_seqs, self._plan.capabilities.max_num_seqs)

    def _sampling_plan(self, slices: Sequence[ScheduledSlice]) -> SamplingPlan:
        """Freeze this step's sampling shape before any forward work exists.

        Without a coordinator the engine keeps the greedy contract: the
        executor may take the argmax fast path. With one, ``plan_for`` pins the
        packed active-row map and reads ``all_greedy``/``any_penalty`` from host
        staging, so no device sync is needed to pick the kernel variant.
        """
        if self._sampling is None:
            num_rows = sum(1 for scheduled in slices if scheduled.sample_last_query)
            return SamplingPlan(num_rows=num_rows, all_greedy=True)
        return self._sampling.plan_for(slices)

    def _prompt_logprob_plan(
        self, slices: Sequence[ScheduledSlice], inputs: Sequence[RequestStepInput]
    ) -> tuple[PromptLogprobSlicePlan, ...]:
        """Resolve which prompt positions each slice's forward can score.

        Scheduler-owned decision (the invariant boundary): position ``p`` is
        scored through hidden row ``p - 1``, so only positions whose
        predecessor is produced by this step are planned. Positions inside the
        prefix-cached region ``[0, num_cached_tokens)`` and position 0 are
        excluded — they surface downstream as no-value markers and are never
        re-forwarded just for reporting.
        """
        if self._sampling is None:
            return ()
        entries: list[PromptLogprobSlicePlan] = []
        for index, (scheduled, value) in enumerate(zip(slices, inputs, strict=True)):
            lifecycle = self._requests.find(scheduled.request_id)
            if lifecycle is None:
                continue
            params = lifecycle.request.sampling
            if params.prompt_logprobs is None:
                continue
            cached = lifecycle.machine.num_cached_tokens
            lo = max(scheduled.query_start, cached + 1, 1)
            hi = min(scheduled.query_end, value.prompt_tokens)
            positions = tuple(range(lo, hi))
            if positions:
                entries.append(
                    PromptLogprobSlicePlan(
                        slice_index=index, k=params.prompt_logprobs, positions=positions
                    )
                )
        return tuple(entries)

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def plan(self) -> ResolvedSchedulerPlan:
        return self._plan

    @property
    def inflight_ticket(self) -> ExecutionTicket | None:
        return self._inflight

    @property
    def resource_blocked(self) -> bool:
        return self._resource_blocked

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running) + len(self._prefilling)

    def sequence_for(self, request_id: str) -> SequenceHandle | None:
        return self._sequences.get(request_id)
