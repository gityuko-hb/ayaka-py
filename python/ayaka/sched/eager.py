"""Continuous-batching scheduler for Ayaka.

The public name ``EagerScheduler`` is preserved for compatibility, but this is
no longer a whole-prompt/one-phase baseline scheduler. It builds one immutable
host plan per iteration, prioritizes decode by default, uses spare budget for
chunked prefill, supports mixed PREFILL+DECODE batches, and delegates physical
KV truth to the runtime.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from ayaka.configs.base import ConfigError
from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.executor.completion import CompletionResult
from ayaka.executor.ticket import ExecutionTicket, TerminalStatus
from ayaka.handles import SequenceHandle
from ayaka.plan import SamplingPlan
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.request.schema import Request
from ayaka.sched.base import BaseScheduler
from ayaka.sched.budget import BatchBudget
from ayaka.sched.outcome import FinishReason, RequestOutcome, RequestReport, SchedulerReport
from ayaka.sched.plan import BatchStepPlan, KVRequirement, Phase, PreparedStep, ScheduledSlice
from ayaka.sched.policy import QueueEntry, order

__all__ = [
    "AdmissionAdvisor",
    "EagerScheduler",
    "OverloadedError",
    "PrefixHintProvider",
    "PreemptionController",
    "SequenceAllocator",
    "StepPrepareError",
    "StepRuntime",
]


class OverloadedError(RuntimeError):
    """Admission backpressure: a request or queue ceiling is already reached."""


class StepPrepareError(RuntimeError):
    """Physical preparation failed.

    ``transient`` means another scheduling choice may succeed later. A
    non-transient error with ``request_id`` identifies a request that can never
    satisfy the current engine contract.
    """

    def __init__(self, message: str, *, transient: bool, request_id: str | None = None) -> None:
        super().__init__(message)
        self.transient = transient
        self.request_id = request_id


class SequenceAllocator(Protocol):
    def create(self, request_id: str) -> SequenceHandle: ...
    def release(self, sequence: SequenceHandle) -> None: ...
    def advance_epoch(self, completed_steps: int) -> int: ...


class StepRuntime(Protocol):
    """Authoritative physical prepare/adopt/cancel boundary."""

    def prepare(self, step: BatchStepPlan) -> PreparedStep: ...
    def adopt(self, prepared: PreparedStep) -> ExecutionTicket: ...
    def cancel(self, ticket: ExecutionTicket, reason: str) -> None: ...


class PrefixHintProvider(Protocol):
    """Cheap, non-owning prefix affinity hint.

    Returning N does *not* make N tokens computed. Physical cache acquisition
    remains in StepRuntime.prepare().
    """

    def estimate_cached_tokens(self, lifecycle: RequestLifecycle) -> int: ...


class AdmissionAdvisor(Protocol):
    """Optional anti-thrashing admission gate (e.g. full-ISL + watermark)."""

    def can_admit(self, lifecycle: RequestLifecycle, *, full_prompt_remaining: int) -> bool: ...


class PreemptionController(Protocol):
    """Optional bridge to the KV/runtime preemption implementation.

    Implementations must atomically update the request's authoritative KV state
    before returning True. The scheduler only changes queue membership/reporting.
    """

    def preempt(
        self, lifecycle: RequestLifecycle, sequence: SequenceHandle, *, mode: str
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class _InflightSlice:
    request_id: str
    phase: Phase
    query_end: int
    prompt_tokens: int


class EagerScheduler(BaseScheduler):
    """Production-core continuous scheduler.

    Core invariants:
    - Exactly one execution ticket is owned at a time (safe baseline for K01/K04).
    - Decode-first continuous batching by default for stable inter-token latency.
    - Remaining token budget is filled with chunked prefill and may form a MIXED batch.
    - Prefix/cache data used for ranking is advisory; prepare() is authoritative.
    - Transient reservation failure shrinks work instead of globally stalling.
    - Request identity is guarded by sequence_epoch + state_version in BatchStepPlan.
    """

    def __init__(
        self,
        plan: ResolvedSchedulerPlan,
        requests: LifecycleManager,
        runtime: StepRuntime,
        allocator: SequenceAllocator,
        *,
        text_stops: bool = False,
        clock: Callable[[], int] | None = None,
        prefix_hints: PrefixHintProvider | None = None,
        admission: AdmissionAdvisor | None = None,
        preemption: PreemptionController | None = None,
        decode_first: bool = True,
        allow_mixed_batches: bool = True,
        prefill_chunk_size: int | None = None,
        max_bypass: int = 64,
    ) -> None:
        if not isinstance(plan, ResolvedSchedulerPlan):
            raise TypeError("plan must be ResolvedSchedulerPlan")
        if not isinstance(requests, LifecycleManager):
            raise TypeError("requests must be LifecycleManager")
        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if max_bypass <= 0:
            raise ValueError("max_bypass must be positive")

        self._plan = plan
        self._requests = requests
        self._runtime = runtime
        self._allocator = allocator
        self._text_stops = text_stops
        self._clock = clock or time.monotonic_ns
        self._prefix_hints = prefix_hints
        self._admission = admission
        self._preemption = preemption
        self._decode_first = decode_first
        self._allow_mixed_batches = allow_mixed_batches
        self._prefill_chunk_size = prefill_chunk_size
        self._max_bypass = max_bypass

        self._waiting: list[QueueEntry] = []
        self._prefilling: dict[str, RequestLifecycle] = {}
        self._running: dict[str, RequestLifecycle] = {}
        self._sequences: dict[str, SequenceHandle] = {}
        self._pending: list[RequestReport] = []
        self._inflight: ExecutionTicket | None = None
        self._inflight_slices: tuple[_InflightSlice, ...] = ()
        self._last_step: BatchStepPlan | None = None
        self._next_ordinal = 0
        self._ordinals: dict[str, int] = {}
        self._round = 0
        self._step_id = -1
        self._report_step_id = -1
        self._resource_blocked = False
        self._abort_pending: set[str] = set()

    def add_request(self, request: Request) -> RequestLifecycle:
        self._plan.validate_request(request)
        sampling = request.sampling
        # Sampling execution is intentionally still delegated to the existing S1
        # SamplingPlan contract. Remove this gate when publishes a full plan.
        if not sampling.is_greedy or sampling.n != 1:
            raise ConfigError(
                "request.sampling",
                "UNSUPPORTED_SAMPLING",
                "scheduler core currently requires the sampling-plan upgrade for non-greedy/n>1",
            )
        if request.stop.stop_strings and not self._text_stops:
            raise ConfigError(
                "request.stop.stop_strings",
                "UNSUPPORTED_STOP",
                "text stop strings require a tokenizer-backed output owner",
            )
        self._check_capacity()
        lifecycle = self._requests.create(request)
        self._requests.advance_to_queue(lifecycle.request_id)
        sequence = self._allocator.create(lifecycle.request_id)
        lifecycle.bind_sequence(sequence, state_version=0, computed_tokens=0)
        self._sequences[lifecycle.request_id] = sequence
        self._waiting.append(self._new_queue_entry(lifecycle, sequence))
        self._queue_report(
            RequestReport(lifecycle.request_id, RequestOutcome.ADMITTED, lifecycle.sequence_epoch)
        )
        return lifecycle

    def abort(self, request_id: str) -> bool:
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or lifecycle.is_terminal:
            return False
        if self._owns_inflight(request_id) and self._inflight is not None:
            lifecycle.token.cancel("aborted by scheduler")
            self._runtime.cancel(self._inflight, f"aborted {request_id}")
            return True

        lifecycle.token.cancel("aborted by scheduler")
        self._drop(request_id)
        if request_id in self._abort_pending:
            return False
        self._abort_pending.add(request_id)
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
        return True

    def schedule(self) -> ExecutionTicket | None:
        """Build one continuous-batch step and reserve it transactionally."""
        if self._inflight is not None:
            return None

        self._round += 1
        self._refresh_cache_hints()
        for plan in self._candidate_plans():
            ticket = self._prepare_and_adopt(plan)
            if ticket is not None:
                return ticket
        return None

    def update_from_output(self, output: CompletionResult) -> None:
        """Settle queue membership after the execution owner settles the ticket."""
        if self._inflight is not None and self._inflight.id == output.ticket_id:
            self._inflight = None
            inflight = self._inflight_slices
            self._inflight_slices = ()

            if output.status is TerminalStatus.SUCCEEDED:
                for item in inflight:
                    lifecycle = self._requests.find(item.request_id)
                    self._prefilling.pop(item.request_id, None)
                    if lifecycle is None or lifecycle.is_terminal or lifecycle.token.is_cancelled:
                        continue
                    if item.phase is Phase.PREFILL:
                        if item.query_end < item.prompt_tokens:
                            self._requeue(lifecycle)
                        else:
                            self._running[item.request_id] = lifecycle
                    else:
                        self._running[item.request_id] = lifecycle
            else:
                # The completion owner is responsible for failure reports. Drop
                # scheduler membership so stale requests cannot be rescheduled.
                for item in inflight:
                    self._drop(item.request_id)
                    self._sequences.pop(item.request_id, None)

        for request_id in output.ignored_requests:
            self._drop(request_id)
            self._sequences.pop(request_id, None)
            self._abort_pending.discard(request_id)

    @property
    def has_unfinished(self) -> bool:
        return self._requests.num_active > 0

    # -- Scheduling ----------------------------------------------------

    def _candidate_plans(self) -> Iterator[BatchStepPlan]:
        if self._allow_mixed_batches:
            plan = self._build_continuous_batch()
            if plan is not None:
                yield plan
            return

        first, second = (
            (self._build_decode_only, self._build_prefill_only)
            if self._decode_first
            else (self._build_prefill_only, self._build_decode_only)
        )
        for builder in (first, second):
            plan = builder()
            if plan is not None:
                yield plan

    def _build_continuous_batch(self) -> BatchStepPlan | None:
        budget = BatchBudget.from_plan(self._plan)
        slices: list[ScheduledSlice] = []
        inputs = []

        if self._decode_first:
            self._append_decodes(budget, slices, inputs)
            self._append_prefills(budget, slices, inputs)
        else:
            self._append_prefills(budget, slices, inputs)
            self._append_decodes(budget, slices, inputs)

        return self._make_plan(slices, inputs) if slices else None

    def _build_decode_only(self) -> BatchStepPlan | None:
        budget = BatchBudget.from_plan(self._plan)
        slices: list[ScheduledSlice] = []
        inputs = []
        self._append_decodes(budget, slices, inputs)
        return self._make_plan(slices, inputs) if slices else None

    def _build_prefill_only(self) -> BatchStepPlan | None:
        budget = BatchBudget.from_plan(self._plan)
        slices: list[ScheduledSlice] = []
        inputs = []
        self._append_prefills(budget, slices, inputs)
        return self._make_plan(slices, inputs) if slices else None

    def _append_decodes(self, budget: BatchBudget, slices: list, inputs: list) -> None:
        for lifecycle in tuple(self._running.values()):
            if lifecycle.is_terminal or lifecycle.token.is_cancelled:
                continue
            if not budget.can_add(1):
                break
            snapshot = lifecycle.snapshot()
            slices.append(
                ScheduledSlice(
                    lifecycle.request_id,
                    snapshot.sequence_epoch,
                    snapshot.state_version,
                    snapshot.computed_tokens,
                    1,
                    Phase.DECODE,
                    sample_last_query=True,
                )
            )
            inputs.append(snapshot)
            budget.consume(1, phase="decode")

    def _append_prefills(self, budget: BatchBudget, slices: list, inputs: list) -> None:
        if budget.remaining_sequences <= 0 or not self._waiting:
            return

        eligible = [
            entry
            for entry in self._waiting
            if not entry.lifecycle.is_terminal
            and not entry.lifecycle.token.is_cancelled
            and entry.ready_ns <= self._clock()
        ]
        ranked = order(
            eligible,
            round_id=self._round,
            now_ns=self._clock(),
            max_bypass=self._max_bypass,
            scheduling_policy=self._plan.scheduling_policy,
        )

        selected: set[str] = set()
        for entry in ranked:
            if budget.remaining_sequences <= 0:
                break
            lifecycle = entry.lifecycle
            snapshot = lifecycle.snapshot()
            remaining = snapshot.prompt_tokens - snapshot.computed_tokens
            if remaining <= 0:
                continue

            if self._admission is not None and not self._admission.can_admit(
                lifecycle, full_prompt_remaining=remaining
            ):
                entry.bypass_count += 1
                continue

            requested = remaining
            if self._prefill_chunk_size is not None:
                requested = min(requested, self._prefill_chunk_size)
            chunk = budget.largest_fittable(requested)
            if chunk <= 0:
                entry.bypass_count += 1
                # Do not break: a later request may need fewer padded tokens.
                continue

            query_end = snapshot.computed_tokens + chunk
            is_last_prompt_chunk = query_end == snapshot.prompt_tokens
            slices.append(
                ScheduledSlice(
                    lifecycle.request_id,
                    snapshot.sequence_epoch,
                    snapshot.state_version,
                    snapshot.computed_tokens,
                    chunk,
                    Phase.PREFILL,
                    sample_last_query=is_last_prompt_chunk,
                )
            )
            inputs.append(snapshot)
            budget.consume(chunk, phase="prefill")
            selected.add(lifecycle.request_id)
            entry.ready_round = self._round
            entry.bypass_count = 0

        # Removal is deferred until prepare/adopt succeeds. This keeps schedule()
        # side-effect free if all candidate preparations fail.

    def _prepare_and_adopt(self, plan: BatchStepPlan) -> ExecutionTicket | None:
        prepared: PreparedStep | None = None
        current: BatchStepPlan | None = plan

        while current is not None:
            try:
                prepared = self._runtime.prepare(current)
            except StepPrepareError as exc:
                if not exc.transient and exc.request_id is not None:
                    sequence = self.fail_request(exc.request_id, str(exc))
                    if sequence is not None:
                        self._allocator.release(sequence)
                    current = self._without_request(current, exc.request_id)
                    continue

                if exc.transient:
                    shrunk = self._shrink_transient(current)
                    if shrunk is not None:
                        current = shrunk
                        continue
                    victim_id = self._try_preempt_one({s.request_id for s in current.slices})
                    if victim_id is not None:
                        current = self._without_request(current, victim_id) or current
                        continue
                    self._resource_blocked = True
                return None
            break

        if current is None or prepared is None:
            return None

        ticket = self._runtime.adopt(prepared)
        self._inflight = ticket
        self._last_step = current
        self._inflight_slices = tuple(
            _InflightSlice(
                s.request_id,
                s.phase,
                s.query_end,
                value.prompt_tokens,
            )
            for s, value in zip(current.slices, current.inputs, strict=True)
        )
        self._resource_blocked = False

        # Queue membership changes only after physical reservation succeeded.
        for scheduled in current.slices:
            if scheduled.phase is Phase.PREFILL:
                self._drop_waiting(scheduled.request_id)
                lifecycle = self._requests.get(scheduled.request_id)
                self._prefilling[scheduled.request_id] = lifecycle
        return ticket

    def _shrink_transient(self, plan: BatchStepPlan) -> BatchStepPlan | None:
        """Reduce transient pressure while preserving decode latency when possible."""
        pairs = list(zip(plan.slices, plan.inputs, strict=True))
        prefill_indices = [i for i, (s, _) in enumerate(pairs) if s.phase is Phase.PREFILL]

        # First shrink the largest prefill chunk; this is the cheapest response to
        # page/padding fragmentation and avoids evicting a decode request.
        if prefill_indices:
            i = max(prefill_indices, key=lambda j: pairs[j][0].query_count)
            scheduled, value = pairs[i]
            if scheduled.query_count > 1:
                new_count = max(1, scheduled.query_count // 2)
                pairs[i] = (
                    ScheduledSlice(
                        scheduled.request_id,
                        scheduled.sequence_epoch,
                        scheduled.expected_state_version,
                        scheduled.query_start,
                        new_count,
                        scheduled.phase,
                        sample_last_query=False,
                    ),
                    value,
                )
                return self._rebuild_plan(plan, pairs)
            pairs.pop(i)
            return self._rebuild_plan(plan, pairs)

        # With pure decode, drop one request and let it run next iteration before
        # considering true KV preemption.
        if len(pairs) > 1:
            pairs.pop()
            return self._rebuild_plan(plan, pairs)
        return None

    def _try_preempt_one(self, current_request_ids: set[str]) -> str | None:
        if self._preemption is None or len(self._running) <= 1:
            return None

        # Prefer a running request that is not part of the already-shrunk step;
        # otherwise preemption would invalidate the plan we are about to retry.
        victims = [
            x
            for x in self._running.values()
            if not x.is_terminal and x.request_id not in current_request_ids
        ]
        if not victims:
            return None
        # Keep higher-priority/older requests. Preempt the least-preferred tail.
        victims.sort(
            key=lambda l: (  # noqa: E741
                int(getattr(l.request, "priority", 0) or 0),
                -int(getattr(l.request, "arrival_ns", 0)),
            )
        )
        victim = victims[0]
        sequence = self._sequences.get(victim.request_id)
        if sequence is None:
            return None

        mode_obj = getattr(self._plan, "preemption_mode", "recompute")
        mode = str(getattr(mode_obj, "value", mode_obj)).lower()
        if not self._preemption.preempt(victim, sequence, mode=mode):
            return None

        self._running.pop(victim.request_id, None)
        outcome = (
            RequestOutcome.PREEMPTED_SWAP if mode == "swap" else RequestOutcome.PREEMPTED_RECOMPUTE
        )
        self._queue_report(RequestReport(victim.request_id, outcome, victim.sequence_epoch))
        self._requeue(victim)
        return victim.request_id

    # -- Plan construction --------------------------------------------

    def _make_plan(self, slices, inputs, step_id: int | None = None) -> BatchStepPlan:
        if step_id is None:
            self._step_id += 1
            step_id = self._step_id
        total = sum(s.query_count for s in slices)
        rows: list[int] = []
        offset = 0
        for scheduled in slices:
            offset += scheduled.query_count
            if scheduled.sample_last_query:
                rows.append(offset - 1)

        per_request = tuple((s.request_id, s.query_count) for s in slices)
        groups = len(self._plan.execution.attention_groups)
        return BatchStepPlan(
            step_id=step_id,
            execution_plan_id=self._plan.execution.execution_plan_id,
            slices=tuple(slices),
            inputs=tuple(inputs),
            padded_num_tokens=self._plan.capabilities.physical_token_slots(total),
            sampling_rows=tuple(rows),
            sampling=SamplingPlan(num_rows=len(rows), all_greedy=True),
            kv_requirements=tuple(
                KVRequirement(group, total, request_tokens=per_request) for group in range(groups)
            ),
            created_ns=self._clock(),
        )

    def _without_request(self, plan: BatchStepPlan, request_id: str) -> BatchStepPlan | None:
        return self._rebuild_plan(
            plan,
            [
                pair
                for pair in zip(plan.slices, plan.inputs, strict=True)
                if pair[0].request_id != request_id
            ],
        )

    def _rebuild_plan(self, plan: BatchStepPlan, pairs) -> BatchStepPlan | None:
        pairs = list(pairs)
        if not pairs:
            return None
        return self._make_plan(
            [scheduled for scheduled, _ in pairs],
            [value for _, value in pairs],
            step_id=plan.step_id,
        )

    # -- Terminal reporting -------------------------------------------

    def finish_request(
        self, request_id: str, reason: FinishReason, detail: str = ""
    ) -> SequenceHandle | None:
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or lifecycle.is_terminal or lifecycle.inflight_slice is not None:
            return None
        self._drop(request_id)
        self._abort_pending.discard(request_id)
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
        if not self._pending:
            return False
        self._report_step_id += 1
        last = self._last_step
        report = SchedulerReport(
            step_id=self._report_step_id,
            reports=tuple(self._pending),
            num_running=len(self._running) + len(self._prefilling),
            num_waiting=len(self._waiting),
            scheduled_prefill_tokens=0 if last is None else last.num_prefill_tokens,
            scheduled_decode_tokens=0 if last is None else last.num_decode_tokens,
            cache_hint_tokens=sum(entry.cache_hint_tokens for entry in self._waiting),
        )
        self._pending.clear()
        self._abort_pending.clear()
        self._last_step = None
        self._requests.apply(report)
        return True

    # -- Inspection ----------------------------------------------------

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

    # -- Internals -----------------------------------------------------

    def _new_queue_entry(self, lifecycle, sequence) -> QueueEntry:
        ordinal = self._ordinals.get(lifecycle.request_id)
        if ordinal is None:
            ordinal = self._next_ordinal
            self._ordinals[lifecycle.request_id] = ordinal
            self._next_ordinal += 1
        entry = QueueEntry(
            lifecycle=lifecycle,
            ordinal=ordinal,
            ready_round=self._round,
            ready_ns=self._clock(),
            sequence=sequence,
        )
        return entry

    def _requeue(self, lifecycle: RequestLifecycle) -> None:
        if any(e.lifecycle.request_id == lifecycle.request_id for e in self._waiting):
            return
        # Preserve the admission ordinal for deterministic FCFS across chunking/preemption.
        self._waiting.append(
            self._new_queue_entry(lifecycle, self._sequences.get(lifecycle.request_id))
        )

    def _refresh_cache_hints(self) -> None:
        if self._prefix_hints is None:
            return
        for entry in self._waiting:
            try:
                value = self._prefix_hints.estimate_cached_tokens(entry.lifecycle)
            except Exception:
                value = 0
            entry.cache_hint_tokens = max(0, int(value))

    def _queue_report(self, report: RequestReport) -> None:
        self._pending = [item for item in self._pending if item.request_id != report.request_id]
        self._pending.append(report)

    def _check_capacity(self) -> None:
        cap = self._plan.max_num_requests
        if cap is not None and self._requests.num_active >= cap:
            raise OverloadedError(f"max_num_requests={cap} reached")
        queued = self._plan.max_queued_requests
        if queued is not None and len(self._waiting) >= queued:
            raise OverloadedError(f"max_queued_requests={queued} reached")

    def _owns_inflight(self, request_id: str) -> bool:
        return any(item.request_id == request_id for item in self._inflight_slices)

    def _drop_waiting(self, request_id: str) -> None:
        if self._waiting:
            self._waiting = [
                entry for entry in self._waiting if entry.lifecycle.request_id != request_id
            ]

    def _drop(self, request_id: str) -> None:
        self._prefilling.pop(request_id, None)
        self._running.pop(request_id, None)
        self._drop_waiting(request_id)
