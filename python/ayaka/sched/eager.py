"""Deterministic reference scheduler.

EagerScheduler intentionally stays simple: waiting requests are considered in
policy order, a prefill slice covers the whole remaining prompt, and decode
schedules one query per running request.  It is the correctness/debug baseline
against which the production ContinuousScheduler can be compared.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.executor.completion import CompletionResult
from ayaka.executor.ticket import ExecutionTicket, TerminalStatus
from ayaka.request.lifecycle import LifecycleManager
from ayaka.request.states import RequestState
from ayaka.sched.core import SchedulerCore
from ayaka.sched.interfaces import (
    OverloadedError,
    RequestPreparer,
    SequenceAllocator,
    StepPrepareError,
    StepRuntime,
)
from ayaka.sched.plan import BatchStepPlan, KVRequirement, Phase, PreparedStep, ScheduledSlice
from ayaka.sched.policy import rank_waiting

if TYPE_CHECKING:
    from ayaka.sampling.engine import SamplingCoordinator

__all__ = [
    "EagerScheduler",
    "OverloadedError",
    "SequenceAllocator",
    "StepPrepareError",
    "StepRuntime",
]

#: Same schedulable-state contract as the continuous scheduler: requests that
#: have not been physically admitted (or were evicted) are deferred, never
#: failed at prepare time.
_SCHEDULABLE_STATES = frozenset(
    {
        RequestState.ADMITTED,
        RequestState.PREFILL,
        RequestState.DECODING,
        RequestState.STREAMING,
    }
)


class EagerScheduler(SchedulerCore):
    """Reference scheduler: whole-prompt prefill, then one-query decode.

    Deliberately unsupported here:
    - chunked prefill;
    - mixed PREFILL+DECODE batches;
    - cache-affinity optimization;
    - preemption;
    - starvation/HOL bypass machinery.

    The implementation remains deterministic and intentionally conservative so
    kernel/KV/runtime bugs can be reproduced without production scheduling
    heuristics in the loop.
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
        sampling: SamplingCoordinator | None = None,
        request_preparer: RequestPreparer | None = None,
    ) -> None:
        super().__init__(
            plan,
            requests,
            runtime,
            allocator,
            text_stops=text_stops,
            clock=clock,
            sampling=sampling,
            request_preparer=request_preparer,
        )
        self._inflight_phases: dict[str, Phase] = {}

    def schedule(self) -> ExecutionTicket | None:
        if self._inflight is not None:
            return None
        for plan in self._candidate_plans():
            ticket = self._prepare_and_adopt(plan)
            if ticket is not None:
                return ticket
        return None

    def _candidate_plans(self) -> Iterator[BatchStepPlan]:
        if self._waiting_ids:
            plan = self._build_prefill()
            if plan is not None:
                yield plan
        if self._running:
            plan = self._build_decode()
            if plan is not None:
                yield plan

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
                if exc.transient and len(current.slices) > 1:
                    current = self._rebuild_plan(
                        current,
                        list(zip(current.slices, current.inputs, strict=True))[:-1],
                    )
                    continue
                self._resource_blocked = exc.transient
                return None
            break

        if current is None or prepared is None:
            return None

        ticket = self._runtime.adopt(prepared)
        self._set_inflight(ticket, (s.request_id for s in current.slices))
        self._inflight_phases = {s.request_id: s.phase for s in current.slices}
        self._resource_blocked = False

        for scheduled in current.slices:
            if scheduled.phase is Phase.PREFILL:
                self._drop_waiting(scheduled.request_id)
                self._prefilling[scheduled.request_id] = self._requests.get(scheduled.request_id)
        return ticket

    def update_from_output(self, output: CompletionResult) -> None:
        if self._sampling is not None and output.published:
            self._sampling.record_published(output.published)
        settled_ids = self._clear_inflight(output.ticket_id)
        if settled_ids is not None:
            phases = self._inflight_phases
            self._inflight_phases = {}
            if output.status is TerminalStatus.SUCCEEDED:
                for request_id in settled_ids:
                    lifecycle = self._requests.find(request_id)
                    self._prefilling.pop(request_id, None)
                    if lifecycle is None or lifecycle.is_terminal or lifecycle.token.is_cancelled:
                        continue
                    if phases.get(request_id) is Phase.PREFILL:
                        self._running_add(request_id, lifecycle)
            else:
                for request_id in settled_ids:
                    if request_id not in self._abort_pending:
                        if self._sampling is not None:
                            self._sampling.release(request_id)
                        self._drop(request_id)
                        self._sequences.pop(request_id, None)
            self._finalize_settled_aborts(settled_ids)

        for request_id in output.ignored_requests:
            if request_id in self._abort_pending:
                self._finalize_abort(request_id)
            else:
                if self._sampling is not None:
                    self._sampling.release(request_id)
                self._drop(request_id)
                self._sequences.pop(request_id, None)

    def _fits(self, total: int, extra: int) -> bool:
        slots = self._plan.capabilities.physical_token_slots(total + extra)
        return (
            slots <= self._plan.max_num_scheduled_tokens
            and slots <= self._plan.execution.compute.max_num_batched_tokens
        )

    def _build_prefill(self) -> BatchStepPlan | None:
        eligible = [
            entry
            for entry in self._iter_waiting()
            if (
                entry.lifecycle.state in _SCHEDULABLE_STATES
                or (
                    self._request_preparer is not None
                    and entry.lifecycle.state is RequestState.PREEMPTED
                )
            )
            and not entry.lifecycle.token.is_cancelled
        ]
        ranked = rank_waiting(eligible, scheduling_policy=self._plan.scheduling_policy)
        slices: list[ScheduledSlice] = []
        inputs = []
        total = 0
        for entry in ranked:
            if len(slices) >= self._seq_cap():
                break
            if self._request_preparer is not None and not self._request_preparer.prepare_request(
                entry.lifecycle
            ):
                continue
            snapshot = entry.lifecycle.snapshot()
            count = snapshot.prompt_tokens - snapshot.computed_tokens
            if count <= 0:
                continue
            # Intentional reference behavior: no chunking/HOL bypass.
            if not self._fits(total, count):
                break
            slices.append(
                ScheduledSlice(
                    entry.lifecycle.request_id,
                    snapshot.sequence_epoch,
                    snapshot.state_version,
                    snapshot.computed_tokens,
                    count,
                    Phase.PREFILL,
                    sample_last_query=(
                        snapshot.computed_tokens + count == len(snapshot.known_tokens)
                    ),
                )
            )
            inputs.append(snapshot)
            total += count
        return self._make_plan(slices, inputs) if slices else None

    def _build_decode(self) -> BatchStepPlan | None:
        slices: list[ScheduledSlice] = []
        inputs = []
        total = 0
        for lifecycle in tuple(self._running.values()):
            if lifecycle.is_terminal or lifecycle.token.is_cancelled:
                continue
            if len(slices) >= self._seq_cap() or not self._fits(total, 1):
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
                    sample_last_query=(snapshot.computed_tokens + 1 == len(snapshot.known_tokens)),
                )
            )
            inputs.append(snapshot)
            total += 1
        return self._make_plan(slices, inputs) if slices else None

    def _make_plan(
        self,
        slices: list[ScheduledSlice],
        inputs: list,
        *,
        step_id: int | None = None,
    ) -> BatchStepPlan:
        if step_id is None:
            step_id = self._next_step_id()
        total = 0
        sampling_rows: list[int] = []
        attribution: list[tuple[str, int]] = []
        for scheduled in slices:
            total += scheduled.query_count
            attribution.append((scheduled.request_id, scheduled.query_count))
            if scheduled.sample_last_query:
                sampling_rows.append(total - 1)
        groups = len(self._plan.execution.attention_groups)
        request_tokens = tuple(attribution)
        return BatchStepPlan(
            step_id=step_id,
            execution_plan_id=self._plan.execution.execution_plan_id,
            slices=tuple(slices),
            inputs=tuple(inputs),
            padded_num_tokens=self._plan.capabilities.physical_token_slots(total),
            sampling_rows=tuple(sampling_rows),
            sampling=self._sampling_plan(slices),
            prompt_logprobs=self._prompt_logprob_plan(slices, inputs),
            kv_requirements=tuple(
                KVRequirement(group, total, request_tokens=request_tokens)
                for group in range(groups)
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
