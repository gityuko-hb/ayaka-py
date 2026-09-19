"""Production continuous-batching scheduler for Ayaka.

Logical scheduling is intentionally separated from physical KV ownership:
``BatchStepPlan`` is immutable and ``StepRuntime.prepare`` is the authority that
can accept, reject, or force the scheduler to reshape a candidate step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.executor.completion import CompletionResult
from ayaka.executor.ticket import ExecutionTicket, TerminalStatus, TicketId
from ayaka.request.lifecycle import LifecycleManager
from ayaka.request.states import RequestState
from ayaka.sched.budget import BatchBudget
from ayaka.sched.core import SchedulerCore
from ayaka.sched.interfaces import (
    AdmissionAdvisor,
    PreemptionController,
    PrefixHintProvider,
    RequestPreparer,
    SequenceAllocator,
    StepPrepareError,
    StepRuntime,
)
from ayaka.sched.outcome import RequestOutcome, RequestReport
from ayaka.sched.plan import (
    BatchStepPlan,
    KVRequirement,
    Phase,
    PreparedStep,
    RequestStepInput,
    ScheduledSlice,
)
from ayaka.sched.policy import QueueEntry, order
from ayaka.sched.preemption import select_preemption_victim

if TYPE_CHECKING:
    from ayaka.sampling.engine import SamplingCoordinator

__all__ = ["ContinuousScheduler", "ContinuousSchedulerStats"]

#: Machine states from which `begin_slice` can legally claim work; anything
#: else (freshly queued but not yet admitted, preempted/evicted, terminal) is
#: deferred to a later step instead of failing the request at prepare time.
_SCHEDULABLE_STATES = frozenset(
    {
        RequestState.ADMITTED,
        RequestState.PREFILL,
        RequestState.DECODING,
        RequestState.STREAMING,
    }
)


@dataclass(frozen=True, slots=True)
class ContinuousSchedulerStats:
    round_id: int
    waiting: int
    prefilling: int
    running: int
    inflight: int
    last_prefill_tokens: int
    last_decode_tokens: int
    last_mixed: bool
    transient_prepare_failures: int
    preemptions: int
    bypasses: int
    admission_rejections: int


@dataclass(frozen=True, slots=True)
class _InflightSlice:
    request_id: str
    phase: Phase
    query_end: int
    prompt_tokens: int


class ContinuousScheduler(SchedulerCore):
    """Decode-first continuous batching with chunked prefill and preemption.

    Capabilities implemented here:
    - one-query-per-request decode with round-robin fairness;
    - chunked prefill;
    - mixed PREFILL+DECODE steps;
    - head-of-line bypass;
    - bounded-starvation aging;
    - optional prefix/cache-affinity ranking hints;
    - advisory physical admission hooks;
    - transient-pressure shrink before preemption;
    - delegated recompute/swap preemption;
    - rollback-safe queue mutation only after prepare/adopt succeeds.

    Up to ``plan.max_inflight`` execution tickets may be adopted at once
    (pipeline microbatches / overlap). A request owns at most one unsettled
    slice: candidates exclude every request with a slice still in flight, and
    preemption victims are only selected outside that set.
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
        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if max_bypass <= 0:
            raise ValueError("max_bypass must be positive")

        self._prefix_hints = prefix_hints
        self._admission = admission
        self._preemption = preemption
        self._decode_first = bool(decode_first)
        self._allow_mixed_batches = bool(allow_mixed_batches)
        chunk_cap = plan.max_prefill_chunk_tokens if plan.enable_chunked_prefill else None
        if prefill_chunk_size is not None:
            if not plan.enable_chunked_prefill:
                raise ValueError("prefill_chunk_size requires chunked prefill in the resolved plan")
            chunk_cap = (
                prefill_chunk_size if chunk_cap is None else min(chunk_cap, prefill_chunk_size)
            )
        self._prefill_chunk_size = chunk_cap
        self._max_bypass = max_bypass

        self._inflight_slices: dict[TicketId, tuple[_InflightSlice, ...]] = {}
        self._last_step: BatchStepPlan | None = None
        self._decode_cursor = 0

        self._transient_prepare_failures = 0
        self._preemptions = 0
        self._bypasses = 0
        self._admission_rejections = 0

    # ------------------------------------------------------------------
    # Scheduling loop
    # ------------------------------------------------------------------

    def schedule(self) -> ExecutionTicket | None:
        if len(self._inflight) >= self._plan.max_inflight:
            return None
        self._round += 1
        self._drain_window()
        for plan in self._candidate_plans():
            ticket = self._prepare_and_adopt(plan)
            if ticket is not None:
                return ticket
        return None

    def update_from_output(self, output: CompletionResult) -> None:
        if self._sampling is not None and output.published:
            self._sampling.record_published(output.published)
        settled_ids = self._clear_inflight(output.ticket_id)
        if settled_ids is not None:
            inflight = self._inflight_slices.pop(output.ticket_id, ())

            if output.status is TerminalStatus.SUCCEEDED:
                for item in inflight:
                    lifecycle = self._requests.find(item.request_id)
                    self._prefilling.pop(item.request_id, None)
                    if lifecycle is None or lifecycle.is_terminal or lifecycle.token.is_cancelled:
                        continue
                    if item.phase is Phase.PREFILL:
                        if item.query_end < item.prompt_tokens:
                            # Partial prefill keeps its concurrency slot between chunks.
                            self._prefilling[item.request_id] = lifecycle
                            self._requeue(lifecycle)
                        else:
                            self._running_add(item.request_id, lifecycle)
                    else:
                        self._running_add(item.request_id, lifecycle)
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

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def _candidate_plans(self) -> Iterator[BatchStepPlan]:
        if self._allow_mixed_batches:
            plan = self._build_continuous_batch()
            if plan is not None:
                yield plan
            return

        builders = (
            (self._build_decode_only, self._build_prefill_only)
            if self._decode_first
            else (self._build_prefill_only, self._build_decode_only)
        )
        for builder in builders:
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

    def _append_decodes(
        self,
        budget: BatchBudget,
        slices: list[ScheduledSlice],
        inputs: list,
    ) -> None:
        # Direct ring walk over the dense running array: no per-step filtering
        # list; dead entries are skipped in place and leave the ring on drop.
        ring = self._running_ids
        count = len(ring)
        if not count:
            return

        start = self._decode_cursor % count
        running = self._running
        scheduled_count = 0
        for offset in range(count):
            lifecycle = running[ring[(start + offset) % count]]
            if lifecycle.is_terminal or lifecycle.token.is_cancelled:
                continue
            if lifecycle.request_id in self._inflight_ids_all:
                # One unsettled slice per request: the adopted ticket keeps
                # its ownership until update_from_output settles it.
                continue
            if not budget.try_consume(1, phase=Phase.DECODE):
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
            scheduled_count += 1

        if scheduled_count:
            self._decode_cursor = (start + scheduled_count) % count

    def _append_prefills(
        self,
        budget: BatchBudget,
        slices: list[ScheduledSlice],
        inputs: list,
    ) -> None:
        if budget.remaining_sequences <= 0 or not self._hot:
            return

        now = self._clock()
        eligible = [
            entry
            for entry in self._hot
            if (
                entry.lifecycle.state in _SCHEDULABLE_STATES
                or (
                    self._request_preparer is not None
                    and entry.lifecycle.state is RequestState.PREEMPTED
                )
            )
            and not entry.lifecycle.token.is_cancelled
            and entry.lifecycle.request_id not in self._inflight_ids_all
            and entry.ready_ns <= now
        ]
        ranked = order(
            eligible,
            round_id=self._round,
            now_ns=now,
            max_bypass=self._max_bypass,
            scheduling_policy=self._plan.scheduling_policy,
        )

        partial_ids = set(self._prefilling)
        # Continue owned partial prompts before admitting another partial prompt.
        ranked.sort(key=lambda entry: entry.lifecycle.request_id not in partial_ids)
        for entry in ranked:
            if budget.remaining_sequences <= 0 or budget.largest_fittable(1) == 0:
                break

            lifecycle = entry.lifecycle
            if self._request_preparer is not None and not self._request_preparer.prepare_request(
                lifecycle
            ):
                continue
            snapshot = lifecycle.snapshot()
            remaining = snapshot.prompt_tokens - snapshot.computed_tokens
            if remaining <= 0:
                continue

            requested = remaining
            if self._prefill_chunk_size is not None:
                requested = min(requested, self._prefill_chunk_size)
            chunk = budget.largest_fittable(requested)
            if chunk <= 0:
                self._mark_bypass(entry)
                continue

            if self._admission is not None and not self._admission.can_admit(
                lifecycle,
                full_prompt_remaining=remaining,
                scheduled_prompt_tokens=chunk,
            ):
                self._admission_rejections += 1
                self._mark_bypass(entry)
                continue

            query_end = snapshot.computed_tokens + chunk
            last_prompt_chunk = query_end == snapshot.prompt_tokens
            if (
                not last_prompt_chunk
                and lifecycle.request_id not in partial_ids
                and len(partial_ids) >= self._plan.max_num_partial_prefills
            ):
                self._mark_bypass(entry)
                continue
            if not budget.try_consume(chunk, phase=Phase.PREFILL):
                self._mark_bypass(entry)
                continue
            if not last_prompt_chunk:
                partial_ids.add(lifecycle.request_id)
            slices.append(
                ScheduledSlice(
                    lifecycle.request_id,
                    snapshot.sequence_epoch,
                    snapshot.state_version,
                    snapshot.computed_tokens,
                    chunk,
                    Phase.PREFILL,
                    sample_last_query=(
                        last_prompt_chunk and query_end == len(snapshot.known_tokens)
                    ),
                )
            )
            inputs.append(snapshot)

        # No queue mutation here. Candidate creation must be rollback-safe.

    # ------------------------------------------------------------------
    # Physical prepare, pressure shrink, preemption
    # ------------------------------------------------------------------

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
                if not exc.transient:
                    self._resource_blocked = False
                    return None

                self._transient_prepare_failures += 1
                shrunk = self._shrink_transient(current)
                if shrunk is not None:
                    current = shrunk
                    continue

                victim_id = self._try_preempt_one(
                    {scheduled.request_id for scheduled in current.slices} | self._inflight_ids_all
                )
                if victim_id is not None:
                    current = self._without_request(current, victim_id) or current
                    continue

                self._resource_blocked = True
                return None
            break

        if current is None or prepared is None:
            return None

        ticket = self._runtime.adopt(prepared)
        self._set_inflight(ticket, (scheduled.request_id for scheduled in current.slices))
        self._last_step = current
        self._inflight_slices[ticket.id] = tuple(
            _InflightSlice(
                scheduled.request_id,
                scheduled.phase,
                scheduled.query_end,
                value.prompt_tokens,
            )
            for scheduled, value in zip(current.slices, current.inputs, strict=True)
        )
        self._resource_blocked = False

        # Only now may logical queue ownership move: physical prepare/adopt won.
        for scheduled in current.slices:
            if scheduled.phase is Phase.PREFILL:
                self._drop_waiting(scheduled.request_id)
                lifecycle = self._requests.get(scheduled.request_id)
                self._prefilling[scheduled.request_id] = lifecycle
        return ticket

    def _shrink_transient(self, plan: BatchStepPlan) -> BatchStepPlan | None:
        """Reduce pressure while protecting decode latency as long as possible."""
        slices = plan.slices
        inputs = plan.inputs
        largest = -1
        for index, scheduled in enumerate(slices):
            if scheduled.phase is not Phase.PREFILL:
                continue
            # At equal width, shrink later/new prompts first. Dropping the only
            # continuation can strand its owned pages while a new request waits.
            priority = (scheduled.query_count, inputs[index].computed_tokens == 0, index)
            if largest < 0 or priority > (
                slices[largest].query_count,
                inputs[largest].computed_tokens == 0,
                largest,
            ):
                largest = index

        # 1. Halve the largest prefill chunk first.
        if largest >= 0:
            scheduled = slices[largest]
            if scheduled.query_count > 1:
                new_count = max(1, scheduled.query_count // 2)
                new_end = scheduled.query_start + new_count
                value = inputs[largest]
                replacement = ScheduledSlice(
                    scheduled.request_id,
                    scheduled.sequence_epoch,
                    scheduled.expected_state_version,
                    scheduled.query_start,
                    new_count,
                    Phase.PREFILL,
                    sample_last_query=(
                        new_end == value.prompt_tokens and new_end == len(value.known_tokens)
                    ),
                )
                return self._make_plan(
                    slices[:largest] + (replacement,) + slices[largest + 1 :],
                    inputs,
                    step_id=plan.step_id,
                )

            # 2. Remove a one-token prefill before sacrificing decode width.
            if len(slices) == 1:
                return None
            return self._make_plan(
                slices[:largest] + slices[largest + 1 :],
                inputs[:largest] + inputs[largest + 1 :],
                step_id=plan.step_id,
            )

        # 3. Pure decode: narrow by one request. Round-robin gives it another turn.
        if len(slices) > 1:
            return self._make_plan(slices[:-1], inputs[:-1], step_id=plan.step_id)
        return None

    def _try_preempt_one(self, current_request_ids: set[str]) -> str | None:
        if self._preemption is None:
            return None

        candidates = dict(self._running)
        candidates.update(self._prefilling)
        # A prefix can already be acquired for a waiting request that did not fit
        # the previous batch. It owns real pages and must also be reclaimable.
        for entry in self._hot:
            lifecycle = entry.lifecycle
            if (
                lifecycle.computed_tokens > 0
                and lifecycle.machine.can(RequestState.PREEMPTED)
                and lifecycle.request_id not in self._pending
            ):
                candidates[lifecycle.request_id] = lifecycle
        victim = select_preemption_victim(
            candidates.values(),
            excluded_request_ids=current_request_ids,
        )
        if victim is None:
            return None
        sequence = self._sequences.get(victim.request_id)
        if sequence is None:
            return None

        mode_obj = getattr(self._plan, "preemption_mode", "recompute")
        mode = str(getattr(mode_obj, "value", mode_obj)).lower()
        if not self._preemption.preempt(victim, sequence, mode=mode):
            return None

        self._running_remove(victim.request_id)
        self._prefilling.pop(victim.request_id, None)
        outcome = (
            RequestOutcome.PREEMPTED_SWAP if mode == "swap" else RequestOutcome.PREEMPTED_RECOMPUTE
        )
        self._queue_report(RequestReport(victim.request_id, outcome, victim.sequence_epoch))
        self._requeue(victim)
        self._preemptions += 1
        return victim.request_id

    # ------------------------------------------------------------------
    # Immutable plan construction
    # ------------------------------------------------------------------

    def _make_plan(
        self,
        slices: Sequence[ScheduledSlice],
        inputs: Sequence[RequestStepInput],
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

    # ------------------------------------------------------------------
    # Prefix hints, fairness, observability
    # ------------------------------------------------------------------

    def _refresh_window_hints(self, admitted: list[QueueEntry]) -> None:
        if self._prefix_hints is None:
            return
        for entry in admitted:
            try:
                value = self._prefix_hints.estimate_cached_tokens(entry.lifecycle)
            except Exception:
                # Ranking failure cannot compromise scheduler availability or be
                # interpreted as physical ownership.
                value = 0
            entry.cache_hint_tokens = max(0, int(value))

    def _mark_bypass(self, entry: QueueEntry) -> None:
        entry.bypass_count += 1
        self._bypasses += 1

    @property
    def stats(self) -> ContinuousSchedulerStats:
        last = self._last_step
        return ContinuousSchedulerStats(
            round_id=self._round,
            waiting=self.num_waiting,
            prefilling=len(self._prefilling),
            running=len(self._running),
            inflight=len(self._inflight),
            last_prefill_tokens=0 if last is None else last.num_prefill_tokens,
            last_decode_tokens=0 if last is None else last.num_decode_tokens,
            last_mixed=False if last is None else last.is_mixed,
            transient_prepare_failures=self._transient_prepare_failures,
            preemptions=self._preemptions,
            bypasses=self._bypasses,
            admission_rejections=self._admission_rejections,
        )
