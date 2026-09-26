"""Serialized engine loop connecting scheduler, completion and output owners.

One engine thread drives: cancel polling, completion settlement, stop detection,
terminal reporting, step scheduling and launch. Physical preparation stays in
the injected step runtime, so this class holds no KV or device state.

Ordering invariant: finish decisions (stop/EOS/stop-string/length) are applied
to the scheduler *before* ``schedule()`` runs in the same iteration, so a
request that exhausted its token budget never enters another batch. As
defense-in-depth, any active request still holding a full output budget is
finished with ``FinishReason.LENGTH`` before scheduling rather than letting the
completion boundary hit ``RequestLifecycle.publish_sample``'s budget guard.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ayaka.configs.base import ConfigError
from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.executor.completion import CompletionCoordinator, ShutdownResult
from ayaka.metrics import IterationStats, StatLogger, StatLoggerManager
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.request.parallel import expand_parallel_request
from ayaka.request.schema import Request
from ayaka.runtime.output import FinishDecision, OutputProcessor
from ayaka.sched.core import SchedulerCore
from ayaka.sched.interfaces import DeadlineExceededError, SequenceAllocator
from ayaka.sched.outcome import FinishReason, RequestOutcome, SchedulerReport
from ayaka.sched.plan import BatchStepPlan
from ayaka.serving.router import RemoteKVPending

__all__ = ["Engine"]


class Engine:
    """Drive one serialized engine loop over injected owners."""

    def __init__(
        self,
        plan: ResolvedSchedulerPlan,
        *,
        requests: LifecycleManager,
        scheduler: SchedulerCore,
        coordinator: CompletionCoordinator,
        output: OutputProcessor,
        allocator: SequenceAllocator,
        remote_kv: RemoteKVPending | None = None,
        stat_loggers: Sequence[StatLogger] = (),
        log_interval: float = 10.0,
        log_clock: Callable[[], float] | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(plan, ResolvedSchedulerPlan):
            raise TypeError("plan must be ResolvedSchedulerPlan")
        self._plan = plan
        self._requests = requests
        self._scheduler = scheduler
        self._coordinator = coordinator
        self._output = output
        self._allocator = allocator
        self._remote_kv = remote_kv
        self._settled = 0
        #: Monotonic nanosecond source for request deadlines; tests inject a
        #: fake clock, production uses the process monotonic clock.
        self._clock = clock or time.monotonic_ns
        self._manager = (
            StatLoggerManager(loggers=stat_loggers, log_interval=log_interval, clock=log_clock)
            if stat_loggers
            else None
        )

    def submit(self, request: Request, *, defer_to_remote_kv: bool = False) -> RequestLifecycle:
        """Register output state, then admit. Failures leave no output state.

        ``defer_to_remote_kv`` parks the request in ``WAITING_REMOTE_KV``; a
        :class:`RemoteKVPending` registry must be wired so the transport can
        drive it out of the park. Parallel families (n>1) register one output
        state per child — the child id set is derived by the shared
        ``expand_parallel_request`` helper, matching the scheduler's fan-out.
        """
        if defer_to_remote_kv and self._remote_kv is None:
            raise ConfigError(
                "engine.remote_kv",
                "REMOTE_KV_REGISTRY_REQUIRED",
                "deferring a request to WAITING_REMOTE_KV requires a remote-KV pending registry",
            )
        deadline = request.deadline_ns
        if deadline is not None and deadline <= self._clock():
            raise DeadlineExceededError(f"{request.request_id}: deadline elapsed before admission")
        children = expand_parallel_request(request) if request.sampling.n > 1 else None
        if children is None:
            self._output.register(request)
            try:
                lifecycle = self._scheduler.add_request(
                    request, defer_to_remote_kv=defer_to_remote_kv
                )
            except BaseException:
                self._output.forget(str(request.request_id))
                raise
            if defer_to_remote_kv and self._remote_kv is not None:
                self._remote_kv.park(str(request.request_id))
            return lifecycle
        registered: list[str] = []
        try:
            for index, child in enumerate(children):
                self._output.register(child, child_index=index)
                registered.append(str(child.request_id))
            lifecycle = self._scheduler.add_request(request, defer_to_remote_kv=defer_to_remote_kv)
        except BaseException:
            for request_id in registered:
                self._output.forget(request_id)
            raise
        if defer_to_remote_kv and self._remote_kv is not None:
            for child in children:
                self._remote_kv.park(str(child.request_id))
        return lifecycle

    def release_remote_kv(self, request_id: str, *, num_cached_tokens: int = 0) -> bool:
        """Remote KV landed: unpark into the local queue with the acquired prefix."""
        if self._remote_kv is None:
            return False
        return self._remote_kv.release(request_id, num_cached_tokens=num_cached_tokens)

    def abandon_remote_kv(self, request_id: str) -> bool:
        """Remote fetch failed: queue locally for a full prefill instead."""
        if self._remote_kv is None:
            return False
        return self._remote_kv.abandon(request_id)

    def abort(self, request_id: str) -> bool:
        """Abort one request; parallel parents fan out to every child."""
        children = self._scheduler.children_of(request_id)
        if children is not None:
            did = False
            for child in children:
                did = self._scheduler.abort(child) or did
            return did
        return self._scheduler.abort(request_id)

    def step(self) -> bool:
        """One serialized iteration; True when any owner changed state."""
        started = time.monotonic()
        settled_before = self._settled
        did = False
        now_ns = self._clock()
        for lifecycle in self._requests:
            if lifecycle.token.is_cancelled:
                if self._scheduler.abort(lifecycle.request_id):
                    did = True
                continue
            deadline = lifecycle.request.deadline_ns
            if deadline is not None and now_ns >= deadline:
                # Expiry cancels publication through the shared abort path: the
                # executor discards sampled tokens for a cancelled token, and
                # the terminal cause survives ticket settlement exactly once.
                if self._scheduler.abort(
                    lifecycle.request_id,
                    reason="deadline exceeded",
                    finish_reason=FinishReason.TIMEOUT,
                ):
                    did = True

        finishes: list[FinishDecision] = []
        published_tokens = 0
        launched_tokens = 0
        for result in self._coordinator.poll():
            did = True
            self._settled += 1
            published_tokens += len(result.published)
            for sample in result.published:
                lifecycle = self._requests.find(sample.request_id)
                if lifecycle is None or lifecycle.is_terminal:
                    continue
                if lifecycle.sequence_epoch != sample.sequence_epoch:
                    continue
                decision = self._output.on_published(
                    sample.request_id,
                    sample.token_id,
                    sequence_epoch=sample.sequence_epoch,
                    generated_tokens=len(lifecycle.output_token_ids),
                    prompt_tokens=lifecycle.request.prompt_len,
                )
                if decision is not None:
                    finishes.append(decision)
            self._scheduler.update_from_output(result)

        for decision in finishes:
            self._finish(decision.request_id, decision.reason, decision.detail)

        report = self._scheduler.flush_reports()
        if report is not None:
            did = True

        if self._finish_exhausted():
            did = True

        while self._coordinator.executor.has_submission_capacity():
            ticket = self._scheduler.schedule()
            if ticket is None:
                break
            self._coordinator.launch(ticket)
            launched_tokens += sum(
                scheduled.query_count for scheduled in ticket.prepared.step.slices
            )
            did = True

        self._allocator.advance_epoch(self._settled)
        if self._manager is not None:
            self._record_stats(
                report,
                settled_before,
                published_tokens,
                launched_tokens,
                time.monotonic() - started,
            )
        # A pending completion fence is progress-capable work: the loop must
        # stay alive until the executor settles or quarantines it.
        return did or self._coordinator.executor.pending_completion

    def _record_stats(
        self,
        report: SchedulerReport | None,
        settled_before: int,
        published_tokens: int,
        launched_tokens: int,
        step_seconds: float,
    ) -> None:
        assert self._manager is not None
        if report is not None:
            outcomes = [r.outcome for r in report.reports]
            stats = IterationStats(
                step_id=report.step_id,
                num_running=report.num_running,
                num_waiting=report.num_waiting,
                num_preempted=report.num_preempted_this_step,
                num_finished=outcomes.count(RequestOutcome.FINISHED),
                num_aborted=outcomes.count(RequestOutcome.ABORTED),
                num_failed=outcomes.count(RequestOutcome.FAILED),
                num_settled=self._settled - settled_before,
                num_new_tokens=published_tokens,
                num_scheduled_tokens=launched_tokens,
                step_seconds=step_seconds,
            )
        else:
            stats = IterationStats(
                step_id=-1,
                num_settled=self._settled - settled_before,
                num_new_tokens=published_tokens,
                num_scheduled_tokens=launched_tokens,
                step_seconds=step_seconds,
            )
        self._manager.record(stats)
        self._manager.maybe_log()

    def run_until_idle(self, *, max_steps: int = 1024) -> int:
        """Run until no active request remains or a step makes no progress."""
        steps = 0
        while self.has_unfinished and steps < max_steps:
            did = self.step()
            steps += 1
            if not did:
                break
        return steps

    def sampling_masks(self, step: BatchStepPlan) -> tuple[frozenset[int], ...]:
        """Pre-sample bans for each sampling row, in ``sampling_rows`` order."""
        masks = []
        for scheduled in step.slices:
            if not scheduled.sample_last_query:
                continue
            lifecycle = self._requests.get(scheduled.request_id)
            masks.append(
                self._output.masked_ids(
                    scheduled.request_id,
                    generated_tokens=len(lifecycle.output_token_ids),
                )
            )
        return tuple(masks)

    def _finish(self, request_id: str, reason: FinishReason, detail: str = "") -> bool:
        sequence = self._scheduler.finish_request(request_id, reason, detail)
        if sequence is None:
            return False
        self._allocator.release(sequence)
        return True

    def _finish_exhausted(self) -> bool:
        """Finish requests whose output budget is spent but no decision arrived."""
        did = False
        for lifecycle in self._requests:
            if lifecycle.is_terminal or lifecycle.token.is_cancelled:
                continue
            if len(lifecycle.output_token_ids) >= lifecycle.request.stop.max_tokens:
                if self._finish(
                    lifecycle.request_id,
                    FinishReason.LENGTH,
                    "output token budget exhausted",
                ):
                    did = True
        return did

    def forget(self, request_id: str) -> None:
        """Drop retained output state once the caller has consumed the text."""
        self._output.forget(request_id)

    @property
    def has_unfinished(self) -> bool:
        return self._scheduler.has_unfinished

    @property
    def scheduler(self) -> SchedulerCore:
        return self._scheduler

    @property
    def requests(self) -> LifecycleManager:
        return self._requests

    @property
    def output(self) -> OutputProcessor:
        return self._output

    def text(self, request_id: str) -> str:
        return self._output.text(request_id)

    def events(self, request_id: str):
        return self._output.events(request_id)

    def finish_reason(self, request_id: str) -> FinishReason | None:
        lifecycle = self._requests.get(request_id)
        return lifecycle.finish_reason

    def close(self) -> ShutdownResult:
        """Drain and settle remaining tickets; call again if not closed."""
        return self._coordinator.shutdown()
