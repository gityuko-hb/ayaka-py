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

from ayaka.configs.base import ConfigError
from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.executor.completion import CompletionCoordinator, ShutdownResult
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.request.schema import Request
from ayaka.runtime.output import FinishDecision, OutputProcessor
from ayaka.sched.core import SchedulerCore
from ayaka.sched.interfaces import SequenceAllocator
from ayaka.sched.outcome import FinishReason
from ayaka.sched.plan import BatchStepPlan

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
    ) -> None:
        if not isinstance(plan, ResolvedSchedulerPlan):
            raise TypeError("plan must be ResolvedSchedulerPlan")
        self._plan = plan
        self._requests = requests
        self._scheduler = scheduler
        self._coordinator = coordinator
        self._output = output
        self._allocator = allocator
        self._settled = 0

    def submit(self, request: Request) -> RequestLifecycle:
        """Register output state, then admit. Failures leave no output state."""
        if request.sampling.n > 1:
            raise ConfigError(
                "request.sampling.n",
                "UNSUPPORTED_PARALLEL_SAMPLING",
                "the output owner binds one request id; n>1 children are expanded "
                "by the scheduler and cannot be registered here",
            )
        self._output.register(request)
        try:
            return self._scheduler.add_request(request)
        except BaseException:
            self._output.forget(str(request.request_id))
            raise

    def abort(self, request_id: str) -> bool:
        return self._scheduler.abort(request_id)

    def step(self) -> bool:
        """One serialized iteration; True when any owner changed state."""
        did = False
        for lifecycle in self._requests.cancelled_requests():
            if self._scheduler.abort(lifecycle.request_id):
                did = True

        finishes: list[FinishDecision] = []
        for result in self._coordinator.poll():
            did = True
            self._settled += 1
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

        if self._scheduler.flush_reports():
            did = True

        if self._finish_exhausted():
            did = True

        if self._coordinator.executor.has_submission_capacity():
            ticket = self._scheduler.schedule()
            if ticket is not None:
                self._coordinator.launch(ticket)
                did = True

        self._allocator.advance_epoch(self._settled)
        # A pending completion fence is progress-capable work: the loop must
        # stay alive until the executor settles or quarantines it.
        return did or self._coordinator.executor.pending_completion

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
