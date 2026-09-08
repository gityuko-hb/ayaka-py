from __future__ import annotations

import time
from collections.abc import Iterator

from ayaka.handles import SequenceHandle
from ayaka.request.cancel import CancellationToken
from ayaka.request.machine import RequestStateMachine, next_request_epoch
from ayaka.request.schema import Request
from ayaka.request.states import RequestState
from ayaka.sched.outcome import (
    FinishReason,
    RequestOutcome,
    RequestReport,
    SchedulerReport,
)
from ayaka.sched.plan import Phase, RequestStepInput, ScheduledSlice
from ayaka.utils.timing import Clock
from ayaka.utils.validation import require_int

_MAX_RETRIES = 3

class UnknownRequest(KeyError):
    """A report referenced a request the manager has never seen."""


class RequestLifecycle:
    """One request's tokens, guarded forward progress and output publication.

    Mutation is serialized by the future engine/completion coordinator. These
    methods do not acquire/free resources or establish device completion.
    Only bind_sequence may accept externally validated resume progress; normal
    progress goes through begin_slice -> commit_computed_range -> publish_sample.
    """

    __slots__ = (
        "finish_reason",
        "machine",
        "request",
        "token",
        "_output_token_ids",
        "_sequence",
        "_binding_epoch",
        "_state_version",
        "_inflight",
        "_input_snapshot",
        "_forward_committed",
        "_last_scheduled_step",
    )

    def __init__(
        self,
        request: Request,
        *,
        max_retries: int = _MAX_RETRIES,
        clock: Clock | None = None,
    ) -> None:
        self.request = request
        self.machine = RequestStateMachine(
            str(request.request_id),
            arrival_ns=request.arrival_ns or None,
            max_retries=max_retries,
            clock=clock,
        )
        self.token = CancellationToken(str(request.request_id))
        self._output_token_ids: list[int] = []
        self.finish_reason: FinishReason | None = None
        self._sequence: SequenceHandle | None = None
        self._binding_epoch = 0
        self._state_version = 0
        self._inflight: tuple[int, ScheduledSlice] | None = None
        self._input_snapshot: RequestStepInput | None = None
        self._forward_committed = False
        self._last_scheduled_step = -1

    @property
    def request_id(self) -> str:
        return str(self.request.request_id)

    @property
    def state(self) -> RequestState:
        return self.machine.state

    @property
    def is_terminal(self) -> bool:
        return self.machine.is_terminal

    @property
    def holds_kv(self) -> bool:
        """Lifecycle hint only; execution leases govern physical lifetime."""
        return self.machine.holds_kv

    @property
    def sequence_epoch(self) -> int:
        return self.machine.sequence_epoch

    @property
    def state_version(self) -> int:
        """Mirrored KV sequence version; it is not a sampling counter."""
        return self._state_version

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Read-only output snapshot; publish_sample is the only writer."""
        return tuple(self._output_token_ids)

    @property
    def known_tokens(self) -> tuple[int, ...]:
        """Prompt plus published output, derived rather than stored twice."""
        return self.request.prompt_token_ids + self.output_token_ids

    @property
    def computed_tokens(self) -> int:
        return self.machine.num_computed_tokens

    @property
    def inflight_slice(self) -> ScheduledSlice | None:
        return None if self._inflight is None else self._inflight[1]

    def bind_sequence(
        self,
        sequence: SequenceHandle,
        *,
        state_version: int,
        computed_tokens: int = 0,
    ) -> None:
        """Bind an acquired sequence snapshot, after KV resume validation.

        A raw prefix match is not sufficient evidence. Rebinding an existing
        request invalidates the previous epoch. Outstanding work must first
        be discarded only after the caller establishes that it is safe.
        """
        if self._inflight is not None:
            raise ValueError("cannot rebind a request with an unfinished slice")
        if self.is_terminal or self.token.is_cancelled:
            raise ValueError("cannot bind a terminal or cancelled request")
        if self.machine.num_prompt_tokens != self.request.prompt_len:
            raise ValueError("request must be tokenized before sequence binding")
        value = RequestStepInput(
            self.request_id,
            sequence,
            self.sequence_epoch,
            state_version,
            self.request.prompt_len,
            self.known_tokens,
            computed_tokens,
            self.request.stop.max_tokens,
        )
        if self._sequence is not None:
            self.machine.sequence_epoch = next_request_epoch()
        self._sequence = sequence
        self._binding_epoch = self.sequence_epoch
        self._state_version = value.state_version
        self.machine.num_computed_tokens = computed_tokens
        self.machine.num_cached_tokens = computed_tokens

    def snapshot(self) -> RequestStepInput:
        """Capture known inputs and the current KV version without side effects."""
        if self._sequence is None:
            raise ValueError("request has no acquired sequence binding")
        if self._binding_epoch != self.sequence_epoch:
            raise ValueError("sequence binding was invalidated; bind a validated KV snapshot")
        if self.machine.num_generated_tokens != len(self._output_token_ids):
            raise ValueError("sample counters were mutated outside publish_sample")
        return RequestStepInput(
            self.request_id,
            self._sequence,
            self.sequence_epoch,
            self._state_version,
            self.request.prompt_len,
            self.known_tokens,
            self.computed_tokens,
            self.request.stop.max_tokens,
        )

    def begin_slice(
        self,
        step_id: int,
        scheduled: ScheduledSlice,
        *,
        now_ns: int | None = None,
    ) -> RequestStepInput:
        """Claim the only unfinished slice, without crediting any computation."""
        value = self.validate_begin_slice(step_id, scheduled)
        if self.state is RequestState.ADMITTED:
            target = (
                RequestState.PREFILL if scheduled.phase is Phase.PREFILL else RequestState.DECODING
            )
            self.machine.transition(target, now_ns=now_ns)
        self._inflight = (step_id, scheduled)
        self._input_snapshot = value
        self._forward_committed = False
        self._last_scheduled_step = step_id
        return value

    def validate_begin_slice(self, step_id: int, scheduled: ScheduledSlice) -> RequestStepInput:
        """Preflight a claim without mutation so a coordinator can validate a whole batch."""
        require_int(step_id, "step_id")
        if self._inflight is not None:
            raise ValueError("request already has an unfinished slice")
        if self.is_terminal or self.token.is_cancelled:
            raise ValueError("request is terminal or cancelled")
        if step_id <= self._last_scheduled_step:
            raise ValueError("a new slice requires a fresh step identity")
        value = self.snapshot()
        value.validate_slice(scheduled)
        self._validate_phase_state(scheduled)
        return value

    def owns_slice(self, step_id: int, scheduled: ScheduledSlice) -> bool:
        """Match the full step/slice identity, including an invalidated incarnation."""
        return self._inflight == (step_id, scheduled)

    def accepts_completion(self, step_id: int, scheduled: ScheduledSlice) -> bool:
        """Whether this incarnation still permits logical completion/publication."""
        return (
            self.owns_slice(step_id, scheduled)
            and self.sequence_epoch == scheduled.sequence_epoch
            and not self.is_terminal
            and not self.token.is_cancelled
        )

    def _validate_phase_state(self, scheduled: ScheduledSlice) -> None:
        allowed = (
            (RequestState.ADMITTED, RequestState.PREFILL)
            if scheduled.phase is Phase.PREFILL
            else (RequestState.ADMITTED, RequestState.DECODING, RequestState.STREAMING)
        )
        if self.state not in allowed:
            raise ValueError("request state is incompatible with the scheduled phase")

    def _require_slice(self, step_id: int, scheduled: ScheduledSlice) -> None:
        if self._inflight != (step_id, scheduled):
            raise ValueError("slice does not match the unfinished step")
        if (
            scheduled.sequence_epoch != self.sequence_epoch
            or self.is_terminal
            or self.token.is_cancelled
        ):
            raise ValueError("slice belongs to an invalidated or cancelled request")

    def validate_forward_commit(
        self,
        step_id: int,
        scheduled: ScheduledSlice,
        *,
        committed_state_version: int,
    ) -> None:
        """Validate a completion before any state/output mutation.

        The caller supplies a version returned by successful KV commit, not a
        guessed version at enqueue time. Device proof/resource ownership is
        enforced by the later completion coordinator.
        """
        self._require_slice(step_id, scheduled)
        if self._forward_committed:
            raise ValueError("forward range was already committed")
        require_int(committed_state_version, "committed_state_version")
        if committed_state_version != scheduled.expected_state_version + 1:
            raise ValueError("committed KV version must advance the reserved version once")
        self._validate_phase_state(scheduled)
        value = self.snapshot()
        value.validate_slice(scheduled)
        if value != self._input_snapshot:
            raise ValueError("request inputs or sequence binding changed during the step")

    def commit_computed_range(
        self,
        step_id: int,
        scheduled: ScheduledSlice,
        *,
        committed_state_version: int,
        now_ns: int | None = None,
    ) -> None:
        """Commit a completed forward, retaining the slice until its sample is published."""
        self.validate_forward_commit(
            step_id,
            scheduled,
            committed_state_version=committed_state_version,
        )
        if scheduled.phase is Phase.PREFILL and self.state is RequestState.ADMITTED:
            self.machine.transition(RequestState.PREFILL, now_ns=now_ns)
        elif scheduled.phase is Phase.DECODE and self.state is RequestState.ADMITTED:
            self.machine.transition(RequestState.DECODING, now_ns=now_ns)
        self.machine.commit_computed_range(scheduled.query_start, scheduled.query_count)
        self._state_version = committed_state_version
        if scheduled.phase is Phase.PREFILL and scheduled.query_end == self.request.prompt_len:
            self.machine.transition(RequestState.DECODING, now_ns=now_ns)
        self._forward_committed = True
        if not scheduled.sample_last_query:
            self._clear_slice()

    def publish_sample(self, step_id: int, scheduled: ScheduledSlice, token_id: int) -> None:
        """Publish exactly one identified sample; the new token has no KV yet."""
        self._require_slice(step_id, scheduled)
        require_int(token_id, "sample token id")
        if not self._forward_committed or not scheduled.sample_last_query:
            raise ValueError("no completed sampling row is awaiting publication")
        if len(self._output_token_ids) >= self.request.stop.max_tokens:
            raise ValueError("output token budget is exhausted")
        self.machine.on_token_generated(1)
        self._output_token_ids.append(token_id)
        self._clear_slice()

    def discard_slice(self, step_id: int, scheduled: ScheduledSlice) -> None:
        """Forget an unsubmitted or fully drained slice and invalidate its epoch.

        This does not release any lease or undo committed KV. If a committed
        sample is discarded and execution will continue, the caller must bind
        a validated recomputation boundary before scheduling another query.
        Every discard requires a fresh binding, even if no work was submitted.
        """
        if self._inflight != (step_id, scheduled):
            raise ValueError("slice does not match the unfinished step")
        self._clear_slice()
        self.machine.sequence_epoch = next_request_epoch()

    def _clear_slice(self) -> None:
        self._inflight = None
        self._input_snapshot = None
        self._forward_committed = False

    def __repr__(self) -> str:
        return (
            f"<RequestLifecycle {self.request_id} {self.state.value} "
            f"epoch={self.sequence_epoch} computed={self.computed_tokens} "
            f"out={len(self._output_token_ids)}>"
        )


#: outcome -> the state it implies, or None when the outcome changes counters
#: without changing state (a prefill chunk, a generated token).  Kept as data so
#: the mapping can be asserted exhaustively in one test instead of being
#: inferred from control flow.
_OUTCOME_STATE: dict[RequestOutcome, RequestState | None] = {
    RequestOutcome.QUEUED: RequestState.WAITING,
    RequestOutcome.WAITING_ON_KV: RequestState.WAITING_KV,
    RequestOutcome.ADMITTED: RequestState.ADMITTED,
    RequestOutcome.PREFILL_CHUNK: RequestState.PREFILL,
    RequestOutcome.PREFILL_DONE: RequestState.DECODING,
    RequestOutcome.DECODED: None,
    RequestOutcome.STREAMED: RequestState.STREAMING,
    RequestOutcome.PREEMPTED_RECOMPUTE: RequestState.PREEMPTED,
    RequestOutcome.PREEMPTED_SWAP: RequestState.SWAPPED,
    RequestOutcome.RESUMED: RequestState.WAITING,
    RequestOutcome.FINISHED: RequestState.FINISHED,
    RequestOutcome.ABORTED: RequestState.CANCELLED,
    RequestOutcome.FAILED: RequestState.FAILED,
}


class LifecycleManager:
    """Registry + the outcome→transition translation."""

    __slots__ = ("_active", "_finished", "_last_step_id", "_max_retries")

    def __init__(self, *, max_retries: int = _MAX_RETRIES) -> None:
        self._active: dict[str, RequestLifecycle] = {}
        self._finished: dict[str, RequestLifecycle] = {}
        self._max_retries = max_retries
        self._last_step_id = -1

    def create(self, request: Request) -> RequestLifecycle:
        rid = str(request.request_id)
        if rid in self._active or rid in self._finished:
            raise ValueError(f"duplicate request_id {rid!r}")
        lc = RequestLifecycle(request, max_retries=self._max_retries)
        self._active[rid] = lc
        return lc

    def get(self, request_id: str) -> RequestLifecycle:
        lc = self._active.get(request_id) or self._finished.get(request_id)
        if lc is None:
            raise UnknownRequest(request_id)
        return lc

    def find(self, request_id: str) -> RequestLifecycle | None:
        return self._active.get(request_id) or self._finished.get(request_id)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._active or request_id in self._finished

    def __iter__(self) -> Iterator[RequestLifecycle]:
        """Iterates the **active** set only.  A snapshot, because cancellation
        from another thread can retire an entry mid-walk."""
        return iter(tuple(self._active.values()))

    @property
    def num_active(self) -> int:
        return len(self._active)

    @property
    def num_finished(self) -> int:
        return len(self._finished)

    # admission side
    def advance_to_queue(self, request_id: str, *, num_cached_tokens: int = 0) -> None:
        """CREATED → … → WAITING.  The pre-scheduler pipeline is synchronous and
        has no scheduler involvement, so it is driven directly rather than
        through a report."""
        m = self.get(request_id).machine
        m.on_validating()
        m.on_validated()
        m.on_tokenized(self.get(request_id).request.prompt_len)
        m.on_cache_looked_up(num_cached_tokens=num_cached_tokens)

    # the translation
    def apply(self, report: SchedulerReport) -> None:
        if report.step_id <= self._last_step_id:
            raise ValueError(
                f"step_id {report.step_id} is not ahead of the last applied "
                f"{self._last_step_id}: reports must be applied in order, or the "
                "state machine replays a step it already accounted for"
            )
        for entry in report.reports:
            self.apply_one(entry, step_id=report.step_id)
        self._last_step_id = report.step_id

    def apply_one(self, entry: RequestReport, *, step_id: int, now_ns: int | None = None) -> None:
        lc = self._active.get(entry.request_id)
        if lc is None:
            # A report for an already-retired request is normal: the scheduler
            # sees the abort one step later than the API thread does.
            if entry.request_id in self._finished:
                return
            raise UnknownRequest(entry.request_id)

        if entry.sequence_epoch != lc.sequence_epoch:
            return
        m = lc.machine
        outcome = entry.outcome

        if outcome in (
            RequestOutcome.PREFILL_CHUNK,
            RequestOutcome.PREFILL_DONE,
            RequestOutcome.DECODED,
        ):
            scheduled = entry.completed_slice
            assert scheduled is not None and entry.committed_state_version is not None
            if scheduled.phase is Phase.PREFILL:
                expected = (
                    RequestOutcome.PREFILL_DONE
                    if scheduled.query_end == lc.request.prompt_len
                    else RequestOutcome.PREFILL_CHUNK
                )
                if outcome is not expected:
                    raise ValueError("prefill outcome disagrees with its prompt boundary")
            lc.validate_forward_commit(
                step_id,
                scheduled,
                committed_state_version=entry.committed_state_version,
            )
            lc.commit_computed_range(
                step_id,
                scheduled,
                committed_state_version=entry.committed_state_version,
                now_ns=now_ns,
            )
            if entry.new_token_ids:
                lc.publish_sample(step_id, scheduled, entry.new_token_ids[0])
            return

        target = _OUTCOME_STATE[outcome]
        if target is None:
            return

        if outcome is RequestOutcome.FINISHED:
            if lc.inflight_slice is not None:
                raise ValueError("cannot finish a request with an unfinished slice")
            lc.finish_reason = entry.finish_reason
            m.on_finished(now_ns=now_ns)
        elif outcome is RequestOutcome.ABORTED:
            lc.token.cancel(entry.detail or "aborted by scheduler")
            m.on_cancelled(now_ns=now_ns)
        elif outcome is RequestOutcome.FAILED:
            m.on_failed(entry.detail or "scheduler reported failure", now_ns=now_ns)
        elif outcome is RequestOutcome.STREAMED:
            m.on_streamed(now_ns=now_ns)
        elif outcome is RequestOutcome.QUEUED and m.state is RequestState.WAITING:
            pass  # already queued; QUEUED is idempotent by design
        else:
            m.transition(target, now_ns=now_ns)

        self._retire_if_terminal(lc)

    def _retire_if_terminal(self, lc: RequestLifecycle) -> None:
        self.retire_terminal(lc)

    def retire_terminal(self, lc: RequestLifecycle) -> None:
        """Retire this exact lifecycle object; never remove a replacement with the same ID."""
        if lc.is_terminal and self._active.get(lc.request_id) is lc:
            self._finished[lc.request_id] = self._active.pop(lc.request_id)

    def reap(self, *, older_than_ns: int = 0) -> tuple[str, ...]:
        """Drop finished entries whose output has been consumed.

        Explicit rather than automatic: at the moment a request goes terminal the
        client has not read its last token yet, so dropping it at the transition
        loses output.
        """
        if not older_than_ns:
            reaped = tuple(self._finished)
            self._finished.clear()
            return reaped
        cutoff = time.monotonic_ns() - older_than_ns
        reaped = tuple(
            rid for rid, lc in self._finished.items() if lc.machine.finished_ns <= cutoff
        )
        for rid in reaped:
            del self._finished[rid]
        return reaped

    def cancel(self, request_id: str, reason: str = "client disconnect") -> bool:
        """Signal cancellation.  Returns True if this call did it.

        Note what this does *not* do: it does not free KV.  The request is still
        in the scheduler's running set and its blocks are still pinned; the
        scheduler notices the token on its next pass and reports ABORTED, and
        only then is the state terminal.  Freeing here would race the executor
        reading those blocks in a step that is already in flight.
        """
        lc = self._active.get(request_id)
        if lc is None:
            return False
        return lc.token.cancel(reason)

    def cancelled_requests(self) -> tuple[RequestLifecycle, ...]:
        """Active requests whose token is set — what the scheduler polls."""
        return tuple(lc for lc in self._active.values() if lc.token.is_cancelled)
