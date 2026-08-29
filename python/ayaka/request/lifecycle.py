from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

from ayaka.request.schema import Request
from ayaka.request.cancel import CancellationToken
from ayaka.request.machine import RequestStateMachine
from ayaka.request.states import RequestState
from ayaka.sched.outcome import (
    FinishReason,
    RequestOutcome,
    RequestReport,
    SchedulerReport,
)

_MAX_RETRIES = 3

class UnknownRequest(KeyError):
    """A report referenced a request the manager has never seen."""


class RequestLifecycle:
    """The frozen IR, the mutable state, and the output, in one place.

    ``__slots__`` because there is one per in-flight request and the engine loop
    walks all of them every step.
    """

    __slots__ = ("finish_reason", "machine", "output_token_ids", "request", "token")

    def __init__(self, request: Request, *, max_retries: int = _MAX_RETRIES) -> None:
        self.request = request
        self.machine = RequestStateMachine(
            str(request.request_id),
            arrival_ns=request.arrival_ns or time.monotonic_ns(),
            max_retries=max_retries,
        )
        self.token = CancellationToken(str(request.request_id))
        self.output_token_ids: list[int] = []
        self.finish_reason: FinishReason | None = None

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
        return self.machine.holds_kv

    def append_tokens(self, token_ids: Iterable[int]) -> None:
        self.output_token_ids.extend(token_ids)

    def __repr__(self) -> str:
        return (
            f"<RequestLifecycle {self.request_id} {self.state.value} "
            f"out={len(self.output_token_ids)}>"
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
    __slots__ = (
        "_active", 
        "_finished", 
        "_last_step_id", 
        "_max_retries"
    )

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
            self.apply_one(entry)
        self._last_step_id = report.step_id
        
    def apply_one(self, entry: RequestReport) -> None:
        lc = self._active.get(entry.request_id)
        if lc is None:
            # A report for an already-retired request is normal: the scheduler
            # sees the abort one step later than the API thread does.
            if entry.request_id in self._finished:
                return
            raise UnknownRequest(entry.request_id)

        m = lc.machine
        outcome = entry.outcome

        # Counters first: a preemption transition rewinds num_computed_tokens,
        # so crediting this step's work afterwards would credit work that the
        # eviction just threw away.
        if outcome is RequestOutcome.PREFILL_CHUNK:
            if m.state is not RequestState.PREFILL:
                m.transition(RequestState.PREFILL)
            m.on_prefill_chunk(entry.num_scheduled_tokens)
            self._retire_if_terminal(lc)
            return
        if outcome is RequestOutcome.DECODED:
            m.on_token_generated(entry.num_new_tokens or 1)
            return
        # PREFILL_DONE arrives in two shapes: the tail of a chunked prefill
        # (already in PREFILL) and a single-shot prefill reported straight from
        # ADMITTED.  Both must credit their tokens, or a non-chunked engine
        # silently reports num_computed_tokens == 0 for every request.
        if outcome is RequestOutcome.PREFILL_DONE and entry.num_scheduled_tokens:
            if m.state is not RequestState.PREFILL:
                m.transition(RequestState.PREFILL)
            m.on_prefill_chunk(entry.num_scheduled_tokens)

        target = _OUTCOME_STATE[outcome]
        if target is None:
            return

        if outcome is RequestOutcome.FINISHED:
            lc.finish_reason = entry.finish_reason
            m.on_finished()
        elif outcome is RequestOutcome.ABORTED:
            lc.token.cancel(entry.detail or "aborted by scheduler")
            m.on_cancelled()
        elif outcome is RequestOutcome.FAILED:
            m.on_failed(entry.detail or "scheduler reported failure")
        elif outcome is RequestOutcome.STREAMED:
            m.on_streamed()
        elif outcome is RequestOutcome.QUEUED and m.state is RequestState.WAITING:
            pass  # already queued; QUEUED is idempotent by design
        else:
            m.transition(target)

        self._retire_if_terminal(lc)
        
    def _retire_if_terminal(self, lc: RequestLifecycle) -> None:
        if lc.is_terminal:
            self._finished[lc.request_id] = self._active.pop(lc.request_id, lc)

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