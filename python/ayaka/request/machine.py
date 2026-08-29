from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from ayaka.request.states import (
    RUNNING_STATES,
    TERMINAL_STATES,
    RequestState,
    is_legal,
    legal_targets,
)
from ayaka.utils.timing import Clock, RequestTiming

_DEFAULT_HISTORY = 16


class IllegalTransition(RuntimeError):
    def __init__(self, request_id: str, src: RequestState, dst: RequestState) -> None:
        allowed = ", ".join(sorted(s.value for s in legal_targets(src))) or "(terminal)"
        super().__init__(
            f"{request_id}: {src.value} -> {dst.value} is not a legal transition; "
            f"from {src.value} the legal targets are: {allowed}"
        )
        self.request_id = request_id
        self.src = src
        self.dst = dst


class RequestStateMachine:
    """Guarded state + the counters the scheduler and the metrics layer read."""

    __slots__ = (
        "_history",
        "_state",
        "clock",
        "timing",
        "cancel_requested",
        "failure_reason",
        "max_retries",
        "num_cached_tokens",
        "num_computed_tokens",
        "preemption_count",
        "request_id",
        "retry_count",
        "swap_count",
    )

    def __init__(
        self,
        request_id: str,
        *,
        arrival_ns: int | None = None,
        clock: Clock | None = None,
        max_retries: int = 3,
        history_size: int = _DEFAULT_HISTORY,
    ) -> None:
        self.request_id = request_id
        self._state = RequestState.CREATED
        self.clock = clock or Clock()
        initial_ns = arrival_ns if arrival_ns is not None else self.clock.now().as_nanos()
        self.timing = RequestTiming(arrival_ns=initial_ns)
        self._history: deque[tuple[RequestState, int]] = deque(
            [(RequestState.CREATED, initial_ns)], maxlen=history_size
        )
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.preemption_count = 0
        self.swap_count = 0
        self.retry_count = 0
        self.max_retries = max_retries
        self.cancel_requested = False
        self.failure_reason = ""

    @property
    def arrival_ns(self) -> int:
        return self.timing.arrival_ns

    @property
    def first_scheduled_ns(self) -> int:
        return self.timing.first_scheduled_ns

    @property
    def first_token_ns(self) -> int:
        return self.timing.first_token_ns

    @property
    def finished_ns(self) -> int:
        return self.timing.finished_ns

    @property
    def num_generated_tokens(self) -> int:
        return self.timing.num_output_tokens

    @property
    def state(self) -> RequestState:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    @property
    def holds_kv(self) -> bool:
        """True while the request owns GPU blocks.
        The engine loop releases on the edge out of this set,
        which is why it is a property of the state and
        not a separate flag that can drift."""
        return self._state in RUNNING_STATES

    @property
    def history(self) -> tuple[tuple[RequestState, int], ...]:
        return tuple(self._history)

    def can(self, dst: RequestState) -> bool:
        return is_legal(self._state, dst)

    def transition(self, dst: RequestState, *, now_ns: int | None = None) -> RequestState:
        """Move to ``dst``.  Raises :class:`IllegalTransition` if the edge does
        not exist.  Returns the previous state so the caller can account for the
        edge without re-reading."""
        src = self._state
        if not is_legal(src, dst):
            raise IllegalTransition(self.request_id, src, dst)

        ts = now_ns if now_ns is not None else self.clock.now().as_nanos()
        self._state = dst
        self._history.append((dst, ts))

        # Latency marks.  Set-once: a preempted request that prefills twice must
        # not report the second prefill as its TTFT.
        if dst in (RequestState.PREFILL, RequestState.DECODING) and not self.first_scheduled_ns:
            self.timing.mark_scheduled(ts)
        if dst is RequestState.STREAMING:
            self.timing.mark_streamed(ts)
        if dst in TERMINAL_STATES:
            self.timing.mark_finished(ts)

        if dst is RequestState.PREEMPTED:
            self.preemption_count += 1
            # Recompute preemption drops the KV, so the prefill progress is gone.
            # Leaving this stale is how a resumed request writes its second
            # prefill on top of block indices it no longer owns.
            self.num_computed_tokens = self.num_cached_tokens
        elif dst is RequestState.SWAPPED:
            self.swap_count += 1  # KV survives in host memory; progress is kept
        elif dst is RequestState.RETRYING:
            self.retry_count += 1

        return src

    # The engine loop calls these instead of naming states, so the scheduler
    # never has to import RequestState to report an outcome.
    def on_validating(self) -> None:
        self.transition(RequestState.VALIDATING)

    def on_validated(self) -> None:
        self.transition(RequestState.TOKENIZING)

    def on_tokenized(self, num_prompt_tokens: int) -> None:
        if num_prompt_tokens < 1:
            raise ValueError(f"{self.request_id}: tokenizer produced 0 tokens")
        self.transition(RequestState.CACHE_LOOKUP)

    def on_cache_looked_up(self, num_cached_tokens: int = 0) -> None:
        self.num_cached_tokens = num_cached_tokens
        self.num_computed_tokens = num_cached_tokens
        self.transition(RequestState.WAITING)

    def on_admitted(self) -> None:
        self.transition(RequestState.ADMITTED)

    def on_prefill_started(self) -> None:
        self.transition(RequestState.PREFILL)

    def on_prefill_chunk(self, num_tokens: int) -> None:
        """A chunk of a chunked prefill.  Not a transition — the state is already
        PREFILL and stays PREFILL."""
        if self._state is not RequestState.PREFILL:
            raise IllegalTransition(self.request_id, self._state, RequestState.PREFILL)
        self.num_computed_tokens += num_tokens

    def on_decode_started(self) -> None:
        self.transition(RequestState.DECODING)

    def on_token_generated(self, count: int = 1) -> None:
        """Also not a transition: a decode step that produces a token leaves the
        request in DECODING.  STREAMING is entered only when a token is actually
        handed to the client."""
        if self._state not in (RequestState.DECODING, RequestState.STREAMING):
            raise IllegalTransition(self.request_id, self._state, RequestState.DECODING)
        self.timing.num_output_tokens += count
        self.num_computed_tokens += count

    def on_streamed(self, *, now_ns: int | None = None) -> None:
        ts = now_ns if now_ns is not None else self.clock.now().as_nanos()
        if self._state is RequestState.STREAMING:
            self.timing.mark_streamed(ts)
            return
        self.transition(RequestState.STREAMING, now_ns=ts)

    def on_preempted(self, *, swap: bool = False) -> None:
        self.transition(RequestState.SWAPPED if swap else RequestState.PREEMPTED)

    def on_requeued(self) -> None:
        self.transition(RequestState.WAITING)

    def on_finished(self) -> None:
        self.transition(RequestState.FINISHED)

    def on_cancelled(self) -> None:
        """Idempotent by design: a cancel racing with completion is normal, and
        making the loser raise would turn a race into an error path."""
        self.cancel_requested = True
        if self.is_terminal:
            return
        self.transition(RequestState.CANCELLED)

    def on_failed(self, reason: str) -> None:
        self.failure_reason = reason
        if self.is_terminal:
            return
        self.transition(RequestState.FAILED)

    def on_error(self, reason: str) -> bool:
        """Transient failure.  Returns True if a retry was started, False if the
        budget is exhausted and the request was failed instead."""
        if self.retry_count >= self.max_retries:
            self.on_failed(f"{reason} (retries exhausted: {self.retry_count})")
            return False
        self.failure_reason = reason
        self.transition(RequestState.RETRYING)
        return True

    def queue_time_ns(self) -> int:
        return self.timing.queue_time_ns() or 0

    def ttft_ns(self) -> int:
        return self.timing.ttft_ns() or 0

    def tpot_ns(self) -> int:
        """Mean inter-token latency, excluding the prefill.  Zero until at least
        two tokens exist, because one token gives no interval."""
        return self.timing.tpot_ns() or 0

    def e2e_ns(self) -> int:
        return self.timing.e2e_ns() or 0

    def __iter__(self) -> Iterator[tuple[RequestState, int]]:
        return iter(self._history)

    def __repr__(self) -> str:
        return (
            f"<RequestStateMachine {self.request_id} {self._state.value} "
            f"computed={self.num_computed_tokens} gen={self.num_generated_tokens} "
            f"preempt={self.preemption_count} swap={self.swap_count}>"
        )
