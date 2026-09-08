from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from itertools import count

from ayaka.request.states import (
    RUNNING_STATES,
    TERMINAL_STATES,
    RequestState,
    is_legal,
    legal_targets,
)
from ayaka.utils.timing import Clock, RequestTiming
from ayaka.utils.validation import require_int

_DEFAULT_HISTORY = 16
_REQUEST_EPOCHS = count(1)
# Process-local incarnations. A restarted/distributed worker needs its own
# worker generation as well; that protocol belongs to the runtime.


def next_request_epoch() -> int:
    """Allocate an incarnation never reused within this process."""
    return next(_REQUEST_EPOCHS)


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
        "num_prompt_tokens",
        "sequence_epoch",
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
        self.num_prompt_tokens = 0
        self.sequence_epoch = next_request_epoch()
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
            self.num_cached_tokens = 0
            self.num_computed_tokens = 0
        elif dst is RequestState.SWAPPED:
            self.swap_count += 1  # KV survives in host memory; progress is kept
        elif dst is RequestState.RETRYING:
            self.retry_count += 1

        if dst in (
            RequestState.PREEMPTED,
            RequestState.SWAPPED,
            RequestState.RETRYING,
            RequestState.CANCELLED,
            RequestState.FAILED,
        ):
            self.sequence_epoch = next_request_epoch()
        return src

    # The engine loop calls these instead of naming states, so the scheduler
    # never has to import RequestState to report an outcome.
    def on_validating(self) -> None:
        self.transition(RequestState.VALIDATING)

    def on_validated(self) -> None:
        self.transition(RequestState.TOKENIZING)

    def on_tokenized(self, num_prompt_tokens: int) -> None:
        require_int(num_prompt_tokens, "num_prompt_tokens", minimum=1)
        self.transition(RequestState.CACHE_LOOKUP)
        self.num_prompt_tokens = num_prompt_tokens

    def on_cache_looked_up(self, num_cached_tokens: int = 0) -> None:
        """Accept already validated/acquired prefix progress, not a raw lookup."""
        require_int(num_cached_tokens, "num_cached_tokens")
        if num_cached_tokens > self.num_prompt_tokens + self.num_generated_tokens:
            raise ValueError("cached progress exceeds known tokens")
        self.transition(RequestState.WAITING)
        self.num_cached_tokens = num_cached_tokens
        self.num_computed_tokens = num_cached_tokens

    def on_admitted(self) -> None:
        self.transition(RequestState.ADMITTED)

    def on_prefill_started(self) -> None:
        self.transition(RequestState.PREFILL)

    def commit_computed_range(self, query_start: int, query_count: int) -> None:
        """Credit a contiguous forward range after successful KV/state commit.

        This low-level counter API does not prove completion. RequestLifecycle
        additionally checks slice identity/version and its input snapshot.
        Sampling is a separate operation and never advances this counter.
        """
        require_int(query_start, "query_start")
        require_int(query_count, "query_count", minimum=1)
        if self._state not in (
            RequestState.PREFILL,
            RequestState.DECODING,
            RequestState.STREAMING,
        ):
            raise ValueError("forward progress requires an executing request")
        if query_start != self.num_computed_tokens:
            raise ValueError("forward range must start at computed progress")
        if query_start + query_count > self.num_prompt_tokens + self.num_generated_tokens:
            raise ValueError("forward range exceeds known tokens")
        self.num_computed_tokens = query_start + query_count

    def on_prefill_chunk(self, num_tokens: int) -> None:
        """Credit a completed prompt chunk; the request remains in PREFILL."""
        if self._state is not RequestState.PREFILL:
            raise IllegalTransition(self.request_id, self._state, RequestState.PREFILL)
        require_int(num_tokens, "num_tokens", minimum=1)
        if self.num_computed_tokens + num_tokens > self.num_prompt_tokens:
            raise ValueError("prefill chunk extends past the prompt")
        self.commit_computed_range(self.num_computed_tokens, num_tokens)

    def on_decode_started(self) -> None:
        if self.num_computed_tokens < self.num_prompt_tokens:
            raise ValueError("decode cannot start before prompt computation completes")
        self.transition(RequestState.DECODING)

    def on_token_generated(self, count: int = 1) -> None:
        """Publish a sample count without claiming KV for that new token.

        Zero is an explicit no-op. P0 supports at most one sample from a
        forward and requires every currently known token to be computed.
        Lifecycle users publish the token ID through publish_sample instead
        of updating this metric independently from their output buffer.
        """
        require_int(count, "count")
        if count == 0:
            return
        if count != 1:
            raise ValueError("P0 supports one sample per completed forward")
        if self._state not in (RequestState.DECODING, RequestState.STREAMING):
            raise IllegalTransition(self.request_id, self._state, RequestState.DECODING)
        if self.num_computed_tokens != self.num_prompt_tokens + self.num_generated_tokens:
            raise ValueError("sampling requires the completed known-token boundary")
        self.timing.num_output_tokens += count

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

    def on_finished(self, *, now_ns: int | None = None) -> None:
        self.transition(RequestState.FINISHED, now_ns=now_ns)

    def on_cancelled(self, *, now_ns: int | None = None) -> None:
        """Idempotent by design: a cancel racing with completion is normal, and
        making the loser raise would turn a race into an error path."""
        self.cancel_requested = True
        if self.is_terminal:
            return
        self.transition(RequestState.CANCELLED, now_ns=now_ns)

    def on_failed(self, reason: str, *, now_ns: int | None = None) -> None:
        self.failure_reason = reason
        if self.is_terminal:
            return
        self.transition(RequestState.FAILED, now_ns=now_ns)

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
