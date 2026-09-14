from __future__ import annotations

import enum
from dataclasses import dataclass

from ayaka.sched.plan import Phase, ScheduledSlice
from ayaka.utils.validation import require_frozen, require_int, require_text

__all__ = ["FinishReason", "RequestOutcome", "RequestReport", "SchedulerReport"]


class RequestOutcome(enum.StrEnum):
    """One thing the scheduler decided about one request in one step.

    These are *observations*, not states.  The mapping to states lives in the
    lifecycle manager and is tested there, so a new outcome cannot silently
    imply a transition nobody reviewed.
    """

    QUEUED = "queued"  # entered the waiting queue
    WAITING_ON_KV = "waiting_on_kv"  # queued, blocked on block-pool capacity
    ADMITTED = "admitted"  # blocks pinned, nothing run yet
    PREFILL_CHUNK = "prefill_chunk"  # ran a chunk, prompt not complete
    PREFILL_DONE = "prefill_done"  # prompt fully computed
    DECODED = "decoded"  # completed one decode input; sampling is explicit
    STREAMED = "streamed"  # a token was handed to the client
    PREEMPTED_RECOMPUTE = "preempted_recompute"  # evicted, KV dropped
    PREEMPTED_SWAP = "preempted_swap"  # evicted, KV parked on the host
    RESUMED = "resumed"  # re-entered the waiting queue after eviction
    FINISHED = "finished"
    ABORTED = "aborted"
    FAILED = "failed"


class FinishReason(enum.StrEnum):
    STOP = "stop"  # stop token or stop string
    LENGTH = "length"  # max_tokens
    EOS = "eos"
    ABORT = "abort"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RequestReport:
    """An incarnation-bound observation, with explicit forward/sample evidence.

    Compute outcomes require the completed slice, the version returned by KV
    commit, and actual sampled IDs (if requested). Token counts are derived;
    a count-only report can no longer manufacture one output token.
    This is data supplied by a completion owner, not a completion proof.
    """

    request_id: str
    outcome: RequestOutcome
    sequence_epoch: int
    completed_slice: ScheduledSlice | None = None
    committed_state_version: int | None = None
    new_token_ids: tuple[int, ...] = ()
    num_cached_tokens: int = 0
    finish_reason: FinishReason | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        require_frozen(self, "request report")
        require_text(self.request_id, "request_id")
        require_int(self.sequence_epoch, "sequence_epoch", minimum=1)
        require_int(self.num_cached_tokens, "num_cached_tokens")
        if not isinstance(self.outcome, RequestOutcome):
            raise TypeError("outcome must be RequestOutcome")
        if self.outcome is RequestOutcome.FINISHED and self.finish_reason is None:
            raise ValueError(f"{self.request_id}: FINISHED report without a finish_reason")
        for token in self.new_token_ids:
            require_int(token, "sample token id")
        computational = self.outcome in (
            RequestOutcome.PREFILL_CHUNK,
            RequestOutcome.PREFILL_DONE,
            RequestOutcome.DECODED,
        )
        if not computational:
            if self.completed_slice is not None or self.committed_state_version is not None:
                raise ValueError("only compute outcomes may credit a forward range")
            if self.new_token_ids:
                raise ValueError("only a completed sampling slice may publish output")
            return
        scheduled = self.completed_slice
        if scheduled is None or self.committed_state_version is None:
            raise ValueError("compute reports require a slice and committed KV version")
        if (
            scheduled.request_id != self.request_id
            or scheduled.sequence_epoch != self.sequence_epoch
        ):
            raise ValueError("report and slice incarnations disagree")
        require_int(self.committed_state_version, "committed_state_version")
        if self.committed_state_version != scheduled.expected_state_version + 1:
            raise ValueError("committed KV version must advance exactly once")
        if (self.outcome is RequestOutcome.DECODED) != (scheduled.phase is Phase.DECODE):
            raise ValueError("report outcome disagrees with the explicit slice phase")
        if self.outcome is RequestOutcome.PREFILL_CHUNK and scheduled.sample_last_query:
            raise ValueError("an unfinished prompt chunk cannot sample")
        if len(self.new_token_ids) != int(scheduled.sample_last_query):
            raise ValueError("sampled token IDs must match the slice sampling requirement")

    @property
    def num_scheduled_tokens(self) -> int:
        return 0 if self.completed_slice is None else self.completed_slice.query_count

    @property
    def num_new_tokens(self) -> int:
        return len(self.new_token_ids)


@dataclass(frozen=True, slots=True)
class SchedulerReport:
    """One step's worth of reports plus the queue snapshot the metrics layer
    reads.  Immutable so it can be fanned out to the logger, the tracer and the
    lifecycle manager without anyone defensively copying it."""

    step_id: int
    reports: tuple[RequestReport, ...] = ()
    num_running: int = 0
    num_waiting: int = 0
    num_free_blocks: int = 0
    num_preempted_this_step: int = 0

    def __post_init__(self) -> None:
        require_frozen(self, "scheduler report")
        require_int(self.step_id, "step_id")
        for name in ("num_running", "num_waiting", "num_free_blocks", "num_preempted_this_step"):
            require_int(getattr(self, name), name)
        if len({r.request_id for r in self.reports}) != len(self.reports):
            raise ValueError("one report per request is allowed in a step")

    def for_request(self, request_id: str) -> RequestReport | None:
        for r in self.reports:
            if r.request_id == request_id:
                return r
        return None
