from __future__ import annotations

import enum
from dataclasses import dataclass

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
    DECODED = "decoded"  # produced one token
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
    request_id: str
    outcome: RequestOutcome
    num_scheduled_tokens: int = 0
    num_cached_tokens: int = 0
    num_new_tokens: int = 0  # tokens generated this step
    finish_reason: FinishReason | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.outcome is RequestOutcome.FINISHED and self.finish_reason is None:
            raise ValueError(f"{self.request_id}: FINISHED report without a finish_reason")
        if self.num_scheduled_tokens < 0 or self.num_new_tokens < 0:
            raise ValueError(f"{self.request_id}: negative token count")


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

    def for_request(self, request_id: str) -> RequestReport | None:
        for r in self.reports:
            if r.request_id == request_id:
                return r
        return None
