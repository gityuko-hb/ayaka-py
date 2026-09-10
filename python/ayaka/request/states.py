from __future__ import annotations

import enum
from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "AUXILIARY_STATES",
    "MAIN_STATES",
    "RUNNING_STATES",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "RequestState",
    "is_legal",
    "legal_targets",
]


class RequestState(enum.StrEnum):
    """Finite State Machine enumeration tracking an inference request's full lifecycle."""

    # 10 Main Pipeline States
    CREATED = "created"
    """The request has been instantiated from the client payload but not yet validated."""

    VALIDATING = "validating"
    """The request parameters
    (sampling parameters, max tokens, stop sequences) are being verified."""

    TOKENIZING = "tokenizing"
    """Raw prompt strings are being encoded into token ID sequences."""

    CACHE_LOOKUP = "cache_lookup"
    """The prefix-caching index (Radix Tree / Hash Table) is being queried for KV block hits."""

    WAITING = "waiting"
    """The request is queued in the scheduler, waiting for compute slots and KV block allocation."""

    ADMITTED = "admitted"
    """KV blocks have been pinned in physical GPU memory,
        but no computation kernel has executed yet."""

    PREFILL = "prefill"
    """The engine is executing context prefill kernels over un-cached prompt tokens."""

    DECODING = "decoding"
    """The engine is executing autoregressive decode iterations generating one token at a time."""

    STREAMING = "streaming"
    """The newly generated token is being pushed to the client stream (SSE / gRPC / WebSocket)."""

    FINISHED = "finished"
    """Terminal state: generation completed successfully
        (hit stop sequence, EOS, or max token limits)."""

    # 9 Auxiliary & Scheduling States
    WAITING_KV = "waiting_kv"
    """The request passed admission criteria but is stalled
        waiting for local GPU KV blocks to free up."""

    WAITING_REMOTE_KV = "waiting_remote_kv"
    """Stalled waiting for remote/disaggregated KV cache blocks to transfer across the network."""

    PREEMPTED = "preempted"
    """Evicted under GPU memory pressure; KV blocks dropped.
        Re-admission requires full prompt recomputation."""

    SWAPPED = "swapped"
    """Evicted under GPU memory pressure; KV blocks swapped to host-pinned RAM.
        Re-admission requires H2D copy only."""

    PAUSED = "paused"
    """Temporarily suspended due to client backpressure or rate limits;
        GPU KV blocks remain held."""

    MIGRATING = "migrating"
    """Active KV state is being transferred across ranks/nodes
        (Prefill-to-Decode disaggregation or rebalancing)."""

    RETRYING = "retrying"
    """Transient engine error encountered;
    request is restarting from the admission queue or cache lookup."""

    CANCELLED = "cancelled"
    """Terminal state: aborted early by client disconnect, timeout,
    or explicit cancellation signal."""

    FAILED = "failed"
    """Terminal state: unrecoverable runtime failure
    (e.g., CUDA OOM, driver crash, model execution fault)."""


MAIN_STATES: frozenset[RequestState] = frozenset(
    {
        RequestState.CREATED,
        RequestState.VALIDATING,
        RequestState.TOKENIZING,
        RequestState.CACHE_LOOKUP,
        RequestState.WAITING,
        RequestState.ADMITTED,
        RequestState.PREFILL,
        RequestState.DECODING,
        RequestState.STREAMING,
        RequestState.FINISHED,
    }
)

AUXILIARY_STATES: frozenset[RequestState] = frozenset(RequestState) - MAIN_STATES

TERMINAL_STATES: frozenset[RequestState] = frozenset(
    {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED}
)

#: States in which the request holds GPU KV blocks.
#: release() must be called
#: exactly once when transitioning to any state outside this set.
RUNNING_STATES: frozenset[RequestState] = frozenset(
    {
        RequestState.ADMITTED,
        RequestState.PREFILL,
        RequestState.DECODING,
        RequestState.STREAMING,
        RequestState.PAUSED,
        RequestState.MIGRATING,
    }
)


def _build() -> Mapping[RequestState, frozenset[RequestState]]:
    S = RequestState
    table: dict[RequestState, set[RequestState]] = {
        S.CREATED: {S.VALIDATING},
        S.VALIDATING: {S.TOKENIZING},
        S.TOKENIZING: {S.CACHE_LOOKUP},
        # Remote prefix lookup is the only asynchronous step before local queueing
        S.CACHE_LOOKUP: {S.WAITING, S.WAITING_REMOTE_KV},
        S.WAITING: {S.ADMITTED, S.WAITING_KV},
        S.WAITING_KV: {S.WAITING, S.ADMITTED},
        S.WAITING_REMOTE_KV: {S.WAITING, S.CACHE_LOOKUP},
        # ADMITTED: blocks pinned, ready for compute
        S.ADMITTED: {S.PREFILL, S.DECODING, S.PREEMPTED, S.SWAPPED, S.MIGRATING},
        # Full prefix-cache hit transitions ADMITTED -> DECODING directly
        S.PREFILL: {
            S.DECODING,
            S.PREEMPTED,
            S.SWAPPED,
            S.PAUSED,
            S.MIGRATING,
            S.RETRYING,
            S.FINISHED,  # max_tokens reached on prompt itself
        },
        S.DECODING: {
            S.STREAMING,
            S.FINISHED,
            S.PREEMPTED,
            S.SWAPPED,
            S.PAUSED,
            S.MIGRATING,
            S.RETRYING,
        },
        S.STREAMING: {
            S.DECODING,
            S.FINISHED,
            S.PAUSED,
            S.PREEMPTED,
            S.SWAPPED,
        },
        # Recompute path: KV is gone, re-enters admission queue from scratch
        S.PREEMPTED: {S.WAITING},
        # Swapped path: KV is in host memory, can be admitted or demoted to preempted
        S.SWAPPED: {S.WAITING, S.ADMITTED, S.PREEMPTED},
        # Paused path: holds KV, can resume or be evicted under memory pressure
        S.PAUSED: {
            S.PREFILL,
            S.DECODING,
            S.STREAMING,
            S.PREEMPTED,
            S.SWAPPED,
        },
        S.MIGRATING: {S.ADMITTED, S.PREFILL, S.DECODING, S.RETRYING},
        # Retry restarts from queue or cache lookup; never PREFILL
        S.RETRYING: {S.WAITING, S.CACHE_LOOKUP},
        S.FINISHED: set(),
        S.CANCELLED: set(),
        S.FAILED: set(),
    }

    # Cancellation and failure are valid exits from any non-terminal state
    for state, targets in table.items():
        if state not in TERMINAL_STATES:
            targets.add(S.CANCELLED)
            targets.add(S.FAILED)

    return MappingProxyType({k: frozenset(v) for k, v in table.items()})


#: Read-only mapping of state to legal successors. Self-transitions are disallowed.
TRANSITIONS: Mapping[RequestState, frozenset[RequestState]] = _build()


def legal_targets(state: RequestState) -> frozenset[RequestState]:
    """Return all valid next states for a given request state."""
    return TRANSITIONS[state]


def is_legal(src: RequestState, dst: RequestState) -> bool:
    """Validate whether transitioning from `src` to `dst` is permitted."""
    return dst in TRANSITIONS[src]
