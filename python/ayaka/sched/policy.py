"""Deterministic, aging-bounded decode priority for continuous batching."""

from dataclasses import dataclass

from ayaka.handles import SequenceHandle
from ayaka.request.lifecycle import RequestLifecycle


@dataclass(slots=True)
class QueueEntry:
    """Queue membership references the authoritative request lifecycle."""

    lifecycle: RequestLifecycle
    ordinal: int
    ready_round: int
    ready_ns: int
    sequence: SequenceHandle | None = None


def order(entries, *, round_id: int, now_ns: int, max_bypass: int):
    """Aged requests first; otherwise decode, slack, priority, arrival, FIFO.

    Among aged requests the oldest last-service round wins, so a stream of new
    arrivals cannot displace an already waiting request. This bounds opportunity
    count for feasible requests, not wall time during capacity exhaustion.
    """

    def key(entry):
        lc = entry.lifecycle
        aged = round_id - entry.ready_round >= max_bypass
        if aged:
            return (0, entry.ready_round, entry.ordinal, 0, 0, 0)
        request = lc.request
        deadline = request.deadline_ns
        slack = deadline - now_ns if deadline is not None else float("inf")
        prefill = lc.computed_tokens < request.prompt_len
        return (1, int(prefill), slack, -request.priority, lc.machine.arrival_ns, entry.ordinal)

    return sorted(entries, key=key)
