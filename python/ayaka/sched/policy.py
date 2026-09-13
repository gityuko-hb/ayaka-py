"""Deterministic, aging-bounded policies for continuous batching."""

from collections.abc import Iterable
from dataclasses import dataclass

from ayaka.configs.scheduler import SchedulingPolicy
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


def order(
    entries: Iterable[QueueEntry],
    *,
    round_id: int,
    now_ns: int,
    max_bypass: int,
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FCFS,
) -> list[QueueEntry]:
    """Order runnable work by phase, selected policy, and bounded aging.

    Among aged requests the oldest last-service round wins, so a stream of new
    arrivals cannot displace an already waiting request. This bounds opportunity
    count for feasible requests, not wall time during capacity exhaustion. Decode
    work remains ahead of prefill work to preserve inter-token latency; the selected
    policy orders requests within each phase.
    """

    if not isinstance(scheduling_policy, SchedulingPolicy):
        raise TypeError("scheduling_policy must be SchedulingPolicy")

    def key(entry):
        lc = entry.lifecycle
        aged = round_id - entry.ready_round >= max_bypass
        if aged:
            return (0, 0, entry.ready_round, entry.ordinal, 0, 0)
        request = lc.request
        prefill = lc.computed_tokens < request.prompt_len

        if scheduling_policy is SchedulingPolicy.FCFS:
            policy_key = (entry.ready_ns, entry.ordinal, 0, 0)
        elif scheduling_policy is SchedulingPolicy.PRIORITY:
            deadline = request.deadline_ns
            slack = deadline - now_ns if deadline is not None else float("inf")
            policy_key = (slack, -request.priority, lc.machine.arrival_ns, entry.ordinal)
        elif scheduling_policy is SchedulingPolicy.LONGEST_PREFIX_MATCH:
            policy_key = (-lc.machine.num_cached_tokens, entry.ready_ns, entry.ordinal, 0)
        elif scheduling_policy is SchedulingPolicy.SHORTEST_REMAINING:
            remaining = request.max_total_len - lc.computed_tokens
            policy_key = (remaining, entry.ready_ns, entry.ordinal, 0)
        else:  # pragma: no cover - forces an update when the enum grows
            raise AssertionError(f"unhandled scheduling policy: {scheduling_policy}")
        return (1, int(prefill), *policy_key)

    return sorted(entries, key=key)
