"""Pure waiting-request ranking and a separate aging override.

Phase budgets and preemption victims belong to the concrete scheduler. None of
these functions allocates KV, looks up prefixes or proves admission progress.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ayaka.configs.base import ConfigError
from ayaka.configs.scheduler import SchedulingPolicy
from ayaka.handles import SequenceHandle
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.utils.validation import require_int


@dataclass(slots=True)
class QueueEntry:
    """Queue membership references the authoritative request lifecycle.

    ordinal is assigned once at admission and preserved on requeue. ready_ns
    controls eligibility, not arrival order; callers pass only eligible entries.
    ready_round tracks the last service/aging reset without rewriting arrival.
    """

    lifecycle: RequestLifecycle
    ordinal: int
    ready_round: int
    ready_ns: int
    sequence: SequenceHandle | None = None


def rank_waiting(
    entries: Iterable[QueueEntry],
    *,
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FCFS,
) -> list[QueueEntry]:
    """Rank eligible requests by arrival or descending priority, then arrival.

    Ties use the admission ordinal. Phase, ready time, deadline and cached-token
    counters do not affect this order. EDF and cache-aware ranking are unsupported.
    """
    if not isinstance(scheduling_policy, SchedulingPolicy):
        raise ConfigError(
            "scheduler.scheduling_policy", "INVALID_ENUM", "expected SchedulingPolicy"
        )
    if scheduling_policy not in (SchedulingPolicy.FCFS, SchedulingPolicy.PRIORITY):
        raise ConfigError(
            "scheduler.scheduling_policy", "UNSUPPORTED_POLICY", "baseline supports FCFS/PRIORITY"
        )

    def key(entry: QueueEntry) -> tuple[int, int, int]:
        request = entry.lifecycle.request
        priority = -request.priority if scheduling_policy is SchedulingPolicy.PRIORITY else 0
        return priority, request.arrival_ns, entry.ordinal

    return sorted(entries, key=key)


def order(
    entries: Iterable[QueueEntry],
    *,
    round_id: int,
    now_ns: int,
    max_bypass: int,
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FCFS,
) -> list[QueueEntry]:
    """Apply aging to waiting rank, without allocating phase or resource budgets.

    Aged requests precede the policy order, oldest service round first. This is an
    opportunity to be considered, not a starvation guarantee: actual reservation
    must enforce admission fairness. now_ns remains for call-site compatibility;
    eligibility is the caller's responsibility and deadlines do not change rank.
    """
    require_int(round_id, "round_id")
    require_int(now_ns, "now_ns")
    require_int(max_bypass, "max_bypass", minimum=1)
    ranked = rank_waiting(entries, scheduling_policy=scheduling_policy)
    aged = sorted(
        (entry for entry in ranked if round_id - entry.ready_round >= max_bypass),
        key=lambda entry: (entry.ready_round, entry.ordinal),
    )
    return aged + [entry for entry in ranked if round_id - entry.ready_round < max_bypass]
