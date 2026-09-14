"""Pure scheduler ordering policy.

This module deliberately does not allocate KV or mutate request lifecycle state.
It can consume cache-affinity *hints*, but physical prefix ownership remains the
runtime/cache manager's responsibility.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ayaka.configs.base import ConfigError
from ayaka.configs.scheduler import SchedulingPolicy
from ayaka.handles import SequenceHandle
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.utils.validation import require_int


@dataclass(slots=True)
class QueueEntry:
    """Stable queue membership plus non-authoritative scheduling hints.

    ``ordinal`` never changes and is the deterministic FCFS tie breaker.
    ``ready_round`` is reset when a request is requeued after service/preemption.
    ``cache_hint_tokens`` is advisory only; it must never be used as proof that
    KV is resident or acquired.
    """

    lifecycle: RequestLifecycle
    ordinal: int
    ready_round: int
    ready_ns: int
    sequence: SequenceHandle | None = None
    cache_hint_tokens: int = 0
    bypass_count: int = 0

    def __post_init__(self) -> None:
        require_int(self.ordinal, "ordinal")
        require_int(self.ready_round, "ready_round")
        require_int(self.ready_ns, "ready_ns")
        require_int(self.cache_hint_tokens, "cache_hint_tokens")
        require_int(self.bypass_count, "bypass_count")


def _policy_name(policy: SchedulingPolicy | str | Any) -> str:
    value = getattr(policy, "value", policy)
    return str(value).lower().replace("-", "_")


def _priority(entry: QueueEntry) -> int:
    return int(getattr(entry.lifecycle.request, "priority", 0) or 0)


def _arrival_ns(entry: QueueEntry) -> int:
    value = getattr(entry.lifecycle.request, "arrival_ns", None)
    return entry.ready_ns if value is None else int(value)


def _max_output_tokens(entry: QueueEntry) -> int:
    request = entry.lifecycle.request
    for obj in (getattr(request, "stop", None), getattr(request, "sampling", None), request):
        if obj is None:
            continue
        for name in ("max_tokens", "max_output_tokens", "max_new_tokens"):
            value = getattr(obj, name, None)
            if type(value) is int:
                return value
    return 0


def _fcfs_key(entry: QueueEntry) -> tuple[int, int]:
    return _arrival_ns(entry), entry.ordinal


def _priority_key(entry: QueueEntry) -> tuple[int, int, int]:
    # Ayaka baseline used larger integer => higher priority; preserve it.
    return -_priority(entry), _arrival_ns(entry), entry.ordinal


def _cache_affinity_key(entry: QueueEntry) -> tuple[int, int, int]:
    return -entry.cache_hint_tokens, _arrival_ns(entry), entry.ordinal


def _longest_output_key(entry: QueueEntry) -> tuple[int, int, int]:
    return -_max_output_tokens(entry), _arrival_ns(entry), entry.ordinal


def rank_waiting(
    entries: Iterable[QueueEntry],
    *,
    scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.FCFS,
) -> list[QueueEntry]:
    """Rank eligible requests without touching resources.

    Supported names are intentionally a superset of the baseline config:
    ``fcfs``, ``priority``, ``lpm``/``cache_affinity`` and ``lof``. Projects
    whose ``SchedulingPolicy`` enum only exposes FCFS/PRIORITY continue to work.

    LPM is cache-*affinity* ranking only. The runtime must still validate and
    acquire the hinted prefix before execution.
    """

    ranked = list(entries)
    name = _policy_name(scheduling_policy)

    key: Callable[[QueueEntry], tuple[int, ...]]
    if name == "fcfs":
        key = _fcfs_key
    elif name == "priority":
        key = _priority_key
    elif name in {"lpm", "longest_prefix", "cache_affinity"}:
        key = _cache_affinity_key
    elif name in {"lof", "longest_output"}:
        key = _longest_output_key
    else:
        raise ConfigError(
            "scheduler.scheduling_policy",
            "UNSUPPORTED_POLICY",
            f"unsupported scheduling policy: {name}",
        )
    return sorted(ranked, key=key)


def order(
    entries: Iterable[QueueEntry],
    *,
    round_id: int,
    now_ns: int,
    max_bypass: int,
    scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.FCFS,
) -> list[QueueEntry]:
    """Apply a bounded-starvation override on top of the selected policy.

    Aging is deterministic: entries that have waited ``max_bypass`` scheduler
    rounds are considered first, oldest service round first. Within the normal
    population the selected policy is preserved.
    """

    require_int(round_id, "round_id")
    require_int(now_ns, "now_ns")
    require_int(max_bypass, "max_bypass", minimum=1)
    ranked = rank_waiting(entries, scheduling_policy=scheduling_policy)
    aged = sorted(
        (entry for entry in ranked if round_id - entry.ready_round >= max_bypass),
        key=lambda entry: (entry.ready_round, -entry.bypass_count, entry.ordinal),
    )
    aged_ids = {id(entry) for entry in aged}
    return aged + [entry for entry in ranked if id(entry) not in aged_ids]
