"""Pure scheduler ordering policy.

Policy ranks eligible requests only.  It never allocates KV, mutates lifecycle
state, or upgrades a prefix hint into authoritative ownership.
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

__all__ = ["QueueEntry", "order", "rank_waiting"]


@dataclass(slots=True)
class QueueEntry:
    """Stable queue identity plus non-authoritative scheduling hints."""

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
        if self.cache_hint_tokens < 0 or self.bypass_count < 0:
            raise ValueError("queue counters must be non-negative")


def _policy_name(policy: SchedulingPolicy | str | Any) -> str:
    value = getattr(policy, "value", policy)
    return str(value).lower().replace("-", "_")


def _priority(entry: QueueEntry) -> int:
    return int(getattr(entry.lifecycle.request, "priority", 0) or 0)


def _arrival_ns(entry: QueueEntry) -> int:
    arrival = getattr(entry.lifecycle.request, "arrival_ns", 0)
    return int(arrival) if arrival else entry.ready_ns


def _max_output_tokens(entry: QueueEntry) -> int:
    request = entry.lifecycle.request
    sampling = getattr(request, "sampling", None)
    for obj in (sampling, getattr(request, "stop", None), request):
        if obj is None:
            continue
        for name in ("max_tokens", "max_output_tokens", "max_new_tokens"):
            value = getattr(obj, name, None)
            if isinstance(value, int):
                return value
    return 0


def _routing_key(entry: QueueEntry) -> str:
    value = getattr(entry.lifecycle.request, "routing_key", "")
    return "" if value is None else str(value)


def _fcfs_key(entry: QueueEntry) -> tuple[int, int]:
    return (_arrival_ns(entry), entry.ordinal)


def _priority_key(entry: QueueEntry) -> tuple[int, int, int]:
    return (-_priority(entry), _arrival_ns(entry), entry.ordinal)


def _cache_affinity_key(entry: QueueEntry) -> tuple[int, int, int, int]:
    return (-entry.cache_hint_tokens, -_priority(entry), _arrival_ns(entry), entry.ordinal)


def _longest_output_key(entry: QueueEntry) -> tuple[int, int, int, int]:
    return (-_max_output_tokens(entry), -_priority(entry), _arrival_ns(entry), entry.ordinal)


def _routing_key_policy(entry: QueueEntry) -> tuple[str, int, int, int]:
    return (_routing_key(entry), -_priority(entry), _arrival_ns(entry), entry.ordinal)


def rank_waiting(
    entries: Iterable[QueueEntry],
    *,
    scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.FCFS,
) -> list[QueueEntry]:
    """Rank eligible requests without touching physical resources.

    Supported policy names:
    - ``fcfs``: oldest arrival first;
    - ``priority``: high request priority first, then FCFS;
    - ``lpm`` / ``longest_prefix`` / ``cache_affinity``: largest cache hint;
    - ``lof`` / ``longest_output``: largest configured output budget;
    - ``routing_key``: stable grouping by routing key, then FCFS.

    LPM here is intentionally only an affinity ranking. True SGLang-style
    radix/DFS weighting and in-batch prefix simulation belong in the prefix
    provider because they require the concrete cache tree.
    """

    ranked = list(entries)
    name = _policy_name(scheduling_policy)

    key: Callable[[QueueEntry], tuple[Any, ...]]
    if name == "fcfs":
        key = _fcfs_key
    elif name == "priority":
        key = _priority_key
    elif name in {"lpm", "longest_prefix", "cache_affinity"}:
        key = _cache_affinity_key
    elif name in {"lof", "longest_output"}:
        key = _longest_output_key
    elif name == "routing_key":
        key = _routing_key_policy
    else:
        raise ConfigError(
            "scheduler.scheduling_policy",
            "UNSUPPORTED_POLICY",
            f"unsupported scheduling policy: {name}",
        )
    ranked.sort(key=key)
    return ranked


def order(
    entries: Iterable[QueueEntry],
    *,
    round_id: int,
    now_ns: int,
    max_bypass: int,
    scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.FCFS,
) -> list[QueueEntry]:
    """Apply bounded-starvation aging on top of policy ranking."""

    require_int(round_id, "round_id")
    require_int(now_ns, "now_ns")
    require_int(max_bypass, "max_bypass", minimum=1)
    ranked = rank_waiting(entries, scheduling_policy=scheduling_policy)
    for entry in ranked:
        if entry.bypass_count >= max_bypass or round_id - entry.ready_round >= max_bypass:
            break
    else:
        return ranked
    aged = [
        entry
        for entry in ranked
        if entry.bypass_count >= max_bypass or round_id - entry.ready_round >= max_bypass
    ]
    aged.sort(
        key=lambda entry: (
            -entry.bypass_count,
            entry.ready_round,
            _arrival_ns(entry),
            entry.ordinal,
        )
    )
    aged_ids = {id(entry) for entry in aged}
    return aged + [entry for entry in ranked if id(entry) not in aged_ids]
