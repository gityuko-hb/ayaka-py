"""Bounded, torch-free diagnostics for one serialized engine.

Metrics never query device fences, retain tickets/tensors, or release resources.
Read snapshots on the engine thread and export them elsewhere. Durations use the
shared monotonic Clock domain and measure host-observed completion, not GPU time.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.utils.timing import Clock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayaka.executor.ticket import ExecutionTicket
    from ayaka.memory.ledger import MemoryLedger

_LIMITS_NS = (
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000
)


@dataclass(frozen=True, slots=True)
class DurationSnapshot:
    """Non-cumulative buckets: <= each limit, followed by an overflow bucket."""

    count: int
    total_ns: int
    max_ns: int
    limits_ns: tuple[int, ...]
    buckets: tuple[int, ...]


class _Duration:
    def __init__(self) -> None:
        self.count = self.total_ns = self.max_ns = 0
        self.buckets = [0] * (len(_LIMITS_NS) + 1)

    def observe(self, value: int) -> None:
        self.count += 1
        self.total_ns += value
        self.max_ns = max(self.max_ns, value)
        index = next((i for i, limit in enumerate(_LIMITS_NS) if value <= limit), len(_LIMITS_NS))
        self.buckets[index] += 1

    def snapshot(self) -> DurationSnapshot:
        return DurationSnapshot(
            self.count, self.total_ns, self.max_ns, _LIMITS_NS, tuple(self.buckets)
        )


@dataclass(frozen=True, slots=True)
class Rejection:
    stage: str
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class MemoryTierMetrics:
    """Ledger charges in bytes; views/pages in an already charged slab add zero."""

    tier: str
    capacity_bytes: int
    committed_bytes: int
    pending_bytes: int
    held_bytes: int
    materialized_bytes: int


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    observed_ns: int
    active_tickets: int
    pending_tickets: int
    quarantined_tickets: int
    state_counts: tuple[tuple[str, int], ...]
    oldest_pending_ns: int
    oldest_drain_ns: int
    counters: tuple[tuple[str, int], ...]
    completion_latency: DurationSnapshot
    drain_duration: DurationSnapshot
    rejection_counts: tuple[tuple[str, int], ...]
    recent_rejections: tuple[Rejection, ...]
    memory: tuple[MemoryTierMetrics, ...]


class RuntimeMetrics:
    """Engine-thread counters with bounded labels/history and injectable clock.

    Share one instance between the engine's executor and ResourceCoordinator.
    Snapshot receives the active ticket registry rather than keeping a second
    lifecycle registry. No request IDs, page IDs or exception strings are used
    as counter labels. Only the latest 16 rejection details (240 chars each)
    remain for diagnostics. Clock failures are counted and use the last valid
    time, so telemetry cannot interrupt ownership transfer or retirement.
    """

    def __init__(self, *, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self._last_ns = 0
        self._counts: Counter[str] = Counter(clock_errors=0)
        self._rejections: Counter[str] = Counter()
        self._recent: deque[Rejection] = deque(maxlen=16)
        self._completion = _Duration()
        self._drain = _Duration()

    def now_ns(self) -> int:
        try:
            value = self.clock.now().nanos
            if type(value) is not int or value < self._last_ns:
                raise ValueError("invalid or regressing monotonic clock")
        except Exception:
            self._counts["clock_errors"] += 1
            return self._last_ns
        self._last_ns = value
        return value

    def increment(self, name: str) -> None:
        """Bound arbitrary diagnostic labels; internal callers use fixed names."""
        name = name[:80]
        if name not in self._counts and len(self._counts) >= 63:
            name = "other"
        self._counts[name] += 1

    def reject(self, stage: str, code: str, detail: str) -> None:
        stage, code = stage[:32], code[:80]
        label = f"{stage}:{code}"
        if label not in self._rejections and len(self._rejections) >= 63:
            label = "other"
        self._rejections[label] += 1
        self._recent.append(Rejection(stage, code, detail[:240]))

    def terminal(self, ticket: ExecutionTicket) -> None:
        """Observe first terminal proof once, including redelivery/cancel races."""
        if ticket._terminal_ns is not None:
            return
        now = self.now_ns()
        ticket._terminal_ns = now
        if ticket._submitted_ns is not None:
            self._completion.observe(max(0, now - ticket._submitted_ns))
        if ticket._drain_started_ns is not None:
            self._drain.observe(max(0, now - ticket._drain_started_ns))

    def snapshot(
        self,
        tickets: Iterable[ExecutionTicket] = (),
        *,
        ledger: MemoryLedger | None = None,
    ) -> RuntimeSnapshot:
        """Copy current gauges without polling a fence or changing accounting.

        pending includes adopted/submitted/draining/quarantined, excludes
        terminal outcomes awaiting retirement. active includes those outcomes.
        A completed ticket that later has a host settlement failure is pending
        again, but does not produce a second completion latency observation.
        """
        from ayaka.executor.ticket import TicketState

        now = self.now_ns()
        active = tuple(t for t in tickets if t.state is not TicketState.RETIRED)
        pending = tuple(t for t in active if t.state is not TicketState.COMPLETED)
        states = Counter(t.state.value for t in active)
        memory = ()
        if ledger is not None:
            view = ledger.snapshot()
            memory = tuple(
                MemoryTierMetrics(
                    tier.name,
                    account.capacity_bytes,
                    account.committed_bytes,
                    account.pending_bytes,
                    account.held_bytes,
                    account.materialized_bytes,
                )
                for tier, account in sorted(view.tiers.items(), key=lambda item: item[0].value)
            )
        return RuntimeSnapshot(
            now,
            len(active),
            len(pending),
            states[TicketState.QUARANTINED.value],
            tuple((state.value, states[state.value]) for state in TicketState),
            max((max(0, now - t._adopted_ns) for t in pending), default=0),
            max(
                (
                    max(0, now - t._drain_started_ns)
                    for t in pending
                    if t._drain_started_ns is not None and t._terminal_ns is None
                ),
                default=0,
            ),
            tuple(sorted(self._counts.items())),
            self._completion.snapshot(),
            self._drain.snapshot(),
            tuple(sorted(self._rejections.items())),
            tuple(self._recent),
            memory,
        )
