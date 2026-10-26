"""Detached load snapshot for routers, probes and dashboards (research load_snapshot).

The serving owner thread rebuilds one immutable snapshot after every iteration
from values it already owns and publishes it by swapping a single reference.
Readers on any thread take the current snapshot without a lock and never touch
live scheduler, KV or stream state. The snapshot is advisory: its free pages are
not a reservation, and ``age_seconds`` tells a reader how stale it is, so a
stalled owner shows up as an aging snapshot instead of plausible fresh numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["LoadSnapshot"]


@dataclass(frozen=True, slots=True)
class LoadSnapshot:
    #: Strictly increasing per service; 0 is the placeholder before the first build.
    version: int
    #: Owner-clock time the values were read.
    taken_ns: int
    ready: bool
    live_generations: int
    #: Commands queued for admission and the prompt tokens they carry.
    queued_commands: int
    queued_prompt_tokens: int
    running: int
    waiting: int
    free_pages: int
    total_pages: int
    #: Age of the longest-waiting unsettled ticket; ``None`` when none is in flight.
    oldest_ticket_age_seconds: float | None = None
    #: The completion watchdog currently holds the service not ready.
    stalled: bool = False

    def age_seconds(self, now_ns: int) -> float:
        return max(0, now_ns - self.taken_ns) / 1e9

    def is_stale(self, max_age_seconds: float, now_ns: int) -> bool:
        return self.age_seconds(now_ns) > max_age_seconds

    def as_dict(self, *, now_ns: int | None = None) -> dict[str, object]:
        data: dict[str, object] = asdict(self)
        if now_ns is not None:
            data["age_seconds"] = self.age_seconds(now_ns)
        return data
