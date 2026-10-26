"""Shadow measurement for bounded preparation ahead (research R1).

Ayaka already prepares independent requests while earlier tickets are in
flight, up to ``max_inflight`` (K). R1 asks whether a larger K would help before
anyone raises it: this recorder counts engine iterations whose launch loop
stopped because the executor had no submission capacity while requests that
own no in-flight ticket were still waiting. It observes only; scheduling,
ownership and fences are untouched. ``ready_independent`` is an upper bound:
such a request may still have been blocked by KV or token budgets.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

__all__ = ["PrepareAheadShadow", "PrepareAheadSnapshot"]


@dataclass(frozen=True, slots=True)
class PrepareAheadSnapshot:
    iterations: int
    #: Iterations whose launch loop ended on the inflight ceiling.
    capacity_blocked: int
    #: Sum over those iterations of unfinished requests owning no ticket.
    blocked_ready_total: int
    blocked_ready_max: int

    def as_dict(self) -> dict[str, int]:
        return {
            "iterations": self.iterations,
            "capacity_blocked": self.capacity_blocked,
            "blocked_ready_total": self.blocked_ready_total,
            "blocked_ready_max": self.blocked_ready_max,
        }


class PrepareAheadShadow:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._iterations = self._blocked = self._ready_total = self._ready_max = 0

    def observe(self, *, capacity_blocked: bool, ready_independent: int) -> None:
        with self._lock:
            self._iterations += 1
            if capacity_blocked and ready_independent:
                self._blocked += 1
                self._ready_total += ready_independent
                self._ready_max = max(self._ready_max, ready_independent)

    def snapshot(self) -> PrepareAheadSnapshot:
        with self._lock:
            return PrepareAheadSnapshot(
                self._iterations, self._blocked, self._ready_total, self._ready_max
            )
