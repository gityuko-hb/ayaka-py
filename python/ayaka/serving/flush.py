"""Deferred prefix-cache flush: one bounded operation, all-owner quiescence (research C04).

A flush drops every cached prefix page. Losing future hits is the point; racing
a reader is not, so a flush runs only on the engine owner thread and only when
no owner can still read or write KV: no live generation, unfinished request,
ticket in flight, reserved or in-flight page, open transaction, lease or
transfer. An empty Python queue is not taken as proof of that.

A busy engine either answers ``BUSY`` at once (``timeout == 0``) or parks a
single operation until quiescence or its deadline. A second flush while one is
parked joins it and receives the same outcome, so a retrying client cannot
queue destructive work. ``drain=True`` additionally holds new admissions while
the operation waits; without it a steady request stream may keep the engine
busy until ``TIMEOUT``. Quiescence observed in the same poll that reaches the
deadline wins: the deadline bounds the wait, not the flush itself.
"""

from __future__ import annotations

import math
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "MAX_FLUSH_TIMEOUT_SECONDS",
    "FlushOperation",
    "FlushResult",
    "FlushStatus",
    "validate_flush_timeout",
]

#: Longest a caller may ask a flush to wait for quiescence.
MAX_FLUSH_TIMEOUT_SECONDS = 3600.0


class FlushStatus(StrEnum):
    """Outcome of one flush operation; only ``FLUSHED`` changed the cache."""

    FLUSHED = "flushed"
    #: Not quiescent and the caller asked not to wait (``timeout == 0``).
    BUSY = "busy"
    #: Never quiescent before the deadline; nothing was flushed.
    TIMEOUT = "timeout"
    #: The clear itself raised. The engine stops admitting work (fail-closed).
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FlushResult:
    status: FlushStatus
    #: Monotonic id of the operation that answered; coalesced callers share it.
    operation: int
    #: Prefix-cache references released (0 unless ``FLUSHED``).
    released_refs: int = 0
    free_pages_before: int = 0
    free_pages_after: int = 0
    #: Callers answered by this operation, the first included.
    callers: int = 1
    waited_seconds: float = 0.0
    #: Owners still busy when a ``BUSY``/``TIMEOUT`` answer was given.
    blockers: tuple[str, ...] = ()
    detail: str = ""

    @property
    def flushed(self) -> bool:
        return self.status is FlushStatus.FLUSHED

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "operation": self.operation,
            "released_refs": self.released_refs,
            "free_pages_before": self.free_pages_before,
            "free_pages_after": self.free_pages_after,
            "callers": self.callers,
            "waited_seconds": self.waited_seconds,
            "blockers": list(self.blockers),
            "detail": self.detail,
        }


def validate_flush_timeout(timeout: object) -> float:
    """Seconds to wait for quiescence: finite, nonnegative and bounded.

    NaN and infinity are refused: a NaN deadline never expires and an infinite
    one turns a busy server into a flush that can only end at idle.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("flush timeout must be a number of seconds")
    value = float(timeout)
    if not math.isfinite(value) or value < 0:
        raise ValueError("flush timeout must be finite and nonnegative")
    if value > MAX_FLUSH_TIMEOUT_SECONDS:
        raise ValueError(f"flush timeout must not exceed {MAX_FLUSH_TIMEOUT_SECONDS:g} seconds")
    return value


@dataclass(slots=True)
class FlushOperation:
    """The single parked flush; owned and mutated by the engine thread only."""

    operation: int
    started_ns: int
    deadline_ns: int
    drain: bool
    waiters: list[Future] = field(default_factory=list)

    def live_waiters(self) -> list[Future]:
        """Drop callers that gave up; a cancelled future is never answered."""
        self.waiters = [waiter for waiter in self.waiters if not waiter.cancelled()]
        return self.waiters

    def answer(self, result: FlushResult) -> int:
        """Resolve every live caller with ``result``; returns how many got it."""
        answered = 0
        for waiter in self.live_waiters():
            try:
                waiter.set_result(result)
            except InvalidStateError:  # cancelled concurrently: nobody to answer
                continue
            answered += 1
        return answered

    def fail(self, error: BaseException) -> None:
        for waiter in self.live_waiters():
            try:
                waiter.set_exception(error)
            except InvalidStateError:
                continue
