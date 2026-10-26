"""Bounded, pull-based prefix-cache event log (research kv_events_publisher).

Telemetry only: an event says what the prefix service published, evicted or
cleared. It never authorizes a free, an admission or a remote fetch, and losing
one never changes a settlement. The engine thread appends in O(1) and never
blocks on a consumer; consumers drain at their own pace from any thread. When
the buffer is full the oldest event is dropped and counted, and the loss is
visible to a consumer as a jump in ``sequence``.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from ayaka.utils.validation import require_int

__all__ = ["CacheEvent", "CacheEventKind", "CacheEventLog"]


class CacheEventKind(StrEnum):
    #: A complete block with this digest is resident after a publish.
    STORED = "stored"
    #: The block with this digest was evicted.
    REMOVED = "removed"
    #: Every resident block was dropped at once (administrative flush).
    CLEARED = "cleared"


@dataclass(frozen=True, slots=True)
class CacheEvent:
    #: Strictly increasing per log; a gap means dropped events.
    sequence: int
    kind: CacheEventKind
    digest: bytes | None = None
    #: Prompt tokens covered up to and including this block (0 if unknown).
    token_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "kind": self.kind.value,
            "digest": None if self.digest is None else self.digest.hex(),
            "token_count": self.token_count,
        }


class CacheEventLog:
    def __init__(self, capacity: int) -> None:
        require_int(capacity, "cache event capacity", minimum=1)
        self._events: deque[CacheEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next = 1
        self.dropped = 0

    @property
    def capacity(self) -> int:
        maxlen = self._events.maxlen
        assert maxlen is not None
        return maxlen

    @property
    def recorded(self) -> int:
        with self._lock:
            return self._next - 1

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._events)

    def record(
        self, kind: CacheEventKind, digest: bytes | None = None, token_count: int = 0
    ) -> None:
        """Append one event; drops (and counts) the oldest when full."""
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self.dropped += 1
            self._events.append(CacheEvent(self._next, kind, digest, token_count))
            self._next += 1

    def drain(self, limit: int | None = None) -> tuple[CacheEvent, ...]:
        """Remove and return up to ``limit`` oldest events (all when ``None``)."""
        if limit is not None:
            require_int(limit, "drain limit", minimum=1)
        with self._lock:
            count = len(self._events) if limit is None else min(limit, len(self._events))
            return tuple(self._events.popleft() for _ in range(count))
