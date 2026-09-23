"""Push-based output delivery from the engine loop to stream consumers.

The engine (owner) thread is the only producer. Each request owns one bounded
:class:`OutputStream`; production never blocks — a slow consumer overflows and
the façade aborts that request instead of stalling the loop. Wakeup follows the
single-waiter contract proven by the serving handle: one ``asyncio.Future``
per stream, woken via ``call_soon_threadsafe``.

The collector is transport-neutral: it carries :class:`OutputEvent` values from
``ayaka.request.stream`` and knows nothing about serving events or protocols.
Events for unattached request ids are dropped silently — the engine may publish
for requests the façade never opened a stream for (scheduler-internal or
already-released incarnations).
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Sequence
from typing import Protocol

from ayaka.request.stream import OutputEvent

__all__ = [
    "OutputCollector",
    "OutputStream",
    "OutputStreamOverflow",
    "OutputListener",
]

#: Terminal marker appended to the internal deque. ``OutputEvent`` is frozen
#: data so ``None`` is an unambiguous close signal.
_CLOSED: None = None


def _wake(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


class OutputStreamOverflow(RuntimeError):
    """The consumer abandoned a stream that overflowed; output was truncated."""


class OutputStream:
    """Bounded, nonblocking producer side of one request's output stream.

    Single-consumer: like the serving handle, only one ``next_event`` waiter
    is tracked at a time. ``put`` never blocks the engine thread; ``close`` is
    idempotent and always accepted so a terminal slot is never refused.
    """

    def __init__(self, request_id: str, *, limit: int = 1024) -> None:
        if limit < 1:
            raise ValueError("stream limit must be at least 1")
        self.request_id = request_id
        self._lock = threading.Lock()
        self._events: deque[OutputEvent | None] = deque()
        self._waiter: asyncio.Future | None = None
        self._limit = limit
        self.overflowed = False

    def put(self, event: OutputEvent) -> bool:
        """Nonblocking push; ``False`` when the buffer is full (nonterminal)."""
        with self._lock:
            if self._events and self._events[-1] is _CLOSED:
                return True  # closed; a late duplicate/stale event is dropped
            if len(self._events) >= self._limit:
                self.overflowed = True
                return False
            self._events.append(event)
            waiter, self._waiter = self._waiter, None
        self._notify(waiter)
        return True

    def close(self) -> None:
        """Append the terminal marker and wake any waiting consumer."""
        with self._lock:
            if self._events and self._events[-1] is _CLOSED:
                return
            self._events.append(_CLOSED)
            waiter, self._waiter = self._waiter, None
        self._notify(waiter)

    def pending(self) -> int:
        with self._lock:
            return len(self._events)

    async def next_event(self) -> OutputEvent:
        """Await the next event; raises ``StopAsyncIteration`` once closed."""
        while True:
            with self._lock:
                if self._events:
                    event = self._events.popleft()
                    if event is _CLOSED:
                        raise StopAsyncIteration
                    if event is not None:
                        return event
                waiter = asyncio.get_running_loop().create_future()
                self._waiter = waiter
            try:
                await waiter
            finally:
                with self._lock:
                    if self._waiter is waiter:
                        self._waiter = None

    def _notify(self, waiter: asyncio.Future | None) -> None:
        if waiter is None:
            return
        try:
            waiter.get_loop().call_soon_threadsafe(_wake, waiter)
        except RuntimeError:
            pass  # the consumer loop is gone; nothing to wake


class OutputListener(Protocol):
    """What ``OutputProcessor`` calls once per published output event."""

    def on_output_event(self, event: OutputEvent) -> None: ...


class OutputCollector:
    """Registry of per-request streams; the ``OutputProcessor`` listener.

    ``attach``/``drop`` happen on the owner thread during admission and
    settlement; ``on_output_event`` fires on the engine thread mid-step. All
    entry points share one lock because attach and publication can interleave
    with the owner's settlement scan.
    """

    def __init__(self, *, limit: int = 1024, downstream: OutputListener | None = None) -> None:
        self._limit = limit
        self._downstream = downstream
        self._lock = threading.Lock()
        self._streams: dict[str, OutputStream] = {}
        self._overflowed: set[str] = set()
        self._child_parents: dict[str, str] = {}

    def attach(self, request_id: str) -> OutputStream:
        with self._lock:
            if request_id in self._streams:
                raise ValueError(f"stream {request_id!r} is already attached")
            stream = OutputStream(request_id, limit=self._limit)
            self._streams[request_id] = stream
            self._overflowed.discard(request_id)
        return stream

    def attach_parallel(self, parent_id: str, child_ids: Sequence[str]) -> OutputStream:
        """One interleaved stream for a parallel-sampling family.

        The stream is keyed by the parent id; child events route into it and
        carry ``child_index``. Child terminal events do NOT auto-close the
        stream — the façade closes it once every child is terminal.
        """
        if not child_ids:
            raise ValueError("a parallel family needs at least one child")
        with self._lock:
            if parent_id in self._streams:
                raise ValueError(f"stream {parent_id!r} is already attached")
            stream = OutputStream(parent_id, limit=self._limit)
            self._streams[parent_id] = stream
            for child_id in child_ids:
                self._child_parents[child_id] = parent_id
            self._overflowed.discard(parent_id)
        return stream

    def get(self, request_id: str) -> OutputStream | None:
        with self._lock:
            return self._streams.get(request_id)

    def drop(self, request_id: str) -> None:
        with self._lock:
            self._streams.pop(request_id, None)
            self._overflowed.discard(request_id)
            for child_id, parent_id in list(self._child_parents.items()):
                if parent_id == request_id:
                    del self._child_parents[child_id]

    def on_output_event(self, event: OutputEvent) -> None:
        with self._lock:
            stream = self._streams.get(event.request_id)
            child = False
            if stream is None:
                parent_id = self._child_parents.get(event.request_id)
                if parent_id is not None:
                    stream = self._streams.get(parent_id)
                    child = True
        if stream is None:
            return
        if not stream.put(event):
            with self._lock:
                self._overflowed.add(stream.request_id)
        if event.finish_reason is not None and not child:
            stream.close()
        if self._downstream is not None:
            # Fan out after local routing: a wire bridge re-emits every event,
            # including ones for streams this collector does not track.
            self._downstream.on_output_event(event)

    def overflowed_requests(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._overflowed)

    @property
    def active_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._streams)
