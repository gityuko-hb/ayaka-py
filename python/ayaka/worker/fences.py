"""Worker-owned completion fences: nonblocking, sync-free and release-safe.

Fences never materialize device values. ``RecordingFence`` caches its terminal
result and returns its event to the owning pool exactly once; a query failure
is unknown quiescence, so it poisons the worker through the supplied callback
instead of claiming success.
"""

from __future__ import annotations

from collections.abc import Callable

from ayaka.device.backend import DeviceBackend
from ayaka.device.events import EventPool
from ayaka.executor.ticket import CompletionFence, FenceResult, WorkState

__all__ = ["FlightFence", "ImmediateFence", "RecordingFence"]


class ImmediateFence:
    """Host proof: the worker completed the work before returning to the caller."""

    __slots__ = ()

    def query(self) -> FenceResult:
        return FenceResult(WorkState.SUCCEEDED, quiescent=True)


class RecordingFence:
    """A recorded device event queried without blocking; releases on proof."""

    __slots__ = ("_backend", "_event", "_on_error", "_pool", "_result")

    def __init__(
        self,
        backend: DeviceBackend,
        event: object,
        pool: EventPool,
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._backend = backend
        self._event = event
        self._pool = pool
        self._result: FenceResult | None = None
        self._on_error = on_error

    def query(self) -> FenceResult:
        if self._result is not None:
            return self._result
        try:
            done = bool(self._backend.query(self._event))
            if done:
                self.release()
        except BaseException as exc:
            if self._on_error is not None:
                self._on_error(exc)
            raise
        if not done:
            return FenceResult(WorkState.PENDING)
        result = FenceResult(WorkState.SUCCEEDED, quiescent=True)
        self._result = result
        return result

    def release(self) -> None:
        """Return the event to the pool once; safe to repeat for a shared flight."""
        event = self._event
        pool = self._pool
        if event is not None and pool is not None:
            pool.release(event)
            self._event = None
            self._pool = None
            # A completed drain also releases sibling fences. Never query an
            # event again after it has been returned and potentially reused.
            self._result = FenceResult(WorkState.SUCCEEDED, quiescent=True)


class FlightFence:
    """Ticket-scoped wrapper that retires one worker flight on quiescence.

    A ticket can own several recorded fences (an execute fence and a drain
    fence created after partial submission). Exactly one of them is polled to
    completion; whichever proves quiescence retires the flight and releases
    every sibling event through the worker callback.
    """

    __slots__ = ("_fence", "_retire", "_retired")

    def __init__(self, fence: CompletionFence, retire: Callable[[], None]) -> None:
        self._fence = fence
        self._retire = retire
        self._retired = False

    def query(self) -> FenceResult:
        result = self._fence.query()
        if result.quiescent and not self._retired:
            self._retire()
            self._retired = True
        return result
