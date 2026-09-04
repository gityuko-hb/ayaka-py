from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ayaka.device.backend import DeviceBackend
from ayaka.utils.torch_utils import no_device_sync

__all__ = ["EventPool", "EventPoolExhausted"]

_DEFAULT_MAX_LIVE = 4096


class EventPoolExhausted(RuntimeError):
    pass


class EventPool:
    __slots__ = ("_backend", "_created", "_free", "_live", "_max_live", "_pending", "_timing")

    def __init__(
        self,
        backend: DeviceBackend,
        *,
        timing: bool = False,
        max_live: int = _DEFAULT_MAX_LIVE,
    ) -> None:
        self._backend = backend
        self._timing = timing
        self._free: list[Any] = []
        self._pending: list[Any] = []
        self._live: set[int] = set()
        self._created = 0
        self._max_live = max_live

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_live(self) -> int:
        return len(self._live)

    @property
    def num_created(self) -> int:
        return self._created

    def _drain_pending(self) -> None:
        if not self._pending:
            return
        still: list[Any] = []
        for event in self._pending:
            if self._backend.query(event):
                self._free.append(event)
            else:
                still.append(event)
        self._pending = still

    def acquire(self) -> Any:
        self._drain_pending()
        if self._free:
            event = self._free.pop()
        else:
            total = self._created
            if total >= self._max_live:
                raise EventPoolExhausted(
                    f"event pool at its ceiling of {self._max_live}: "
                    f"{self.num_live} live, {self.num_pending} pending, 0 free. "
                    "Either a release is missing or no stream is making progress."
                )
            event = self._backend.create_event(timing=self._timing)
            self._created += 1
        self._live.add(id(event))
        return event

    def release(self, event: Any) -> None:
        key = id(event)
        if key not in self._live:
            raise RuntimeError("release of an event this pool did not hand out")
        self._live.discard(key)
        if self._backend.query(event):
            self._free.append(event)
        else:
            self._pending.append(event)

    @contextmanager
    def borrow(self) -> Iterator[Any]:
        event = self.acquire()
        try:
            yield event
        finally:
            self.release(event)

    def record_on(self, stream: Any) -> Any:
        event = self.acquire()
        self._backend.record(event, stream)
        return event

    @contextmanager
    def fence(self, src_stream: Any, dst_stream: Any, *, assert_async: bool = False) -> Iterator[Any]:
        """Tạo device-side dependency. Nếu assert_async=True, kích hoạt no_device_sync để bắt lỗi sync ngầm."""
        event = self.record_on(src_stream)
        try:
            if assert_async:
                with no_device_sync(strict=True):
                    self._backend.wait(dst_stream, event)
            else:
                self._backend.wait(dst_stream, event)
            yield event
        finally:
            self.release(event)

    def elapsed_ms(self, start: Any, end: Any) -> float:
        if not self._timing:
            raise RuntimeError(
                "this pool creates events with timing disabled; construct it with "
                "timing=True for profiling"
            )
        return self._backend.elapsed_ms(start, end)

    def close(self) -> None:
        if self._live:
            raise RuntimeError(
                f"close() with {len(self._live)} events still checked out — release them first"
            )
        for event in (*self._free, *self._pending):
            self._backend.destroy_event(event)
        self._free.clear()
        self._pending.clear()
        self._created = 0

    def __repr__(self) -> str:
        return (
            f"<EventPool free={self.num_free} pending={self.num_pending} "
            f"live={self.num_live} created={self._created}>"
        )