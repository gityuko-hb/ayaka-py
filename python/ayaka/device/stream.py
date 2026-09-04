from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ayaka.device.backend import DeviceBackend
from ayaka.device.events import EventPool
from ayaka.types import StreamRole
from ayaka.utils.torch_memory import peak_memory_bytes, pinned_empty

__all__ = ["ROLE_PRIORITY", "StreamPool"]

ROLE_PRIORITY: dict[StreamRole, int] = {
    StreamRole.COMPUTE: 0,
    StreamRole.H2D: 0,
    StreamRole.D2H: 0,
    StreamRole.P2P: 0,
    StreamRole.COMM: -2,
    StreamRole.KV: -1,
}


class StreamPool:
    __slots__ = ("_backend", "_closed", "_events", "_index", "_priority_range", "_streams")

    def __init__(
        self,
        backend: DeviceBackend,
        index: int = 0,
        *,
        events: EventPool | None = None,
    ) -> None:
        self._backend = backend
        self._index = index
        self._streams: dict[StreamRole, Any] = {}
        self._events = events if events is not None else EventPool(backend)
        self._closed = False
        self._priority_range = backend.stream_priority_range()
        _warn_on_connection_limit()

    @property
    def events(self) -> EventPool:
        return self._events

    @property
    def num_streams(self) -> int:
        return len(self._streams)

    def get(self, role: StreamRole) -> Any:
        if self._closed:
            raise RuntimeError("stream pool is closed")
        stream = self._streams.get(role)
        if stream is None:
            stream = self._backend.create_stream(self._index, self._priority_for(role))
            self._streams[role] = stream
        return stream

    def _priority_for(self, role: StreamRole) -> int:
        least, greatest = self._priority_range
        offset = ROLE_PRIORITY.get(role, 0)
        return max(greatest, min(least, least + offset))

    def priority_of(self, role: StreamRole) -> int:
        return self._priority_for(role)

    def created_roles(self) -> tuple[StreamRole, ...]:
        return tuple(self._streams)

    def ensure_all(self) -> None:
        for role in StreamRole:
            self.get(role)

    def allocate_pinned_buffer(
        self, shape: tuple[int, ...], dtype: Any, *, allow_pageable: bool = True
    ) -> tuple[Any, bool]:
        """Tiện ích cấp phát buffer trên Host dùng cho H2D/D2H transfer không bị chặn."""
        return pinned_empty(shape, dtype, allow_pageable=allow_pageable)

    @contextmanager
    def profile_peak_memory(self) -> Iterator[list[int]]:
        """Context manager đo đỉnh bộ nhớ cấp phát khi chạy các tác vụ trên pool này."""
        with peak_memory_bytes(self._index) as peak:
            yield peak

    def order(self, after: StreamRole, before: StreamRole) -> None:
        src = self.get(before)
        dst = self.get(after)
        if src is dst:
            return
        with self._events.fence(src, dst):
            pass

    @contextmanager
    def joined(self, *roles: StreamRole, into: StreamRole = StreamRole.COMPUTE) -> Iterator[None]:
        yield
        for role in roles:
            self.order(after=into, before=role)

    def synchronize(self, role: StreamRole) -> None:
        self._backend.synchronize_stream(self.get(role))

    def synchronize_all(self) -> None:
        for stream in self._streams.values():
            self._backend.synchronize_stream(stream)

    def close(self) -> None:
        if self._closed:
            return
        self.synchronize_all()
        self._events.close()
        for stream in self._streams.values():
            self._backend.destroy_stream(stream)
        self._streams.clear()
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __enter__(self) -> StreamPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        roles = ",".join(r.value for r in self._streams)
        return f"<StreamPool cuda:{self._index} [{roles or '-'}] {self._events!r}>"


_WARNED = False


def _warn_on_connection_limit() -> None:
    global _WARNED
    if _WARNED:
        return
    raw = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")
    if raw is None:
        return
    try:
        limit = int(raw)
    except ValueError:
        return
    if limit < len(StreamRole):
        _WARNED = True
        warnings.warn(
            f"CUDA_DEVICE_MAX_CONNECTIONS={limit} is below the {len(StreamRole)} stream "
            "roles Ayaka uses: streams will still be ordered correctly but they will "
            "share hardware queues, so compute/copy/comm overlap is lost.",
            RuntimeWarning,
            stacklevel=3,
        )