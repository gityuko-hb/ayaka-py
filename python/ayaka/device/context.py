from __future__ import annotations

import os
import threading
from typing import Any

from ayaka.device.backend import DeviceBackend, get_backend
from ayaka.distributed.device import DeviceCapability, DeviceRef
from ayaka.types import DeviceKind
from ayaka.utils.torch_memory import DeviceMemory, device_memory, empty_cache
from ayaka.utils.torch_utils import synchronize


class ForkedContextError(RuntimeError):
    pass


_CONTEXTS: dict[tuple[int, int], DeviceContext] = {}
_LOCK = threading.Lock()

class DeviceContext:
    __slots__ = ("_backend", "_capability", "_closed", "_index", "_pid", "_ref")

    _index: int
    _pid: int
    _backend: DeviceBackend
    _capability: DeviceCapability
    _ref: DeviceRef
    _closed: bool

    def __new__(cls, index: int = 0, *, backend: DeviceBackend | None = None) -> DeviceContext:
        key = (os.getpid(), index)
        with _LOCK:
            existing = _CONTEXTS.get(key)
            if existing is not None and not existing._closed:
                if backend is not None and backend is not existing._backend:
                    raise RuntimeError(
                        f"device {index} already has a {existing._backend.name} context in "
                        f"this process; two backends on one device cannot order work "
                        f"against each other"
                    )
                return existing
            obj = super().__new__(cls)
            obj._index = index
            obj._pid = os.getpid()
            obj._backend = backend if backend is not None else get_backend()
            obj._closed = False
            obj._capability = obj._backend.probe(index)
            obj._ref = DeviceRef(kind=DeviceKind.CUDA, index=index, uuid="")
            _CONTEXTS[key] = obj
            return obj

    def __init__(self, index: int = 0, *, backend: DeviceBackend | None = None) -> None:
        del index, backend

    @property
    def index(self) -> int:
        return self._index

    @property
    def ref(self) -> DeviceRef:
        return self._ref

    @property
    def backend(self) -> DeviceBackend:
        self._check_pid()
        return self._backend

    @property
    def capability(self) -> DeviceCapability:
        return self._capability

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _check_pid(self) -> None:
        if os.getpid() != self._pid:
            raise ForkedContextError(
                f"device context for cuda:{self._index} was created in pid {self._pid} "
                f"and is being used in pid {os.getpid()}: a CUDA context does not "
                "survive fork. Create the context after the fork, or use spawn."
            )

    def bind(self) -> None:
        self._check_pid()
        self._backend.set_device(self._index)

    def memory_info(self) -> tuple[int, int]:
        self._check_pid()
        return self._backend.memory_info(self._index)

    def detailed_memory(self) -> DeviceMemory:
        """Lấy trạng thái chi tiết gồm driver_used, allocated, reserved và fragmentation."""
        self._check_pid()
        if self._backend.name == "torch-cuda":
            return device_memory(self._index)
        free, total = self.memory_info()
        return DeviceMemory(
            device_index=self._index,
            driver_free=free,
            driver_total=total,
            allocated=0,
            reserved=0,
        )

    def empty_cache(self) -> None:
        """Xả vùng nhớ đệm unallocated của Caching Allocator về lại cho driver."""
        self._check_pid()
        if self._backend.name == "torch-cuda":
            empty_cache()

    @property
    def free_bytes(self) -> int:
        return self.memory_info()[0]

    @property
    def total_bytes(self) -> int:
        return self.memory_info()[1]

    def synchronize(self) -> None:
        self._check_pid()
        synchronize(self._index)

    def budget_bytes(self, utilization: float) -> int:
        if not 0.0 < utilization <= 1.0:
            raise ValueError(f"utilization {utilization} must be in (0, 1]")
        return int(self.total_bytes * utilization)

    def supports(self, dtype: Any) -> bool:
        return bool(self._capability.supports(dtype))

    def close(self) -> None:
        with _LOCK:
            self._closed = True
            _CONTEXTS.pop((self._pid, self._index), None)

    def __repr__(self) -> str:
        cap = self._capability
        return (
            f"<DeviceContext cuda:{self._index} {cap.name or 'unknown'} "
            f"sm_{cap.sm_major}{cap.sm_minor} backend={self._backend.name}>"
        )


def reset_contexts() -> None:
    with _LOCK:
        for ctx in tuple(_CONTEXTS.values()):
            ctx._closed = True
        _CONTEXTS.clear()