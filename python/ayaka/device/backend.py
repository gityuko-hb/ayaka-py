from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from ayaka.configs.hardware import CC_LIMITS
from ayaka.distributed.device import DeviceCapability
from ayaka.utils.torch_utils import (
    compute_capability,
    cuda_available,
    require_torch,
    synchronize,
)
from ayaka.utils.torch_utils import (
    device_count as get_cuda_device_count,
)

__all__ = ["DeviceBackend", "NullBackend", "TorchCudaBackend", "get_backend"]


@runtime_checkable
class DeviceBackend(Protocol):
    @property
    def name(self) -> str: ...
    def device_count(self) -> int: ...
    def probe(self, index: int) -> DeviceCapability: ...
    def set_device(self, index: int) -> None: ...
    def memory_info(self, index: int) -> tuple[int, int]: ...
    def stream_priority_range(self) -> tuple[int, int]: ...
    def create_stream(self, index: int, priority: int) -> Any: ...
    def destroy_stream(self, stream: Any) -> None: ...
    def create_event(self, *, timing: bool) -> Any: ...
    def destroy_event(self, event: Any) -> None: ...
    def record(self, event: Any, stream: Any) -> None: ...
    def wait(self, stream: Any, event: Any) -> None: ...
    def query(self, event: Any) -> bool: ...
    def elapsed_ms(self, start: Any, end: Any) -> float: ...
    def synchronize_stream(self, stream: Any) -> None: ...
    def synchronize_device(self, index: int) -> None: ...


class NullBackend(DeviceBackend):
    __slots__ = ("_capability", "_clock", "_events", "_next_id", "_recorded", "_streams")

    def __init__(self, capability: DeviceCapability | None = None) -> None:
        self._clock = 0
        self._next_id = 1
        self._streams: dict[int, int] = {}  # handle -> priority
        self._events: dict[int, bool] = {}  # handle -> timing
        self._recorded: dict[int, int] = {} # handle -> clock at record
        self._capability = capability or DeviceCapability(
            name="null", sm_major=8, sm_minor=6, num_sms=16, hbm_bytes=8 << 30
        )

    @property
    def name(self) -> str:
        return "null"

    @property
    def live_streams(self) -> int:
        return len(self._streams)

    @property
    def live_events(self) -> int:
        return len(self._events)

    def advance(self, ticks: int = 1) -> None:
        self._clock += ticks

    def _alloc(self) -> int:
        handle = self._next_id
        self._next_id += 1
        return handle

    def device_count(self) -> int:
        return 1

    def probe(self, index: int) -> DeviceCapability:
        return self._capability

    def set_device(self, index: int) -> None:
        return None

    def memory_info(self, index: int) -> tuple[int, int]:
        return self._capability.hbm_bytes, self._capability.hbm_bytes

    def stream_priority_range(self) -> tuple[int, int]:
        return 0, -5

    def create_stream(self, index: int, priority: int) -> Any:
        handle = self._alloc()
        self._streams[handle] = priority
        return handle

    def destroy_stream(self, stream: Any) -> None:
        if self._streams.pop(int(stream), None) is None:
            raise RuntimeError(f"double free of stream {stream}")

    def create_event(self, *, timing: bool) -> Any:
        handle = self._alloc()
        self._events[handle] = timing
        return handle

    def destroy_event(self, event: Any) -> None:
        if self._events.pop(int(event), None) is None:
            raise RuntimeError(f"double free of event {event}")
        self._recorded.pop(int(event), None)

    def record(self, event: Any, stream: Any) -> None:
        if int(event) not in self._events:
            raise RuntimeError(f"record on destroyed event {event}")
        self._recorded[int(event)] = self._clock + 1

    def wait(self, stream: Any, event: Any) -> None:
        if int(event) not in self._events:
            raise RuntimeError(f"wait on destroyed event {event}")

    def query(self, event: Any) -> bool:
        recorded = self._recorded.get(int(event))
        return True if recorded is None else self._clock >= recorded

    def elapsed_ms(self, start: Any, end: Any) -> float:
        return float(self._recorded.get(int(end), 0) - self._recorded.get(int(start), 0))

    def synchronize_stream(self, stream: Any) -> None:
        self.advance()

    def synchronize_device(self, index: int) -> None:
        self.advance()


class TorchCudaBackend(DeviceBackend):
    __slots__ = ("_torch",)

    def __init__(self) -> None:
        self._torch = require_torch(
            capability="TorchCudaBackend",
            remedy="install ayaka[cuda] or install torch with CUDA support",
        )
        if not cuda_available():
            raise RuntimeError("no CUDA device available")

    @property
    def name(self) -> str:
        return "torch-cuda"

    def device_count(self) -> int:
        return get_cuda_device_count()

    def probe(self, index: int) -> DeviceCapability:
        props = self._torch.cuda.get_device_properties(index)
        cc = compute_capability(index)
        major, minor = cc if cc is not None else (int(props.major), int(props.minor))
        limits = CC_LIMITS.get((major, minor), {})

        return DeviceCapability(
            name=str(props.name),
            sm_major=major,
            sm_minor=minor,
            num_sms=int(props.multi_processor_count),
            hbm_bytes=int(props.total_memory),
            l2_bytes=int(getattr(props, "L2_cache_size", 0) or 0),
            shared_mem_per_block=int(
                getattr(props, "shared_memory_per_block_optin", 0)
                or getattr(props, "shared_memory_per_block", 0)
                or limits.get("shared_per_block", 0)
            ),
            shared_mem_per_sm=int(
                getattr(props, "shared_memory_per_multiprocessor", 0)
                or limits.get("shared_per_sm", 0)
            ),
            registers_per_sm=int(
                getattr(props, "regs_per_multiprocessor", 0) or limits.get("regs_per_sm", 0)
            ),
            max_threads_per_sm=int(
                getattr(props, "max_threads_per_multi_processor", 0)
                or limits.get("max_threads_per_sm", 0)
            ),
            warp_size=int(getattr(props, "warp_size", 32)),
            pci_bus_id=f"{getattr(props, 'pci_bus_id', 0)}",
        )

    def set_device(self, index: int) -> None:
        self._torch.cuda.set_device(index)

    def memory_info(self, index: int) -> tuple[int, int]:
        free, total = self._torch.cuda.mem_get_info(index)
        return int(free), int(total)

    def stream_priority_range(self) -> tuple[int, int]:
        return 0, -5

    def create_stream(self, index: int, priority: int) -> Any:
        return self._torch.cuda.Stream(device=index, priority=priority)

    def destroy_stream(self, stream: Any) -> None:
        return None

    def create_event(self, *, timing: bool) -> Any:
        return self._torch.cuda.Event(enable_timing=timing)

    def destroy_event(self, event: Any) -> None:
        return None

    def record(self, event: Any, stream: Any) -> None:
        event.record(stream)

    def wait(self, stream: Any, event: Any) -> None:
        stream.wait_event(event)

    def query(self, event: Any) -> bool:
        return bool(event.query())

    def elapsed_ms(self, start: Any, end: Any) -> float:
        return float(start.elapsed_time(end))

    def synchronize_stream(self, stream: Any) -> None:
        stream.synchronize()

    def synchronize_device(self, index: int) -> None:
        synchronize(index)


def get_backend(prefer_cuda: bool = True) -> DeviceBackend:
    """Real backend when a device is usable, null otherwise.

    ``AYAKA_FORCE_NULL_DEVICE=1`` pins the null backend, which is how the fast
    lane stays deterministic on a runner that happens to have a GPU.
    """
    if os.environ.get("AYAKA_FORCE_NULL_DEVICE") not in (None, "", "0"):
        return NullBackend()
    if prefer_cuda and cuda_available():
        try:
            return TorchCudaBackend()
        except Exception:
            pass
    return NullBackend()
