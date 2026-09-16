from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_utils import (
    cuda_available,
    device_index,
    dtype_name,
    require_torch,
    resolve_device,
    torch_dtype,
)


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    device_index: int
    driver_free: int
    driver_total: int
    allocated: int
    reserved: int

    @property
    def driver_used(self) -> int:
        """Bytes the driver considers in use, by every process on the device."""
        return self.driver_total - self.driver_free

    @property
    def allocator_overhead(self) -> int:
        """Reserved but unallocated: caching-allocator fragmentation."""
        return self.reserved - self.allocated

    def describe(self) -> str:
        """One-line summary in MiB, for bring-up and pressure logs."""
        mib = 2**20
        return (
            f"cuda:{self.device_index} "
            f"driver {self.driver_used / mib:.0f}/{self.driver_total / mib:.0f} MiB, "
            f"allocated {self.allocated / mib:.0f} MiB, "
            f"reserved {self.reserved / mib:.0f} MiB "
            f"(+{self.allocator_overhead / mib:.0f} MiB overhead)"
        )


def device_memory(device: Any = None) -> DeviceMemory:
    """Sample a device's memory at both levels.

    Raises:
        CapabilityError: if the resolved device is not CUDA -- there is no
            equivalent for CPU, and returning zeros would let a caller
            reconcile a ledger against nothing.
    """
    resolved = resolve_device(device, capability="device_memory")
    if resolved.type != "cuda":
        raise CapabilityError(
            "device_memory",
            detail=f"{resolved} is not a CUDA device",
            remedy="query memory only for CUDA devices",
        )
    module = require_torch()
    free, total = module.cuda.mem_get_info(resolved.index)
    return DeviceMemory(
        device_index=resolved.index,
        driver_free=int(free),
        driver_total=int(total),
        allocated=int(module.cuda.memory_allocated(resolved.index)),
        reserved=int(module.cuda.memory_reserved(resolved.index)),
    )


def empty_cache() -> None:
    """Return the caching allocator's free segments to the driver.

    Rarely the right call: it is synchronous, it destroys the reuse the
    allocator exists to provide, and the next allocation pays full driver cost.
    Legitimate uses are between phases (after weight loading, before KV
    materialization) and in tests. A no-op without CUDA.
    """
    if cuda_available():
        require_torch().cuda.empty_cache()


@contextmanager
def peak_memory_bytes(device: Any = None) -> Generator[list[int]]:
    """Measure peak allocator usage across a block.

    Yields a one-element list that is filled on exit, so the value survives the
    ``with``::

        with peak_memory_bytes() as peak:
            run_profiling_forward()
        activation_bytes = peak[0]

    This is how the activation budget is measured during bootstrap profiling
    with a dummy KV cache -- the step that breaks the KV/activation
    bootstrap cycle. Resets the peak counter on entry, so nested use is not
    supported.
    """
    result: list[int] = [0]
    if not cuda_available():
        yield result
        return
    module = require_torch()
    index = device_index(device)
    module.cuda.reset_peak_memory_stats(index)
    try:
        yield result
    finally:
        module.cuda.synchronize(index)
        result[0] = int(module.cuda.max_memory_allocated(index))


def empty_host_tensor(
    shape: tuple[int, ...] | int,
    dtype: Any,
    *,
    pinned: bool,
    allow_pageable: bool = False,
) -> tuple[Any, bool]:
    """Allocate a host tensor, page-locked when ``pinned`` asks for it.

    Returns ``(tensor, actually_pinned)``.  The flag is the *actual* tier, not
    the request: a caller that charges a memory ledger must charge what it got,
    and a predicted tier is not a materialized one.

    Args:
        shape: Tensor shape.
        dtype: Dtype or dtype name.
        pinned: Whether page-locked memory is required.
        allow_pageable: Fall back to ordinary host memory when pinning is
            impossible. Off by default, because the caller that charged a
            ledger for pinned bytes must not silently receive pageable memory —
            the two differ in whether DMA from the buffer is possible at all.

    Raises:
        CapabilityError: ``pinned`` was requested, the result would not be
            page-locked, and ``allow_pageable`` is False.
    """
    module = require_torch(capability="pinned_host_memory")
    resolved = torch_dtype(dtype)
    if not pinned:
        return module.empty(shape, dtype=resolved, device="cpu"), False
    if not cuda_available():
        if not allow_pageable:
            raise CapabilityError(
                "pinned_host_memory",
                detail="page-locked host memory requires CUDA",
                remedy="run on a CUDA host, or allow pageable fallback",
            )
        return module.empty(shape, dtype=resolved, device="cpu"), False
    try:
        tensor = module.empty(shape, dtype=resolved, device="cpu", pin_memory=True)
    except RuntimeError as exc:
        if not allow_pageable:
            raise CapabilityError(
                "pinned_host_memory",
                detail=f"cannot page-lock {shape} of {dtype_name(resolved)}: {exc}",
                remedy="lower the host tier size, or raise the RLIMIT_MEMLOCK ceiling",
            ) from exc
        return module.empty(shape, dtype=resolved, device="cpu"), False
    # ``pin_memory=True`` is a request, not a guarantee; report what exists.
    if bool(tensor.is_pinned()):
        return tensor, True
    if not allow_pageable:
        raise CapabilityError(
            "pinned_host_memory",
            detail="torch returned pageable memory for a pinned request",
            remedy="lower the host tier size, or raise the RLIMIT_MEMLOCK ceiling",
        )
    return tensor, False


def pinned_empty(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    allow_pageable: bool = True,
) -> tuple[Any, bool]:
    """Host buffer, page-locked when possible; see :func:`empty_host_tensor`."""
    return empty_host_tensor(shape, dtype, pinned=True, allow_pageable=allow_pageable)
