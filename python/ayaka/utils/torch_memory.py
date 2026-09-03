from __future__ import annotations
 
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final
 
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
def peak_memory_bytes(device: Any = None) -> Iterator[list[int]]:
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
        
_MEMINFO: Final[str] = "/proc/meminfo"
_CGROUP_V2_MAX: Final[str] = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_MAX: Final[str] = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

def _host_total_bytes() -> int | None:
    """Total host RAM, honouring a cgroup limit when one applies.
 
    A container's ``MemTotal`` is the *host's*, not the container's, so a
    pinned-memory ceiling computed from ``/proc/meminfo`` alone will happily
    exceed the cgroup limit and get the process OOM-killed by the kernel rather
    than refused by an allocator.
    """
    limits: list[int] = []
    try:
        with open(_MEMINFO, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    limits.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    for path in (_CGROUP_V2_MAX, _CGROUP_V1_MAX):
        try:
            with open(path, encoding="ascii") as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        value = int(raw)
        # cgroup v1 writes a sentinel near 2**63 to mean "unlimited".
        if 0 < value < 2**62:
            limits.append(value)
    return min(limits) if limits else None
 
 
def host_pinned_ceiling_bytes(
    *,
    requested_bytes: int | None = None,
    fraction: float = 0.25,
    reserve_bytes: int = 2 * 2**30,
) -> int:
    """Safe upper bound on page-locked host memory.
 
    Pinned memory is not swappable: over-pinning does not degrade, it takes the
    kernel's reclaimable pool away and ends in an OOM kill that no Python
    handler sees. So the ceiling is the minimum of what the operator asked for
    and what the system can survive.
 
    Args:
        requested_bytes: What the operator configured, if anything.
        fraction: Share of total host memory that may be pinned.
        reserve_bytes: Absolute floor left to the rest of the system.
 
    Returns:
        The byte ceiling, never negative.
 
    >>> host_pinned_ceiling_bytes(requested_bytes=0)
    0
    """
    total = _host_total_bytes()
    if total is None:
        # Unknown host size: trust only what was explicitly asked for.
        return max(requested_bytes or 0, 0)
    system_ceiling = max(int(total * fraction), 0)
    system_ceiling = min(system_ceiling, max(total - reserve_bytes, 0))
    if requested_bytes is None:
        return system_ceiling
    return max(min(requested_bytes, system_ceiling), 0)
 
 
def pinned_empty(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    allow_pageable: bool = True,
) -> tuple[Any, bool]:
    """Allocate a host buffer, page-locked when possible.
 
    Returns ``(tensor, pinned)``. The flag is the *actual* tier, not the
    request: a caller that charges a memory ledger must charge what it got, and
    a predicted tier is not a materialized one.
 
    Args:
        shape: Tensor shape.
        dtype: Dtype or dtype name.
        allow_pageable: Fall back to ordinary host memory when pinning fails.
            Set False when only DMA-capable memory is acceptable.
 
    Raises:
        CapabilityError: when pinning fails and ``allow_pageable`` is False.
    """
    module = require_torch(capability="pinned_host_memory")
    resolved = torch_dtype(dtype)
    if cuda_available():
        try:
            return module.empty(shape, dtype=resolved, device="cpu", pin_memory=True), True
        except RuntimeError as exc:
            if not allow_pageable:
                raise CapabilityError(
                    "pinned_host_memory",
                    detail=f"cannot page-lock {shape} of {dtype_name(resolved)}: {exc}",
                    remedy="lower the host tier size, or raise the RLIMIT_MEMLOCK ceiling",
                ) from exc
    elif not allow_pageable:
        raise CapabilityError(
            "pinned_host_memory",
            detail="page-locked host memory requires CUDA",
            remedy="run on a CUDA host, or allow pageable fallback",
        )
    return module.empty(shape, dtype=resolved, device="cpu"), False