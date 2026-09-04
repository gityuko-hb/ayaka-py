"""NVML utility layer — safe wrapper around pynvml integrated with import_utils.

Guarantees:
1. Never imports torch.
2. Does not initialize CUDA runtime context (fully safe across fork).
3. Leverages LazyModule and has_module to delay and isolate imports.
4. Degrades gracefully: returns None or safe defaults instead of crashing control-plane commands.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Final

from ayaka.utils.import_utils import LazyModule, has_module

#: Lazy handle for pynvml so top-level imports remain zero-overhead
nvml: Final[Any] = LazyModule(
    "pynvml",
    capability="nvml",
    remedy="pip install pynvml",
)

@dataclass(frozen=True, slots=True)
class NvmlDeviceRaw:
    index: int
    name: str
    sm_major: int
    sm_minor: int
    num_sms: int
    hbm_bytes: int
    uuid: str
    pci_bus_id: str
    
def nvml_available() -> bool:
    """Check if pynvml package is available in the environment without executing it."""
    return has_module("pynvml")


@contextlib.contextmanager
def nvml_session() -> Generator[Any | None]:
    """Context manager handling nvmlInit and guaranteed nvmlShutdown."""
    if not nvml_available():
        yield None
        return

    try:
        nvml.nvmlInit()
    except Exception:
        yield None
        return

    try:
        yield nvml
    finally:
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()


def get_driver_version(pynvml_mod: Any) -> str:
    """Retrieve driver version string safely."""
    try:
        drv = pynvml_mod.nvmlSystemGetDriverVersion()
        return drv.decode("utf-8", errors="replace") if isinstance(drv, bytes) else str(drv)
    except Exception:
        return ""


def get_attribute_int(pynvml_mod: Any, handle: Any, attr_name: str) -> int:
    """Safely fetch an NVML device attribute; returns 0 on missing/failure."""
    try:
        get = getattr(pynvml_mod, "nvmlDeviceGetAttribute")
        attr = getattr(pynvml_mod, attr_name)
        return int(get(handle, attr))
    except Exception:
        return 0


def probe_device(pynvml_mod: Any, index: int, handle: Any) -> NvmlDeviceRaw:
    """Extract raw hardware metrics for a single GPU handle."""
    name_raw = pynvml_mod.nvmlDeviceGetName(handle)
    name = name_raw.decode("utf-8", errors="replace") if isinstance(name_raw, bytes) else str(name_raw)

    major, minor = pynvml_mod.nvmlDeviceGetCudaComputeCapability(handle)
    mem = pynvml_mod.nvmlDeviceGetMemoryInfo(handle)

    try:
        uuid_raw = pynvml_mod.nvmlDeviceGetUUID(handle)
        uuid = uuid_raw.decode("utf-8", errors="replace") if isinstance(uuid_raw, bytes) else str(uuid_raw)
    except Exception:
        uuid = ""

    try:
        pci = pynvml_mod.nvmlDeviceGetPciInfo(handle)
        bus_id = pci.busId.decode("utf-8", errors="replace") if isinstance(pci.busId, bytes) else str(pci.busId)
    except Exception:
        bus_id = ""

    num_sms = get_attribute_int(pynvml_mod, handle, "NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT")

    return NvmlDeviceRaw(
        index=index,
        name=name,
        sm_major=int(major),
        sm_minor=int(minor),
        num_sms=num_sms,
        hbm_bytes=int(mem.total),
        uuid=uuid,
        pci_bus_id=bus_id,
    )


def probe_nvlink_matrix(pynvml_mod: Any, handles: list[Any]) -> list[list[bool]]:
    """Build an NxN adjacency boolean matrix representing NVLink connectivity."""
    n = len(handles)
    matrix = [[False] * n for _ in range(n)]

    try:
        get_status = getattr(pynvml_mod, "nvmlDeviceGetP2PStatus")
        nvlink_ok = getattr(pynvml_mod, "NVML_P2P_STATUS_OK")
        cap = getattr(pynvml_mod, "NVML_P2P_CAPS_INDEX_NVLINK")
    except AttributeError:
        return matrix

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                if int(get_status(handles[i], handles[j], cap)) == int(nvlink_ok):
                    matrix[i][j] = True
            except Exception:
                continue
    return matrix