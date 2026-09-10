from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from ayaka.utils.import_utils import CapabilityError

logger = logging.getLogger(__name__)

__all__ = [
    "Platform",
    "capability_check",
    "cuda_is_initialized",
    "current_platform",
    "device_count_stateless",
    "platform_report",
    "probe_device_properties",
    "require_capability",
    "reset_platform_cache",
    "supported_archs",
]

#: Shared with the build system. CMakeLists.txt reads the same file to
#: populate CMAKE_CUDA_ARCHITECTURES, so runtime checks agree with build targets.
_ARCH_FILE = Path(__file__).resolve().parents[3] / "cuda_archs.json"


def capability_check(
    condition: bool,
    feature: str,
    *,
    detail: str,
    remedy: str | None = None,
) -> None:
    """Assert condition or raise CapabilityError with clear operator guidance."""
    if not condition:
        raise CapabilityError(
            feature,
            detail=detail,
            remedy=remedy or f"install or configure {feature} support",
        )


def cuda_is_initialized() -> bool:
    """Check whether PyTorch CUDA context has been initialized in this process."""
    try:
        import torch

        return bool(torch.cuda.is_initialized())
    except Exception:
        return False


def device_count_stateless() -> int:
    """Return visible GPU device count without initializing a CUDA context."""
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def probe_device_properties(device: int, property_names: list[str]) -> list[Any]:
    """Query low-level hardware attributes for a device."""
    import torch

    props = torch.cuda.get_device_properties(device)
    result: list[Any] = []
    for name in property_names:
        if name == "major":
            result.append(int(props.major))
        elif name == "minor":
            result.append(int(props.minor))
        elif name == "name":
            result.append(str(props.name))
        elif name == "total_memory":
            result.append(int(props.total_memory))
        else:
            result.append(getattr(props, name, None))
    return result


@lru_cache(maxsize=1)
def supported_archs() -> list[int]:
    """Compute capabilities Ayaka builds kernels for, as ``[80, 89, 90]``."""
    try:
        data = json.loads(_ARCH_FILE.read_text(encoding="utf-8"))
        return sorted(int(arch) for arch in data["architectures"])
    except FileNotFoundError:
        logger.warning(
            "cuda_archs.json not found at %s; defaulting to [80, 89, 90]",
            _ARCH_FILE,
        )
        return [80, 89, 90]
    except Exception as exc:
        logger.warning("Error reading %s: %s; fallback to [80, 89, 90]", _ARCH_FILE, exc)
        return [80, 89, 90]


class Platform:
    """A resolved backend platform, with its operations already bound.

    Attribute lookup is instant: detection happens once in ``current_platform()``
    and is cached, preventing import-time CUDA context creation.
    """

    __slots__ = ("_empty_cache", "_synchronize", "device_type", "name")

    def __init__(
        self,
        name: str,
        device_type: str,
        empty_cache: Any,
        synchronize: Any,
    ) -> None:
        self.name = name
        self.device_type = device_type
        self._empty_cache = empty_cache
        self._synchronize = synchronize

    # -- identity ------------------------------------------------------

    @property
    def is_cuda(self) -> bool:
        return self.name == "cuda"

    @property
    def is_rocm(self) -> bool:
        return self.name == "rocm"

    @property
    def is_gpu(self) -> bool:
        return self.name in ("cuda", "rocm")

    @property
    def is_cpu(self) -> bool:
        return self.name == "cpu"

    def __repr__(self) -> str:
        return f"<Platform {self.name} devices={self.device_count()}>"

    # -- operations ----------------------------------------------------

    def empty_cache(self) -> None:
        self._empty_cache()

    def synchronize(self) -> None:
        self._synchronize()

    def device_count(self) -> int:
        return device_count_stateless() if self.is_gpu else 0

    # -- capability queries --------------------------------------------

    def compute_capability(self, device: int = 0) -> tuple[int, int]:
        """``(major, minor)`` compute capability for `device`."""
        capability_check(
            self.is_gpu,
            "gpu",
            detail=f"platform is {self.name}",
        )
        return _compute_capability(device)

    def sm_version(self, device: int = 0) -> int:
        """Compute capability as the integer ``80``, ``89``, ``90``."""
        major, minor = self.compute_capability(device)
        return major * 10 + minor

    def device_name(self, device: int = 0) -> str:
        return _device_name(device)

    def total_memory(self, device: int = 0) -> int:
        """Total device memory in bytes."""
        return _total_memory(device)

    def supports_bf16(self, device: int = 0) -> bool:
        return self.is_gpu and self.sm_version(device) >= 80

    def supports_fp8(self, device: int = 0) -> bool:
        """Native fp8 tensor-core support: Ada (89) and Hopper (90) onward."""
        return self.is_gpu and self.sm_version(device) >= 89

    def is_pin_memory_available(self) -> bool:
        """Whether pinned host allocations are usable.

        WSL reports pinned memory as available but performs catastrophically
        over PCIe virtualization, so it is treated as disabled under WSL.
        """
        if not self.is_gpu:
            return False
        if _in_wsl():
            logger.warning(
                "Pinned memory is disabled under WSL: host-to-device transfers "
                "are significantly slower than native Linux."
            )
            return False
        return True

    def free_memory(self, device: int = 0, fraction: float = 1.0) -> int:
        """Free device memory in bytes, discounted by `fraction`.

        Computed against total memory to preserve fixed headroom for NCCL,
        cuBLAS workspace, and memory fragmentation.
        """
        if not self.is_gpu:
            try:
                import psutil

                return int(psutil.virtual_memory().available * fraction)
            except Exception:
                return 0

        import torch

        free, total = torch.cuda.mem_get_info(device)
        return max(0, int(free - (1.0 - fraction) * total))


@lru_cache(maxsize=16)
def _compute_capability(device: int) -> tuple[int, int]:
    major, minor = probe_device_properties(device, ["major", "minor"])
    return int(major), int(minor)


@lru_cache(maxsize=16)
def _device_name(device: int) -> str:
    (name,) = probe_device_properties(device, ["name"])
    return str(name)


@lru_cache(maxsize=16)
def _total_memory(device: int) -> int:
    (total,) = probe_device_properties(device, ["total_memory"])
    return int(total)


def _in_wsl() -> bool:
    import platform

    return "microsoft" in platform.uname().release.lower()


def _noop(*args: Any, **kwargs: Any) -> None:
    pass


@lru_cache(maxsize=1)
def current_platform() -> Platform:
    """Detect platform backend lazily on first call."""
    try:
        import torch

        if getattr(torch.version, "hip", None) is not None:
            platform = Platform(
                "rocm", "cuda", torch.cuda.empty_cache, torch.cuda.synchronize
            )
        elif (
            getattr(torch.version, "cuda", None) is not None
            and torch.cuda.is_available()
        ):
            platform = Platform(
                "cuda", "cuda", torch.cuda.empty_cache, torch.cuda.synchronize
            )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            platform = Platform(
                "xpu", "xpu", torch.xpu.empty_cache, torch.xpu.synchronize
            )
        else:
            platform = Platform("cpu", "cpu", _noop, _noop)
    except Exception:
        platform = Platform("cpu", "cpu", _noop, _noop)

    logger.debug("Detected platform %s", platform.name)
    return platform


def reset_platform_cache() -> None:
    """Clear platform caches. For tests only."""
    current_platform.cache_clear()
    supported_archs.cache_clear()
    _compute_capability.cache_clear()
    _device_name.cache_clear()
    _total_memory.cache_clear()


def require_capability(
    minimum_sm: int,
    feature: str,
    *,
    device: int = 0,
    remedy: str | None = None,
) -> None:
    """Raise CapabilityError unless device meets minimum_sm and is built."""
    platform = current_platform()
    capability_check(
        platform.is_gpu,
        feature,
        detail=f"requires a GPU, platform is {platform.name}",
        remedy=remedy,
    )

    actual = platform.sm_version(device)
    capability_check(
        actual >= minimum_sm,
        feature,
        detail=f"requires sm_{minimum_sm}, device has sm_{actual}",
        remedy=remedy,
    )

    archs = supported_archs()
    if archs and not any(actual >= arch for arch in archs):
        raise CapabilityError(
            feature,
            detail=(
                f"device sm_{actual} is below every architecture this build "
                f"targets ({archs})"
            ),
            remedy=f"add {actual} to {_ARCH_FILE.name} and rebuild",
        )


def platform_report() -> dict[str, Any]:
    """Summary of hardware and execution platform for diagnostics."""
    platform = current_platform()
    report: dict[str, Any] = {
        "platform": platform.name,
        "device_count": platform.device_count(),
        "cuda_initialized": cuda_is_initialized() if platform.is_gpu else False,
        "build_architectures": supported_archs(),
        "devices": [],
    }
    for index in range(platform.device_count()):
        try:
            report["devices"].append(
                {
                    "index": index,
                    "name": platform.device_name(index),
                    "sm": platform.sm_version(index),
                    "total_memory": platform.total_memory(index),
                    "bf16": platform.supports_bf16(index),
                    "fp8": platform.supports_fp8(index),
                }
            )
        except Exception as exc:
            report["devices"].append({"index": index, "error": str(exc)})
    return report
