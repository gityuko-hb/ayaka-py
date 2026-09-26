"""Lazy wrapper around the legacy CUDA IPC native extension

Importing this module never imports PyTorch, never builds the native
extension and never initializes a CUDA context. The extension is JIT-built on
first real use through ``torch.utils.cpp_extension.load`` against a CUDA
runtime whose major version matches the installed torch build, so an
incompatible host toolkit (the CUDA 12.8 ``nvcc`` here vs. torch cu130) is
never pulled into the build.

The six primitives and their contract:

``ipc_supported``          host/device can export IPC memory
``legacy_ipc_capable``     fail-closed probe: tensor sits in an exportable
                           (non-VMM) allocation; never raises
``export_allocation``      -> ``(handle, byte_offset, allocation_nbytes, device)``
``open_allocation``        import a handle as an owning uint8 CUDA tensor
``byte_view``              flat uint8 view that keeps its owner alive
``ipc_handle_size``        runtime ``cudaIpcMemHandle_t`` size

Every public failure raises :class:`CapabilityError` with the ``cuda_ipc``
capability and an actionable remedy; native errors are chained as the cause
instead of being swallowed into a boolean.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_utils import cuda_available, torch_available

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

__all__ = [
    "ipc_available",
    "ipc_handle_size",
    "byte_view",
    "export_allocation",
    "legacy_ipc_capable",
    "open_allocation",
    "reset_cache",
]

logger = logging.getLogger(__name__)

CAPABILITY = "cuda_ipc"

_DISABLE_ENV = "AYAKA_DISABLE_CUDA_IPC"
_VERBOSE_ENV = "AYAKA_IPC_VERBOSE"
_BUILD_DIR_ENV = "AYAKA_IPC_BUILD_DIR"
_EXTENSION_NAME = "ayaka_ipc_ext"
_SOURCE = Path(__file__).resolve().parent / "csrc" / "ipc" / "ipc_ext.cpp"

_REMEDY = (
    "run on Linux with a CUDA runtime matching torch.version.cuda "
    "(pip nvidia-cuda-runtime-cuXX) or install ayaka with the cuda extra"
)

_lock = threading.Lock()
_extension: Any | None = None


def reset_cache() -> None:
    """Forget the loaded extension. Used by tests and kill-switch toggles."""
    global _extension
    with _lock:
        _extension = None


def _disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _platform_supported() -> bool:
    """Legacy CUDA IPC v1 is a Linux/NVIDIA lane; Windows import stays clean."""
    return sys.platform.startswith("linux")


def _optional_torch() -> Any | None:
    if not torch_available():
        return None
    import torch as _torch

    return _torch


def _unavailable_reason() -> str | None:
    """Return why IPC cannot be attempted here, or ``None`` when it can."""
    if _disabled():
        return f"{_DISABLE_ENV} is set"
    if not _platform_supported():
        return f"legacy CUDA IPC requires Linux, host platform is {sys.platform!r}"
    torch = _optional_torch()
    if torch is None:
        return "PyTorch is not importable"
    if getattr(torch.version, "hip", None):
        return "the ROCm/HIP IPC lane is outside the v1 target"
    if not cuda_available():
        return "no CUDA device is available"
    return None


def _require_available() -> Any:
    reason = _unavailable_reason()
    if reason is not None:
        raise CapabilityError(CAPABILITY, detail=reason, remedy=_REMEDY)
    return _load_extension()


def _find_cudart(root: Path) -> Path | None:
    """Locate the CUDA runtime shared library in ``root``.

    pip wheels ship only the versioned ``libcudart.so.13`` and no ``.so``
    linker symlink, so the loader links the exact path when the symlink is
    absent.
    """
    if not root.is_dir():
        return None
    for pattern in ("libcudart.so", "libcudart.so.*", "cudart64_*.dll"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _cudart_link_flags(lib: str) -> list[str]:
    library = _find_cudart(Path(lib))
    if library is not None:
        return [str(library)]
    return [f"-L{lib}", "-lcudart"]


def _cuda_runtime_paths() -> tuple[str, str]:
    """Locate CUDA headers/libs matching the torch CUDA major version.

    The pip ``nvidia/cuXX`` runtime is preferred over ``CUDA_HOME`` because
    torch may be built against a newer toolkit than the host's ``nvcc``; the
    extension is host-only C++ and only needs matching headers/libs, not nvcc.
    """
    torch = _optional_torch()
    cuda_version = None if torch is None else getattr(torch.version, "cuda", None)
    if not cuda_version:
        raise CapabilityError(
            CAPABILITY,
            detail="torch does not report a CUDA runtime version",
            remedy="install a CUDA build of torch (cuda extra)",
        )
    major = str(cuda_version).split(".", 1)[0]
    if torch is not None:
        site = Path(torch.__file__).resolve().parent.parent
        root = site / "nvidia" / f"cu{major}"
        include, lib = root / "include", root / "lib"
        if (include / "cuda_runtime_api.h").is_file() and _find_cudart(lib) is not None:
            return str(include), str(lib)
    for env_name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        home = os.environ.get(env_name)
        if not home:
            continue
        include = Path(home) / "include"
        for lib_name in ("lib64", "lib"):
            lib = Path(home) / lib_name
            if (include / "cuda_runtime_api.h").is_file() and _find_cudart(lib) is not None:
                return str(include), str(lib)
    raise CapabilityError(
        CAPABILITY,
        detail=(
            f"no CUDA {major}.x headers/libs found next to torch or under "
            "CUDA_HOME/CUDA_PATH/CUDA_ROOT"
        ),
        remedy=f"pip install nvidia-cuda-runtime-cu{major} matching torch.version.cuda",
    )


def _crt_include_dirs() -> list[Path]:
    """Include roots that contain a complete CUDA ``crt`` tree."""
    candidates: list[Path] = []
    try:
        import triton

        candidates.append(
            Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "include"
        )
    except ImportError:  # pragma: no cover - triton is a cuda-extra dependency
        pass
    for env_name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        home = os.environ.get(env_name)
        if home:
            candidates.append(Path(home) / "include")
    candidates.append(Path("/usr/local/cuda/include"))
    return [
        candidate for candidate in candidates if (candidate / "crt" / "host_defines.h").is_file()
    ]


def _crt_include_fallback(include: str) -> str | None:
    """Work around pip CUDA wheels that flatten ``crt/*.h``.

    The ``nvidia-cuda-runtime-cuXX`` wheels ship ``host_defines.h`` et al. at
    the include root while the headers still ``#include "crt/host_defines.h"``.
    A second, *stable* include root with a complete ``crt`` tree (Triton's
    bundled NVIDIA headers or a host toolkit) lets gcc resolve those includes.
    Stable paths matter: they become part of the ninja command line, and a
    random temp directory would force a rebuild in every process.
    """
    include_path = Path(include)
    if (include_path / "crt" / "host_defines.h").is_file():
        return None
    if not (include_path / "host_defines.h").is_file():
        return None
    for candidate in _crt_include_dirs():
        return str(candidate)
    return None


def _load_extension() -> Any:
    """JIT-build (once) and return the ``ayaka_ipc_ext`` module."""
    global _extension
    with _lock:
        if _extension is not None:
            return _extension
        if not _SOURCE.is_file():
            raise CapabilityError(
                CAPABILITY,
                detail=f"native source is missing from the installation: {_SOURCE}",
                remedy="reinstall a wheel that packages ayaka.kernel/csrc",
            )
        try:
            from torch.utils.cpp_extension import load
        except Exception as exc:  # pragma: no cover - torch always ships it
            raise CapabilityError(
                CAPABILITY,
                detail=f"torch.utils.cpp_extension is unavailable: {exc}",
                remedy="reinstall torch",
            ) from exc

        include, lib = _cuda_runtime_paths()
        include_paths = [include]
        fallback = _crt_include_fallback(include)
        if fallback is not None:
            include_paths.append(fallback)
        build_directory = os.environ.get(_BUILD_DIR_ENV) or None
        verbose = os.environ.get(_VERBOSE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        try:
            module = load(
                name=_EXTENSION_NAME,
                sources=[str(_SOURCE)],
                extra_cflags=["-O2"],
                extra_ldflags=_cudart_link_flags(lib),
                extra_include_paths=include_paths,
                build_directory=build_directory,
                with_cuda=False,
                verbose=verbose,
            )
        except Exception as exc:
            raise CapabilityError(
                CAPABILITY,
                detail=f"JIT build of {_EXTENSION_NAME} failed: {exc}",
                remedy=(
                    "check the compiler/toolchain and that the CUDA headers match "
                    "torch.version.cuda; set AYAKA_IPC_VERBOSE=1 for the build log"
                ),
            ) from exc
        _extension = module
        return module


def _wrap_native(operation: str, call: Any) -> Any:
    try:
        return call()
    except CapabilityError:
        raise
    except Exception as exc:
        raise CapabilityError(
            CAPABILITY,
            detail=f"{operation} failed: {exc}",
            remedy=_REMEDY,
        ) from exc


def ipc_available() -> bool:
    """Whether legacy CUDA IPC is usable on this host.

    Returns ``False`` (without building anything) for the kill-switch,
    unsupported platforms, unavailable CUDA and the ROCm lane. A native build
    failure on an otherwise eligible host raises :class:`CapabilityError`
    rather than being reported as a silent ``False``.
    """
    if _unavailable_reason() is not None:
        return False
    return bool(_wrap_native("ipc_supported", lambda: _load_extension().ipc_supported()))


def legacy_ipc_capable(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` lives in an exportable (non-VMM) allocation.

    Deliberately fail-closed: CPU tensors, non-tensors, empty tensors and any
    probed error answer ``False`` so capability checks never take the process
    down. Never triggers a JIT build for CPU inputs.
    """
    torch = _optional_torch()
    if torch is None or not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return False
    if _unavailable_reason() is not None:
        return False
    try:
        return bool(_load_extension().legacy_ipc_capable(tensor))
    except Exception:
        logger.debug("legacy_ipc_capable probe failed", exc_info=True)
        return False


def export_allocation(tensor: torch.Tensor) -> tuple[bytes, int, int, int]:
    """Export the allocation containing ``tensor``.

    Returns ``(handle, byte_offset, allocation_nbytes, device)`` where the
    offset locates ``tensor.data_ptr()`` inside the exported allocation. The
    caller (registry/communicator) owns keeping the tensor alive.
    """
    _require_cuda_tensor("export_allocation", tensor)
    if not tensor.is_contiguous():
        raise CapabilityError(
            CAPABILITY,
            detail="export_allocation requires a dense contiguous tensor",
            remedy="materialize the region before exporting it",
        )
    if tensor.numel() == 0:
        raise CapabilityError(
            CAPABILITY,
            detail="export_allocation requires a non-empty tensor",
            remedy="allocate at least one element",
        )
    _require_available()
    result = _wrap_native(
        "export_allocation",
        lambda: _load_extension().export_allocation(tensor),
    )
    handle, offset, allocation_nbytes, device = result
    return bytes(handle), int(offset), int(allocation_nbytes), int(device)


def open_allocation(handle: bytes, allocation_nbytes: int, device: int) -> torch.Tensor:
    """Import ``handle`` as an owning uint8 CUDA tensor of ``allocation_nbytes``.

    The returned tensor owns exactly one mapping: dropping it closes the
    mapping. Callers narrow it to their byte range with a view.
    """
    if type(allocation_nbytes) is not int or allocation_nbytes <= 0:
        raise CapabilityError(
            CAPABILITY,
            detail=f"invalid IPC allocation size {allocation_nbytes!r}",
            remedy="pass the allocation extent reported by export_allocation",
        )
    if type(device) is not int or device < 0:
        raise CapabilityError(
            CAPABILITY,
            detail=f"invalid IPC device {device!r}",
            remedy="pass the local CUDA device ordinal to open on",
        )
    if not isinstance(handle, (bytes, bytearray, memoryview)):
        raise CapabilityError(
            CAPABILITY,
            detail=f"IPC handle must be bytes, got {type(handle).__name__}",
            remedy="pass the opaque handle exactly as returned by export_allocation",
        )
    module = _require_available()
    raw = bytes(handle)
    expected = int(module.ipc_handle_size())
    if len(raw) != expected:
        raise CapabilityError(
            CAPABILITY,
            detail=f"unexpected IPC handle size: got {len(raw)} bytes, expected {expected}",
            remedy="pass the opaque handle exactly as returned by export_allocation",
        )
    return _wrap_native(
        f"open_allocation(device={device})",
        lambda: module.open_allocation(raw, allocation_nbytes, device),
    )


def byte_view(tensor: torch.Tensor, nbytes: int) -> torch.Tensor:
    """Flat uint8 view over the first ``nbytes`` of ``tensor``.

    The view keeps the owning tensor alive, so it can never dangle after the
    caller drops its own reference.
    """
    _require_cuda_tensor("byte_view", tensor)
    if type(nbytes) is not int or nbytes < 0:
        raise CapabilityError(
            CAPABILITY,
            detail=f"invalid byte_view size {nbytes!r}",
            remedy="pass a non-negative byte length",
        )
    limit = tensor.numel() * tensor.element_size()
    if nbytes > limit:
        raise CapabilityError(
            CAPABILITY,
            detail=f"byte_view of {nbytes} bytes exceeds the tensor's {limit} bytes",
            remedy="pass a length within the tensor",
        )
    if not tensor.is_contiguous():
        raise CapabilityError(
            CAPABILITY,
            detail="byte_view requires a dense contiguous tensor",
            remedy="materialize the region first",
        )
    module = _require_available()
    return _wrap_native("byte_view", lambda: module.byte_view(tensor, nbytes))


def ipc_handle_size() -> int:
    """Runtime size of an opaque IPC handle in bytes."""
    module = _require_available()
    return int(_wrap_native("ipc_handle_size", module.ipc_handle_size))


def _open_mapping_count() -> int:
    """Test-only: live native IPC mappings owned by this process."""
    module = _require_available()
    return int(_wrap_native("_open_mapping_count", module._open_mapping_count))


def _require_cuda_tensor(operation: str, tensor: Any) -> None:
    torch = _optional_torch()
    if torch is None or not isinstance(tensor, torch.Tensor):
        raise CapabilityError(
            CAPABILITY,
            detail=f"{operation} requires a torch.Tensor, got {type(tensor).__name__}",
            remedy="pass a CUDA tensor",
        )
    if not tensor.is_cuda:
        raise CapabilityError(
            CAPABILITY,
            detail=f"{operation} requires a CUDA tensor, got device {tensor.device}",
            remedy="move the tensor to a CUDA device first",
        )
