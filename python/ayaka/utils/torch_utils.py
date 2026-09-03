from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from types import ModuleType
from typing import Any, Final, Generator

from ayaka.utils.import_utils import CapabilityError, LazyModule

#: Lazy handle. Import this instead of ``torch`` in modules that must stay
#: importable without PyTorch.
torch: Final[Any] = LazyModule(
    "torch",
    capability="torch",
    remedy="pip install torch",
)

#: Compute-capability floors worth naming, because the bare tuples read as
#: magic numbers at the call sites that gate on them.
SM_AMPERE: Final[tuple[int, int]] = (8, 0)
"""bf16, TF32, ``cp.async``, FlashAttention-2."""
SM_ADA: Final[tuple[int, int]] = (8, 9)
"""Native FP8 arithmetic."""
SM_HOPPER: Final[tuple[int, int]] = (9, 0)
"""TMA (``cp.async.bulk``), ``wgmma``, FP8 tensor cores."""

@cache
def torch_available() -> bool:
    """Whether PyTorch can actually be imported.

    Cached, so the import happens at most once per process. Unlike
    ``has_module("torch")`` this executes the module, which is the only way to
    catch an installation whose shared libraries are missing.

    >>> torch_available() in (True, False)
    True
    """
    try:
        import torch as _torch
    except Exception:
        # Deliberately broad: a broken CUDA install raises OSError, a partial
        # wheel raises ImportError, and a version mismatch can raise RuntimeError
        # from torch's own bootstrap. All of them mean the same thing here.
        return False
    return True


def require_torch(*, capability: str | None = None, remedy: str | None = None) -> ModuleType:
    """Return the real ``torch`` module or raise :class:`CapabilityError`.

    Args:
        capability: Stable identifier for the feature that needs torch, used in
            the error message. Defaults to ``"torch"``.
        remedy: Install hint shown to the operator.

    >>> try:
    ...     require_torch(capability="kv_storage", remedy="pip install torch>=2.4")
    ... except CapabilityError as exc:
    ...     print(exc.capability, "|", exc.remedy)  # doctest: +SKIP
    """
    try:
        import torch as _torch
    except Exception as exc:
        raise CapabilityError(
            capability or "torch",
            detail=f"cannot import torch: {exc}",
            remedy=remedy or "pip install torch",
        ) from exc
    return _torch


@cache
def torch_version() -> tuple[int, int, int]:
    """Return ``(major, minor, patch)`` of the installed torch.

    Local version suffixes (``2.4.1+cu121``, ``2.6.0.dev20240101``) are
    stripped; a missing component reads as 0.

    Raises:
        CapabilityError: if torch is unavailable.
    """
    raw = require_torch(capability="torch_version").__version__
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(raw))
    if match is None:  # pragma: no cover - torch has always been PEP 440
        raise CapabilityError(
            "torch_version",
            detail=f"cannot parse torch version {raw!r}",
            remedy="install a release build of torch",
        )
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def require_torch_version(
    minimum: tuple[int, ...],
    *,
    capability: str,
    remedy: str | None = None,
) -> None:
    """Assert an installed torch at or above ``minimum``.

    Tuple comparison is lexicographic, so ``(2, 4) <= (2, 4, 1)`` holds and a
    two-element floor does the obvious thing.

    Raises:
        CapabilityError: when torch is missing or too old.
    """
    installed = torch_version()
    if installed[: len(minimum)] < tuple(minimum):
        wanted = ".".join(str(part) for part in minimum)
        have = ".".join(str(part) for part in installed)
        raise CapabilityError(
            capability,
            detail=f"requires torch >= {wanted}, found {have}",
            remedy=remedy or f"pip install --upgrade 'torch>={wanted}'",
        )


@cache
def cuda_available() -> bool:
    """Whether a usable CUDA device is present.

    False on a torch-free host, on a CPU-only build, and on a machine with no
    driver -- callers get one predicate instead of three.
    """
    if not torch_available():
        return False
    return bool(require_torch().cuda.is_available())


@cache
def device_count() -> int:
    """Number of visible CUDA devices; 0 when CUDA is unavailable.

    Reflects ``CUDA_VISIBLE_DEVICES``, which is what a tensor-parallel rank
    actually sees -- do not use it to size a world.
    """
    return require_torch().cuda.device_count() if cuda_available() else 0

def resolve_device(device: Any = None, *, capability: str = "device") -> Any:
    """Normalize a device specification into a ``torch.device``.

    ``None`` means "the best available": the current CUDA device when one
    exists, CPU otherwise. A bare ``"cuda"`` is pinned to the *current* index
    rather than left ambiguous, because an unindexed device silently follows
    whatever ``set_device`` ran last, and a KV slab that moves between steps is
    a class of bug worth making impossible.

    Args:
        device: ``None``, a string, an int index, or a ``torch.device``.
        capability: Name used if torch turns out to be unavailable.

    Raises:
        CapabilityError: if torch is unavailable, or CUDA was requested and is
            not present.
    """
    module = require_torch(capability=capability)
    if device is None:
        if cuda_available():
            return module.device(f"cuda:{module.cuda.current_device()}")
        return module.device("cpu")
    if isinstance(device, int) and not isinstance(device, bool):
        device = f"cuda:{device}"
    resolved = module.device(device)
    if resolved.type == "cuda":
        if not cuda_available():
            raise CapabilityError(
                capability,
                detail=f"device {device!r} requires CUDA, which is unavailable",
                remedy="run on a CUDA host, or pass device='cpu'",
            )
        if resolved.index is None:
            resolved = module.device(f"cuda:{module.cuda.current_device()}")
        if resolved.index >= device_count():
            raise CapabilityError(
                capability,
                detail=f"cuda:{resolved.index} is outside the {device_count()} visible devices",
                remedy="check CUDA_VISIBLE_DEVICES",
            )
    return resolved


def device_index(device: Any = None) -> int | None:
    """Return the CUDA index of ``device``, or None for a non-CUDA device."""
    resolved = resolve_device(device)
    return resolved.index if resolved.type == "cuda" else None


@contextmanager
def device_guard(device: Any) -> Generator[Any]:
    """Run a block with ``device`` current, restoring the previous one after.

    A no-op for CPU. Yields the resolved device so the body does not resolve it
    a second time.
    """
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        yield resolved
        return
    module = require_torch()
    previous = module.cuda.current_device()
    module.cuda.set_device(resolved.index)
    try:
        yield resolved
    finally:
        module.cuda.set_device(previous)

@cache
def compute_capability(index: int = 0) -> tuple[int, int] | None:
    """Return ``(major, minor)`` for a CUDA device, or None when unavailable.

    None rather than an exception, so a caller on a CPU host takes the same
    branch as one on an unknown device. Every predicate below treats None as
    "not supported", which is the safe direction.
    """
    if not cuda_available() or index >= device_count():
        return None
    major, minor = require_torch().cuda.get_device_capability(index)
    return int(major), int(minor)


def _at_least(floor: tuple[int, int], index: int) -> bool:
    capability = compute_capability(index)
    return capability is not None and capability >= floor


def supports_bf16(index: int = 0) -> bool:
    """bf16 tensor cores: Ampere and newer."""
    return _at_least(SM_AMPERE, index)


def supports_tf32(index: int = 0) -> bool:
    """TF32 matmul path: Ampere and newer."""
    return _at_least(SM_AMPERE, index)


def supports_flash_attention(index: int = 0) -> bool:
    """FlashAttention-2 kernels: Ampere and newer."""
    return _at_least(SM_AMPERE, index)


def supports_fp8(index: int = 0) -> bool:
    """Native FP8 arithmetic: Ada (sm_89) and newer.

    Note the gap this predicate is really about: FP8 *storage* works anywhere
    the dtype exists, because it is bytes. What needs sm_89 is arithmetic. A KV
    cache in FP8 on an Ampere card is legal and slow, not illegal -- see
    ``ayaka.cache.kv.storage.validation``.
    """
    return _at_least(SM_ADA, index)


def supports_tma(index: int = 0) -> bool:
    """TMA (``cp.async.bulk``) and ``wgmma``: Hopper and newer."""
    return _at_least(SM_HOPPER, index)


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Everything a backend selector needs about one device, resolved once.

    Backend selection is a global engine mode, not a per-request decision, so
    it happens at bring-up and this record is what it reads. Frozen and
    hashable, so it can key a compiled-kernel cache.
    """

    index: int
    name: str
    compute_capability: tuple[int, int] | None
    total_memory_bytes: int
    multiprocessor_count: int

    @property
    def is_cuda(self) -> bool:
        return self.compute_capability is not None

    @property
    def sm(self) -> str:
        """Architecture string, e.g. ``"sm_86"``. ``"cpu"`` off-device."""
        if self.compute_capability is None:
            return "cpu"
        major, minor = self.compute_capability
        return f"sm_{major}{minor}"

    def describe(self) -> str:
        """One-line summary for bring-up logs."""
        if not self.is_cuda:
            return "cpu"
        return (
            f"cuda:{self.index} {self.name} {self.sm} "
            f"{self.total_memory_bytes / 2**30:.1f} GiB "
            f"{self.multiprocessor_count} SMs"
        )


def device_profile(index: int = 0) -> DeviceProfile:
    """Collect a :class:`DeviceProfile`, or a CPU placeholder off-device."""
    if not cuda_available() or index >= device_count():
        return DeviceProfile(
            index=-1,
            name="cpu",
            compute_capability=None,
            total_memory_bytes=0,
            multiprocessor_count=0,
        )
    properties = require_torch().cuda.get_device_properties(index)
    return DeviceProfile(
        index=index,
        name=properties.name,
        compute_capability=compute_capability(index),
        total_memory_bytes=int(properties.total_memory),
        multiprocessor_count=int(properties.multi_processor_count),
    )

def has_dtype(name: str) -> bool:
    """Whether this torch build provides a dtype by name.

    Older builds have no ``float8_e4m3fn``; the check belongs at bring-up, not
    inside a constructor where it surfaces as ``AttributeError: module 'torch'
    has no attribute``.
    """
    if not torch_available():
        return False
    return getattr(require_torch(), name.removeprefix("torch."), None) is not None


def torch_dtype(name: str, *, capability: str = "dtype") -> Any:
    """Resolve a canonical dtype name to a ``torch.dtype``.

    Accepts ``"float16"`` or ``"torch.float16"``, and passes an actual
    ``torch.dtype`` straight through so callers can accept either.

    Raises:
        CapabilityError: if torch is unavailable or this build lacks the dtype.
    """
    module = require_torch(capability=capability)
    if isinstance(name, module.dtype):
        return name
    resolved = getattr(module, str(name).removeprefix("torch."), None)
    if not isinstance(resolved, module.dtype):
        raise CapabilityError(
            capability,
            detail=f"this PyTorch build does not provide dtype {name!r}",
            remedy="upgrade torch, or choose a dtype this build supports",
        )
    return resolved


def dtype_name(dtype: Any) -> str:
    """Canonical name of a ``torch.dtype``.

    ``str(torch.float16)`` is ``"torch.float16"``; this returns
    ``"float16"``, which is the vocabulary configs and checkpoints use.

    >>> dtype_name("torch.bfloat16")
    'bfloat16'
    """
    return str(dtype).removeprefix("torch.")


def dtype_bytes(dtype: Any) -> int:
    """Element size in bytes of a dtype or dtype name.

    Derived from torch itself rather than a table, so it stays correct for
    dtypes this module has never heard of.
    """
    module = require_torch(capability="dtype_bytes")
    resolved = torch_dtype(dtype) if not isinstance(dtype, module.dtype) else dtype
    return module.empty(0, dtype=resolved).element_size()

def synchronize(device: Any = None) -> None:
    """Block until every kernel queued on ``device`` has completed.

    A no-op without CUDA. Correct before releasing a slab whose streams may
    still be reading it; wrong anywhere on a decode path.
    """
    if cuda_available():
        require_torch().cuda.synchronize(device_index(device))


@contextmanager
def no_device_sync(*, strict: bool = True) -> Generator[None]:
    """Fail (or warn) on any implicit device-to-host synchronization inside.

    Wrap a hot path with this and every ``.item()``, ``.cpu()``, ``bool(t)``,
    ``int(t)`` and ``print(t)`` hiding in it raises. Those calls do two kinds of
    damage: each drains the pipeline for hundreds of microseconds, and none of
    them can be captured into a CUDA graph -- so a single one makes a whole
    decode step ungraphable.

    This is a debugging and CI tool, not something to leave in a hot path::

        with no_device_sync():
            engine.write_kv(layer, slots, key, value)

    Args:
        strict: Raise on a sync point. False downgrades to a warning, which is
            useful for a first pass over code that has many.

    A no-op on a torch build without ``set_sync_debug_mode`` (pre-1.11) and
    without CUDA.
    """
    module = require_torch(capability="sync_debug") if torch_available() else None
    setter = getattr(getattr(module, "cuda", None), "set_sync_debug_mode", None)
    if module is None or setter is None or not cuda_available():
        yield
        return
    previous = module.cuda.get_sync_debug_mode()
    setter("error" if strict else "warn")
    try:
        yield
    finally:
        setter(previous)

def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> int:
    """Seed Python, NumPy (if present) and torch, returning the seed.

    ``deterministic_algorithms`` additionally forces torch onto deterministic
    kernels. It is off by default because it is genuinely expensive and it
    makes some ops raise rather than run -- turn it on to reproduce a numerics
    bug, not in serving.

    Seeding does **not** make an LLM engine reproducible on its own: batch
    composition changes reduction order, and floating-point addition is not
    associative. Two runs with the same seed and different batching will
    diverge. Use it to make a *test* reproducible, and do not promise more.

    >>> seed_everything(1234)
    1234
    """
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed % (2**32))
    if torch_available():
        module = require_torch()
        module.manual_seed(seed)
        if cuda_available():
            module.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            module.use_deterministic_algorithms(True, warn_only=True)
    return seed