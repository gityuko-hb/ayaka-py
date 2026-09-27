from __future__ import annotations

import logging
import re
from enum import StrEnum
from functools import lru_cache
from typing import Any

from ayaka.utils.import_utils import CapabilityError

logger = logging.getLogger(__name__)

__all__ = [
    "ArchReachability",
    "Platform",
    "arch_reachability",
    "capability_check",
    "cuda_is_initialized",
    "current_platform",
    "device_count_stateless",
    "framework_architectures",
    "framework_exact_architectures",
    "framework_ptx",
    "platform_report",
    "probe_device_properties",
    "require_capability",
    "reset_platform_cache",
]


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


#: A framework arch-list entry. The optional suffix decides how far the entry's
#: compatibility actually reaches, so it cannot be stripped before reading the
#: number: ``sm_90a`` is arch-specific and runs on exactly 9.0, while plain
#: ``sm_90`` and family ``sm_90f`` both run on 9.0 and every later minor.
_ARCH_NAME = re.compile(r"^sm_(\d+)([af])?$")


@lru_cache(maxsize=1)
def framework_architectures() -> tuple[int, ...]:
    """Compute capabilities the installed framework ships broadly compatible cubins for.

    Read from the framework itself rather than from a file this project
    maintains. A project-side list can only ever disagree with the wheel that is
    actually installed, and it has no way to influence what gets compiled -- so
    it could add false claims and never remove one.

    Entries here are plain ``sm_NN`` and family ``sm_NNf`` targets, both of
    which run on the same major revision with a minor at least as high. Entries
    whose compatibility is narrower than that are excluded and reported by
    :func:`framework_exact_architectures` instead.

    Returns the ``major*10+minor`` form, e.g. ``[75, 80, 86, 90, 100, 120]``.
    Empty when the framework has no CUDA build. This is a build-time property of
    the wheel, so it resolves without a visible device.
    """
    broad, exact = _parse_framework_arches()
    return broad


@lru_cache(maxsize=1)
def framework_exact_architectures() -> tuple[int, ...]:
    """Compute capabilities covered only on the exact revision, if any.

    An ``sm_NNa`` target is architecture-specific: it runs on that compute
    capability and on nothing else, not even a later minor of the same major.
    Folding it into :func:`framework_architectures` would make a device look
    supported when its kernel cannot load there.
    """
    _broad, exact = _parse_framework_arches()
    return exact


def _parse_framework_arches() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the framework's arch list into broadly compatible and exact-only."""
    try:
        from ayaka.utils.torch_utils import torch

        names = [str(name) for name in torch.cuda.get_arch_list()]
    except Exception:
        return (), ()
    broad: set[int] = set()
    exact: set[int] = set()
    for name in names:
        match = _ARCH_NAME.match(name)
        if match is None:
            # An entry we cannot read precisely is excluded rather than guessed.
            # Guessing here is how a device gets labelled supported and then
            # fails to launch.
            logger.debug("unrecognised framework arch entry %r; excluded", name)
            continue
        arch = int(match.group(1))
        if match.group(2) == "a":
            exact.add(arch)
        else:
            broad.add(arch)
    return tuple(sorted(broad)), tuple(sorted(exact))


@lru_cache(maxsize=1)
def framework_ptx() -> int | None:
    """Compute capability of the framework's embedded PTX, if it embeds any.

    Derived from the framework's own gencode flags, so it reflects what is
    actually shipped. Many binary wheels embed cubins only and no PTX at all,
    and in that case there is nothing to JIT from -- which is why this may
    correctly be ``None``.
    """
    try:
        from ayaka.utils.torch_utils import torch

        flags = str(torch.cuda.get_gencode_flags())
    except Exception:
        return None
    targets = {int(digits) for digits in re.findall(r"code=compute_(\d+)", flags) if digits}
    if not targets:
        return None
    # Several PTX targets can be embedded; the highest is the one that reaches
    # the furthest, since PTX compatibility is upward only.
    return max(targets)


class ArchReachability(StrEnum):
    """How the installed framework can reach a device, strongest first.

    ``NATIVE`` is the only level that proves a cubin exists, but it does not
    promise the architecture's own fast paths: a plain ``sm_120`` cubin runs on a
    12.0 device without the family-specific (``sm_120f``) features, so the
    ``supports_*`` predicates remain the authority on what is accelerated.
    ``PTX_JIT_ONLY`` works but pays a JIT cost on first launch and is not a
    claim of qualification. ``UNREACHABLE`` means a kernel launch would fail.
    """

    NATIVE = "native"
    PTX_JIT_ONLY = "ptx_jit_only"
    UNREACHABLE = "unreachable"


def arch_reachability(sm: int) -> ArchReachability:
    """Classify how the installed framework reaches compute capability ``sm``.

    ``sm`` uses the ``major*10+minor`` form, matching
    :meth:`Platform.sm_version`. Two distinct rules apply, and using only one of
    them is how a device gets mislabelled in both directions:

    * A **cubin** for ``X.y`` runs on a device of ``X.y'`` only when the major
      revision matches and ``y' >= y``. So ``sm_86`` covers ``sm_89`` (Ada), and
      ``sm_86`` does not cover ``sm_90``.
    * **PTX** for ``compute_X`` runs on any device at ``X`` or above, crossing
      major revisions as well. Many binary wheels embed no PTX at all, in which
      case nothing above the highest cubin is reachable.
    """
    if type(sm) is not int or isinstance(sm, bool) or sm < 20:
        raise ValueError("sm must be a compute capability as major*10+minor")
    major, minor = _split_cc(sm)
    for built in framework_architectures():
        built_major, built_minor = _split_cc(built)
        if built_major == major and built_minor <= minor:
            return ArchReachability.NATIVE
    # An architecture-specific target covers its own revision and nothing else,
    # not even a later minor of the same major.
    if sm in framework_exact_architectures():
        return ArchReachability.NATIVE
    ptx = framework_ptx()
    if ptx is not None:
        ptx_major, ptx_minor = _split_cc(ptx)
        if (major, minor) >= (ptx_major, ptx_minor):
            return ArchReachability.PTX_JIT_ONLY
    return ArchReachability.UNREACHABLE


def _split_cc(sm: int) -> tuple[int, int]:
    """Split the ``major*10+minor`` form into a comparable pair."""
    return sm // 10, sm % 10


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
        return device_count_stateless() if self.is_cuda else 0

    # -- capability queries --------------------------------------------

    def compute_capability(self, device: int = 0) -> tuple[int, int]:
        """``(major, minor)`` compute capability for `device`."""
        capability_check(
            self.is_cuda,
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
        return self.is_cuda and self.sm_version(device) >= 80

    def supports_fp8(self, device: int = 0) -> bool:
        """Native fp8 tensor-core support: Ada (89) and Hopper (90) onward."""
        return self.is_cuda and self.sm_version(device) >= 89

    def is_pin_memory_available(self) -> bool:
        """Whether pinned host allocations are usable.

        WSL reports pinned memory as available but performs catastrophically
        over PCIe virtualization, so it is treated as disabled under WSL.
        """
        if not self.is_cuda:
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
        if not self.is_cuda:
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
    """Detect platform backend lazily on first call: CUDA, else CPU fallback."""
    try:
        import torch

        if getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available():
            platform = Platform("cuda", "cuda", torch.cuda.empty_cache, torch.cuda.synchronize)
        else:
            platform = Platform("cpu", "cpu", _noop, _noop)
    except Exception:
        platform = Platform("cpu", "cpu", _noop, _noop)

    logger.debug("Detected platform %s", platform.name)
    return platform


def reset_platform_cache() -> None:
    """Clear platform caches. For tests only."""
    current_platform.cache_clear()
    framework_architectures.cache_clear()
    framework_exact_architectures.cache_clear()
    framework_ptx.cache_clear()
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
        platform.is_cuda,
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

    reach = arch_reachability(actual)
    if reach is ArchReachability.UNREACHABLE:
        ptx = framework_ptx()
        raise CapabilityError(
            feature,
            detail=(
                f"device sm_{actual} is not covered by the installed framework: it "
                f"ships cubins for {framework_architectures()} and "
                + (f"PTX for sm_{ptx}" if ptx is not None else "no PTX at all")
            ),
            remedy=(
                f"install a torch build that includes sm_{actual}; a cubin only "
                "covers the same major revision with a minor at least as high, "
                "and PTX only reaches upward"
            ),
        )
    if reach is ArchReachability.PTX_JIT_ONLY:
        # Reachable, but a first-launch JIT cost is a real cost. Do not hide it.
        logger.info(
            "device sm_%s has no cubin; the framework will JIT it from PTX on first launch",
            actual,
        )


def platform_report() -> dict[str, Any]:
    """Summary of hardware and execution platform for diagnostics.

    Never raises. The architecture facts come from the installed framework and
    are safe to read even with no visible device, so a headless bring-up still
    gets a complete picture.
    """
    platform = current_platform()
    archs = framework_architectures()
    ptx = framework_ptx()
    report: dict[str, Any] = {
        "platform": platform.name,
        "device_count": platform.device_count(),
        "cuda_initialized": cuda_is_initialized() if platform.is_cuda else False,
        "framework_architectures": archs,
        "framework_ptx_architecture": ptx,
        "framework_has_cuda": bool(archs),
        "devices": [],
    }
    for index in range(platform.device_count()):
        try:
            sm = platform.sm_version(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": platform.device_name(index),
                    "sm": sm,
                    "reachability": str(arch_reachability(sm).value),
                    "total_memory": platform.total_memory(index),
                    "bf16": platform.supports_bf16(index),
                    "fp8": platform.supports_fp8(index),
                }
            )
        except Exception as exc:
            report["devices"].append({"index": index, "error": str(exc)})
    return report
