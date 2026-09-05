"""Device / dtype / layout validation, without allocating anything.

Runs before a single tensor exists, so a configuration that cannot execute
fails at bring-up with a list of reasons rather than at the first decode with
an ``AttributeError`` or, worse, with quietly wrong numbers.

The design is **collect then decide**: every issue is gathered and returned
together. Raising on the first one makes an operator fix a config, rerun, and
discover the next problem, once per round trip.

Two changes from the version this replaces:

* ``LAYOUT_UNSUPPORTED`` is now actually reachable. It was declared and never
  emitted, while layout errors came out of ``__post_init__`` as bare
  ``ValueError`` -- so the structured code existed for a path that could not
  produce it, and the real path bypassed the structure entirely. The two
  questions are genuinely different and are now answered in the right places:
  "is this layout meaningful for this family" is a construction-time invariant,
  "does this layout have a backend on this device" is a validation-time fact.
* ``KIND_UNSUPPORTED`` classifies a recurrent model instead of letting it fall
  through an unreachable branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ayaka.kvcache.storage.dtypes import is_fp8_storage_dtype
from ayaka.kvcache.storage.errors import KVStorageCompatibilityError
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec
from ayaka.types import KVLayoutKind
from ayaka.utils.torch_utils import has_dtype, torch_available

#: Compute capability floor for native FP8 arithmetic: sm_89 (Ada) and above.
#:
#: Tuple comparison is lexicographic, so sm_86 (Ampere, e.g. RTX 3050/A100)
#: sorts below and is refused, while sm_90 (Hopper) passes. Below the floor FP8
#: is still exercisable through the dequantized reference path, which is why
#: this is a validation issue and not a hard import-time error.
MINIMUM_NATIVE_FP8_COMPUTE_CAPABILITY: Final[tuple[int, int]] = (8, 9)

#: Layouts with a validated backend, per device class.
#:
#: ``HND`` is deliberately absent everywhere: it is constructible, so the
#: ``LAYOUT_UNSUPPORTED`` path is reachable and testable rather than being dead
#: code that decays.
_VALIDATED_LAYOUTS: Final[dict[str, frozenset[KVLayoutKind]]] = {
    "cuda": frozenset({KVLayoutKind.NHD, KVLayoutKind.NLD}),
    "cpu": frozenset({KVLayoutKind.NHD, KVLayoutKind.NLD}),
}


class KVStorageIssueCode(StrEnum):
    """Structured reasons a storage configuration cannot run on a device."""

    DTYPE_UNAVAILABLE = "dtype_unavailable"
    """PyTorch is not installed, or this build lacks the dtype."""
    DEVICE = "device"
    """The dtype or format requires a device class the target is not."""
    COMPUTE_CAPABILITY = "compute_capability"
    """Native execution requires a CUDA compute capability floor."""
    LAYOUT_UNSUPPORTED = "layout_unsupported"
    """The layout has no validated backend on the target device."""
    KIND_UNSUPPORTED = "kind_unsupported"
    """No storage implementation exists for this cache family."""
    QUANTIZATION = "quantization"
    """The quantization policy is incoherent with the dtype."""


@dataclass(frozen=True, slots=True)
class KVStorageCompatibilityIssue:
    """One reason a configuration was rejected.

    Attributes:
        code: Machine-readable category, for callers that branch.
        message: Human-readable detail, for the operator who has to fix it.
    """

    code: KVStorageIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class KVStorageValidationResult:
    """Outcome of validating one storage spec against one device."""

    spec: BaseKVStorageSpec
    device_type: str
    compute_capability: tuple[int, int] | None
    issues: tuple[KVStorageCompatibilityIssue, ...]

    @property
    def compatible(self) -> bool:
        """True exactly when no issues were found."""
        return not self.issues

    def codes(self) -> frozenset[KVStorageIssueCode]:
        """The distinct issue categories, for callers that branch on them."""
        return frozenset(issue.code for issue in self.issues)

    def require_compatible(self) -> None:
        """Raise when incompatible.

        Raises:
            KVStorageCompatibilityError: carrying this result on ``.result``.
        """
        if self.issues:
            raise KVStorageCompatibilityError(self)


def validate_kv_storage_support(
    spec: BaseKVStorageSpec,
    *,
    device_type: str,
    compute_capability: tuple[int, int] | None = None,
) -> KVStorageValidationResult:
    """Validate dtype, device, layout and quantization without allocating.

    The support matrix: fp32, fp16 and bf16 run anywhere. FP8 needs a CUDA
    device at compute capability 8.9 or above for native execution; below that
    it is only exercisable through the dequantized reference path.

    Args:
        spec: Geometry to validate.
        device_type: Target device class, e.g. ``"cuda"`` or ``"cpu"``.
        compute_capability: CUDA compute capability, required for FP8 decisions.

    Returns:
        The structured result. Call
        :meth:`~KVStorageValidationResult.require_compatible` to fail loudly.

    Raises:
        TypeError: if ``spec`` is not a :class:`KVStorageSpec`. This is a
            nominal ``isinstance`` against a base class: one MRO walk, and it
            checks the actual type. The ``runtime_checkable`` Protocol it
            replaces compared attribute *names* only, so an object with the
            right names and entirely wrong signatures passed.
        ValueError: for an empty ``device_type``.
    """
    if not isinstance(spec, BaseKVStorageSpec):
        raise TypeError(f"spec must be a BaseKVStorageSpec, got {type(spec).__name__}")
    normalized_device = device_type.strip().lower()
    if not normalized_device:
        raise ValueError("device_type must not be empty")

    issues: list[KVStorageCompatibilityIssue] = []

    # family
    if not spec.kind.is_implemented:
        issues.append(
            KVStorageCompatibilityIssue(
                KVStorageIssueCode.KIND_UNSUPPORTED,
                f"no storage implementation exists for cache family {spec.kind.value}",
            )
        )

    # torch build
    if not torch_available():
        issues.append(
            KVStorageCompatibilityIssue(
                KVStorageIssueCode.DTYPE_UNAVAILABLE,
                "PyTorch is not installed",
            )
        )
    elif not has_dtype(spec.dtype):
        issues.append(
            KVStorageCompatibilityIssue(
                KVStorageIssueCode.DTYPE_UNAVAILABLE,
                f"this PyTorch build does not provide {spec.dtype}",
            )
        )

    # FP8 gate
    if is_fp8_storage_dtype(spec.dtype):
        if normalized_device != "cuda":
            issues.append(
                KVStorageCompatibilityIssue(
                    KVStorageIssueCode.DEVICE,
                    "native FP8 KV execution requires a CUDA device",
                )
            )
        elif compute_capability is None:
            issues.append(
                KVStorageCompatibilityIssue(
                    KVStorageIssueCode.COMPUTE_CAPABILITY,
                    "CUDA compute capability is required to decide native FP8 KV support",
                )
            )
        elif compute_capability < MINIMUM_NATIVE_FP8_COMPUTE_CAPABILITY:
            issues.append(
                KVStorageCompatibilityIssue(
                    KVStorageIssueCode.COMPUTE_CAPABILITY,
                    f"native FP8 KV requires compute capability at least "
                    f"{MINIMUM_NATIVE_FP8_COMPUTE_CAPABILITY}, got {compute_capability}",
                )
            )

    # quantization coherence
    quantization = spec.quantization
    if quantization is None:
        issues.append(
            KVStorageCompatibilityIssue(
                KVStorageIssueCode.QUANTIZATION,
                f"{type(spec).__name__} did not initialize its quantization policy",
            )
        )
    else:
        # Unreachable for a spec built through the public constructors, which
        # enforce the same invariant. Kept because a spec can also arrive from
        # deserialization or from a subclass written elsewhere, and a silent
        # pass there would put an unscaled FP8 cache into production.
        try:
            quantization.validate_for_dtype(spec.dtype)
        except ValueError as exc:
            issues.append(KVStorageCompatibilityIssue(KVStorageIssueCode.QUANTIZATION, str(exc)))

    # layout backend
    supported = _VALIDATED_LAYOUTS.get(normalized_device)
    if supported is None:
        issues.append(
            KVStorageCompatibilityIssue(
                KVStorageIssueCode.DEVICE,
                f"unknown device class {normalized_device!r}; "
                f"known: {', '.join(sorted(_VALIDATED_LAYOUTS))}",
            )
        )
    elif spec.layout not in supported:
        issues.append(
            KVStorageCompatibilityIssue(
                KVStorageIssueCode.LAYOUT_UNSUPPORTED,
                f"layout {spec.layout.value} has no validated backend on {normalized_device}; "
                f"supported: {', '.join(sorted(item.value for item in supported))}",
            )
        )

    return KVStorageValidationResult(
        spec=spec,
        device_type=normalized_device,
        compute_capability=compute_capability,
        issues=tuple(issues),
    )
