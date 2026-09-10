"""KV quantization policy.

This module exists to close a hole in the previous design. That version had a
full FP8 apparatus -- a dtype registry entry, an ``is_fp8`` predicate, a
compute-capability gate at sm_89, and ``uint8`` view workarounds on both the
scatter and gather paths -- but **nowhere to put a scale**. Dequantization was
a bare ``tensor.to(dtype)`` cast.

That is not a missing feature, it is silent corruption. ``float8_e4m3fn``
saturates at 448.0. Real K/V activations exceed that routinely, so an
unscaled store clamps or produces NaN, and the damage shows up as slightly
wrong logits many layers downstream with no exception anywhere.

Three decisions are encoded here:

1. **Scale is part of the spec, not of attention metadata.** The capacity
   planner has to know the scale tensors exist (they cost bytes), and the
   attention backend has to fetch them together with the cache. Putting them
   anywhere else means both reach into storage anyway.

2. **FP8 without a scheme is unconstructible.** The invariant lives in
   ``__post_init__``, so "FP8 with no scale" is not a bug you can write.

3. **Scales are static and calibrated, not computed per write.** A dynamic
   per-write scale would change the meaning of pages *already stored* under
   the old scale. Because a KV cache is append-only across many steps, that
   retroactively corrupts every earlier token. See
   :meth:`KVQuantization.explain_static_requirement`.

Nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .dtypes import is_fp8_storage_dtype, normalize_storage_dtype, storage_dtype_max

#: Scales are always fp32 regardless of the storage dtype. They are read once
#: per kernel launch and broadcast, so their precision is free, while a fp16
#: scale would itself round-trip and reintroduce error.
SCALE_DTYPE: Final[str] = "float32"
SCALE_DTYPE_BYTES: Final[int] = 4

class QuantizationScheme(StrEnum):
    """Granularity at which a scale is applied."""

    NONE = "none"
    """Values are stored as-is. The only legal scheme for fp32/fp16/bf16."""

    PER_TENSOR = "per_tensor"
    """One scalar scale per (layer, plane).

    Matches what vLLM and SGLang carry as ``k_scale`` / ``v_scale``, and what
    FP8 checkpoints publish in their quantization config. Cheapest to apply:
    the kernel loads one fp32 and broadcasts it, costing nothing measurable
    against the memory traffic of the cache itself.
    """

    PER_HEAD = "per_head"
    """One scale per (layer, plane, head).

    Declared as a typed axis so that adding it later is a change of value, not
    a change of shape in every signature. Currently rejected at construction --
    no kernel in Ayaka consumes a per-head scale vector yet, and accepting the
    config while ignoring the extra scales would be worse than refusing.
    """

    @property
    def is_device_resident(self) -> bool:
        """Whether the scales need a device tensor.

        A per-tensor scale is one Python float applied as a scalar multiplier,
        so it costs no device memory and forces no device read on the write
        path. A per-head scheme would need a vector per (layer, plane) and
        would therefore show up in the capacity budget -- which is why this is
        a property on the scheme rather than an assumption baked into the
        planner.
        """
        return self is QuantizationScheme.PER_HEAD

@dataclass(frozen=True, slots=True)
class KVQuantization:
    """Quantization policy attached to a storage spec.

    Attributes:
        scheme: Scale granularity.
        require_calibration: When true (the default for any real scheme),
            :meth:`writing <ayaka.kvcache.storage.backend.PagedKVStorage.write>`
            into a plane whose scale has not been set raises instead of
            silently using 1.0. Reference tests that only check round-tripping
            of small values can turn this off explicitly.
        storage_dtype_max: Largest finite magnitude of the storage dtype,
            resolved at construction so the hot path never re-derives it.

    Construct through :meth:`none` or :meth:`per_tensor` rather than calling
    the initializer positionally.
    """

    scheme: QuantizationScheme = QuantizationScheme.NONE
    require_calibration: bool = True
    storage_dtype_max: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, QuantizationScheme):
            raise TypeError(
                f"scheme must be a QuantizationScheme, got {type(self.scheme).__name__}")
        if self.scheme is QuantizationScheme.PER_HEAD:
            raise NotImplementedError(
                "per-head KV scales are declared but not implemented: no Ayaka kernel "
                "consumes a per-head scale vector yet"
            )
        if self.storage_dtype_max < 0.0:
            raise ValueError("storage_dtype_max must not be negative")
        if self.scheme is not QuantizationScheme.NONE and self.storage_dtype_max <= 0.0:
            raise ValueError(
                "a quantized scheme needs the storage dtype's max magnitude; "
                "build it with KVQuantization.for_dtype()"
            )


    @classmethod
    def none(cls) -> KVQuantization:
        """Policy for an unquantized cache."""
        return cls(scheme=QuantizationScheme.NONE, require_calibration=False, storage_dtype_max=0.0)

    @classmethod
    def per_tensor(cls, dtype: str, *, require_calibration: bool = True) -> KVQuantization:
        """Policy with one scalar scale per (layer, plane).

        Args:
            dtype: The storage dtype the scale will be applied against.
            require_calibration: Refuse writes until a scale is set.
        """
        return cls(
            scheme=QuantizationScheme.PER_TENSOR,
            require_calibration=require_calibration,
            storage_dtype_max=storage_dtype_max(dtype),
        )

    @classmethod
    def for_dtype(cls, dtype: str, *, require_calibration: bool = True) -> KVQuantization:
        """Return the default policy implied by a storage dtype.

        FP8 gets :meth:`per_tensor`; everything else gets :meth:`none`. This is
        what a spec uses when the caller does not pass a policy explicitly, so
        the common path cannot end up with FP8 and no scale.
        """
        normalized = normalize_storage_dtype(dtype)
        if is_fp8_storage_dtype(normalized):
            return cls.per_tensor(normalized, require_calibration=require_calibration)
        return cls.none()

    @property
    def is_quantized(self) -> bool:
        """Whether stored values must be scaled on write and unscaled on read."""
        return self.scheme is not QuantizationScheme.NONE

    def scale_bytes(self, *, num_layers: int, num_planes: int, num_heads: int = 0) -> int:
        """**Device** bytes the scales occupy for a whole storage instance.

        Zero for :attr:`QuantizationScheme.PER_TENSOR`, because those scales are host
        floats -- see :attr:`QuantizationScheme.is_device_resident`. Reporting a
        non-zero figure for them would make the planner reserve memory nothing
        ever writes to, and make ``materialized_bytes`` disagree with the
        tensors it claims to measure.
        """
        if not self.scheme.is_device_resident:
            return 0
        return num_layers * num_planes * max(num_heads, 1) * SCALE_DTYPE_BYTES

    def validate_for_dtype(self, dtype: str) -> None:
        """Assert that this policy is coherent with a storage dtype.

        The invariant is two-sided on purpose:

        * FP8 **must** carry a scheme -- otherwise values above 448.0 (e4m3fn)
          or 57344.0 (e5m2) saturate without any diagnostic.
        * A non-FP8 dtype **must not** carry one -- a scale on fp16 storage is
          either dead weight or, worse, applied by one code path and not
          another, which reads as a numerics bug in the model.

        Raises:
            ValueError: when the policy and dtype disagree.
        """
        normalized = normalize_storage_dtype(dtype)
        quantizable = is_fp8_storage_dtype(normalized)
        if quantizable and not self.is_quantized:
            raise ValueError(
                f"{normalized} storage requires a quantization scheme: values beyond "
                f"±{storage_dtype_max(normalized)} saturate silently. "
                "Use KVQuantization.per_tensor()."
            )
        if not quantizable and self.is_quantized:
            raise ValueError(
                f"{normalized} storage must not carry a quantization scheme "
                f"(got {self.scheme}); scales on an unquantized cache are dead weight "
                "at best and an inconsistently applied factor at worst"
            )
        if self.is_quantized and self.storage_dtype_max != storage_dtype_max(normalized):
            raise ValueError(
                "quantization policy was built for a different dtype "
                f"(max {self.storage_dtype_max} vs "
                f"{storage_dtype_max(normalized)} for {normalized})"
            )

    @staticmethod
    def explain_static_requirement() -> str:
        """Why scales are calibrated up front rather than derived per write.

        Exposed as a method so the reasoning travels with the code and can be
        quoted in an error message instead of living only in a commit body.
        """
        return (
            "KV pages are append-only across many decode steps. A scale chosen "
            "from the current write applies retroactively to every token already "
            "stored under the previous scale, so re-deriving it per write corrupts "
            "history. Calibrate once, then treat the scale as immutable for the "
            "lifetime of the data."
        )
