from __future__ import annotations

from typing import Final, cast

from ayaka.types import DType
from ayaka.utils.torch_utils import dtype_name

#: Dtypes the KV storage layer accepts. Deliberately an allowlist: integer and
#: sub-byte formats cannot carry a scale (``KVQuantization.validate_for_dtype``
#: refuses a scheme on any non-FP8 dtype), so every write would cast real
#: activations through a truncating ``.to(int8)`` with no diagnostic. Until a
#: calibrated integer scheme exists, refusing the dtype is the fail-closed answer.
_KV_STORAGE_DTYPES: Final[tuple[DType, ...]] = (
    DType.FP32,
    DType.FP16,
    DType.BF16,
    DType.FP8_E4M3,
    DType.FP8_E5M2,
)

#: Derived once so forward and reverse conversion cannot drift apart. Element
#: size, floating-point classification, and max magnitude all come from the
#: :class:`~ayaka.types.DType` registry; nothing about the type is repeated here.
_DTYPE_BY_NAME: Final[dict[str, DType]] = {
    cast(str, dtype.torch_name): dtype for dtype in _KV_STORAGE_DTYPES
}

#: Public, ordered view of every supported dtype name. Useful in error messages and
#: ``--help`` output; do not mutate.
STORAGE_DTYPES: Final[tuple[str, ...]] = tuple(_DTYPE_BY_NAME)


def normalize_storage_dtype(dtype: str) -> str:
    """Normalize and validate a public storage dtype without importing PyTorch.

    Args:
        dtype: A dtype name such as ``float16``, ``torch.float16``, or ``Float16``.

    Returns:
        The canonical lowercase name.

    Raises:
        TypeError: If ``dtype`` is not a string.
        ValueError: If the normalized name is not a supported storage dtype.
    """
    if not isinstance(dtype, str):
        raise TypeError(f"storage dtype must be a string, got {type(dtype).__name__}")
    normalized = dtype_name(dtype)
    if normalized not in _DTYPE_BY_NAME:
        raise ValueError(
            f"unsupported storage dtype {dtype!r}; supported: {', '.join(STORAGE_DTYPES)}"
        )
    return normalized


def to_storage_dtype(dtype: DType) -> str:
    """Translate a canonical enum into its KV storage name.

    Raises:
        ValueError: If ``dtype`` has no KV storage representation. Sub-byte weight
            formats and integer formats are intentionally unsupported rather than
            silently promoted or stored unscaled.
    """
    if dtype not in _KV_STORAGE_DTYPES:
        raise ValueError(f"{dtype.label} has no KV storage representation")
    return cast(str, dtype.torch_name)


def from_storage_dtype(name: str) -> DType:
    """Translate a supported storage dtype name into the canonical enum."""
    return _DTYPE_BY_NAME[normalize_storage_dtype(name)]


def storage_dtype_bytes(dtype: str) -> int:
    """Return element size in bytes, derived from :attr:`DType.itemsize`."""
    return int(from_storage_dtype(dtype).itemsize)


def storage_dtype_max(dtype: str) -> float:
    """Return the largest finite magnitude representable by ``dtype``."""
    info = from_storage_dtype(dtype)
    if info.max_finite is None:
        raise ValueError(f"{info.label} has no finite magnitude")
    return info.max_finite


def is_fp8_storage_dtype(dtype: str) -> bool:
    """Return whether ``dtype`` is an FP8 storage format."""
    return from_storage_dtype(dtype) in (DType.FP8_E4M3, DType.FP8_E5M2)


def is_float_storage_dtype(dtype: str) -> bool:
    """Return whether ``dtype`` holds floating-point values."""
    return from_storage_dtype(dtype).is_floating_point
