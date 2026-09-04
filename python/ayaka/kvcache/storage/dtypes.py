from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ayaka.types import DType
from ayaka.utils.torch_utils import dtype_name


@dataclass(frozen=True, slots=True)
class _StorageDTypeInfo:
    torch_name: str
    max_finite: float


#: Information specific to KV storage that cannot be derived from :class:`DType`.
#:
#: Names deliberately match PyTorch attributes. ``max_finite`` lets quantization clamp
#: values before casting; element size and floating-point classification come directly
#: from :class:`DType` and are not repeated here.
_STORAGE_INFO: Final[dict[DType, _StorageDTypeInfo]] = {
    DType.FP32: _StorageDTypeInfo("float32", 3.4028234663852886e38),
    DType.FP16: _StorageDTypeInfo("float16", 65504.0),
    DType.BF16: _StorageDTypeInfo("bfloat16", 3.3895313892515355e38),
    DType.FP8_E4M3: _StorageDTypeInfo("float8_e4m3fn", 448.0),
    DType.FP8_E5M2: _StorageDTypeInfo("float8_e5m2", 57344.0),
    DType.INT8: _StorageDTypeInfo("int8", 127.0),
}

#: Derived once so forward and reverse conversion cannot drift apart.
_DTYPE_BY_NAME: Final[dict[str, DType]] = {
    info.torch_name: dtype for dtype, info in _STORAGE_INFO.items()
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
        ValueError: If ``dtype`` has no KV storage representation. Sub-byte weight formats
            are intentionally unsupported rather than silently promoted to int8.
    """
    info = _STORAGE_INFO.get(dtype)
    if info is None:
        raise ValueError(f"{dtype.label} has no KV storage representation")
    return info.torch_name


def from_storage_dtype(name: str) -> DType:
    """Translate a supported storage dtype name into the canonical enum."""
    return _DTYPE_BY_NAME[normalize_storage_dtype(name)]


def storage_dtype_bytes(dtype: str) -> int:
    """Return element size in bytes, derived from :attr:`DType.itemsize`."""
    return int(from_storage_dtype(dtype).itemsize)


def storage_dtype_max(dtype: str) -> float:
    """Return the largest finite magnitude representable by ``dtype``."""
    return _STORAGE_INFO[from_storage_dtype(dtype)].max_finite


def is_fp8_storage_dtype(dtype: str) -> bool:
    """Return whether ``dtype`` is an FP8 storage format."""
    return from_storage_dtype(dtype) in (DType.FP8_E4M3, DType.FP8_E5M2)


def is_float_storage_dtype(dtype: str) -> bool:
    """Return whether ``dtype`` holds floating-point values."""
    return from_storage_dtype(dtype).is_floating_point
