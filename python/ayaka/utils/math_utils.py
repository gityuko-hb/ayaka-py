from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from ayaka.utils.torch_utils import torch

__all__ = [
    "FP8_E4M3_MAX",
    "FP8_E4M3_MIN",
    "INT8_MAX",
    "INT8_MIN",
    "LOCAL_ABSMAX_EPS",
    "MXFP4_BLOCK_SIZE",
    "NVFP4_BLOCK_SIZE",
    "align_down",
    "align_up",
    "div_ceil",
    "fp4_reference_values",
    "median",
    "next_power_of_2",
    "percentiles",
    "require_cuda",
    "require_cuda_contiguous",
    "require_last_dim_contiguous",
    "validate_output_dtype",
]

#: Largest finite magnitude of ``torch.float8_e4m3fn``. Mirrors
#: ``DType.FP8_E4M3.max_finite`` (``ayaka.types``); pinned by
#: ``tests/test_math_utils.py`` so the two cannot drift.
FP8_E4M3_MAX: Final[float] = 448.0

FP8_E4M3_MIN: Final[float] = -448.0

INT8_MAX: Final[int] = 127

INT8_MIN: Final[int] = -128

#: Floor for a per-tensor local absmax before its reciprocal is taken.
LOCAL_ABSMAX_EPS: Final[float] = 1.0e-10

#: Elements sharing one E8M0 scale in an MXFP4 quantization block.
MXFP4_BLOCK_SIZE: Final[int] = 32

#: Elements sharing one E4M3 scale in an NVFP4 quantization block.
NVFP4_BLOCK_SIZE: Final[int] = 16


def div_ceil(a: int, b: int) -> int:
    """Return ``a / b`` rounded toward positive infinity.

    Args:
        a: Dividend. May be negative; the double-negation form keeps the
            result correct for every sign combination.
        b: Divisor. Must be a positive integer.

    Returns:
        The smallest integer ``q`` with ``q * b >= a``.

    Raises:
        TypeError: if either operand is not an integer.
        ValueError: if ``b`` is not positive.
    """
    if type(a) is not int:
        raise TypeError("value must be an integer")
    if type(b) is not int:
        raise TypeError("divisor must be an integer")
    if b <= 0:
        raise ValueError("divisor must be positive")
    return -(-a // b)


def next_power_of_2(num: int) -> int:
    """Return the smallest power of two that is at least ``num``.

    ``0`` and ``1`` are returned unchanged, matching
    ``triton.next_power_of_2`` so launch-geometry helpers can be shared
    between host and kernel code.

    Raises:
        TypeError: if ``num`` is not an integer.
        ValueError: if ``num`` is negative.
    """
    if type(num) is not int:
        raise TypeError("num must be an integer")
    if num < 0:
        raise ValueError("num must not be negative")
    if num <= 1:
        return num
    return 1 << (num - 1).bit_length()


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``.

    Kept general (``alignment`` need not be a power of two) because it only
    runs in planning code, never per step. The bitmask form
    ``(v + a - 1) & ~(a - 1)`` would be faster but would silently produce
    garbage for a non-power-of-two alignment.

    Raises:
        TypeError: if either argument is not an integer.
        ValueError: for a non-positive alignment or a negative value.
    """
    if type(value) is not int:
        raise TypeError("value must be an integer")
    if type(alignment) is not int:
        raise TypeError("alignment must be an integer")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if value < 0:
        raise ValueError("value must not be negative")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    """Round ``value`` down to the previous multiple of ``alignment``.

    Counterpart of :func:`align_up`; ``0`` stays ``0`` for every valid
    alignment.

    Raises:
        TypeError: if either argument is not an integer.
        ValueError: for a non-positive alignment or a negative value.
    """
    if type(value) is not int:
        raise TypeError("value must be an integer")
    if type(alignment) is not int:
        raise TypeError("alignment must be an integer")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if value < 0:
        raise ValueError("value must not be negative")
    return value - value % alignment


def median(values: Sequence[float]) -> float:
    """Return the arithmetic median of ``values``.

    Raises:
        ValueError: if ``values`` is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """Return min/mean/p50/p90/p95/p99/max for ``values``.

    An empty sequence yields an all-zero mapping so benchmark reports do not
    need a special case.
    """
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def require_cuda(x: Any, name: str) -> None:
    """Require a CUDA tensor, rejecting CPU and non-tensor inputs.

    Raises:
        ValueError: if ``x`` does not live on a CUDA device.
    """
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def require_cuda_contiguous(x: Any, name: str) -> None:
    """Require a CUDA tensor with a dense, row-major memory layout.

    Raises:
        ValueError: if ``x`` is not CUDA or not contiguous.
    """
    require_cuda(x, name)
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def require_last_dim_contiguous(x: Any, name: str) -> None:
    """Require a CUDA tensor whose last dimension has stride 1.

    Kernels that vectorize along the last dimension accept non-contiguous
    outer dimensions but never a strided innermost one.

    Raises:
        ValueError: if ``x`` is not CUDA or its last stride is not 1.
    """
    require_cuda(x, name)
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must have contiguous last dimension")


def validate_output_dtype(dtype: Any) -> None:
    """Require a supported kernel output dtype.

    Raises:
        ValueError: if ``dtype`` is not float16, bfloat16, or float32.
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("out_dtype must be float16, bfloat16, or float32")


def fp4_reference_values(device: Any = None) -> Any:
    """Return the eight representable E2M1 finite magnitudes in code order.

    Used to check FP4 decode/unpack kernels against the codebook.
    """
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
