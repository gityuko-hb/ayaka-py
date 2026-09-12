"""Setup and tensor contracts shared by the dense inference layers."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import torch

from ayaka.utils.import_utils import require_module

type LayerBackend = Literal["triton", "torch"]

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def load_kernel(backend: LayerBackend, module: str, name: str) -> Callable[..., Any] | None:
    """Resolve public handles during construction, without automatic fallback."""
    if backend == "torch":
        return None
    if backend != "triton":
        raise ValueError("backend must be 'triton' or 'torch'")
    kernels = require_module(
        f"ayaka.kernel.triton.{module}",
        capability=f"layers.{module}.triton",
        remedy="install the cuda extra or explicitly use backend='torch'",
    )
    return getattr(kernels, name)


def positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def check_dtype(dtype: torch.dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError("dtype must be float16, bfloat16, or float32")


def check_input(x: torch.Tensor, name: str, backend: LayerBackend) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    check_dtype(x.dtype)
    if backend == "triton" and x.device.type != "cuda":
        raise ValueError(f"{name} must be CUDA for backend='triton'")
    if backend == "torch" and x.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} must be on CPU or CUDA")


def check_rows(x: torch.Tensor, name: str, width: int, backend: LayerBackend) -> None:
    check_input(x, name, backend)
    if x.ndim < 2 or x.shape[-1] != width:
        raise ValueError(f"{name} must have shape [..., {width}] with at least two dimensions")
    if x.stride(-1) != 1 or (x.ndim > 2 and not x.is_contiguous()):
        raise ValueError(f"{name} must have contiguous rows; rank > 2 must be contiguous")
    if x.ndim == 2 and x.shape[0] > 1 and x.stride(0) < width:
        raise ValueError(f"{name} rows must not overlap")


def check_out(out: torch.Tensor | None, x: torch.Tensor, shape: tuple[int, ...]) -> None:
    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out.shape != shape or out.device != x.device or out.dtype != x.dtype:
        raise ValueError("out must match the output shape and input device/dtype")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")


def write_output(result: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return result
    out.copy_(result)
    return out
