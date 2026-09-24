"""Host-side contract helpers shared by the Triton kernel modules.

This module must stay importable without Triton: CPU-only installs and the
reference paths import it. It holds the checks every launcher otherwise
copies -- tensor/device/dtype/rank/shape guards, output allocation and
meta-kernel allocation -- composed from small predicates so a call site can
pick exactly the checks it needs while the error fragments tests match on
stay stable.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def dtype_list(dtypes: Iterable[torch.dtype]) -> str:
    """Human-readable dtype enumeration, e.g. ``"float16, bfloat16, or float32"``."""
    names = [str(dtype).removeprefix("torch.") for dtype in dtypes]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return ", ".join(names[:-1]) + f", or {names[-1]}"


def require_tensor(x: object, name: str) -> None:
    """Require ``x`` to be a ``torch.Tensor``.

    Raises:
        TypeError: if ``x`` is not a tensor.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def require_cuda(x: torch.Tensor, name: str) -> None:
    """Require ``x`` to be a CUDA tensor.

    Raises:
        TypeError: if ``x`` is not a tensor.
        ValueError: if ``x`` does not live on a CUDA device.
    """
    require_tensor(x, name)
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def require_dtype(x: torch.Tensor, name: str, dtypes: tuple[torch.dtype, ...]) -> None:
    """Require ``x.dtype`` to be one of ``dtypes``.

    Raises:
        TypeError: if the dtype is not allowed.
    """
    if x.dtype not in dtypes:
        raise TypeError(f"{name} must have dtype {dtype_list(dtypes)}; got {x.dtype}")


def require_ndim(x: torch.Tensor, name: str, ndim: int | tuple[int, ...]) -> None:
    """Require ``x`` to have one of the allowed ranks.

    Raises:
        ValueError: if the rank is not allowed.
    """
    allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
    if x.dim() not in allowed:
        expected = " or ".join(str(rank) for rank in allowed)
        raise ValueError(f"{name} must have rank {expected}, got {x.dim()}")


def require_shape(x: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
    """Require ``x`` to have exactly ``shape``.

    Raises:
        ValueError: if the shape differs.
    """
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(x.shape)}")


def require_device(x: torch.Tensor, name: str, device: torch.device) -> None:
    """Require ``x`` to live on ``device``.

    Raises:
        ValueError: if the device differs.
    """
    if x.device != device:
        raise ValueError(f"{name} must be on {device}, got {x.device}")


def require_same_device(
    x: torch.Tensor,
    name: str,
    reference: torch.Tensor,
    reference_name: str,
) -> None:
    """Require ``x`` and ``reference`` to share one CUDA device.

    Raises:
        ValueError: if the devices differ.
    """
    if x.device != reference.device:
        raise ValueError(f"{name} and {reference_name} must be on the same CUDA device")


def require_last_dim_stride1(x: torch.Tensor, name: str) -> None:
    """Require the innermost dimension of ``x`` to have stride 1.

    Kernels that vectorize along the last dimension accept non-contiguous
    outer dimensions but never a strided innermost one.

    Raises:
        ValueError: if ``x`` is scalar or its last stride is not 1.
    """
    if x.dim() == 0 or x.stride(-1) != 1:
        raise ValueError(f"{name} must have stride(-1) == 1")


def require_contiguous(x: torch.Tensor, name: str) -> None:
    """Require ``x`` to have a dense, row-major layout.

    Raises:
        ValueError: if ``x`` is not contiguous.
    """
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def prepare_output(
    like: torch.Tensor,
    out: torch.Tensor | None,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    dtypes: tuple[torch.dtype, ...] | None = None,
    name: str = "out",
    like_name: str = "input",
) -> torch.Tensor:
    """Return ``out`` after contract checks, or allocate a fresh tensor.

    Args:
        like: Reference tensor providing the device, default shape and dtype.
        out: Caller-provided output, or ``None`` to allocate.
        shape: Required output shape. Defaults to ``like.shape``.
        dtype: Allocation dtype, and required dtype when ``dtypes`` is None.
        dtypes: Allowed ``out`` dtypes, when the caller accepts several.
        name: Argument name used in error messages.
        like_name: Reference name used in the device error message.

    Returns:
        The validated ``out``, or a fresh ``torch.empty`` tensor.
    """
    resolved_shape = tuple(like.shape) if shape is None else tuple(shape)
    resolved_dtype = like.dtype if dtype is None else dtype
    if out is None:
        return torch.empty(resolved_shape, device=like.device, dtype=resolved_dtype)
    require_tensor(out, name)
    require_same_device(out, name, like, like_name)
    allowed = (resolved_dtype,) if dtypes is None else tuple(dtypes)
    if out.dtype not in allowed:
        raise TypeError(f"{name} must have dtype {dtype_list(allowed)}; got {out.dtype}")
    require_shape(out, name, resolved_shape)
    return out


def fake_tensor(
    like: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Allocate an uninitialized meta tensor mirroring ``like``."""
    resolved_shape = tuple(like.shape) if shape is None else tuple(shape)
    return torch.empty(
        resolved_shape,
        dtype=like.dtype if dtype is None else dtype,
        device=like.device,
    )


def fake_output(
    like: torch.Tensor,
    out: torch.Tensor | None,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Meta-kernel output: the caller's ``out`` when given, else a fresh tensor."""
    if out is not None:
        return out
    return fake_tensor(like, shape=shape, dtype=dtype)
