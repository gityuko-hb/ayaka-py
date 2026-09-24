"""Host-side contracts shared by the sampling kernel modules.

``topk_topp``, ``fused`` and ``gumbel`` all accept a ``[B, V]`` probability or
logit batch plus per-row parameter tensors, and all register the same
``(ids, valid)`` meta kernel. Those checks live here once; each module keeps
its own kernels, references and launch geometry.
"""

from __future__ import annotations

import torch


def validate_probs(probs: torch.Tensor, name: str = "probs") -> tuple[int, int]:
    """Validate a ``[B, V]`` CUDA float batch.

    Args:
        probs: Candidate probability/logit batch.
        name: Argument name used in error messages.

    Returns:
        Tuple ``(batch, vocab)`` of Python ints.

    Raises:
        TypeError: If ``probs`` is not a tensor or not floating-point.
        ValueError: If ``probs`` is not CUDA or not 2-D.
    """
    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not probs.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not probs.is_floating_point():
        raise TypeError(f"{name} must be a float dtype; got {probs.dtype}")
    if probs.dim() != 2:
        raise ValueError(f"{name} must have shape [B, V]; got {tuple(probs.shape)}")
    return probs.size(0), probs.size(1)


def validate_param_tensor(
    tensor: torch.Tensor, name: str, batch: int, device: torch.device
) -> None:
    """Validate a ``[B]`` per-row parameter tensor on the batch device.

    Args:
        tensor: Candidate per-row parameter (thresholds, seeds, offsets).
        name: Argument name used in error messages.
        batch: Expected leading dimension.
        device: Expected device (the batch device).

    Raises:
        TypeError: If ``tensor`` is not a tensor.
        ValueError: If the shape is not ``[B]`` or the device mismatches.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 1 or tensor.size(0) != batch:
        raise ValueError(f"{name} must have shape [B] matching the batch")
    if tensor.device != device:
        raise ValueError(f"{name} and probs must be on the same device")


def validate_opt_seed_offset(
    arr: torch.Tensor | None, name: str, batch: int, device: torch.device
) -> None:
    """Validate an optional ``[B]`` int64 seed/offset tensor.

    Args:
        arr: Seed/offset tensor, or ``None`` for the scalar fallback.
        name: Argument name used in error messages.
        batch: Expected leading dimension.
        device: Expected device (the batch device).

    Raises:
        TypeError: If ``arr`` is a tensor with dtype other than int64.
        ValueError: If the shape is not ``[B]`` or the device mismatches.
    """
    if arr is None:
        return
    validate_param_tensor(arr, name, batch, device)
    if arr.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64; got {arr.dtype}")


def resolve_seed_offset(
    probs: torch.Tensor,
    arr: torch.Tensor | None,
    val: int,
) -> torch.Tensor:
    """Materialize a ``[B]`` int64 seed/offset stream.

    Args:
        probs: ``[B, V]`` batch providing the batch size and device.
        arr: Per-row tensor stream, or ``None`` to broadcast ``val``.
        val: Scalar fallback used when ``arr`` is ``None``.

    Returns:
        Int64 ``[B]`` tensor on the batch device.
    """
    if arr is not None:
        return arr
    return torch.full((probs.size(0),), val, dtype=torch.int64, device=probs.device)


def sampling_fake(
    probs: torch.Tensor, *args: object, **kwargs: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta kernel shared by all sampling ops: fresh ``(ids, valid)`` buffers.

    Args:
        probs: ``[B, V]`` batch providing the batch size and device.
        *args: Ignored positional launcher arguments.
        **kwargs: Ignored launcher keyword arguments.

    Returns:
        Tuple ``(token_ids, valid)`` with uninitialized int32 ``[B]`` and
        bool ``[B]`` tensors, for shape inference under ``torch.compile``.
    """
    batch = probs.shape[0]
    return (
        torch.empty(batch, dtype=torch.int32, device=probs.device),
        torch.empty(batch, dtype=torch.bool, device=probs.device),
    )


def prepare_sampling_outputs(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate fresh ``(output, valid)`` buffers for a probs batch.

    Args:
        probs: ``[batch, vocab]`` tensor providing device and batch size.

    Returns:
        Tuple ``(output, valid)`` with int32 ``[batch]`` indices and bool
        ``[batch]`` flags on the same device as ``probs``.
    """
    batch_size = probs.shape[0]
    output = torch.empty((batch_size,), device=probs.device, dtype=torch.int32)
    valid = torch.empty((batch_size,), device=probs.device, dtype=torch.bool)
    return output, valid


def normalize_threshold(
    value: torch.Tensor | float, name: str, batch: int, device: torch.device
) -> torch.Tensor:
    """Broadcast a scalar threshold to ``[B]`` float32.

    Args:
        value: Per-row ``[B]`` tensor (validated, cast to float32) or a
            Python scalar broadcast to the batch.
        name: Argument name used in error messages.
        batch: Expected leading dimension.
        device: Expected device.

    Returns:
        Float32 ``[B]`` tensor on ``device``.

    Raises:
        TypeError: If a tensor ``value`` fails validation.
        ValueError: If a tensor ``value`` has the wrong shape/device.
    """
    if isinstance(value, torch.Tensor):
        validate_param_tensor(value, name, batch, device)
        return value.to(torch.float32)
    return torch.full((batch,), float(value), dtype=torch.float32, device=device)
