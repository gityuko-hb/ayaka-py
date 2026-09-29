"""Host-side contracts shared by the LoRA shrink/expand kernels.

The BGMV (decode) and SGMV (prefill) launchers accept the same operand
contract, so the shape/dtype/stride checks and the fixed block sizes live here
once. This module imports the Torch-only helpers from :mod:`._host`; kernel
modules import it, never the other way around. Every check is a host-side
Python predicate: no device readback happens on the forward path.
"""

from __future__ import annotations

import torch

from ayaka.kernel.triton._host import (
    dtype_list,
    require_contiguous,
    require_cuda,
    require_dtype,
    require_last_dim_stride1,
    require_ndim,
    require_same_device,
    require_tensor,
)
from ayaka.utils.math_utils import div_ceil, next_power_of_2

__all__ = [
    "BGMV_BLOCK_N",
    "EXPAND_BLOCK_N",
    "FLOAT_DTYPES",
    "SGMV_BLOCK_M",
    "SHRINK_BLOCK_K",
    "cdiv",
    "rank_block",
    "validate_bgmv",
    "validate_expand",
    "validate_grouped_expand",
    "validate_grouped_shrink",
    "validate_shrink",
    "validate_sgmv",
]

#: Floating dtypes accepted by every LoRA kernel.
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

#: K tile of both shrink kernels; the SGMV ``tl.dot`` needs K >= 16.
SHRINK_BLOCK_K = 128

#: Output-column tile of the SGMV expand kernel.
EXPAND_BLOCK_N = 64

#: Output-column tile of the BGMV expand kernel.
BGMV_BLOCK_N = 128

#: Grouped-row tile of the SGMV kernels; ``tl.dot`` needs M >= 16.
SGMV_BLOCK_M = 16


def cdiv(a: int, b: int) -> int:
    """Ceiling division, mirroring the launch-grid arithmetic."""
    return div_ceil(a, b)


def rank_block(rank_capacity: int) -> int:
    """Power-of-two rank tile covering ``rank_capacity`` (1..64).

    The tile is part of the captured launch; actual ranks below it are masked
    on device so replay stays correct while the bank contents change.
    """
    if not 1 <= rank_capacity <= 64:
        raise ValueError("rank capacity must be within 1..64")
    return max(16, next_power_of_2(int(rank_capacity)))


def _require_float(tensor: object, name: str, *, ndim: int) -> torch.Tensor:
    require_tensor(tensor, name)
    assert isinstance(tensor, torch.Tensor)
    require_cuda(tensor, name)
    require_dtype(tensor, name, FLOAT_DTYPES)
    require_ndim(tensor, name, ndim)
    return tensor


def _require_index(tensor: object, name: str, *, ndim: int = 1) -> torch.Tensor:
    require_tensor(tensor, name)
    assert isinstance(tensor, torch.Tensor)
    require_cuda(tensor, name)
    require_dtype(tensor, name, (torch.int64, torch.int32))
    require_ndim(tensor, name, ndim)
    require_contiguous(tensor, name)
    return tensor


def _require_offset(offset: object) -> int:
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise TypeError("output slice offset must be an int")
    return offset


def _shrink_core(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    low: torch.Tensor,
    tokens: int,
) -> tuple[int, int, int, int]:
    """Shared shrink operand checks for ``tokens`` routed rows."""
    _require_float(x, "x", ndim=2)
    require_last_dim_stride1(x, "x")
    _require_float(a, "a", ndim=3)
    require_contiguous(a, "a")
    _require_index(ranks, "ranks")
    _require_float(low, "low", ndim=2)
    require_last_dim_stride1(low, "low")
    if tokens < 1:
        raise ValueError("LoRA requires at least one token row")
    slots = int(a.shape[0])
    rank_capacity = int(a.shape[1])
    if a.shape[2] != x.shape[1]:
        raise ValueError("x and A disagree on the input width")
    if ranks.numel() != slots:
        raise ValueError("rank bank must have one entry per slot including base")
    if low.shape[0] < tokens or low.shape[1] < rank_capacity:
        raise ValueError("latent workspace is smaller than the requested tile")
    if a.dtype != x.dtype or low.dtype != x.dtype:
        raise TypeError(f"LoRA operands must share one dtype from {dtype_list(FLOAT_DTYPES)}")
    for name, tensor in (("a", a), ("ranks", ranks), ("low", low)):
        require_same_device(tensor, name, x, "x")
    return tokens, slots, rank_capacity, int(a.shape[2])


def _expand_core(
    low: torch.Tensor,
    b: torch.Tensor,
    ranks: torch.Tensor,
    output: torch.Tensor,
    offset: int,
    tokens: int,
) -> tuple[int, int, int]:
    """Shared expand operand checks for ``tokens`` routed rows."""
    _require_float(low, "low", ndim=2)
    require_last_dim_stride1(low, "low")
    _require_float(b, "b", ndim=3)
    require_contiguous(b, "b")
    _require_index(ranks, "ranks")
    _require_float(output, "output", ndim=2)
    require_last_dim_stride1(output, "output")
    offset = _require_offset(offset)
    if tokens < 1:
        raise ValueError("LoRA requires at least one token row")
    slots = int(b.shape[0])
    rank_capacity = int(b.shape[2])
    width = int(b.shape[1])
    if ranks.numel() != slots:
        raise ValueError("rank bank must have one entry per slot including base")
    if low.shape[0] < tokens or low.shape[1] < rank_capacity:
        raise ValueError("latent workspace is smaller than the requested tile")
    if output.shape[0] < tokens:
        raise ValueError("output has fewer rows than the token batch")
    if offset < 0 or offset + width > output.shape[1]:
        raise ValueError("output slice offset is outside the projection width")
    if b.dtype != low.dtype or output.dtype != low.dtype:
        raise TypeError(f"LoRA operands must share one dtype from {dtype_list(FLOAT_DTYPES)}")
    for name, tensor in (("b", b), ("ranks", ranks), ("output", output)):
        require_same_device(tensor, name, low, "low")
    return slots, rank_capacity, width


def validate_shrink(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    rows: torch.Tensor,
    low: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate the BGMV shrink operands; returns ``(tokens, slots, rank, input)``."""
    tokens = int(x.shape[0])
    result = _shrink_core(x, a, ranks, low, tokens)
    _require_index(rows, "rows")
    if rows.numel() != tokens:
        raise ValueError("rows must have one slot id per token")
    require_same_device(rows, "rows", x, "x")
    return result


def validate_expand(
    low: torch.Tensor,
    b: torch.Tensor,
    ranks: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> tuple[int, int, int, int]:
    """Validate the BGMV expand operands; returns ``(tokens, slots, rank, output)``."""
    _require_index(rows, "rows")
    tokens = int(rows.numel())
    slots, rank_capacity, width = _expand_core(low, b, ranks, output, offset, tokens)
    require_same_device(rows, "rows", low, "low")
    return tokens, slots, rank_capacity, width


def validate_bgmv(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    rows: torch.Tensor,
    low: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> tuple[int, int, int, int]:
    """Validate one BGMV shrink+expand pair before any kernel is enqueued."""
    tokens, slots, rank_capacity, _ = validate_shrink(x, a, ranks, rows, low)
    expanded = validate_expand(low, b, ranks, rows, output, offset)
    if expanded[1] != slots or expanded[2] != rank_capacity:
        raise ValueError("A and B banks disagree on slots or rank capacity")
    if expanded[0] != tokens:
        raise ValueError("shrink and expand disagree on the token batch")
    return tokens, slots, rank_capacity, expanded[3]


def _grouped_tables(
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    slots: int,
    tokens: int,
    reference: torch.Tensor,
) -> None:
    _require_index(permutation, "permutation")
    if permutation.numel() < tokens:
        raise ValueError("token permutation is shorter than the token batch")
    _require_index(counts, "counts")
    _require_index(offsets, "offsets")
    if counts.numel() != slots or offsets.numel() != slots:
        raise ValueError("grouped tables must have one entry per slot including base")
    for name, tensor in (
        ("counts", counts),
        ("offsets", offsets),
        ("permutation", permutation),
    ):
        require_same_device(tensor, name, reference, "reference")


def validate_grouped_shrink(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    low: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate the SGMV shrink operands; returns ``(tokens, slots, rank, input)``."""
    tokens = int(x.shape[0])
    result = _shrink_core(x, a, ranks, low, tokens)
    if result[3] < 16:
        raise ValueError("grouped SGMV requires an input width of at least 16")
    _grouped_tables(counts, offsets, permutation, result[1], tokens, x)
    return result


def validate_grouped_expand(
    low: torch.Tensor,
    b: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> tuple[int, int, int, int]:
    """Validate the SGMV expand operands; returns ``(tokens, slots, rank, output)``."""
    tokens = int(low.shape[0])
    slots, rank_capacity, width = _expand_core(low, b, ranks, output, offset, tokens)
    _grouped_tables(counts, offsets, permutation, slots, tokens, low)
    return tokens, slots, rank_capacity, width


def validate_sgmv(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    low: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> tuple[int, int, int, int]:
    """Validate one SGMV shrink+expand pair before any kernel is enqueued.

    Every segment is bounded by its own device-side count, so stale padding
    metadata in the permutation tail is never read; the host only checks the
    table shapes and dtypes.
    """
    tokens, slots, rank_capacity, _ = validate_grouped_shrink(
        x, a, ranks, counts, offsets, permutation, low
    )
    expanded = validate_grouped_expand(low, b, ranks, counts, offsets, permutation, output, offset)
    if expanded[0] != tokens or expanded[1] != slots or expanded[2] != rank_capacity:
        raise ValueError("shrink and expand disagree on the routed batch")
    return tokens, slots, rank_capacity, expanded[3]
