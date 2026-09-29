"""Plain-torch oracles for the LoRA shrink/expand kernels.

Each function mirrors the numerics of one public op: FP32 accumulation, a
single rounding to the workspace/output dtype, and no write to a row whose
actual rank is zero. They are passed as ``reference=`` to
:func:`ayaka.kernel.ops.custom_op`, so ``AYAKA_FORCE_REFERENCE_OPS`` and
``verify_against_reference`` exercise them on any device.
"""

from __future__ import annotations

import torch

__all__ = [
    "bgmv_expand_ref",
    "bgmv_shrink_ref",
    "sgmv_expand_ref",
    "sgmv_shrink_ref",
]


def bgmv_shrink_ref(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    rows: torch.Tensor,
    low: torch.Tensor,
) -> None:
    """``low[row, :rank] = x[row] @ A[slot, :rank].T`` in FP32."""
    for row in range(x.shape[0]):
        slot = int(rows[row].item())
        rank = int(ranks[slot].item())
        if rank <= 0:
            continue
        low[row, :rank] = (x[row].float() @ a[slot, :rank].float().T).to(low.dtype)


def bgmv_expand_ref(
    low: torch.Tensor,
    b: torch.Tensor,
    ranks: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> None:
    """``output[row, offset + :] += low[row, :rank] @ B[slot, :, :rank].T``."""
    width = b.shape[1]
    for row in range(rows.numel()):
        slot = int(rows[row].item())
        rank = int(ranks[slot].item())
        if rank <= 0:
            continue
        delta = low[row, :rank].float() @ b[slot, :, :rank].float().T
        target = output[row, offset : offset + width]
        target.copy_((target.float() + delta).to(output.dtype))


def sgmv_shrink_ref(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    low: torch.Tensor,
) -> None:
    """Grouped shrink: ``low[start + i] = x[token] @ A[seg, :rank].T``."""
    for seg in range(a.shape[0]):
        count = int(counts[seg].item())
        rank = int(ranks[seg].item())
        if count <= 0 or rank <= 0:
            continue
        start = int(offsets[seg].item())
        for i in range(count):
            token = int(permutation[start + i].item())
            low[start + i, :rank] = (x[token].float() @ a[seg, :rank].float().T).to(low.dtype)


def sgmv_expand_ref(
    low: torch.Tensor,
    b: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> None:
    """Grouped expand: scatter each segment's delta back to its token rows."""
    width = b.shape[1]
    for seg in range(b.shape[0]):
        count = int(counts[seg].item())
        rank = int(ranks[seg].item())
        if count <= 0 or rank <= 0:
            continue
        start = int(offsets[seg].item())
        for i in range(count):
            token = int(permutation[start + i].item())
            delta = low[start + i, :rank].float() @ b[seg, :, :rank].float().T
            target = output[token, offset : offset + width]
            target.copy_((target.float() + delta).to(output.dtype))
