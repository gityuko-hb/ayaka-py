"""Fused row stats: single-read row-wise log-sum-exp on GPU.

Backs ``_canonicalize_rep_logsumexp`` in ``ayaka.sampling.ops.penalties``,
which needs the full-row logsumexp for rows with ``rep_penalty > 0``. The
previous implementation wasted bandwidth twice: ``torch.nonzero`` caused
an implicit host sync to count active rows, then ``index_select`` built a
materialized ``[active, V]`` copy before logsumexp read it again.

This module uses one kernel with one read per row. It takes a ``row_gate``
of ``[n]``: rows with ``gate <= 0`` are masked out of the loads (no memory
traffic) and yield ``0.0`` with no prior count, sync, or copy. The torch
fallback keeps the same semantics with boolean-mask indexing and runs only
when Triton/CUDA is unavailable.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op

_HAS_TRITON_FUSED = True


@triton.jit
def _row_logsumexp_kernel(
    logits_ptr,
    gate_ptr,
    out_ptr,
    vocab_size,
    stride_row,
    BLOCK: tl.constexpr,
):
    """Online row-wise logsumexp in fp32 with per-row gating.

    Args:
        logits_ptr: Pointer to ``[n, vocab]`` logits of any float dtype.
        gate_ptr: Pointer to ``[n]`` per-row gate; rows with ``<= 0`` are
            skipped and produce ``0.0``.
        out_ptr: Pointer to ``[n]`` fp32 output.
        vocab_size: Number of vocabulary columns to scan.
        stride_row: Row stride (in elements) of ``logits_ptr``.
        BLOCK: Tile width along the vocabulary dimension.

    Note:
        Single pass with online ``max``/``sum`` rescaling, so each logits
        element is read exactly once. Out-of-bounds lanes read as ``-inf``.
        Inactive rows yield ``s == 0`` whose raw ``lse`` is NaN; the final
        ``where`` buries it and stores ``0.0`` to match the legacy
        ``lse == 0`` semantics for rows without a representative.
    """
    row = tl.program_id(0).to(tl.int64)
    gate = tl.load(gate_ptr + row) > 0

    m = float("-inf")
    s = 0.0
    for start in range(0, vocab_size, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inb = cols < vocab_size
        x = tl.load(
            logits_ptr + row * stride_row + cols,
            mask=inb & gate,
            other=float("-inf"),
        ).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    lse = m + tl.log(tl.maximum(s, 1e-30))
    # Row inactive: s = 0 gives lse = nan (inf - inf) — where buries NaN, stores 0.
    tl.store(out_ptr + row, tl.where(gate, lse, 0.0))


def _row_logsumexp_ref(logits: torch.Tensor, row_gate: torch.Tensor | None) -> torch.Tensor:
    """Plain-torch reference with identical semantics (CPU-runnable)."""
    n = logits.size(0)
    if row_gate is None:
        return torch.logsumexp(logits.to(torch.float32), dim=-1)
    gate = row_gate > 0
    lse = torch.zeros(n, dtype=torch.float32, device=logits.device)
    if bool(gate.any()):
        lse[gate] = torch.logsumexp(logits[gate].to(torch.float32), dim=-1)
    return lse


def _row_logsumexp_fake(logits: torch.Tensor, row_gate: torch.Tensor | None) -> torch.Tensor:
    """Meta kernel: fresh fp32 ``[n]`` without touching memory."""
    return torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)


@custom_op(
    namespace="ayaka",
    reference=_row_logsumexp_ref,
    fake_impl=_row_logsumexp_fake,
    dispatch_key="CUDA",
)
def row_logsumexp_gpu(logits: torch.Tensor, row_gate: torch.Tensor | None) -> torch.Tensor:
    """Compute row-wise logsumexp on GPU, skipping gated-off rows.

    Args:
        logits: ``[n, vocab]`` float tensor of any float dtype.
        row_gate: ``[n]`` float or bool gate, or ``None`` for all rows
            active. Rows with values ``<= 0`` produce ``0.0``.

    Returns:
        Fresh fp32 ``[n]`` tensor on the same device as ``logits``.

    Note:
        Pure function of ``logits`` and ``row_gate``: inputs are never
        mutated and the output is freshly allocated. One program per row
        (``grid=(n,)``) with ``BLOCK = next_power_of_2(min(V, 4096))``.
        Reference: ``torch.logsumexp(logits.float(), dim=-1)`` with gated
        rows forced to ``0.0``, suitable for ``verify_against_reference``.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be a float dtype; got {logits.dtype}")
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [n, vocab]; got {tuple(logits.shape)}")
    if row_gate is not None:
        if not isinstance(row_gate, torch.Tensor):
            raise TypeError("row_gate must be a torch.Tensor or None")
        if row_gate.dim() != 1 or row_gate.size(0) != logits.size(0):
            raise ValueError("row_gate must have shape [n] matching logits rows")
        if row_gate.device != logits.device:
            raise ValueError("row_gate and logits must be on the same device")
    n, v = logits.shape
    gate = (
        row_gate
        if row_gate is not None
        else torch.ones(n, dtype=torch.float32, device=logits.device)
    )
    if gate.dtype == torch.bool:
        gate = gate.to(torch.float32)
    out = torch.empty(n, dtype=torch.float32, device=logits.device)
    block = triton.next_power_of_2(min(v, 4096))
    _row_logsumexp_kernel[(n,)](logits, gate, out, v, logits.stride(0), BLOCK=block)
    return out
