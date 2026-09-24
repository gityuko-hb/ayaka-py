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

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton._host import fake_tensor
from ayaka.kernel.triton.reference.sampling import row_logsumexp_ref
from ayaka.kernel.triton.sampling._common import validate_param_tensor, validate_probs

_HAS_TRITON_FUSED = True

# Split-count policy for the small-batch/large-vocab path.
#
# ``_MIN_SPLITS`` preserves the historical 16-way split, which is the count
# that was tuned and measured on 16-SM parts. A 16-SM device therefore keeps
# exactly the old split count for every shape in the trigger range. Devices
# with more SMs scale the count up to ``_MAX_SPLITS``, bounded so that no
# chunk shrinks below ``_MIN_SPLIT_CHUNK`` elements (per-program loop and
# partial-store overhead) and so that large row counts do not queue waves.
_MIN_SPLITS = 16
_MAX_SPLITS = 128
_MIN_SPLIT_CHUNK = 1000


def _floor_pow2(value: int) -> int:
    """Largest power of two <= ``value`` (``value >= 1``)."""
    return 1 << max(0, int(value).bit_length() - 1)


def _split_count_from_sm(vocab: int, rows: int, sm_count: int) -> int:
    """Pure split-count policy for a given SM count (unit-testable on CPU).

    ``max_by_chunk`` is the largest power of two whose chunk is still at least
    ``_MIN_SPLIT_CHUNK`` elements, so the documented bound holds exactly
    instead of overshooting by one power-of-two step.
    """
    sm_count = max(1, int(sm_count) or _MIN_SPLITS)

    max_by_chunk = _floor_pow2(max(1, vocab // _MIN_SPLIT_CHUNK))
    max_by_rows = max(_MIN_SPLITS, triton.next_power_of_2(max(1, sm_count // max(rows, 1))))

    splits = triton.next_power_of_2(max(_MIN_SPLITS, sm_count))
    return max(1, min(splits, _MAX_SPLITS, max_by_chunk, max_by_rows))


def _row_logsumexp_num_splits(vocab: int, rows: int, device: torch.device) -> int:
    """Choose a power-of-two split count for the small-batch path.

    The split path exists to parallelize a wide vocabulary scan across SMs
    when the batch is too small to fill the device. The count scales with
    ``multi_processor_count`` (capped at ``_MAX_SPLITS``) and shrinks when
    the resulting chunk would fall below ``_MIN_SPLIT_CHUNK`` elements or
    when the row count already covers the device.
    """
    props = torch.cuda.get_device_properties(device)
    sm_count = int(getattr(props, "multi_processor_count", 0) or _MIN_SPLITS)
    return _split_count_from_sm(vocab, rows, sm_count)


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


@triton.jit
def _row_logsumexp_split_kernel(
    logits_ptr,
    gate_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    vocab_size,
    stride_row,
    chunk_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split_id = tl.program_id(1).to(tl.int64)
    num_splits = tl.num_programs(1)

    gate = tl.load(gate_ptr + row) > 0
    if not gate:
        tl.store(partial_max_ptr + row * num_splits + split_id, float("-inf"))
        tl.store(partial_sum_ptr + row * num_splits + split_id, 0.0)
        return

    split_start = split_id * chunk_size
    split_end = tl.minimum(split_start + chunk_size, vocab_size)

    m = float("-inf")
    s = 0.0
    for start in range(split_start, split_end, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inb = cols < split_end
        x = tl.load(
            logits_ptr + row * stride_row + cols,
            mask=inb,
            other=float("-inf"),
        ).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    tl.store(partial_max_ptr + row * num_splits + split_id, m)
    tl.store(partial_sum_ptr + row * num_splits + split_id, s)


@triton.jit
def _row_logsumexp_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    out_ptr,
    gate_ptr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    gate = tl.load(gate_ptr + row) > 0
    if not gate:
        tl.store(out_ptr + row, 0.0)
        return

    lanes = tl.arange(0, BLOCK)
    mask = lanes < num_splits
    m_vals = tl.load(partial_max_ptr + row * num_splits + lanes, mask=mask, other=float("-inf"))
    s_vals = tl.load(partial_sum_ptr + row * num_splits + lanes, mask=mask, other=0.0)

    m_global = tl.max(m_vals, axis=0)
    scaled_sums = s_vals * tl.exp(m_vals - m_global)
    s_global = tl.sum(tl.where(mask, scaled_sums, 0.0), axis=0)

    lse = m_global + tl.log(tl.maximum(s_global, 1e-30))
    tl.store(out_ptr + row, lse)


def _row_logsumexp_fake(logits: torch.Tensor, row_gate: torch.Tensor | None) -> torch.Tensor:
    """Meta kernel: fresh fp32 ``[n]`` without touching memory."""
    return fake_tensor(logits, dtype=torch.float32)


@custom_op(
    namespace="ayaka",
    reference=row_logsumexp_ref,
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
    validate_probs(logits, "logits")
    if row_gate is not None:
        validate_param_tensor(row_gate, "row_gate", logits.size(0), logits.device)
    n, v = logits.shape
    gate = (
        row_gate
        if row_gate is not None
        else torch.ones(n, dtype=torch.float32, device=logits.device)
    )
    if gate.dtype == torch.bool:
        gate = gate.to(torch.float32)
    out = torch.empty(n, dtype=torch.float32, device=logits.device)

    # When batch is small (e.g. n < 8) and vocab is large, split the row scan
    # across SMs; the count follows the device's SM count, not a fixed 16.
    if n < 8 and v >= 16384:
        num_splits = _row_logsumexp_num_splits(v, n, logits.device)
        chunk_size = triton.cdiv(v, num_splits)
        partial_max = torch.empty((n, num_splits), dtype=torch.float32, device=logits.device)
        partial_sum = torch.empty((n, num_splits), dtype=torch.float32, device=logits.device)
        block = min(4096, triton.next_power_of_2(chunk_size))
        with torch.cuda.device(logits.device):
            cast(
                Any,
                _row_logsumexp_split_kernel[(n, num_splits)](
                    logits,
                    gate,
                    partial_max,
                    partial_sum,
                    v,
                    logits.stride(0),
                    chunk_size,
                    BLOCK=block,  # type: ignore[reportArgumentType]
                ),
            )
            cast(
                Any,
                _row_logsumexp_reduce_kernel[(n,)](
                    partial_max,
                    partial_sum,
                    out,
                    gate,
                    num_splits=num_splits,  # type: ignore[reportArgumentType]
                    BLOCK=num_splits,  # type: ignore[reportArgumentType]
                ),
            )
    else:
        block = triton.next_power_of_2(min(v, 4096))
        with torch.cuda.device(logits.device):
            cast(
                Any,
                _row_logsumexp_kernel[(n,)](logits, gate, out, v, logits.stride(0), BLOCK=block),  # type: ignore[reportArgumentType]
            )
    return out
