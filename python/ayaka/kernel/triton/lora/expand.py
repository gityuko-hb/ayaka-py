"""LoRA expand kernels: ``output[token, offset:] += z @ B[slot].T``.

The mirrored pair of :mod:`ayaka.kernel.triton.lora.shrink`:

* :func:`bgmv_expand` -- one program per ``(row, output tile)`` with the slot
  read from the per-token table.
* :func:`sgmv_expand` -- one program per ``(segment, row block, output tile)``;
  the grouped latent is scattered back to the token rows named by the stable
  permutation. Each output element has exactly one writer.

Both accumulate in FP32, add to the existing projection output after a single
rounding, and never load B for a base row or a zero-rank slot.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.caps import Cap
from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton.lora._common import (
    BGMV_BLOCK_N,
    EXPAND_BLOCK_N,
    SGMV_BLOCK_M,
    cdiv,
    rank_block,
    validate_expand,
    validate_grouped_expand,
)
from ayaka.kernel.triton.reference.lora import bgmv_expand_ref, sgmv_expand_ref

__all__ = ["bgmv_expand", "sgmv_expand"]


@triton.jit
def _bgmv_expand_kernel(
    low_ptr,
    b_ptr,
    rank_ptr,
    rows_ptr,
    out_ptr,
    N,
    offset,
    stride_zm,
    stride_bs,
    stride_bn,
    stride_om,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    slot = tl.load(rows_ptr + row)
    rank = tl.load(rank_ptr + slot)
    offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_r = tl.arange(0, BLOCK_R)
    r_mask = offs_r < rank
    z = tl.load(low_ptr + row * stride_zm + offs_r, mask=r_mask, other=0.0).to(tl.float32)
    b = tl.load(
        b_ptr + slot * stride_bs + offs_n[:, None] * stride_bn + offs_r[None, :],
        mask=n_mask[:, None] & r_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(b * z[None, :], axis=1)
    out_ptrs = out_ptr + row * stride_om + offset + offs_n
    current = tl.load(out_ptrs, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(out_ptrs, (current + acc).to(out_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _sgmv_expand_kernel(
    low_ptr,
    b_ptr,
    rank_ptr,
    counts_ptr,
    offsets_ptr,
    perm_ptr,
    out_ptr,
    N,
    offset,
    stride_zm,
    stride_bs,
    stride_bn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seg = tl.program_id(0)
    block_m = tl.program_id(1)
    block_n = tl.program_id(2)
    count = tl.load(counts_ptr + seg)
    start = tl.load(offsets_ptr + seg)
    row_start = block_m * BLOCK_M
    if row_start < count:
        rank = tl.load(rank_ptr + seg)
        offs_m = row_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < count
        token = tl.load(perm_ptr + start + offs_m, mask=m_mask, other=0)
        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < rank
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        z = tl.load(
            low_ptr + (start + offs_m)[:, None] * stride_zm + offs_r[None, :],
            mask=m_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + seg * stride_bs + offs_n[:, None] * stride_bn + offs_r[None, :],
            mask=n_mask[:, None] & r_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.dot(z, tl.trans(b), input_precision="ieee")
        out_ptrs = out_ptr + token[:, None] * stride_om + offset + offs_n[None, :]
        mask = m_mask[:, None] & n_mask[None, :]
        current = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptrs, (current + acc).to(out_ptr.dtype.element_ty), mask=mask)


@custom_op(
    name="lora_bgmv_expand",
    mutates_args=["output"],
    reference=bgmv_expand_ref,
    caps=Cap.CUDAGRAPH_SAFE,
)
def bgmv_expand(
    low: torch.Tensor,
    b: torch.Tensor,
    ranks: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> None:
    """Per-row expand into ``output[:, offset:]``; base/zero-rank rows are no-ops."""
    tokens, _, rank_capacity, width = validate_expand(low, b, ranks, rows, output, offset)
    grid = (tokens, cdiv(width, BGMV_BLOCK_N))
    cast(Any, _bgmv_expand_kernel)[grid](
        low,
        b,
        ranks,
        rows,
        output,
        width,
        offset,
        low.stride(0),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        BLOCK_R=rank_block(rank_capacity),
        BLOCK_N=BGMV_BLOCK_N,
        num_warps=4,
    )


@custom_op(
    name="lora_sgmv_expand",
    mutates_args=["output"],
    reference=sgmv_expand_ref,
    caps=Cap.CUDAGRAPH_SAFE,
)
def sgmv_expand(
    low: torch.Tensor,
    b: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    output: torch.Tensor,
    offset: int,
) -> None:
    """Grouped expand; each token row is written by exactly one segment program."""
    tokens, slots, rank_capacity, width = validate_grouped_expand(
        low, b, ranks, counts, offsets, permutation, output, offset
    )
    grid = (slots, cdiv(tokens, SGMV_BLOCK_M), cdiv(width, EXPAND_BLOCK_N))
    cast(Any, _sgmv_expand_kernel)[grid](
        low,
        b,
        ranks,
        counts,
        offsets,
        permutation,
        output,
        width,
        offset,
        low.stride(0),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        BLOCK_M=SGMV_BLOCK_M,
        BLOCK_R=rank_block(rank_capacity),
        BLOCK_N=EXPAND_BLOCK_N,
        num_warps=4,
    )
