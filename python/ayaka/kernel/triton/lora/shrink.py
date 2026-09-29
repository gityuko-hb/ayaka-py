"""LoRA shrink kernels: ``z = x @ A[slot].T`` with FP32 accumulation.

Two algorithms share the same operand contract:

* :func:`bgmv_shrink` -- one program per routed row, slot read from the
  per-token table. Decode and any ungrouped caller use this path.
* :func:`sgmv_shrink` -- one program per ``(segment, row block)``; tokens are
  gathered through the stable permutation and written back in grouped order.
  Prefill uses this path so the launch grid is capacity-fixed while the
  device-side segment counts decide the active rows.

Both store the latent in the workspace dtype after a single FP32 rounding, and
neither loads A for a base row or a slot whose actual rank is zero.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.caps import Cap
from ayaka.kernel.ops import custom_op
from ayaka.kernel.triton.lora._common import (
    SGMV_BLOCK_M,
    SHRINK_BLOCK_K,
    cdiv,
    rank_block,
    validate_grouped_shrink,
    validate_shrink,
)
from ayaka.kernel.triton.reference.lora import bgmv_shrink_ref, sgmv_shrink_ref

__all__ = ["bgmv_shrink", "sgmv_shrink"]


@triton.jit
def _bgmv_shrink_kernel(
    x_ptr,
    a_ptr,
    rank_ptr,
    rows_ptr,
    out_ptr,
    K,
    stride_xm,
    stride_as,
    stride_ar,
    stride_om,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(rows_ptr + row)
    rank = tl.load(rank_ptr + slot)
    offs_r = tl.arange(0, BLOCK_R)
    r_mask = offs_r < rank
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(x_ptr + row * stride_xm + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        a = tl.load(
            a_ptr + slot * stride_as + offs_r[:, None] * stride_ar + offs_k[None, :],
            mask=r_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a * x[None, :], axis=1)
    tl.store(out_ptr + row * stride_om + offs_r, acc.to(out_ptr.dtype.element_ty), mask=r_mask)


@triton.jit
def _sgmv_shrink_kernel(
    x_ptr,
    a_ptr,
    rank_ptr,
    counts_ptr,
    offsets_ptr,
    perm_ptr,
    out_ptr,
    K,
    stride_xm,
    stride_as,
    stride_ar,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    seg = tl.program_id(0)
    block = tl.program_id(1)
    count = tl.load(counts_ptr + seg)
    start = tl.load(offsets_ptr + seg)
    row_start = block * BLOCK_M
    if row_start < count:
        rank = tl.load(rank_ptr + seg)
        offs_m = row_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < count
        token = tl.load(perm_ptr + start + offs_m, mask=m_mask, other=0)
        offs_r = tl.arange(0, BLOCK_R)
        r_mask = offs_r < rank
        acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x = tl.load(
                x_ptr + token[:, None] * stride_xm + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            a = tl.load(
                a_ptr + seg * stride_as + offs_r[:, None] * stride_ar + offs_k[None, :],
                mask=r_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = tl.dot(x, tl.trans(a), acc=acc, input_precision="ieee")
        out_ptrs = out_ptr + (start + offs_m)[:, None] * stride_om + offs_r[None, :]
        tl.store(
            out_ptrs,
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & r_mask[None, :],
        )


@custom_op(
    name="lora_bgmv_shrink",
    mutates_args=["low"],
    reference=bgmv_shrink_ref,
    caps=Cap.CUDAGRAPH_SAFE,
)
def bgmv_shrink(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    rows: torch.Tensor,
    low: torch.Tensor,
) -> None:
    """Per-row shrink into ``low``; base rows and zero ranks write nothing."""
    tokens, _, rank_capacity, input_width = validate_shrink(x, a, ranks, rows, low)
    cast(Any, _bgmv_shrink_kernel)[(tokens,)](
        x,
        a,
        ranks,
        rows,
        low,
        input_width,
        x.stride(0),
        a.stride(0),
        a.stride(1),
        low.stride(0),
        BLOCK_K=SHRINK_BLOCK_K,
        BLOCK_R=rank_block(rank_capacity),
        num_warps=4,
    )


@custom_op(
    name="lora_sgmv_shrink",
    mutates_args=["low"],
    reference=sgmv_shrink_ref,
    caps=Cap.CUDAGRAPH_SAFE,
)
def sgmv_shrink(
    x: torch.Tensor,
    a: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    permutation: torch.Tensor,
    low: torch.Tensor,
) -> None:
    """Grouped shrink; segments and their token rows come from device tables."""
    tokens, slots, rank_capacity, _ = validate_grouped_shrink(
        x, a, ranks, counts, offsets, permutation, low
    )
    grid = (slots, cdiv(tokens, SGMV_BLOCK_M))
    cast(Any, _sgmv_shrink_kernel)[grid](
        x,
        a,
        ranks,
        counts,
        offsets,
        permutation,
        low,
        int(a.shape[2]),
        x.stride(0),
        a.stride(0),
        a.stride(1),
        low.stride(0),
        BLOCK_M=SGMV_BLOCK_M,
        BLOCK_K=SHRINK_BLOCK_K,
        BLOCK_R=rank_block(rank_capacity),
        num_warps=4,
    )
