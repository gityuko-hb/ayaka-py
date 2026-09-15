"""Fused row stats — Triton kernel cho row_logsumexp một lượt đọc.

MỤC ĐÍCH (A3): _canonicalize_rep_logsumexp trong ops/penalties.py cần
logsumexp của TOÀN ROW cho các row có rep_penalty > 0. Bản cũ làm hai việc
lãng phí:
  1. torch.nonzero(rep_full > 0) — host sync ngầm để lấy số active rows;
  2. logits.index_select(0, active_rows) — MATERIALIZED COPY [active × V]
     trước khi logsumexp đọc lần nữa ⇒ 2× bandwidth cho rep.

Bản này: MỘT kernel, MỘT lượt đọc/row. Kernel nhận `row_gate` [n_rows] — row
có gate <= 0 bị MASK KHỎI LOAD (không có memory traffic) và cho lse = 0 mà
không cần biết trước bao nhiêu row active, không nonzero, không copy.

Fallback torch giữ nguyên semantics: boolean-mask index (không host sync ở
đường assignment), chỉ dùng khi không có Triton/CUDA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

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
    """MỘT program/row. Online max/sum trong fp32 — mỗi phần tử logits đọc
    đúng MỘT lần. Row có gate <= 0: mọi load bị mask (không traffic) và out
    = 0, khớp semantics cũ (lse = 0 cho row không có rep)."""
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
    # Row inactive: s = 0 ⇒ lse = nan (inf - inf) — where chôn nan, ghi 0.
    tl.store(out_ptr + row, tl.where(gate, lse, 0.0))


def row_logsumexp_gpu(
    logits: torch.Tensor, row_gate: torch.Tensor | None
) -> torch.Tensor:
    """Đường Triton: logsumexp theo row, chỉ đọc row có gate > 0 (hoặc toàn
    bộ khi row_gate=None). logits [n, V] bất kỳ dtype float; trả fp32 [n]."""
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
