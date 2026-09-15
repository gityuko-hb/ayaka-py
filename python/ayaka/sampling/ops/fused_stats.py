"""row_logsumexp — MỘT nguồn cho stats theo row của penalties (A3).

Dispatch: Triton (kernel/triton/sampling/fused.py) trên CUDA, torch fallback
nếu không. Cùng semantics hai đường:
  * row_gate=None: logsumexp toàn bộ rows;
  * row_gate cho trước: chỉ row có gate > 0 được đọc và có lse khác 0; row
    còn lại cho lse = 0 — TRÁNH nonzero + index_select copy của bản cũ.

fp32 accumulate bất kể dtype logits vào (khớp bản cũ: sub.to(float32)).
"""

from __future__ import annotations

import torch

try:  # pragma: no cover
    from ayaka.kernel.triton.sampling.fused import _HAS_TRITON_FUSED, row_logsumexp_gpu
except Exception:  # pragma: no cover
    _HAS_TRITON_FUSED = False
    row_logsumexp_gpu = None

__all__ = ["row_logsumexp"]


def _row_logsumexp_torch(logits: torch.Tensor, row_gate: torch.Tensor | None) -> torch.Tensor:
    """Fallback thuần torch — oracle của đường Triton, không host sync."""
    n = logits.size(0)
    if row_gate is None:
        return torch.logsumexp(logits.to(torch.float32), dim=-1)
    gate = row_gate != 0
    lse = torch.zeros(n, dtype=torch.float32, device=logits.device)
    if bool(gate.any()):
        lse[gate] = torch.logsumexp(logits[gate].to(torch.float32), dim=-1)
    return lse


def row_logsumexp(logits: torch.Tensor, *, row_gate: torch.Tensor | None = None) -> torch.Tensor:
    """Logsumexp theo row với gate tùy chọn — xem docstring module.

    Args:
        logits: [n_rows, vocab], dtype float bất kỳ.
        row_gate: [n_rows] (bool hoặc số) — row có gate > 0 mới được tính;
            None = tất cả row.

    Returns:
        Float32 [n_rows]; lse = 0 cho row bị gate chặn.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits phải 2 chiều [n, V], nhận shape {tuple(logits.shape)}")
    if row_gate is not None and row_gate.size(0) != logits.size(0):
        raise ValueError(f"row_gate size {row_gate.size(0)} != logits rows {logits.size(0)}")
    if _HAS_TRITON_FUSED and row_logsumexp_gpu is not None and logits.is_cuda:  # pragma: no cover
        result: torch.Tensor = row_logsumexp_gpu(logits, row_gate)
        return result
    return _row_logsumexp_torch(logits, row_gate)
