"""Pipeline sampling THAM CHIẾU — spec thứ tự các bước, KHÔNG phải runtime path.

Sau M0 đường runtime duy nhất là ``ayaka.sampling.plan.Sampler``. File này giữ
một bản cài đặt độc lập của CÙNG thứ tự để tests so parity giữa bản runtime và
bản spec (tests/test_pipeline_order.py). Đổi Sampler mà không đổi bản này (hoặc
ngược lại) phải làm parity test đỏ.

THỨ TỰ CANONICAL (Sampler.__call__ cài đúng thứ tự này; đổi ở đây thì PHẢI đổi
docstring Sampler và test_pipeline_order.py cùng lúc):
  1. penalty         (penalties.py, apply_penalties_)      — trên logit thô
  2. bitmask         (bitmask.py, apply_allow_bitmask_)     — NEG_INF=-inf
  3. temperature     — chia ĐÚNG MỘT LẦN, trước stats và lọc
  4. stats           (topk_topp.py, softmax_stats_scaled)   — trên scaled
  5. filter + sample (topk_topp.py hoặc gumbel.py)

GHI CHÚ lịch sử: bản trước tính stats TRÊN logits chưa chia temperature rồi
truyền logits thô vào sampler — hai nơi lệch nhau. M0 hợp nhất; stats giờ nằm
trên distribution thật dùng để sample.
"""

from __future__ import annotations

from ayaka.sampling.ops.sampling import (
    apply_allow_bitmask_,
    gumbel_sample,
    softmax_stats_scaled,
    topk_topp_sample,
)
import torch

from ayaka.sampling.metadata import SamplingMetadata
from ayaka.sampling.ops.penalties import PenaltyState, apply_penalties_

STAGES = ("penalty", "bitmask", "temperature", "stats", "filter_sample")


def run_sampling_pipeline(
    logits: torch.Tensor,
    md: SamplingMetadata,
    penalty_state: PenaltyState,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    *,
    sampler: str = "inverse_cdf",  # "inverse_cdf" | "gumbel"
    compute_stats: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Reference: chạy đúng thứ tự canonical ở trên. Trả (token, stats_hoac_None).

    logits bị sửa IN-PLACE bởi penalty + bitmask — caller phải clone() trước nếu
    cần giữ logits gốc (LogprobProcessor raw snapshot phải clone TRƯỚC hàm này).
    """
    apply_penalties_(logits, md, penalty_state)

    if mask is not None:
        assert row_indices is not None, "mask không row_indices không rõ ràng buộc row nào"
        apply_allow_bitmask_(logits, mask, row_indices, logits.size(1))

    scaled = logits / temperature.clamp_min(1e-6).unsqueeze(1)

    stats = softmax_stats_scaled(scaled) if compute_stats else None

    if sampler == "gumbel":
        tok = gumbel_sample(scaled, top_k, top_p, min_p, seed, offset)
    elif sampler == "inverse_cdf":
        tok = topk_topp_sample(scaled, top_k, top_p, min_p, seed, offset)
    else:
        raise ValueError(f"sampler không hợp lệ: {sampler!r}")

    return tok, stats
