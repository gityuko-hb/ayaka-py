"""Mirostat v2 — kiểm soát perplexity trực tiếp thay vì top_k/top_p tĩnh.

Port từ `ayaka-sampling` (Rust, đã nghỉ hưu, có Mirostat v2 + choose_gumbel).
CHƯA đối chiếu lại với bản Rust gốc — nếu sai khác công thức update mu hoặc
đơn vị surprisal (bit vs nat), lấy bản Rust làm chuẩn, không phải file này.

Nguyên lý (Basu et al., "Mirostat: A Neural Text Decoding Algorithm that
Directly Controls Perplexity"):
  1. Cắt phân phối tại vị trí đầu tiên mà surprisal (-log2 p) vượt mu.
  2. Sample trong tập đã cắt (renormalize, dùng lại counter_uniform).
  3. Đo surprisal THẬT của token vừa chọn, cập nhật mu theo sai số so với tau.

KHÔNG dùng filter_probs (top_k/top_p/min_p) — Mirostat tự quyết truncation
theo surprisal per-step, không theo rank hay cumulative-prob cố định. Hai cơ
chế lọc này KHÔNG kết hợp cùng nhau trong một lần gọi (chọn một).

state `mu` PHẢI persistent theo slot — giống PenaltyState, không phải giá
trị tính lại từ đầu mỗi step. Caller (core.plan hoặc tương đương) chịu trách
nhiệm lưu mu_moi giữa các step, giống cách PenaltyState được caller giữ qua
record()/move()/reset().
"""

from __future__ import annotations

import torch

from ayaka.sampling.rng import counter_uniform


def mirostat_v2_step(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    tau: torch.Tensor,
    mu: torch.Tensor,
    eta: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trả (token_id [B], mu_moi [B]). Caller lưu mu_moi cho step kế tiếp.

    tau: surprisal mục tiêu, đơn vị bit (thường 3.0-5.0, tương ứng
         perplexity mục tiêu ~2^tau). eta: learning rate cập nhật mu
         (thường ~0.1). mu khởi tạo thường = 2*tau (theo paper gốc).
    """
    v = logits.size(1)
    x = logits.to(torch.float32) / temperature.clamp_min(1e-6).unsqueeze(1)
    sorted_x, sorted_idx = x.sort(dim=-1, descending=True)

    m = sorted_x[:, :1]
    e = torch.exp(sorted_x - m)
    cum_e = e.cumsum(dim=-1)
    z = cum_e[:, -1:].clamp_min(torch.finfo(torch.float32).tiny)
    p = e / z  # phân phối thật, đã sort giảm dần theo rank
    surprisal_bits = -torch.log2(p.clamp_min(torch.finfo(p.dtype).tiny))

    # Vị trí cắt: token đầu tiên có surprisal > mu bị loại, LUÔN giữ >=1 token
    # (kể cả khi token top-1 đã có surprisal > mu — không được cắt về rỗng).
    keep = surprisal_bits <= mu.unsqueeze(1)
    keep[:, 0] = True
    k = keep.to(torch.int64).cumprod(dim=-1).sum(dim=-1, keepdim=True).clamp_(min=1)
    # cumprod thay vì .sum(keep) trực tiếp: đảm bảo cắt tại vi phạm ĐẦU TIÊN,
    # không phải đếm tổng số token thoả mãn rải rác (surprisal không đảm bảo
    # đơn điệu tuyệt đối do fp rounding ở đuôi phân phối rất phẳng).

    rank = torch.arange(v, device=logits.device).unsqueeze(0)
    trunc_p = torch.where(rank < k, p, torch.zeros_like(p))
    trunc_p = trunc_p / trunc_p.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)

    u = counter_uniform(seed, offset).to(trunc_p.dtype).unsqueeze(1)
    pos = (trunc_p.cumsum(dim=-1) < u).sum(dim=-1).clamp_(max=v - 1)
    tok = sorted_idx.gather(1, pos.unsqueeze(1)).squeeze(1)

    observed_surprisal = surprisal_bits.gather(1, pos.unsqueeze(1)).squeeze(1)
    mu_new = mu - eta * (observed_surprisal - tau)
    return tok, mu_new


def mirostat_v2_init(tau: torch.Tensor) -> torch.Tensor:
    """mu khởi tạo = 2*tau, theo paper gốc. Gọi một lần khi bắt đầu request,
    kết quả được caller lưu persistent giống PenaltyState."""
    return 2.0 * tau
