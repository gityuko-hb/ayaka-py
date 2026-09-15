"""DRY ("Don't Repeat Yourself") — phạt token sẽ KÉO DÀI một n-gram đã lặp,
khác penalties.py (phạt theo TẦN SUẤT token, không quan tâm thứ tự/pattern).

CHƯA có tham chiếu canonical để so bit-exact (không có bản Rust/reference nào
của Ayaka cho DRY — khác gumbel.py/mirostat.py là port lại từ ayaka-sampling).
Tham số (multiplier, base, allowed_length) theo quy ước phổ biến trong cộng
đồng llama.cpp/koboldcpp — CẦN đối chiếu lại nếu muốn khớp một implementation
cụ thể nào đó.

Thuật toán: với mỗi vị trí i trong lịch sử (0 <= i < T-1), đo độ dài khớp lùi
k giữa history[i-k:i] và history[T-k:T] (đuôi hiện tại). Nếu k đạt
allowed_length, token history[i] (token từng "theo sau" pattern khớp đó)
nhận penalty = multiplier * base^(k - allowed_length). Nhiều vị trí cùng phạt
một token thì lấy MAX (không cộng dồn).

CỐ Ý viết bằng vòng lặp Python (O(B × T × max_ngram), không vectorize) —
đây là oracle tham chiếu, không phải hot path. Khác PenaltyState (A1 đã
delta-flush trên bảng device), DRY cần THỨ TỰ history nên giữ vòng lặp host;
đây là ứng viên tối ưu/kernel hoá về sau nếu profiling cho thấy cần, không
phải bây giờ.
"""

from __future__ import annotations

import torch


def dry_bias(
    history: list[list[int]],
    vocab_size: int,
    multiplier: torch.Tensor,
    base: torch.Tensor,
    allowed_length: torch.Tensor,
    *,
    max_ngram: int = 32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Trả [B, V] bias (<=0) để CỘNG vào logits (không phải sign-branch như
    penalties.py — DRY luôn additive, không có vấn đề gauge dependence).

    history[b]: list[int] token đã sinh của request b, THEO ĐÚNG THỨ TỰ
    (khác PenaltyState — DRY cần thứ tự, không chỉ tần suất).
    """
    b = len(history)
    bias = torch.zeros(b, vocab_size, device=device)
    mult = multiplier.tolist()
    bse = base.tolist()
    allow = allowed_length.tolist()

    for row, hist in enumerate(history):
        t = len(hist)
        if t < 2:
            continue
        cap = min(max_ngram, t - 1)
        al = int(allow[row])
        for i in range(t - 1):
            k = 0
            while (
                k < cap
                and i - 1 - k >= 0
                and (t - 1 - k) >= 0
                and hist[i - 1 - k] == hist[t - 1 - k]
            ):
                k += 1
            if k >= al:
                tok = hist[i]
                penalty = mult[row] * (bse[row] ** (k - al))
                if -penalty < bias[row, tok]:
                    bias[row, tok] = -penalty
    return bias


def apply_dry_(
    logits: torch.Tensor,
    history: list[list[int]],
    multiplier: torch.Tensor,
    base: torch.Tensor,
    allowed_length: torch.Tensor,
    *,
    max_ngram: int = 32,
) -> torch.Tensor:
    """In-place, additive — không cần canonicalize như apply_penalties_'s rep
    branch vì đây không phải sign-branch trên logit thô."""
    bias = dry_bias(
        history,
        logits.size(1),
        multiplier,
        base,
        allowed_length,
        max_ngram=max_ngram,
        device=logits.device,
    )
    logits += bias
    return logits
