"""seeded_gumbel — sample bằng Gumbel-max trick thay vì inverse-CDF.

Gumbel-max: argmax_i(log p_i + g_i), g_i ~ Gumbel(0,1) độc lập, cho đúng phân
phối softmax(logit). Vì argmax bất biến khi cộng hằng số, và log p_i =
x_i - LSE(x) (x đã chia nhiệt độ), việc CHUẨN HOÁ p_i về phân phối thật
(chia cho tổng) không bắt buộc cho quyết định sample — chỉ cần thứ tự tương
đối đúng. Khác _reference_sample (topk_topp.py) ở đúng một chỗ: bước sample
cuối (cumsum+threshold → noise+argmax). filter_probs (top_k→renorm→top_p→
min_p) GIỮ NGUYÊN, dùng lại y hệt — đây không phải bộ lọc mới, chỉ là cách
chọn khác trong tập đã lọc.

TẠI SAO đáng làm kernel riêng thay vì dùng FlashInfer's rejection sampling
sẵn có:
  - Rejection sampling tiêu thụ số vòng biến thiên ⇒ khó CUDA-graph hơn.
    Gumbel-max chi phí CỐ ĐỊNH mỗi row (1 argmax trên tập đã biết kích thước).
  - Philox4x32-10 hiện có là hàm THUẦN của địa chỉ ⇒ sinh noise độc lập theo
    (row, candidate) không tốn state, song song hoàn hảo — hợp gumbel-max
    (cần NHIỀU số ngẫu nhiên/row) hơn hẳn so với chỉ 1 số/row của inverse-CDF.
  - Không phụ thuộc FlashInfer cho primitive sampling lõi (nhất quán với
    "zero external deps" đã áp cho FlashAttention/Marlin).

QUYẾT ĐỊNH PERF QUAN TRỌNG NHẤT (dành cho bản Triton, xem
ayaka/kernel/triton/sampling/gumbel.py): sinh noise CHỈ cho tập sống sót sau
filter_probs (bị chặn bởi top_k tối đa cho phép, thường vài trăm), KHÔNG cho
toàn V (~150K) — chênh lệch có thể 1000x+ số lần draw RNG. Oracle dưới đây cố
tình sinh noise cho CẢ V (đơn giản hơn, đây không phải hot path, đúng tinh
thần "reference được phép chậm").
"""

from __future__ import annotations

import torch

from ayaka.sampling.ops.topk_topp import filter_probs
from ayaka.sampling.rng import counter_uniform_cols

try:  # pragma: no cover
    from ayaka.kernel.triton.sampling.gumbel import HAS_TRITON_GUMBEL, fused_gumbel_sample
except Exception:  # pragma: no cover
    HAS_TRITON_GUMBEL = False
    fused_gumbel_sample = None


def _reference_gumbel_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Oracle CPU/torch. Cùng cấu trúc _reference_sample bên topk_topp.py,
    dùng chung filter_probs, chỉ thay bước sample cuối.

    KHÔNG cần chuẩn hoá sp thành phân phối thật (khác _reference_sample): sp
    chưa chuẩn hoá (tổng < 1 vì đã lọc) vẫn cho đúng argmax, vì hằng số
    -log(tổng) cộng đều mọi ứng viên trong row không đổi thứ tự.
    """
    v = logits.size(1)
    sp, si = filter_probs(logits, top_k, top_p, min_p)

    degenerate = sp.sum(dim=-1) <= 0

    u = counter_uniform_cols(seed, offset, v).to(sp.dtype)
    u = u.clamp(min=torch.finfo(sp.dtype).tiny, max=1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(u))

    log_sp = torch.where(
        sp > 0,
        torch.log(sp.clamp_min(torch.finfo(sp.dtype).tiny)),
        torch.full_like(sp, float("-inf")),
    )
    winner = torch.argmax(log_sp + gumbel, dim=-1)
    tok = si.gather(1, winner.unsqueeze(1)).squeeze(1)

    if bool(degenerate.any()):
        tok = torch.where(degenerate, logits.argmax(dim=-1), tok)
    return tok


def gumbel_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    *,
    force_reference: bool = False,
) -> torch.Tensor:
    """Dispatch: Triton (ayaka/kernel/triton/sampling/gumbel.py) khi có, oracle
    torch khi không — cùng pattern dispatch như topk_topp_sample."""
    if (
        HAS_TRITON_GUMBEL
        and fused_gumbel_sample is not None
        and logits.is_cuda
        and not force_reference
    ):  # pragma: no cover
        result: torch.Tensor = fused_gumbel_sample(logits, top_k, top_p, min_p, seed, offset)
        return result
    return _reference_gumbel_sample(logits, top_k, top_p, min_p, seed, offset)
