"""Triton kernel cho seeded_gumbel — P1 kernel-workflow template.

THIẾT KẾ (đã thảo luận, không phải mặc định ngẫu nhiên): sinh gumbel noise
CHỈ cho TOPK_BOUND vị trí đầu (sau khi đã sort+filter bởi filter_probs),
KHÔNG cho toàn V. Input là (sp, si) — output CỦA filter_probs (đã sort giảm
dần, đã lọc). Lý do tách khỏi việc filter (không tự làm top_k/top_p/min_p ở
đây): filter là phần phức tạp, nhiều pass, đã có bản torch đúng và test kỹ
(topk_topp.py); kernel này CHỈ làm phần mới (noise + argmax), tái dùng filter
đã có qua đường gọi ở gumbel.py (dispatch), không phải reimplement.

Convention import triton trực tiếp (không try/except) giống
ayaka/kernel/triton/paged_attention.py: triton nằm trong extra ``cuda``, nên
module này chỉ import được khi có triton; phía gọi (ops/gumbel.py) tự bắt
ImportError và fallback về oracle torch.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.kernel.triton.sampling.philox import philox_u01

HAS_TRITON_GUMBEL = True


@triton.jit
def _gumbel_argmax_kernel(
    sp_ptr,
    si_ptr,
    out_ptr,
    seed_ptr,
    offset_ptr,
    stride_row,
    TOPK_BOUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MỘT program/row. Đọc TOPK_BOUND phần tử đầu của sp/si (đã sort
    giảm dần bởi filter_probs), sinh noise cho từng vị trí, argmax, ghi
    token id thật (không phải rank) vào out.

    sp=0 ở một vị trí (đã bị filter loại) ⇒ log(sp)=-inf ⇒ không bao giờ
    thắng argmax dù noise là gì — khớp bất biến "không chọn token bị lọc"
    đã test ở gumbel.py's oracle.
    """
    row = tl.program_id(0)
    seed = tl.load(seed_ptr + row)
    offset_base = tl.load(offset_ptr + row)

    col = tl.arange(0, BLOCK)
    mask = col < TOPK_BOUND
    sp = tl.load(sp_ptr + row * stride_row + col, mask=mask, other=0.0)

    logp = tl.where(sp > 0, tl.log(sp), float("-inf"))

    offset = offset_base * TOPK_BOUND + col  # dia chi hoa rieng cho gumbel,
    # giong het counter_uniform_cols ben gumbel.py (khong dung offset+col
    # truc tiep de tranh dam do voi cac loi goi counter_uniform 1-chieu khac)
    u = philox_u01(seed, offset)
    u = tl.minimum(tl.maximum(u, 1e-12), 1.0 - 1e-7)
    gumbel = -tl.log(-tl.log(u))

    score = tl.where(mask, logp + gumbel.to(tl.float32), float("-inf"))
    best_local = tl.argmax(score, axis=0)
    winner_id = tl.load(si_ptr + row * stride_row + best_local)
    tl.store(out_ptr + row, winner_id)


def fused_gumbel_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    *,
    topk_bound: int = 512,
) -> torch.Tensor:
    """Entry point gọi từ gumbel.py. Vẫn gọi filter_probs (torch) cho phần
    filter — chỉ kernel hoá phần noise+argmax.

    topk_bound PHẢI >= max thực tế của top_k đang dùng trong batch, nếu
    không candidate ngoài topk_bound bị cắt oan ÂM THẦM.
    """
    from ayaka.sampling.ops.topk_topp import filter_probs

    sp, si = filter_probs(logits, top_k, top_p, min_p)
    b, v = sp.shape
    block = triton.next_power_of_2(min(topk_bound, v))
    out = torch.empty(b, dtype=torch.int64, device=logits.device)

    cast(Any, _gumbel_argmax_kernel)[(b,)](
        sp,
        si,
        out,
        seed,
        offset,
        sp.stride(0),
        TOPK_BOUND=min(topk_bound, v),
        BLOCK=block,
    )
    return out
