"""Top-k / top-p / min-p + multinomial.

RNG LÀ COUNTER-BASED, KHÔNG PHẢI torch.Generator. Đây là quyết định thiết kế,
không phải chi tiết:

  * torch.Generator có state ẩn ⇒ kết quả của row i phụ thuộc vào có bao nhiêu
    row khác trong batch và chúng tiêu thụ bao nhiêu số ngẫu nhiên. Rejection
    sampling tiêu thụ số lượng biến thiên ⇒ không batch-invariant.
  * Generator không capture được vào CUDA graph.

splitmix64(seed, offset) cho mỗi row một dòng số độc lập, thuần hàm, tái lập
được, và (seed, offset) là tensor nên FlashInfer capture graph được — chính là
lý do FlashInfer yêu cầu seed/offset dạng tensor thay vì int.

CẬP NHẬT: splitmix64 giờ import từ ayaka.sampling.rng (nguồn canonical, hợp nhất
với bản Python-int trong columns.py) thay vì tự định nghĩa ở đây — hai bản
từng trùng lặp, xem docstring ayaka/sampling/rng.py.
"""

from __future__ import annotations

import torch

from ayaka.sampling.rng import counter_uniform

try:  # pragma: no cover
    import flashinfer.sampling as _fi  # type: ignore[import-not-found]

    _HAS_FLASHINFER = True
except Exception:  # pragma: no cover
    _fi = None
    _HAS_FLASHINFER = False


def has_flashinfer() -> bool:
    return _HAS_FLASHINFER


def topk_topp_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    *,
    force_reference: bool = False,
) -> torch.Tensor:
    if _HAS_FLASHINFER and _fi is not None and logits.is_cuda and not force_reference:
        # FlashInfer 0.6 top_k_top_p_sampling_from_logits KHÔNG có min_p. Đi tiếp
        # vào đây khi có row min_p > 0 là bỏ lọc min_p âm thầm — sai phân phối
        # mà không lỗi. Rẽ về oracle thay vì trả kết quả sai.
        if bool((min_p > 0.0).any()):
            return _reference_sample(logits, top_k, top_p, min_p, seed, offset)
        # FlashInfer: rejection sampling không sort, nhiều vòng gộp trong MỘT
        # kernel. seed/offset phải là tensor int64 mới capture CUDA graph được.
        sampled: torch.Tensor = _fi.top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            filter_apply_order="top_k_first",
            seed=seed,
            offset=offset,
        )
        return sampled
    return _reference_sample(logits, top_k, top_p, min_p, seed, offset)


def filter_probs(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lọc → trả (sorted_probs đã chuẩn hoá, sorted_indices).

    Tách filter khỏi sample là có chủ ý: filter là phần TẤT ĐỊNH, so được
    bit-exact với HuggingFace; sample là phần ngẫu nhiên, chỉ so được bằng
    thống kê. Gộp hai thứ vào một hàm thì không còn oracle chặt cho phần một.

    THỨ TỰ THEO ĐÚNG HUGGINGFACE: top_k → RENORM → top_p → min_p.

    Bước RENORM ở giữa KHÔNG được bỏ. HF cài mỗi warper là một
    LogitsProcessor riêng ghi -inf vào scores, và warper sau gọi
    softmax(scores) LẠI TỪ ĐẦU ⇒ prob được chuẩn hoá trên tập đã sống sót
    sau top_k trước khi top_p cắt. Bỏ renorm thì cumsum của top_p chạy trên
    tổng < 1 ⇒ nucleus cắt rộng hơn HF (giữ nhiều token hơn mức đáng lẽ).
    min_p thì bất biến với renorm (cả p_i lẫn p_max cùng scale) nên không
    cần renorm lần hai.
    """
    v = logits.size(1)
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)

    if bool((top_k > 0).any()):
        rank = torch.arange(v, device=logits.device).unsqueeze(0)
        k = top_k.to(torch.long).unsqueeze(1)
        sp = sp.masked_fill((k > 0) & (rank >= k), 0.0)
        sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(sp.dtype).tiny)

    if bool((top_p < 1.0).any()):
        cum = sp.cumsum(dim=-1)
        drop = (cum - sp) > top_p.unsqueeze(1)
        # BUG ĐÃ SỬA (oracle HF bắt được, 2026-08-20):
        # `(top_p < 1.0).any()` là kiểm tra TOÀN BATCH, nên nhánh này chạy cho
        # MỌI row — kể cả row có top_p == 1.0 nghĩa là "không lọc gì". Với
        # top_p == 1.0, cumsum tích luỹ sai số fp32 nên (cum - sp) ở đuôi ra
        # 1.0000001 > 1.0 và cắt oan vài token cuối.
        #   Triệu chứng: row top_p=1.0 mất ~9/256 token có prob nhỏ nhất, IM
        #   LẶNG. Chỉ lộ ra khi batch KHÔNG đồng nhất (row khác có top_p<1) —
        #   batch đồng nhất không bao giờ vào nhánh này nên test cũ luôn xanh.
        drop &= (top_p < 1.0).unsqueeze(1)
        drop[:, 0] = False  # min_tokens_to_keep=1, khớp HF
        sp = sp.masked_fill(drop, 0.0)

    if bool((min_p > 0.0).any()):
        sp = sp.masked_fill(sp < min_p.unsqueeze(1) * sp[:, :1], 0.0)

    return sp, si


def filter_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """Bản [B, V] không sort của filter_probs — chỉ dùng để so với oracle."""
    sp, si = filter_probs(logits, top_k, top_p, min_p)
    out = torch.zeros_like(sp)
    out.scatter_(1, si, sp)
    return out


def _reference_sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Đường tham chiếu: sort-based. Chậm, nhưng đúng tuyệt đối — nó là oracle
    để verify đường FlashInfer bằng chi-square."""
    v = logits.size(1)
    sp, si = filter_probs(logits, top_k, top_p, min_p)

    total = sp.sum(dim=-1, keepdim=True)
    degenerate = (total <= 0).squeeze(1)
    sp = sp / total.clamp_min(torch.finfo(sp.dtype).tiny)

    u = counter_uniform(seed, offset).to(sp.dtype).unsqueeze(1)
    pos = (sp.cumsum(dim=-1) < u).sum(dim=-1).clamp_(max=v - 1)
    tok = si.gather(1, pos.unsqueeze(1)).squeeze(1)

    if bool(degenerate.any()):
        # Chỉ xảy ra khi mọi token bị lọc sạch — vi phạm hợp đồng ở tầng trên.
        # Fallback argmax thay vì trả NaN im lặng.
        tok = torch.where(degenerate, logits.argmax(dim=-1), tok)
    return tok


def softmax_stats(logits: torch.Tensor, temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper chia temperature rồi tính stats — hợp đồng cũ cho reference path."""
    scaled = logits.to(torch.float32) / temperature.clamp_min(1e-6).unsqueeze(1)
    return softmax_stats_scaled(scaled)


def softmax_stats_scaled(
    scaled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stats trên logits ĐÃ chia temperature — canonical Sampler gọi bản này.

    Trả (max, sum_exp, entropy) trong MỘT lần duyệt.

    Ba số này là điểm mở rộng có chủ ý: min-p, dynamic min-p, typical sampling,
    mirostat v2 và entropy-based sampling ĐỀU chỉ là hàm thuần trên chúng. Tính
    ở đây thì mọi strategy về sau tốn thêm ~0 và KHÔNG cần round-trip host để
    giữ state — đây chính là chỗ rtp-llm sẽ vỡ nếu muốn thêm mirostat, vì mọi
    state của nó đều nằm ở host.
    """
    x = scaled.to(torch.float32)
    m = x.max(dim=-1, keepdim=True).values
    d = x - m
    finite = torch.isfinite(d)
    e = torch.where(finite, torch.exp(d), torch.zeros_like(d))
    s = e.sum(dim=-1, keepdim=True)
    ent = torch.log(s) - (e * torch.where(finite, d, torch.zeros_like(d))).sum(
        dim=-1, keepdim=True
    ) / s.clamp_min(torch.finfo(torch.float32).tiny)
    return m.squeeze(1), s.squeeze(1), ent.squeeze(1)
