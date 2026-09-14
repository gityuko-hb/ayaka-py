"""Repetition / frequency / presence penalty — COORDINATE-BASED, không dense.

SGLang materialize logit_bias thành tensor (batch_size, vocab_size) trên GPU.
Với B=256, V=152K, fp32 = 155 MB cho dữ liệu 99.99% là 0. rtp-llm làm tệ hơn:
dense u8 [B,V] trên PAGEABLE memory, cấp phát lại mỗi step.

Ở đây state là (row, token, count) thưa. Chi phí O(số token duy nhất đã sinh)
thay vì O(B×V).

--- BẢN VÁ (so với file gốc đã review) ---

1. unrecord() — rollback cho speculative decoding. record() optimistic-update
   khi draft token được đề xuất; nếu draft bị reject, phải undo. Core đã có
   bug "missing speculative decode rollback" đã fix ở lớp khác — file này
   thiếu đúng năng lực đó, giờ thêm vào. Fail-closed: unrecord token chưa
   từng record là lỗi logic ở caller, raise chứ không im lặng bỏ qua.

2. Repetition penalty (sign-branch) giờ áp trên log-softmax đã chuẩn hoá thay
   vì logit thô — loại gauge dependence. Lý do: softmax bất biến khi cộng
   hằng số vào mọi logit ⇒ điểm gốc (zero-point) của logit là gauge tuỳ ý mà
   training không ràng buộc, nhưng `torch.where(v<0, v*rep, v/rep)` lại ĐỌC
   đúng điểm gốc tuỳ ý đó. Theo arXiv 2607.09791 (07/2026): re-center logit
   bằng hằng số là no-op ở theta=1 nhưng đổi 58-96% token greedy ở theta=1.3
   thường dùng; kết hợp constrained decoding (đúng use case của bitmask.py),
   trên 200 JSON schema thật, theta=1.3 làm tỷ lệ output hợp lệ rớt từ 97%
   xuống 23%. Paper khảo sát thấy công thức y hệt (sign-branch, không
   normalize) ở khoảng chục engine, trong đó có SGLang. freq/pres giữ nguyên
   additive/subtractive — paper xác nhận phần đó không bị gauge dependence.

   Cài đặt: canonicalize bằng cách TRỪ logsumexp(row) trước khi so sánh dấu
   và nhân/chia, rồi CỘNG LẠI logsumexp(row) để giữ scale nhất quán với phần
   logit không bị đụng tới trong row (softmax(row) không đổi bởi phép trừ-rồi-
   cộng một hằng số như nhau cho mọi phần tử được ghi). CHI PHÍ: cần
   logsumexp của TOÀN ROW (không chỉ token đã xuất hiện) cho các row có
   rep_penalty > 0 — quay lại O(active_rows × V) READ (không phải WRITE) cho
   riêng phần rep, freq/pres vẫn thuần O(unique tokens). Nếu chi phí này đáng
   kể ở batch lớn, cân nhắc fuse với softmax_stats (topk_topp.py) thay vì tính
   logsumexp riêng — không làm ở đây để giữ file độc lập, dễ test.

3. coords_padded() — biến thể shape CỐ ĐỊNH của coords(), cần cho CUDA graph
   capture (graph đòi static shape, coords() gốc trả shape đổi theo số token
   unique). Fail-closed nếu vượt max_unique (không âm thầm cắt bớt penalty).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ayaka.sampling.metadata import SamplingMetadata

__all__ = ["PenaltyState", "apply_penalties_", "apply_penalties_padded_"]


class PenaltyState:
    """Đếm token đã sinh, per-slot. Slot ổn định theo persistent batch.

    Hợp đồng structural (record/unrecord/reset/move) giữ nguyên từ bản port;
    repo này không có Protocol tương ứng, caller chỉ cần đúng chữ ký đó —
    không cần kế thừa ABC nào.
    """

    __slots__ = ("_counts", "max_batch_size")

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self._counts: list[dict[int, int]] = [{} for _ in range(max_batch_size)]

    def record(self, slot: int, tokens: Iterable[int]) -> None:
        d = self._counts[slot]
        for t in tokens:
            d[int(t)] = d.get(int(t), 0) + 1

    def unrecord(self, slot: int, tokens: Iterable[int]) -> None:
        """Undo record() — dùng khi speculative decode reject draft token đã
        optimistically record(). Giảm count, xoá key nếu về 0.

        Fail-closed: unrecord token chưa từng record ở slot này là lỗi logic
        ở caller (rollback sai token/sai slot/gọi hai lần) — raise thay vì
        lặng lẽ no-op, đúng tinh thần I4 (capability error over silent
        fallback). Không cho count âm.
        """
        d = self._counts[slot]
        for t in tokens:
            t = int(t)
            if t not in d or d[t] <= 0:
                raise KeyError(
                    f"unrecord token {t} ở slot {slot} nhưng chưa từng record "
                    "(hoặc đã unrecord hết) — khả năng cao là bug ở caller."
                )
            d[t] -= 1
            if d[t] == 0:
                del d[t]

    def reset(self, slot: int) -> None:
        self._counts[slot].clear()

    def move(self, src: int, dst: int) -> None:
        if src != dst:
            self._counts[dst] = self._counts[src]
            self._counts[src] = {}

    def unique_tokens(self, n_active: int) -> int:
        return sum(len(self._counts[i]) for i in range(n_active))

    def count(self, slot: int, token: int) -> int:
        """Occurrences of ``token`` recorded for ``slot`` (prompt + output)."""
        return self._counts[slot].get(int(token), 0)

    def coords(
        self, n_active: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, cols, cnts = [], [], []
        for r in range(n_active):
            for t, c in self._counts[r].items():
                rows.append(r)
                cols.append(t)
                cnts.append(c)
        if not rows:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, torch.empty(0, dtype=torch.float32, device=device)
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(cols, dtype=torch.long, device=device),
            torch.tensor(cnts, dtype=torch.float32, device=device),
        )

    def coords_padded(
        self, n_active: int, device: torch.device, max_unique: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Biến thể shape CỐ ĐỊNH — cần cho CUDA graph capture.

        Trả thêm `valid` [max_unique] bool: True cho phần tử thật, False cho
        padding. Padding được set cols=0 (vị trí bất kỳ, không quan trọng vì
        apply_penalties_padded_ sẽ trung hoà mọi hiệu ứng bằng `valid`, KHÔNG
        dựa vào (row,col) padding trỏ tới đâu).

        Fail-closed nếu tổng unique tokens vượt max_unique: raise, không âm
        thầm cắt bớt (cắt bớt sẽ làm mất một phần penalty mà không ai biết —
        đúng lớp lỗi mà project này nhất quán tránh ở mọi nơi khác).
        """
        rows, cols, cnts = self.coords(n_active, device)
        total = rows.numel()
        if total > max_unique:
            raise ValueError(
                f"unique tokens ({total}) vượt max_unique ({max_unique}) đã "
                "cấp cho CUDA graph capture — tăng max_unique hoặc graph-break "
                "quanh bước penalty."
            )
        pad = max_unique - total
        if pad > 0:
            rows = torch.cat([rows, rows.new_zeros(pad)])
            cols = torch.cat([cols, cols.new_zeros(pad)])
            cnts = torch.cat([cnts, cnts.new_zeros(pad)])
        valid = torch.arange(max_unique, device=device) < total
        return rows, cols, cnts, valid


def _canonicalize_rep_logsumexp(logits: torch.Tensor, rep_full: torch.Tensor) -> torch.Tensor:
    """logsumexp theo row cho các row có rep_penalty > 0, 0 cho row còn lại.
    O(active_rows × V) READ — không ghi lại logits (khác việc log_softmax
    toàn row), nên không tốn thêm bộ nhớ/bandwidth ghi."""
    lse = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float32)
    active_rows = torch.nonzero(rep_full > 0, as_tuple=True)[0]
    if active_rows.numel() > 0:
        sub = logits.index_select(0, active_rows).to(torch.float32)
        lse[active_rows] = torch.logsumexp(sub, dim=-1)
    return lse


def apply_penalties_(
    logits: torch.Tensor, md: SamplingMetadata, state: PenaltyState
) -> torch.Tensor:
    """In-place. CHẠY TRƯỚC MASK — xem ghi chú NEG_INF trong ops/bitmask.py.

    rep: sign-branch áp trên log-softmax đã chuẩn hoá (xem docstring module).
    freq/pres: giữ nguyên additive/subtractive như bản gốc.
    """
    n = md.n_active
    if logits.size(0) != n:
        raise ValueError(
            f"logits.size(0)={logits.size(0)} != md.n_active={n} — caller đưa "
            "nhầm logits không khớp batch của metadata (vd đã slice logits mà "
            "quên slice md/state theo cùng tập row). Fail-closed thay vì "
            "IndexError khó hiểu ở bước sau."
        )
    if n == 0 or not md.any_penalty:
        return logits
    rows, cols, cnt = state.coords(n, logits.device)
    if rows.numel() == 0:
        return logits

    rep_full = md.active("rep_penalty")
    freq = md.active("freq_penalty").index_select(0, rows)
    pres = md.active("pres_penalty").index_select(0, rows)
    rep_row = rep_full.index_select(0, rows)

    lse = _canonicalize_rep_logsumexp(logits, rep_full)
    lse_row = lse.index_select(0, rows)

    vals = logits[rows, cols].to(torch.float32)
    shifted = vals - lse_row
    penalized = torch.where(shifted < 0, shifted * rep_row, shifted / rep_row)
    vals = torch.where(rep_row > 0, penalized + lse_row, vals)
    vals = vals - freq * cnt - pres * (cnt > 0).to(vals.dtype)
    logits[rows, cols] = vals.to(logits.dtype)
    return logits


def apply_penalties_padded_(
    logits: torch.Tensor,
    md: SamplingMetadata,
    rows: torch.Tensor,
    cols: torch.Tensor,
    cnt: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Biến thể graph-capturable của apply_penalties_ — nhận trực tiếp output
    của PenaltyState.coords_padded() thay vì tự gọi coords() (coords() không
    static-shape nên không được gọi trong graph).

    QUAN TRỌNG — lý do KHÔNG dùng `logits[rows,cols] = vals` như bản không
    padding: padding luôn đặt (row,col)=(0,0), CÓ THỂ alias một entry thật ở
    đúng vị trí đó (row 0 thật sự có token 0 trong cửa sổ penalty). Ghi đè
    trực tiếp với index trùng nhau là undefined behavior — ai thắng không
    đảm bảo, đặc biệt trên GPU (dict trong coords() đảm bảo (row,col) thật
    không trùng nhau, nhưng padding thì có thể trùng MỘT entry thật).

    Sửa bằng cách ghi DELTA (new - old) qua index_put_(..., accumulate=True)
    thay vì ghi giá trị mới: cộng dồn có thứ tự xác định (well-defined) cho
    index trùng, và delta của padding bị ép về 0 ở dòng cuối — cộng 0 vào bất
    cứ vị trí nào, kể cả trùng với entry thật, là an toàn tuyệt đối. Nhờ vậy
    không cần trung hoà rep/freq/pres/cnt cho padding từng phần tử như draft
    trước (rủi ro hơn, dễ lệch nếu quên trung hoà đúng chỗ) — chỉ cần chặn ở
    một điểm duy nhất, cuối cùng.
    """
    rep_full = md.active("rep_penalty")
    freq_full = md.active("freq_penalty")
    pres_full = md.active("pres_penalty")

    rep_row = rep_full.index_select(0, rows)
    freq = freq_full.index_select(0, rows)
    pres = pres_full.index_select(0, rows)

    lse = _canonicalize_rep_logsumexp(logits, rep_full)
    lse_row = lse.index_select(0, rows)

    old_vals = logits[rows, cols].to(torch.float32)
    shifted = old_vals - lse_row
    penalized = torch.where(shifted < 0, shifted * rep_row, shifted / rep_row)
    new_vals = torch.where(rep_row > 0, penalized + lse_row, old_vals)
    new_vals = new_vals - freq * cnt - pres * (cnt > 0).to(new_vals.dtype)

    # inf/nan có thể xuất hiện ở nhánh KHÔNG được chọn (vd rep_row=0 cho một
    # padding tình cờ thừa hưởng row có rep=0) — an toàn vì torch.where không
    # lan NaN/Inf từ nhánh không chọn, và delta bị ép 0 ngay sau đây.
    delta = torch.where(valid, new_vals - old_vals, torch.zeros_like(new_vals))
    logits.index_put_((rows, cols), delta.to(logits.dtype), accumulate=True)
    return logits
