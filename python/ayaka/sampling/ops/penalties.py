"""Repetition / frequency / presence penalty — COORDINATE-BASED, không dense.

SGLang materialize logit_bias thành tensor (batch_size, vocab_size) trên GPU.
Với B=256, V=152K, fp32 = 155 MB cho dữ liệu 99.99% là 0. rtp-llm làm tệ hơn:
dense u8 [B,V] trên PAGEABLE memory, cấp phát lại mỗi step.

Ở đây state là (row, token, count) thưa. Chi phí O(số token duy nhất đã sinh)
thay vì O(B×V).

--- BẢN VÁ (so với file gốc đã review) ---

1. unrecord() — rollback cho speculative decoding. record() optimistic-update
   khi draft token được đề xuất; nếu draft bị reject, phải undo. Fail-closed:
   unrecord token chưa từng record là lỗi logic ở caller — raise thay vì im
   lặng bỏ qua. Không cho count âm.

2. Repetition penalty (sign-branch) áp trên log-softmax đã chuẩn hoá thay vì
   logit thô — loại gauge dependence (arXiv 2607.09791): TRỪ logsumexp(row)
   trước khi so dấu/nhân-chia, rồi CỘNG LẠI để giữ scale cho phần logit không
   đụng tới. logsumexp TOÀN ROW cho row có rep > 0 — tính qua row_logsumexp
   (ops/fused_stats.py, Triton 1-pass, một lượt đọc/row).

3. A1 — delta-flush, KHÔNG dict Python per-slot. record/unrecord/reset/move
   chỉ ENQUEUE delta trên host, O(delta), không lookup. coords()/coords_
   padded() gọi _flush() MỘT lần/step: gộp delta theo (slot, token) → lookup
   vị trí entry trên bảng device → index_put_(accumulate=True) cho entry cũ,
   cấp entry mới cho token chưa có, compact các entry count→0 (tổng quát hoá
   swap-remove). Bảng device: cols [B, cap] int64, cnts [B, cap] fp32, n_used
   [B] int64; entry thật gọn trong [0, n_used), không thứ tự.

   SEMANTIC DELTA so với bản dict (tuyến tính nên CHỈ khác ở điểm raise):
   count CUỐI khớp chạy tuần tự từng delta; "unrecord token chưa từng
   record" raise LÚC FLUSH và CHỈ khi net âm với token vắng mặt — pattern
   (record t; unrecord t) cùng step là net 0 ⇒ no-op, không raise (đúng nhu
   cầu optimistic draft-reject).

4. coords_padded() — shape TỈNH [n_active, cap] + valid — cho CUDA graph
   capture. Không còn max_unique: bảng device chính là bound.

5. A2 — promote dense: slot có occupancy chạm per_slot_cap được SCATTER sang
   một row dense [max_dense_slots, V] (mặc định 4); delta về sau đi thẳng
   vào dense (new = old + net). coords()/coords_padded() LOẠI row dense khỏi
   phần thưa. per_slot_cap mặc định = promote_fraction × vocab (0.25·V) —
   "ngưỡng" thể hiện qua kích thước bảng, không phải một phép kiểm tra
   runtime riêng (và nhờ vậy flush không phải sync đọc occupancy mỗi step).
   Pool đầy → fail-closed raise.

6. MỘT nguồn công thức duy nhất: apply_penalties_ dựng coords rồi đi vào
   cùng _apply_penalties_core với apply_penalties_padded_ — công thức
   rep/freq/pres viết đúng một lần, ở đường graph-capturable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from ayaka.sampling.metadata import SamplingMetadata
from ayaka.sampling.ops.fused_stats import row_logsumexp

__all__ = ["PenaltyState", "apply_penalties_", "apply_penalties_padded_"]


class PenaltyState:
    """Đếm token đã sinh, per-slot, bảng device [B, cap] + delta host.

    Hợp đồng structural (record/unrecord/reset/move) giữ nguyên từ bản port;
    state nội bộ là bảng device — host work mỗi step phẳng theo |delta|,
    không theo số unique token tích luỹ.
    """

    __slots__ = (
        "_cols",
        "_cnts",
        "_dense_cnts",
        "_dense_members",
        "_dense_slot_ids",
        "_dense_used",
        "_device",
        "_n_used",
        "_pending",
        "max_batch_size",
        "per_slot_cap",
        "promote_fraction",
        "vocab_size",
    )

    def __init__(
        self,
        max_batch_size: int,
        *,
        vocab_size: int | None = None,
        per_slot_cap: int | None = None,
        promote_fraction: float = 0.25,
        max_dense_slots: int = 4,
        device: torch.device | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if vocab_size is not None and vocab_size <= 0:
            raise ValueError("vocab_size phải > 0 khi có")
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.promote_fraction = promote_fraction
        self._device = (
            torch.device(device) if isinstance(device, str) else device
        ) or torch.device("cpu")
        if per_slot_cap is None:
            per_slot_cap = max(1, math.ceil(promote_fraction * vocab_size)) if vocab_size else 512
        if per_slot_cap <= 0:
            raise ValueError("per_slot_cap phải > 0")
        self.per_slot_cap = per_slot_cap
        if max_dense_slots < 0:
            raise ValueError("max_dense_slots phải >= 0")

        cap = per_slot_cap
        self._cols = torch.zeros(max_batch_size, cap, dtype=torch.int64, device=self._device)
        self._cnts = torch.zeros(max_batch_size, cap, dtype=torch.float32, device=self._device)
        self._n_used = torch.zeros(max_batch_size, dtype=torch.int64, device=self._device)
        self._dense_cnts: torch.Tensor | None = None
        self._dense_slot_ids = torch.full(
            (max_dense_slots,), -1, dtype=torch.int64, device=self._device
        )
        self._dense_used = 0
        self._dense_members: set[int] = set()
        self._pending: list[tuple] = []

    # ------------------------------------------------------------------
    # Structural contract — enqueue host delta, O(delta), không lookup
    # ------------------------------------------------------------------
    def record(self, slot: int, tokens: Iterable[int]) -> None:
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        pending = self._pending
        for token in tokens:
            pending.append(("delta", slot, int(token), 1))

    def unrecord(self, slot: int, tokens: Iterable[int]) -> None:
        """Undo record() — rollback speculative decode. Delta -1 được enqueue;
        token thiếu (net âm với token vắng trong bảng) raise LÚC _flush() —
        xem SEMANTIC DELTA ở docstring module."""
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        pending = self._pending
        for token in tokens:
            pending.append(("delta", slot, int(token), -1))

    def reset(self, slot: int) -> None:
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._pending.append(("reset", slot))

    def move(self, src: int, dst: int) -> None:
        if not (0 <= src < self.max_batch_size and 0 <= dst < self.max_batch_size):
            raise IndexError("move slot out of range")
        if src != dst:
            self._pending.append(("move", src, dst))

    def unique_tokens(self, n_active: int) -> int:
        """Số token unique đang đếm của n_active row đầu (sparse + dense)."""
        if n_active <= 0:
            return 0
        self._flush()
        sparse = int(self._keep_mask(n_active).sum().item())
        dense = 0
        if self._dense_cnts is not None:
            ids = self._dense_slot_ids.tolist()
            for i, slot in enumerate(ids):
                if 0 <= slot < n_active:
                    dense += int((self._dense_cnts[i] > 0).sum().item())
        return sparse + dense

    def count(self, slot: int, token: int) -> int:
        """Occurrences of ``token`` recorded for ``slot`` (prompt + output)."""
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._flush()
        d_idx = self._dense_index_of(slot)
        if d_idx is not None:
            assert self._dense_cnts is not None
            return int(self._dense_cnts[d_idx, token].item())
        used = int(self._n_used[slot].item())
        if used == 0:
            return 0
        hit = (self._cols[slot, :used] == token).nonzero()
        if hit.numel() == 0:
            return 0
        return int(self._cnts[slot, int(hit[0])].item())

    # ------------------------------------------------------------------
    # Coords — MỘT flush/step, boolean-index trên device
    # ------------------------------------------------------------------
    def coords(
        self, n_active: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact [N] coords cho phần THƯA; row promoted cho rỗng (dense xử
        lý riêng qua dense branch). N = tổng unique sparse."""
        self._flush()
        if n_active <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            cnts = torch.empty(0, dtype=torch.float32, device=device)
            return empty, empty, cnts
        keep = self._keep_mask(n_active)
        b_idx = keep.nonzero(as_tuple=True)[0].to(torch.long)
        rows = b_idx
        cols = self._cols.narrow(0, 0, n_active)[keep]
        cnts = self._cnts.narrow(0, 0, n_active)[keep]
        if device != self._device:
            rows = rows.to(device)
            cols = cols.to(device)
            cnts = cnts.to(device)
        return rows, cols, cnts

    def coords_padded(
        self, n_active: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shape TỈNH [n_active, cap] + valid — cho CUDA graph capture.

        `valid` [n_active, cap] bool: True cho entry thật. Padding đặt
        cols=0/cnts=0; apply_penalties_padded_ ép delta padding = 0 qua
        `valid`, KHÔNG dựa vào (row, col) padding trỏ tới đâu. Không cần
        max_unique: bảng device chính là bound — occupancy một slot không
        thể vượt per_slot_cap (promote/fail-closed trước khi tràn).
        """
        self._flush()
        n = max(n_active, 0)
        rows = (
            torch.arange(n, device=self._device, dtype=torch.long)
            .unsqueeze(1)
            .expand(n, self.per_slot_cap)
        )
        cols = self._cols.narrow(0, 0, n)
        cnts = self._cnts.narrow(0, 0, n)
        valid = torch.arange(self.per_slot_cap, device=self._device).unsqueeze(
            0
        ) < self._n_used.narrow(0, 0, n).unsqueeze(1)
        if device != self._device:
            rows = rows.to(device)
            cols = cols.to(device)
            cnts = cnts.to(device)
            valid = valid.to(device)
        return rows, cols, cnts, valid

    # ------------------------------------------------------------------
    # Dense pool (A2)
    # ------------------------------------------------------------------
    @property
    def dense_slot_ids(self) -> torch.Tensor:
        """[max_dense_slots] int64 — logits row của mỗi dense slot, -1 = trống."""
        return self._dense_slot_ids

    @property
    def dense_counts(self) -> torch.Tensor | None:
        """[max_dense_slots, vocab] fp32, hoặc None khi pool chưa cấp."""
        return self._dense_cnts

    def _dense_index_of(self, slot: int) -> int | None:
        for i in range(self._dense_slot_ids.size(0)):
            if int(self._dense_slot_ids[i].item()) == slot:
                return i
        return None

    def _ensure_dense_pool(self) -> torch.Tensor:
        if self._dense_cnts is None:
            if not self.vocab_size:
                raise ValueError(
                    "PenaltyState được tạo mà không có vocab_size nên không "
                    "thể promote sang dense — tăng per_slot_cap hoặc truyền "
                    "vocab_size khi dựng state."
                )
            self._dense_cnts = torch.zeros(
                self._dense_slot_ids.size(0),
                self.vocab_size,
                dtype=torch.float32,
                device=self._device,
            )
        return self._dense_cnts

    def _promote(self, slot: int) -> int:
        """Scatter bảng thưa của slot sang một row dense; trả dense index."""
        if self._dense_used >= self._dense_slot_ids.size(0):
            raise ValueError(
                f"dense pool đầy ({self._dense_used} slot) — fail-closed, "
                "không âm thầm bỏ penalty; tăng max_dense_slots."
            )
        dense = self._ensure_dense_pool()
        d_idx = self._dense_used
        used = int(self._n_used[slot].item())
        if used:
            toks = self._cols[slot, :used]
            if int(toks.max().item()) >= (self.vocab_size or 0) or int(toks.min().item()) < 0:
                raise ValueError(
                    "token id ngoài [0, vocab_size) khi promote — state không "
                    "thể biểu diễn dense cho token này"
                )
            dense[d_idx].index_add_(0, toks, self._cnts[slot, :used])
        self._n_used[slot] = 0
        self._dense_slot_ids[d_idx] = slot
        self._dense_used += 1
        self._dense_members.add(slot)
        return d_idx

    # ------------------------------------------------------------------
    # Flush pipeline
    # ------------------------------------------------------------------
    def _flush(self) -> None:
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        batch: list[tuple[int, int, int]] = []

        def apply_batch() -> None:
            if batch:
                self._apply_delta_batch(batch)
                batch.clear()

        for item in pending:
            if item[0] == "delta":
                batch.append((item[1], item[2], item[3]))
            else:
                apply_batch()
                if item[0] == "reset":
                    self._apply_reset(item[1])
                else:
                    self._apply_move(item[1], item[2])
        apply_batch()
        self._compact()

    def _apply_reset(self, slot: int) -> None:
        self._n_used[slot] = 0
        self._cnts[slot].zero_()
        d_idx = self._dense_index_of(slot)
        if d_idx is not None:
            assert self._dense_cnts is not None
            self._dense_cnts[d_idx].zero_()
            self._dense_slot_ids[d_idx] = -1
            self._dense_used -= 1
            self._dense_members.discard(slot)

    def _apply_move(self, src: int, dst: int) -> None:
        d_idx = self._dense_index_of(src)
        self._cols[dst].copy_(self._cols[src])
        self._cnts[dst].copy_(self._cnts[src])
        self._n_used[dst] = self._n_used[src]
        self._n_used[src] = 0
        self._cnts[src].zero_()
        if d_idx is not None:
            # Row dense chuyển QUYỀN cho dst — counts giữ nguyên, không xoá.
            self._dense_slot_ids[d_idx] = dst
            self._dense_members.discard(src)
            self._dense_members.add(dst)

    def _keep_mask(self, n_active: int) -> torch.Tensor:
        """[n_active, cap] bool — entry thật của phần thưa."""
        return (
            torch.arange(self.per_slot_cap, device=self._device).unsqueeze(0)
            < self._n_used.narrow(0, 0, n_active).unsqueeze(1)
        ) & (self._cnts.narrow(0, 0, n_active) > 0)

    def _apply_delta_batch(self, batch: list[tuple[int, int, int]]) -> None:
        """Gộp delta theo (slot, token), lookup trên bảng device, cập nhật.

        Sync CHỈ khi có net âm (unrecord) hoặc bound chạm cap — đường
        record-thuần (steady state) không sync. Lookup là so sánh thuần
        device ([U, cap]); U = |delta gộp| của step nên O(delta).
        """
        net: dict[tuple[int, int], int] = {}
        for slot, token, c in batch:
            key = (slot, token)
            net[key] = net.get(key, 0) + c
        pairs = [(s, t, n) for (s, t), n in net.items() if n != 0]
        if not pairs:
            return

        slots = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=self._device)
        toks = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=self._device)
        nets = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=self._device)

        # Lookup: token nằm trong entry đang dùng của slot? (thuần device)
        rows = self._cols.index_select(0, slots)
        in_used = torch.arange(self.per_slot_cap, device=self._device).unsqueeze(
            0
        ) < self._n_used.index_select(0, slots).unsqueeze(1)
        match = (rows == toks.unsqueeze(1)) & in_used
        found = match.any(dim=1)
        pos = match.to(torch.long).argmax(dim=1)

        # Overflow: bound host (mọi pair net>0 đều có thể là alloc mới) chạm
        # cap mới kiểm tra thật; slot thật sự cần cấp thêm → promote dense.
        alloc_bound = torch.bincount(slots[nets > 0], minlength=self.max_batch_size)
        pressure = (self._n_used + alloc_bound) > self.per_slot_cap
        if bool(pressure.any().item()):
            for slot_id in torch.nonzero(pressure).flatten().tolist():
                self._promote(int(slot_id))

        s_list = slots.tolist()
        t_list = toks.tolist()
        n_list = nets.tolist()
        f_list = found.tolist()
        p_list = pos.tolist()
        is_dense = [int(s) in self._dense_members for s in s_list]
        dense_pairs = [
            (int(s_list[i]), int(t_list[i]), int(n_list[i]))
            for i in range(len(pairs))
            if is_dense[i]
        ]

        # Validation phần sparse (dense tự validate trong _apply_dense_pairs):
        # net âm phải khớp count hiện có, không cho count âm.
        if bool((nets < 0).any().item()):
            old = torch.zeros(len(pairs), dtype=torch.float32, device=self._device)
            fi = [i for i in range(len(pairs)) if (not is_dense[i]) and f_list[i]]
            if fi:
                f_idx = torch.tensor(fi, dtype=torch.long, device=self._device)
                s_idx = torch.tensor(
                    [int(s_list[i]) for i in fi], dtype=torch.long, device=self._device
                )
                p_idx = torch.tensor([p_list[i] for i in fi], dtype=torch.long, device=self._device)
                old[f_idx] = self._cnts[s_idx, p_idx]
            bad = (old + nets.to(torch.float32)) < 0
            if bool(bad.any().item()):
                i = int(torch.nonzero(bad)[0])
                raise KeyError(
                    f"unrecord token {int(t_list[i])} ở slot {int(s_list[i])} "
                    "nhưng không đủ count trong bảng (chưa từng record hoặc "
                    "đã unrecord hết) — khả năng cao là bug ở caller."
                )

        # Sparse: cập nhật entry cũ (pos unique per slot).
        upd = [
            i for i in range(len(pairs)) if (not is_dense[i]) and f_list[i] and int(n_list[i]) != 0
        ]
        if upd:
            u_slot = torch.tensor(
                [int(s_list[i]) for i in upd], dtype=torch.long, device=self._device
            )
            u_pos = torch.tensor([p_list[i] for i in upd], dtype=torch.long, device=self._device)
            u_new = self._cnts[u_slot, u_pos] + torch.tensor(
                [float(n_list[i]) for i in upd], device=self._device
            )
            self._cnts.index_put_((u_slot, u_pos), u_new, accumulate=True)

        # Sparse: cấp entry mới cho token chưa có — vị trí = n_used + rank
        # trong nhóm cùng slot (sort stable), không đụng entry cũ.
        alloc = [
            i
            for i in range(len(pairs))
            if (not is_dense[i]) and (not f_list[i]) and int(n_list[i]) > 0
        ]
        if alloc:
            a_slot = torch.tensor(
                [int(s_list[i]) for i in alloc], dtype=torch.long, device=self._device
            )
            a_tok = torch.tensor(
                [int(t_list[i]) for i in alloc], dtype=torch.long, device=self._device
            )
            a_net = torch.tensor(
                [float(n_list[i]) for i in alloc],
                dtype=torch.float32,
                device=self._device,
            )
            order = torch.argsort(a_slot, stable=True)
            s_sorted = a_slot[order]
            boundary = torch.cat(
                [
                    torch.ones(1, dtype=torch.bool, device=self._device),
                    s_sorted[1:] != s_sorted[:-1],
                ]
            )
            idx = torch.arange(order.numel(), device=self._device)
            run_start = torch.cummax(
                torch.where(boundary, idx, -torch.ones_like(idx)), dim=0
            ).values
            rank = idx - run_start
            base = self._n_used.index_select(0, s_sorted)
            positions = base + rank
            self._cols.index_put_((s_sorted, positions), a_tok[order])
            self._cnts.index_put_((s_sorted, positions), a_net[order])
            self._n_used.index_add_(0, s_sorted, torch.ones_like(s_sorted, dtype=torch.int64))

        if dense_pairs:
            self._apply_dense_pairs(dense_pairs)

    def _apply_dense_pairs(self, dense_pairs: list[tuple[int, int, int]]) -> None:
        """Delta cho slot đã promote: new = old + net qua index_put_ thường
        (pair unique theo (slot, token) nên không cần accumulate)."""
        dense = self._ensure_dense_pool()
        ids = self._dense_slot_ids.tolist()
        d_rows = [ids.index(s) for s, _, _ in dense_pairs]
        d_idx = torch.tensor(d_rows, dtype=torch.long, device=self._device)
        toks = torch.tensor([t for _, t, _ in dense_pairs], dtype=torch.long, device=self._device)
        nets = torch.tensor(
            [float(n) for _, _, n in dense_pairs], dtype=torch.float32, device=self._device
        )
        old = dense[d_idx, toks]
        new = old + nets
        if bool((new < 0).any().item()):
            i = int(torch.nonzero(new < 0)[0])
            raise KeyError(
                f"unrecord token {int(toks[i])} ở slot {int(self._dense_slot_ids[d_idx[i]])} "
                "nhưng không đủ count trong bảng dense — khả năng cao là bug ở caller."
            )
        dense.index_put_((d_idx, toks), new)

    def _compact(self) -> None:
        """Compact per slot: entry count→0 bị loại, phần còn lại dồn về đầu.

        Tổng quát hoá swap-remove cho nhiều lần xoá trong cùng một flush —
        sort stable theo key "còn sống đứng trước" giữ thứ tự tương đối;
        toàn bộ trên device, không host sync."""
        b, cap = self._cols.shape
        if b == 0 or cap == 0:
            return
        j = torch.arange(cap, device=self._device).unsqueeze(0)
        keep = (j < self._n_used.unsqueeze(1)) & (self._cnts > 0)
        n_keep = keep.sum(dim=1)
        key = torch.where(keep, j, j + cap)
        order = key.sort(dim=1, stable=True).indices
        cols_new = self._cols.gather(1, order)
        cnts_new = self._cnts.gather(1, order)
        tail = j >= n_keep.unsqueeze(1)
        cols_new = torch.where(tail, torch.zeros_like(cols_new), cols_new)
        cnts_new = torch.where(tail, torch.zeros_like(cnts_new), cnts_new)
        self._cols.copy_(cols_new)
        self._cnts.copy_(cnts_new)
        self._n_used.copy_(n_keep)


def _canonicalize_rep_logsumexp(logits: torch.Tensor, rep_full: torch.Tensor) -> torch.Tensor:
    """logsumexp theo row cho các row có rep_penalty > 0, 0 cho row còn lại.

    A3 — MỘT lượt đọc/row: row_logsumexp (ops/fused_stats.py) nhận row_gate
    trực tiếp trên device — KHÔNG torch.nonzero (host sync ngầm) và KHÔNG
    index_select materialize bản copy [active × V] như bản cũ."""
    return row_logsumexp(logits, row_gate=rep_full)


def _apply_penalties_core(
    logits: torch.Tensor,
    md: SamplingMetadata,
    rows: torch.Tensor,
    cols: torch.Tensor,
    cnt: torch.Tensor,
    valid: torch.Tensor,
    dense_cnts: torch.Tensor | None,
    dense_slot_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Công thức penalty — MỘT bản duy nhất, dùng chung hai đường.

    rows/cols/cnt: [N] hoặc [B, S]; valid cùng shape. Padding (valid=False)
    có thể alias entry thật → ghi DELTA (new - old) qua
    index_put_(accumulate=True): cộng dồn có thứ tự cho index trùng, delta
    padding ép 0 — cộng 0 vào đâu cũng an toàn."""
    rep_full = md.active("rep_penalty")
    freq_full = md.active("freq_penalty")
    pres_full = md.active("pres_penalty")

    lse = _canonicalize_rep_logsumexp(logits, rep_full)

    rows_f = rows.reshape(-1)
    cols_f = cols.reshape(-1)
    cnt_f = cnt.reshape(-1).to(torch.float32)
    valid_f = valid.reshape(-1).to(torch.bool)

    rep_row = rep_full.index_select(0, rows_f)
    freq = freq_full.index_select(0, rows_f)
    pres = pres_full.index_select(0, rows_f)
    lse_row = lse.index_select(0, rows_f)

    old_vals = logits[rows_f, cols_f].to(torch.float32)
    shifted = old_vals - lse_row
    penalized = torch.where(shifted < 0, shifted * rep_row, shifted / rep_row)
    new_vals = torch.where(rep_row > 0, penalized + lse_row, old_vals)
    new_vals = new_vals - freq * cnt_f - pres * (cnt_f > 0).to(new_vals.dtype)

    # inf/nan có thể xuất hiện ở nhánh KHÔNG được chọn — an toàn vì
    # torch.where không lan NaN/Inf từ nhánh không chọn, và delta ép 0 ngay.
    delta = torch.where(valid_f, new_vals - old_vals, torch.zeros_like(new_vals))
    if rows_f.numel():
        logits.index_put_((rows_f, cols_f), delta.to(logits.dtype), accumulate=True)

    # A2 — dense branch: row đã promote áp trực tiếp trên toàn [V].
    if dense_cnts is not None and dense_slot_ids is not None:
        d = dense_slot_ids.size(0)
        n_rows = logits.size(0)
        if d and n_rows:
            act = (dense_slot_ids >= 0) & (dense_slot_ids < n_rows)
            sid = dense_slot_ids.clamp_min(0)
            rep = rep_full.index_select(0, sid) * act.to(rep_full.dtype)
            freq_d = freq_full.index_select(0, sid) * act.to(freq_full.dtype)
            pres_d = pres_full.index_select(0, sid) * act.to(pres_full.dtype)
            cnt_d = dense_cnts * act.to(dense_cnts.dtype).unsqueeze(1)
            v_old = logits.index_select(0, sid).to(torch.float32)
            lse_d = lse.index_select(0, sid).unsqueeze(1)
            shifted_d = v_old - lse_d
            penalized_d = torch.where(
                shifted_d < 0, shifted_d * rep.unsqueeze(1), shifted_d / rep.unsqueeze(1)
            )
            # rep branch CHỈ đụng token đã được đếm (cnt > 0) — khớp sparse:
            # coords chỉ tồn tại cho token có count.
            counted = (cnt_d > 0) & (rep.unsqueeze(1) > 0)
            v_new = torch.where(counted, penalized_d + lse_d, v_old)
            v_new = (
                v_new
                - freq_d.unsqueeze(1) * cnt_d
                - pres_d.unsqueeze(1) * (cnt_d > 0).to(v_new.dtype)
            )
            delta_d = torch.where(act.unsqueeze(1), v_new - v_old, torch.zeros_like(v_new))
            v_sz = logits.size(1)
            col_idx = (
                torch.arange(v_sz, device=logits.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(d, v_sz)
            )
            logits.index_put_(
                (sid.unsqueeze(1).expand(d, v_sz), col_idx),
                delta_d.to(logits.dtype),
                accumulate=True,
            )
    return logits


def apply_penalties_(
    logits: torch.Tensor, md: SamplingMetadata, state: PenaltyState
) -> torch.Tensor:
    """In-place. CHẠY TRƯỚC MASK — xem ghi chú NEG_INF trong ops/bitmask.py.

    rep: sign-branch áp trên log-softmax đã chuẩn hoá (xem docstring module).
    freq/pres: giữ nguyên additive/subtractive như bản gốc.

    Thân hàm dựng coords (1D compact, đã flush) rồi đi vào
    _apply_penalties_core — cùng đường với apply_penalties_padded_."""
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
    dense_cnts = state.dense_counts if state._dense_used else None
    dense_ids = state.dense_slot_ids if state._dense_used else None
    if rows.numel() == 0 and dense_cnts is None:
        return logits
    valid = torch.ones(rows.numel(), dtype=torch.bool, device=rows.device)
    return _apply_penalties_core(logits, md, rows, cols, cnt, valid, dense_cnts, dense_ids)


def apply_penalties_padded_(
    logits: torch.Tensor,
    md: SamplingMetadata,
    rows: torch.Tensor,
    cols: torch.Tensor,
    cnt: torch.Tensor,
    valid: torch.Tensor,
    *,
    dense_cnts: torch.Tensor | None = None,
    dense_slot_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Biến thể graph-capturable của apply_penalties_ — nhận trực tiếp output
    của PenaltyState.coords_padded() (2D [B, S] tĩnh) thay vì tự gọi coords()
    (coords() dynamic-shape, không được gọi trong graph).

    Slot đã promote sang dense KHÔNG xuất hiện trong coords_padded — caller
    PHẢI truyền dense_cnts/dense_slot_ids (cùng shape tĩnh) để phần dense
    được áp trong vùng capture; thiếu chúng thì penalty của row dense bị bỏ
    qua — với apply_penalties_ điều đó không thể xảy ra (nó luôn truyền).
    """
    return _apply_penalties_core(logits, md, rows, cols, cnt, valid, dense_cnts, dense_slot_ids)
