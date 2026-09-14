from __future__ import annotations

import torch

NEG_INF = float("-inf")

try:  # pragma: no cover - environment dependent
    import xgrammar as _xgr  # type: ignore[import-not-found]

    _HAS_XGRAMMAR = True
except Exception:  # pragma: no cover
    _xgr = None
    _HAS_XGRAMMAR = False


def has_xgrammar_kernel() -> bool:
    return _HAS_XGRAMMAR


def rows_fully_masked(mask: torch.Tensor) -> torch.Tensor:
    """[n_rows] bool. O(n_rows × W), không phải O(n_rows × V)."""
    return ~(mask != 0).any(dim=1)


def apply_allow_bitmask_(
    logits: torch.Tensor,
    mask: torch.Tensor,
    row_indices: torch.Tensor,
    vocab_size: int,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """In-place. bit=1 giữ nguyên, bit=0 → NEG_INF.

    `row_indices` là compact-row gather: mask row k ràng buộc logits row
    row_indices[k]. Nhờ nó mà 64 stream grammar là MỘT lần apply chứ không phải
    64 lần tuần tự — đúng thứ rtp-llm không làm được vì khoá cứng batch_size==1.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits phải 2 chiều, nhận {logits.dim()}")
    if mask.dim() != 2 or mask.size(0) != row_indices.size(0):
        raise ValueError("mask phải [n_rows, W] và khớp row_indices")
    if validate:
        bad = rows_fully_masked(mask)
        if bool(bad.any()):
            idx = int(torch.nonzero(bad)[0])
            raise ValueError(
                f"mask row {idx} không cho phép token nào. Producer phải fail-closed "
                "sang EOS; để nguyên sẽ ra NaN sau softmax."
            )

    if _HAS_XGRAMMAR and _xgr is not None and logits.is_cuda:  # pragma: no cover
        _xgr.apply_token_bitmask_inplace(
            logits, mask, vocab_size=vocab_size, indices=row_indices.tolist()
        )
        return logits

    return _apply_reference_(logits, mask, row_indices, vocab_size)


def _apply_reference_(
    logits: torch.Tensor,
    mask: torch.Tensor,
    row_indices: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Đường tham chiếu bằng torch thuần.

    Luôn phải tồn tại và luôn phải được test SONG SONG với đường kernel — nó là
    oracle. Hai backend sampling độc lập không có oracle chung là cách bug trốn
    (xem: TensorRT-LLM có TRT backend và PyTorch backend làm cùng việc).
    """
    device = logits.device
    tok = torch.arange(vocab_size, device=device)
    word_idx = torch.div(tok, 32, rounding_mode="floor")  # [V]
    bit_idx = (tok % 32).to(torch.int32)  # [V]
    sel = mask.index_select(1, word_idx)  # [n_rows, V]
    allowed = ((sel >> bit_idx) & 1).to(torch.bool)  # [n_rows, V]

    rows = row_indices.to(torch.long)
    target = logits.index_select(0, rows)
    target = torch.where(allowed, target, torch.full_like(target, NEG_INF))
    logits.index_copy_(0, rows, target)
    return logits


def bitmask_to_allowed_ids(mask_row: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Giải nén một row thành danh sách token id. Dùng cho test và cho đường
    sparse (chưa nối vào sampler — xem README)."""
    tok = torch.arange(vocab_size, device=mask_row.device)
    word = mask_row.index_select(0, torch.div(tok, 32, rounding_mode="floor"))
    bit = (tok % 32).to(torch.int32)
    return tok[((word >> bit) & 1).to(torch.bool)]
