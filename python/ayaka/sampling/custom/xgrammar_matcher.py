"""XGrammarMatcher — adapter từ xgrammar.GrammarMatcher sang hợp đồng
ConstraintMatcher của Tier-1 mask pipeline (ayaka.sampling.mask.grammar).

Hợp đồng matcher (protocol, xem grammar.py):
  * accept_token(token) -> bool
  * rollback(k)
  * is_terminated() -> bool
  * fill_bitmask(rows: MaskRows, i) -> None
  * num_accepted() -> int
  * allowed_count_hint() -> int | None

XGrammar phát bitmask int32 với bit=1 = token ALLOWED — trùng convention
của apply_allow_bitmask_ (bit=1 giữ nguyên, bit=0 → NEG_INF), nên buffer
int32 đọc lại bằng view uint32 là bit-identical với MaskRows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:  # pragma: no cover - environment dependent
    import xgrammar as _xgr  # type: ignore[import-not-found]

    _HAS_XGRAMMAR = True
except Exception:  # pragma: no cover
    _xgr = None
    _HAS_XGRAMMAR = False

from ayaka.sampling.mask.producer import MaskRows

__all__ = ["XGrammarMatcher", "has_xgrammar"]


def has_xgrammar() -> bool:
    return _HAS_XGRAMMAR


class XGrammarMatcher:
    """Bọc xgrammar.GrammarMatcher — matcher per request, không thread-safe."""

    def __init__(self, compiled_grammar: Any, *, vocab_size: int) -> None:
        if not _HAS_XGRAMMAR or _xgr is None:
            raise ImportError("xgrammar chưa được cài — thêm xgrammar vào dev group")
        self._grammar = compiled_grammar
        self._m = _xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size = vocab_size
        self._accepted = 0

    def accept_token(self, token: int) -> bool:
        accepted = bool(self._m.accept_token(int(token)))
        if accepted:
            self._accepted += 1
        return accepted

    def rollback(self, k: int) -> None:
        if k <= 0:
            return
        self._m.rollback(int(k))
        self._accepted = max(0, self._accepted - int(k))

    def is_terminated(self) -> bool:
        return bool(self._m.is_terminated())

    def num_accepted(self) -> int:
        return self._accepted

    def allowed_count_hint(self) -> int | None:
        return None

    def fill_bitmask(self, rows: MaskRows, i: int) -> None:
        """Điền bitmask cho trạng thái hiện tại vào row `i` của MaskRows."""
        words = rows.words
        # fill_next_token_bitmask nhận torch tensor int32 [batch, words];
        # bit=1 = allowed — trùng convention MaskRows nên view uint32 là
        # bit-identical.
        tmp = torch.zeros(1, words, dtype=torch.int32)
        self._m.fill_next_token_bitmask(tmp, 0)
        row: Any = rows.raw(i)
        row[:] = tmp[0].numpy().view(np.uint32)
        vocab = self._vocab_size
        last = vocab >> 5
        if last < words and vocab & 31:
            row[last] &= np.uint32((1 << (vocab & 31)) - 1)
        if last + 1 < words:
            row[last + 1 :] = 0
