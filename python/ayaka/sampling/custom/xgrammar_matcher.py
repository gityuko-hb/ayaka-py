"""Adapter bridging `xgrammar.GrammarMatcher` to the sampling constraint matcher protocol.

Provides an implementation of `ConstraintMatcher` defined in `ayaka.sampling.mask.grammar`,
enabling structured and grammar-constrained decoding via the XGrammar library.

Matcher Protocol Methods:
    * `accept_token(token) -> bool`: Advances matcher state with candidate token.
    * `rollback(k)`: Rewinds matcher state by `k` steps for speculative decoding.
    * `is_terminated() -> bool`: Indicates whether grammar has reached a terminal state.
    * `fill_bitmask(rows, i)`: Writes bitmask for current valid transitions into slot `i`.
    * `num_accepted() -> int`: Number of committed tokens accepted so far.
    * `allowed_count_hint() -> int | None`: Optional hint on allowed token cardinality.

Bitmask Alignment:
    XGrammar emits int32 bitmasks where bit=1 represents an allowed token. This matches
    the bitmask convention of `apply_allow_bitmask_` (bit=1 retained, bit=0 masked to -inf),
    allowing zero-copy bit-identical reinterpretation via uint32 views in `MaskRows`.
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
    """Check whether the xgrammar runtime dependency is available in the environment."""
    return _HAS_XGRAMMAR


class XGrammarMatcher:
    """Wrapper around `xgrammar.GrammarMatcher` adhering to the `ConstraintMatcher` protocol.

    Instantiated per request stream. Instances are stateful and not thread-safe.
    """

    def __init__(self, compiled_grammar: Any, *, vocab_size: int) -> None:
        if not _HAS_XGRAMMAR or _xgr is None:
            raise ImportError("xgrammar chưa được cài — thêm xgrammar vào dev group")
        self._grammar = compiled_grammar
        self._m = _xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size = vocab_size
        self._accepted = 0

    def accept_token(self, token: int) -> bool:
        """Advance matcher state with an emitted token.

        Args:
            token: Emitted token id to validate and accept.

        Returns:
            True if the token was legally accepted by the grammar, False otherwise.
        """
        accepted = bool(self._m.accept_token(int(token)))
        if accepted:
            self._accepted += 1
        return accepted

    def rollback(self, k: int) -> None:
        """Rewind matcher state by `k` steps for speculative decoding recovery.

        Args:
            k: Number of accepted steps to roll back.
        """
        if k <= 0:
            return
        self._m.rollback(int(k))
        self._accepted = max(0, self._accepted - int(k))

    def is_terminated(self) -> bool:
        """Check whether the grammar has reached an accepting terminal state."""
        return bool(self._m.is_terminated())

    def num_accepted(self) -> int:
        """Return the number of accepted tokens committed to the matcher."""
        return self._accepted

    def allowed_count_hint(self) -> int | None:
        """Return optional cardinality hint of allowed tokens for current state."""
        return None

    def fill_bitmask(self, rows: MaskRows, i: int) -> None:
        """Populate the bitmask of allowed tokens for the current state into row `i` of MaskRows.

        Args:
            rows: Destination `MaskRows` buffer storage.
            i: Target row index in the mask buffer.
        """
        words = rows.words
        # fill_next_token_bitmask expects an int32 tensor [batch_size, words] where
        # bit=1 denotes allowed tokens. Viewing this as uint32 matches MaskRows layout.
        tmp = torch.zeros(1, words, dtype=torch.int32)
        self._m.fill_next_token_bitmask(tmp, 0)
        row: Any = rows.raw(i)
        row[:] = tmp[0].numpy().view(np.uint32)
        # Mask out any out-of-vocabulary bits in the final word.
        vocab = self._vocab_size
        last = vocab >> 5
        if last < words and vocab & 31:
            row[last] &= np.uint32((1 << (vocab & 31)) - 1)
        # Clear any trailing padding words beyond the vocabulary limit.
        if last + 1 < words:
            row[last + 1 :] = 0
