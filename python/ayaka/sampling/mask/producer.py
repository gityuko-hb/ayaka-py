"""Tier-1 mask producer contract and bitmask row views.

This module defines the boundary between constraint logic (Tier-1) and the
sampling runtime (Tier-2):

* :class:`MaskRows` is a mutable, zero-copy view over a ``uint32`` bitmask
  buffer owned by :class:`~ayaka.sampling.mask.arena.BitmaskArena`. Each row
  holds one decode position; bit ``t`` of row ``i`` is set when token ``t``
  is allowed. Bits at and beyond ``vocab_size`` are padding and must stay
  zero.
* :class:`MaskProducer` is the protocol every constraint source must
  implement. Producers only clear bits, never add tokens, so combining
  producers is always a bitwise AND.

See :mod:`ayaka.caps` for the meaning of each :class:`~ayaka.caps.Cap`
flag advertised by producers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ayaka.caps import Cap

__all__ = ["Cap", "MaskRows", "MaskProducer"]


class MaskRows:
    """Mutable view over a 2-D ``uint32`` bitmask buffer.

    The buffer is owned by the arena; this object only wraps it. All
    mutating helpers operate on a single row index ``i``.

    Attributes:
        n_rows: Number of addressable rows in the view.
        words: Number of ``uint32`` words per row.
        vocab_size: Vocabulary size. Valid token ids are
            ``[0, vocab_size)``; higher bits are padding.
    """

    __slots__ = ("_buf", "_vocab_size")

    def __init__(self, buf: np.ndarray, vocab_size: int):
        """Wrap an existing bitmask buffer.

        Args:
            buf: 2-D ``uint32`` array of shape ``(n_rows, words)``.
            vocab_size: Vocabulary size used for bounds checks and
                padding handling.

        Raises:
            TypeError: If ``buf`` is not a 2-D ``uint32`` array.
        """
        if buf.ndim != 2 or buf.dtype != np.uint32:
            raise TypeError(f"MaskRows can buffer uint32 2 chieu, nhan {buf.dtype} ndim={buf.ndim}")
        self._buf = buf
        self._vocab_size = vocab_size

    @property
    def n_rows(self) -> int:
        """Return the number of rows in this view."""
        return int(self._buf.shape[0])

    @property
    def words(self) -> int:
        """Return the number of ``uint32`` words per row."""
        return int(self._buf.shape[1])

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size used for bounds checks."""
        return self._vocab_size

    def window(self, base: int, span: int) -> MaskRows:
        """Return a sub-view covering ``[base, base + span)`` rows.

        The window shares backing memory with this view, so writes
        through either handle are immediately visible to the other.

        Args:
            base: First row of the window.
            span: Number of rows in the window.

        Returns:
            A new :class:`MaskRows` sharing the same buffer and
            ``vocab_size``.

        Raises:
            IndexError: If the window is empty or extends past the buffer.
        """
        if base < 0 or span <= 0 or base + span > self._buf.shape[0]:
            raise IndexError(f"window({base}, {span}) ngoai arena {self._buf.shape[0]} row")
        return MaskRows(self._buf[base : base + span], self._vocab_size)

    def raw(self, i: int) -> np.ndarray:
        """Return a zero-copy view of row ``i``.

        Args:
            i: Row index. Must be within ``[0, n_rows)``; out-of-range
                access raises the underlying NumPy ``IndexError``.

        Returns:
            1-D ``uint32`` view of the requested row.
        """
        row: np.ndarray = self._buf[i]
        return row

    def allow_all(self, i: int) -> None:
        """Allow every in-vocabulary token in row ``i``.

        Sets all bits and then clears padding bits beyond ``vocab_size``.
        """
        self._buf[i] = np.uint32(0xFFFFFFFF)
        self.clear_padding(i)

    def deny_all(self, i: int) -> None:
        """Deny every token in row ``i``."""
        self._buf[i] = np.uint32(0)

    def allow_only(self, i: int, tokens) -> None:
        """Reset row ``i`` and allow exactly ``tokens``.

        Out-of-vocabulary and negative ids are silently filtered out. An
        empty (or fully filtered) input leaves the row as deny-all.

        Args:
            i: Row index to rewrite.
            tokens: Token ids to allow. Any sized sequence of ints or
                integer array is accepted.
        """
        row = self._buf[i]
        row[:] = 0
        if len(tokens) == 0:
            return
        idx = np.asarray(tokens, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < self._vocab_size)]
        if idx.size == 0:
            return
        np.bitwise_or.at(row, idx >> 5, (np.uint32(1) << (idx & 31).astype(np.uint32)))

    def allow(self, i: int, token: int) -> None:
        """Set the bit for ``token`` in row ``i``.

        Out-of-vocabulary ids are ignored (no-op).
        """
        if 0 <= token < self._vocab_size:
            self._buf[i, token >> 5] |= np.uint32(1) << np.uint32(token & 31)

    def deny(self, i: int, token: int) -> None:
        """Clear the bit for ``token`` in row ``i``.

        Out-of-vocabulary ids are ignored (no-op).
        """
        if 0 <= token < self._vocab_size:
            self._buf[i, token >> 5] &= ~(np.uint32(1) << np.uint32(token & 31))

    def allows(self, i: int, token: int) -> bool:
        """Check whether row ``i`` allows ``token``.

        Args:
            i: Row index.
            token: Token id to test.

        Returns:
            True when ``token`` is in vocabulary and its bit is set;
            False otherwise (including out-of-vocabulary ids).
        """
        if not 0 <= token < self._vocab_size:
            return False
        return bool(self._buf[i, token >> 5] & (np.uint32(1) << np.uint32(token & 31)))

    def intersect(self, i: int, other: np.ndarray) -> None:
        """Intersect row ``i`` in place with ``other`` (bitwise AND).

        Used to combine the masks of several producers sharing one
        logits row.

        Args:
            i: Row index to update.
            other: Bitmask row broadcastable against one row of the
                internal buffer.
        """
        self._buf[i] &= other

    def clear_padding(self, i: int) -> None:
        """Zero padding bits at and beyond ``vocab_size`` in row ``i``.

        Words fully past the vocabulary are zeroed; the partially used
        tail word keeps only its ``vocab_size & 31`` low bits. This keeps
        the sampler from ever selecting a non-existent token id.
        """
        v = self._vocab_size
        last = v >> 5
        if last >= self.words:
            return
        rem = v & 31
        if rem:
            self._buf[i, last] &= np.uint32((1 << rem) - 1)
            last += 1
        if last < self.words:
            self._buf[i, last:] = 0


@runtime_checkable
class MaskProducer(Protocol):
    """Protocol for Tier-1 constraint sources.

    Attributes:
        caps: Capability flags (see :mod:`ayaka.caps`) describing what
            the planner and sampler may assume about this producer.
    """

    caps: Cap

    def density_hint(self) -> int | None:
        """Estimate how many tokens the current position allows.

        Returns:
            Allowed-token count, or None when the producer cannot
            estimate it cheaply. Used for scheduling heuristics only.
        """
        ...

    def emit(self, draft: Sequence[int], out: MaskRows) -> int:
        """Write ``len(draft) + 1`` mask rows and verify the draft.

        Args:
            draft: Speculative token ids to verify, one per row except
                the last (bonus) row.
            out: Destination window with exactly ``len(draft) + 1`` rows.

        Returns:
            Number of leading draft tokens accepted. The producer must
            roll back any speculative state for the accepted prefix, so
            the caller never needs cross-producer rollback.
        """
        ...

    def commit(self, tokens: Sequence[int]) -> None:
        """Advance producer state with already-sampled tokens.

        Args:
            tokens: Tokens to commit in order.

        Raises:
            ValueError: If a token violates the constraint or arrives
                after the constraint terminated.
        """
        ...

    def committed_len(self) -> int:
        """Return how many tokens have been committed so far."""
        ...
