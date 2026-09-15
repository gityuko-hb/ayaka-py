"""Hop dong cua Tier 1 -- mask producer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ayaka.caps import Cap

__all__ = ["Cap", "MaskRows", "MaskProducer"]


class MaskRows:
    __slots__ = ("_buf", "_vocab_size")

    def __init__(self, buf: np.ndarray, vocab_size: int):
        if buf.ndim != 2 or buf.dtype != np.uint32:
            raise TypeError(f"MaskRows can buffer uint32 2 chieu, nhan {buf.dtype} ndim={buf.ndim}")
        self._buf = buf
        self._vocab_size = vocab_size

    @property
    def n_rows(self) -> int:
        return int(self._buf.shape[0])

    @property
    def words(self) -> int:
        return int(self._buf.shape[1])

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def window(self, base: int, span: int) -> MaskRows:
        if base < 0 or span <= 0 or base + span > self._buf.shape[0]:
            raise IndexError(f"window({base}, {span}) ngoai arena {self._buf.shape[0]} row")
        return MaskRows(self._buf[base : base + span], self._vocab_size)

    def raw(self, i: int) -> np.ndarray:
        row: np.ndarray = self._buf[i]
        return row

    def allow_all(self, i: int) -> None:
        self._buf[i] = np.uint32(0xFFFFFFFF)
        self.clear_padding(i)

    def deny_all(self, i: int) -> None:
        self._buf[i] = np.uint32(0)

    def allow_only(self, i: int, tokens) -> None:
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
        if 0 <= token < self._vocab_size:
            self._buf[i, token >> 5] |= np.uint32(1) << np.uint32(token & 31)

    def deny(self, i: int, token: int) -> None:
        if 0 <= token < self._vocab_size:
            self._buf[i, token >> 5] &= ~(np.uint32(1) << np.uint32(token & 31))

    def allows(self, i: int, token: int) -> bool:
        if not 0 <= token < self._vocab_size:
            return False
        return bool(self._buf[i, token >> 5] & (np.uint32(1) << np.uint32(token & 31)))

    def intersect(self, i: int, other: np.ndarray) -> None:
        self._buf[i] &= other

    def clear_padding(self, i: int) -> None:
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
    caps: Cap

    def density_hint(self) -> int | None: ...
    def emit(self, draft: Sequence[int], out: MaskRows) -> int: ...
    def commit(self, tokens: Sequence[int]) -> None: ...
    def committed_len(self) -> int: ...
