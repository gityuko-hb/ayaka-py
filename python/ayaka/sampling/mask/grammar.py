"""GrammarMaskProducer -- Tier 1 producer boc bat ky ConstraintMatcher nao."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ayaka.sampling.mask.producer import Cap, MaskRows


@runtime_checkable
class ConstraintMatcher(Protocol):
    def accept_token(self, token: int) -> bool: ...
    def rollback(self, k: int) -> None: ...
    def is_terminated(self) -> bool: ...
    def fill_bitmask(self, rows: MaskRows, i: int) -> None: ...
    def num_accepted(self) -> int: ...
    def allowed_count_hint(self) -> int | None: ...


class GrammarMaskProducer:
    caps = Cap.SPEC_VERIFIABLE | Cap.CUDAGRAPH_SAFE

    __slots__ = ("_committed", "_eos", "_m")

    def __init__(self, matcher: ConstraintMatcher, eos_token_id: int):
        if eos_token_id < 0:
            raise ValueError(f"eos_token_id phai >= 0, nhan {eos_token_id}.")
        self._m = matcher
        self._eos = eos_token_id
        self._committed = 0

    def density_hint(self) -> int | None:
        return self._m.allowed_count_hint()

    def committed_len(self) -> int:
        return self._committed

    def _fill_row(self, out: MaskRows, i: int) -> None:
        if self._m.is_terminated():
            out.allow_only(i, (self._eos,))
            return
        try:
            self._m.fill_bitmask(out, i)
        except Exception:
            out.allow_only(i, (self._eos,))
            return
        out.clear_padding(i)
        if not out.raw(i).any():
            out.allow_only(i, (self._eos,))

    def emit(self, draft: Sequence[int], out: MaskRows) -> int:
        if out.n_rows != len(draft) + 1:
            raise ValueError(f"can {len(draft) + 1} row, arena cap {out.n_rows}")

        accepted = 0
        try:
            for i in range(out.n_rows):
                self._fill_row(out, i)
                if i >= len(draft):
                    break
                tok = draft[i]
                if not out.allows(i, tok) or not self._m.accept_token(tok):
                    for j in range(i + 1, out.n_rows):
                        out.allow_only(j, (self._eos,))
                    break
                accepted += 1
        finally:
            if accepted:
                self._m.rollback(accepted)
        return accepted

    def commit(self, tokens: Sequence[int]) -> None:
        n = 0
        try:
            for tok in tokens:
                if self._m.is_terminated():
                    raise ValueError("commit token sau khi matcher da terminate")
                if not self._m.accept_token(tok):
                    raise ValueError(f"grammar tu choi token da commit: {tok}")
                n += 1
        except Exception:
            if n:
                self._m.rollback(n)
            raise
        self._committed += n
