"""Bridge compiled GBNF to Ayaka's existing Tier-1 mask producer protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.sampling.mask.grammar import GrammarMaskProducer

from .gbnf import GrammarError
from .runtime import GrammarConstraint

if TYPE_CHECKING:
    from ayaka.sampling.mask.producer import MaskRows


class GrammarMatcher:
    """Mutable request-owned adapter; invalid accept leaves the state unchanged."""

    def __init__(self, state: GrammarConstraint) -> None:
        self.state = state.fork()
        self._consumed: list[bool] = []

    def accept_token(self, token: int) -> bool:
        if self.state.stopped() or not self.state.vocabulary.is_usable(token):
            return False
        if token in self.state.vocabulary.stops:
            self.state.mask()
        consumed = not (token in self.state.vocabulary.stops and self.state._matcher.is_stopped())
        try:
            candidate = self.state.advance((token,))
        except GrammarError as error:
            if error.kind != "violation":
                raise
            return False
        self.state = candidate
        self._consumed.append(consumed)
        return True

    def rollback(self, k: int) -> None:
        if not 0 <= k <= len(self._consumed):
            raise ValueError("rollback exceeds generated-token history")
        if not k:
            return
        state = self.state.fork()
        native_steps = sum(self._consumed[-k:])
        if native_steps and not state._matcher.rollback(native_steps):
            raise GrammarError("binding", "native matcher rollback failed")
        if state._matcher.is_error():
            raise GrammarError("binding", state._matcher.get_error())
        state.position -= k
        state._terminal = False
        state._mask = None
        self.state = state
        del self._consumed[-k:]

    def is_terminated(self) -> bool:
        return self.state.stopped()

    def num_accepted(self) -> int:
        return self.state.position

    def allowed_count_hint(self) -> int | None:
        return None

    def fill_bitmask(self, rows: MaskRows, i: int) -> None:
        width = self.state.vocabulary.projection
        if rows.vocab_size != width or rows.words < (width + 31) // 32:
            raise GrammarError("binding", "mask row has incompatible model projection")
        row = rows.raw(i)
        mask = self.state.mask()
        row[:] = 0
        row[: len(mask)] = mask
        rows.clear_padding(i)


class GBNFMaskProducer(GrammarMaskProducer):
    """Transactional grammar masks; runtime errors propagate instead of fabricating EOS.

    EOS is allowed only by the grammar, or after a previously consumed stop.
    This avoids silently returning incomplete structured output on parser errors.
    """

    def __init__(self, state: GrammarConstraint) -> None:
        super().__init__(GrammarMatcher(state), min(state.vocabulary.stops))

    def _fill_row(self, out: MaskRows, i: int) -> None:
        if self._m.is_terminated():
            out.allow_only(i, (self._eos,))
            return
        self._m.fill_bitmask(out, i)
        if not out.raw(i).any():
            raise GrammarError("binding", "grammar has no usable continuation in this vocabulary")
