"""Grammar-guided mask producer wrapping any constraint matcher.

:class:`GrammarMaskProducer` adapts a stateful ``ConstraintMatcher`` (for
example an XGrammar matcher or :class:`~ayaka.sampling.mask.trie.TrieMatcher`)
to the :class:`~ayaka.sampling.mask.producer.MaskProducer` protocol. The
producer is fail-closed: whenever the constraint is terminated, the backend
fails, or a row would otherwise be empty, the row allows only the EOS token.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ayaka.sampling.mask.producer import Cap, MaskRows


@runtime_checkable
class ConstraintMatcher(Protocol):
    """Protocol for backends that can enumerate allowed tokens.

    Implementations hold their own parse state; this producer drives exactly
    one matcher instance and never shares it across threads.
    """

    def accept_token(self, token: int) -> bool:
        """Consume ``token`` if it is a valid continuation.

        Returns:
            True on success (state advances); False without changing state.
        """
        ...

    def rollback(self, k: int) -> None:
        """Undo the last ``k`` accepted tokens."""
        ...

    def is_terminated(self) -> bool:
        """Check whether the constraint accepts no further tokens."""
        ...

    def fill_bitmask(self, rows: MaskRows, i: int) -> None:
        """Write the current allow-list into row ``i`` of ``rows``."""
        ...

    def num_accepted(self) -> int:
        """Return the number of currently accepted tokens."""
        ...

    def allowed_count_hint(self) -> int | None:
        """Estimate allowed tokens, or None when unavailable."""
        ...


class GrammarMaskProducer:
    """Tier-1 producer that emits grammar masks from a matcher.

    Attributes:
        caps: Always ``SPEC_VERIFIABLE | CUDAGRAPH_SAFE | ARGMAX_INVARIANT``.
            ``ARGMAX_INVARIANT`` holds because the mask only removes tokens:
            when the unmasked argmax winner is still allowed, masking cannot
            change the argmax. The sampler relies on this for its greedy
            fast path (argmax first, full mask apply only when the winner
            is blocked).
    """

    caps = Cap.SPEC_VERIFIABLE | Cap.CUDAGRAPH_SAFE | Cap.ARGMAX_INVARIANT

    __slots__ = ("_committed", "_eos", "_m")

    def __init__(self, matcher: ConstraintMatcher, eos_token_id: int):
        """Create a producer over an existing matcher.

        Args:
            matcher: Stateful constraint backend. Ownership stays with the
                caller, but the producer must have exclusive access.
            eos_token_id: Fallback token allowed when the constraint is
                terminated, fails, or would yield an empty row.

        Raises:
            ValueError: If ``eos_token_id`` is negative.
        """
        if eos_token_id < 0:
            raise ValueError(f"eos_token_id phai >= 0, nhan {eos_token_id}.")
        self._m = matcher
        self._eos = eos_token_id
        self._committed = 0

    def density_hint(self) -> int | None:
        """Forward the matcher's allowed-count hint.

        Returns:
            Allowed-token estimate, or None when the matcher cannot
            provide one cheaply.
        """
        return self._m.allowed_count_hint()

    def committed_len(self) -> int:
        """Return how many tokens have been committed via :meth:`commit`."""
        return self._committed

    def _fill_row(self, out: MaskRows, i: int) -> None:
        """Fill row ``i`` with the current allow-list, fail-closed to EOS.

        The row allows only EOS when the matcher is terminated, when
        ``fill_bitmask`` raises, or when the filled row is empty. Padding
        bits beyond the vocabulary are always cleared.

        Args:
            out: Destination bitmask view.
            i: Row index to fill.
        """
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
        """Write ``len(draft) + 1`` rows and verify the draft tokens.

        Each row is filled from the current matcher state, then the
        corresponding draft token is tested both against the freshly
        written bit and against ``accept_token``. The first rejection
        poisons all later rows to EOS-only. Accepted-prefix state is
        rolled back before returning, so speculative verification never
        leaks into the committed parse state.

        Args:
            draft: Speculative token ids, one per row except the final
                (bonus) row.
            out: Destination window with exactly ``len(draft) + 1`` rows.

        Returns:
            Number of leading draft tokens accepted.

        Raises:
            ValueError: If ``out`` does not hold exactly
                ``len(draft) + 1`` rows.
        """
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
        """Advance the matcher with already-sampled tokens.

        The commit is atomic: on any failure the already-advanced prefix
        is rolled back and ``_committed`` is left unchanged.

        Args:
            tokens: Tokens to commit in order.

        Raises:
            ValueError: If a token arrives after termination or the
                matcher rejects an already-sampled token.
        """
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
