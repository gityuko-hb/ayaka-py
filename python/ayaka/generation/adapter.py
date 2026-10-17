"""Live paged-runner integration of generation controls and staged constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.generation.contracts import Constraint
from ayaka.generation.controls import Controlled, EndOfGeneration

if TYPE_CHECKING:
    import torch

    from ayaka.request.schema import Request


class GenerationControls:
    """One control state per request, advanced from committed publication only.

    Repeated sampling attempts at the same boundary do not advance the budget.
    A failed forward, replay, preemption or discarded speculative row cannot
    spend reasoning tokens. Masks intersect the runner's existing grammar/bans.
    """

    def __init__(self, vocabulary: int) -> None:
        self.vocabulary = vocabulary
        self._states: dict[str, tuple[Constraint, tuple[int, ...]]] = {}

    def register(self, request: Request) -> None:
        request_id = str(request.request_id)
        if request_id in self._states:
            raise ValueError("duplicate generation controls request")
        constraint = Controlled.compose(
            None,
            EndOfGeneration.STOP,
            frozenset(),
            frozenset(request.suppressed_token_ids),
            request.reasoning_budget,
            self.vocabulary,
        )
        if constraint is not None:
            self._states[request_id] = constraint, ()

    def forget(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def contains(self, request_id: str) -> bool:
        return request_id in self._states

    def selection_mask(self, request_id: str, published: tuple[int, ...]) -> tuple[int, ...]:
        state, committed = self._states[request_id]
        if published[: len(committed)] != committed:
            raise ValueError("generation control publication history changed")
        successor = state.stage(published[len(committed) :]) if published != committed else state
        mask = successor.mask()
        self._states[request_id] = successor, published
        return mask

    def apply(self, request_id: str, published: tuple[int, ...], logits: torch.Tensor) -> None:
        import torch

        mask = self.selection_mask(request_id, published)
        if logits.ndim != 1 or logits.numel() != self.vocabulary:
            raise ValueError("generation controls need one full vocabulary row")
        words = torch.tensor(mask, dtype=torch.int64, device=logits.device)
        ids = torch.arange(self.vocabulary, device=logits.device)
        allowed = ((words[ids // 32] >> (ids % 32)) & 1).bool()
        logits.masked_fill_(~allowed, float("-inf"))
        if not bool(torch.isfinite(logits).any()):
            raise ValueError("generation controls and sampling constraints leave no valid token")


class ConstraintProducer:
    """Adapt a staged Constraint to Ayaka's MaskProducer protocol.

    The core port has no NumPy/Torch dependency. Import the mask protocol only
    when emitting into a live mask arena. Preview never commits draft tokens.
    """

    def __init__(self, constraint: Constraint) -> None:
        from ayaka.caps import Cap

        self.caps = Cap.ARGMAX_INVARIANT | Cap.SPEC_VERIFIABLE | Cap.COMMUTATIVE
        self._state = constraint

    def density_hint(self) -> int:
        return sum(word.bit_count() for word in self._state.mask())

    def emit(self, draft, out) -> int:
        if out.n_rows != len(draft) + 1:
            raise ValueError("constraint producer needs draft rows plus one bonus row")
        preview, accepted, poisoned = self._state.fork(), 0, False
        for row in range(len(draft) + 1):
            mask = preview.mask()
            if len(mask) != out.words:
                raise ValueError("constraint mask width disagrees with mask arena")
            out.raw(row)[:] = mask
            out.clear_padding(row)
            if row < len(draft) and not poisoned:
                try:
                    successor = preview.stage((draft[row],))
                except ValueError:
                    poisoned = True
                else:
                    if successor.position() != preview.position() + 1:
                        raise ValueError("constraint preview advanced to an invalid position")
                    preview, accepted = successor, accepted + 1
        return accepted

    def commit(self, tokens) -> None:
        self._state = self._state.stage(tuple(tokens))

    def committed_len(self) -> int:
        return self._state.position()
