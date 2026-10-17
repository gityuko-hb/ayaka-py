"""Transactional grammar composition, token suppression and reasoning limits."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum

from ayaka.generation.contracts import Constraint, bounded, checked_add


class EndOfGeneration(StrEnum):
    STOP = "stop"
    SUPPRESS = "suppress"


@dataclass(frozen=True, slots=True)
class ReasoningBudget:
    """Count tokens inside start/end tags, then force the remaining end tag."""

    tokens: int
    start: tuple[int, ...]
    end: tuple[int, ...]
    open: bool = False

    def __post_init__(self) -> None:
        bounded(self.tokens, "reasoning tokens", (1 << 32) - 1, 1)
        object.__setattr__(self, "start", tuple(self.start))
        object.__setattr__(self, "end", tuple(self.end))
        if not self.start or not self.end or type(self.open) is not bool:
            raise ValueError("reasoning budget needs nonempty tags and a boolean open flag")
        for token in self.start + self.end:
            bounded(token, "reasoning tag token", (1 << 32) - 1)

    def validate(self, vocabulary: int) -> None:
        if any(token >= vocabulary for token in self.start + self.end):
            raise ValueError("reasoning budget requires in-vocabulary tags")


class _Budget:
    def __init__(self, spec: ReasoningBudget) -> None:
        self.spec = spec
        self.phase = "reasoning" if spec.open else "before"
        self.count = 0
        self.recent: tuple[int, ...] = ()

    def forced_token(self) -> int | None:
        return self.spec.end[self.count] if self.phase == "closing" else None

    def accept(self, token: int) -> None:
        self.recent = (*self.recent, token)[-max(len(self.spec.start), len(self.spec.end)) :]
        if self.phase == "before":
            if self.recent[-len(self.spec.start) :] == self.spec.start:
                self.phase, self.count = "reasoning", 0
        elif self.phase == "reasoning":
            if self.recent[-len(self.spec.end) :] == self.spec.end:
                self.phase = "after"
            else:
                self.count += 1
                if self.count >= self.spec.tokens:
                    self.phase = "closing"
                    self.count = next(
                        (
                            length
                            for length in range(len(self.spec.end) - 1, 0, -1)
                            if self.recent[-length:] == self.spec.end[:length]
                        ),
                        0,
                    )
        elif self.phase == "closing":
            if token != self.spec.end[self.count]:
                raise ValueError("token violates forced reasoning closure")
            self.count += 1
            if self.count == len(self.spec.end):
                self.phase = "after"

    def admits(self, tokens: tuple[int, ...]) -> int:
        preview = copy.copy(self)
        for index, token in enumerate(tokens):
            if preview.forced_token() is not None:
                return index
            preview.accept(token)
        return len(tokens)


class Controlled:
    """Request-local controls composed with an optional staged grammar.

    Masks are packed uint32 words. Forced closure intersects the grammar too:
    an impossible grammar/budget combination fails before accepting any token.
    """

    def __init__(
        self,
        inner: Constraint | None,
        suppressed: frozenset[int],
        budget: _Budget | None,
        vocabulary: int,
        position: int = 0,
    ) -> None:
        self.inner = inner
        self.suppressed = suppressed
        self.budget = budget
        self.vocabulary = vocabulary
        self._position = position

    @classmethod
    def compose(
        cls,
        inner: Constraint | None,
        end_of_generation: EndOfGeneration,
        stop_tokens: frozenset[int],
        never: frozenset[int],
        budget: ReasoningBudget | None,
        vocabulary: int,
    ) -> Constraint | None:
        bounded(vocabulary, "vocabulary", (1 << 31) - 1, 1)
        suppressed = frozenset(never) | (
            frozenset(stop_tokens) if end_of_generation is EndOfGeneration.SUPPRESS else frozenset()
        )
        for token in suppressed:
            bounded(token, "suppressed token", vocabulary - 1)
        if not suppressed and budget is None:
            return inner
        if budget is not None:
            budget.validate(vocabulary)
            if any(token in suppressed for token in budget.end):
                raise ValueError("reasoning end tag cannot use a suppressed token")
        return cls(
            inner,
            suppressed,
            None if budget is None else _Budget(budget),
            vocabulary,
            0 if inner is None else inner.position(),
        )

    def position(self) -> int:
        return self._position

    def fork(self) -> Controlled:
        return Controlled(
            None if self.inner is None else self.inner.fork(),
            self.suppressed,
            copy.copy(self.budget),
            self.vocabulary,
            self._position,
        )

    def stage(self, tokens: tuple[int, ...]) -> Controlled:
        successor = self.fork()
        for token in tokens:
            bounded(token, "selected token", self.vocabulary - 1)
            if token in self.suppressed:
                raise ValueError("suppressed token was selected")
            if successor.budget is not None:
                successor.budget.accept(token)
        if successor.inner is not None:
            successor.inner = successor.inner.stage(tokens)
            if successor.inner.position() != self._position + len(tokens):
                raise ValueError("inner constraint successor has incorrect position")
        successor._position = checked_add(self._position, len(tokens))
        return successor

    def forced(self, limit: int) -> tuple[int, ...]:
        bounded(limit, "forced-token allowance", minimum=1)
        if self.budget is not None and self.budget.phase == "closing":
            forced = self.budget.spec.end[self.budget.count : self.budget.count + limit]
            if self.inner is not None:
                self.inner.stage(forced)
            return forced
        forced = () if self.inner is None else tuple(self.inner.forced(limit))
        for index, token in enumerate(forced):
            if token in self.suppressed:
                forced = forced[:index]
                break
        if self.budget is not None:
            forced = forced[: self.budget.admits(forced)]
        return forced

    def mask(self) -> tuple[int, ...]:
        mask = (
            list(self.inner.mask())
            if self.inner is not None
            else [0xFFFFFFFF] * ((self.vocabulary + 31) // 32)
        )
        if len(mask) != (self.vocabulary + 31) // 32:
            raise ValueError("constraint mask has incorrect vocabulary width")
        if self.vocabulary % 32:
            mask[-1] &= (1 << (self.vocabulary % 32)) - 1
        if self.budget is not None and (token := self.budget.forced_token()) is not None:
            allowed = bool(mask[token // 32] & (1 << (token % 32)))
            mask = [0] * len(mask)
            if allowed:
                mask[token // 32] = 1 << (token % 32)
        for token in self.suppressed:
            mask[token // 32] &= ~(1 << (token % 32))
        if not any(mask):
            raise ValueError("generation controls and grammar leave no valid token")
        return tuple(mask)
