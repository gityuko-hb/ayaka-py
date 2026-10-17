"""Measured generated/verified width control and shared target row allocation."""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass, field
from enum import StrEnum

from ayaka.generation.contracts import bounded

MEMORY = 0.25
COST_MEMORY = 0.25
LOOK_SHARE = 0.01
U32_MAX = (1 << 32) - 1


def power_of_two(value: int) -> int:
    return 1 << (max(value, 1) - 1).bit_length()


def draft_ladder(widest: int) -> tuple[int, ...]:
    """Prepared widths: powers of two and the exact configured ceiling."""
    bounded(widest, "widest proposals", 255, 1)
    return tuple(sorted({widest, *(1 << i for i in range(widest.bit_length()))}))


class TimedLaunch(StrEnum):
    DRAFT = "draft"
    TARGET = "target"


@dataclass(slots=True)
class LaunchCosts:
    draft: dict[int, float] = field(default_factory=dict)
    verify: dict[int, float] = field(default_factory=dict)

    @staticmethod
    def record(costs: dict[int, float], size: int, seconds: float) -> None:
        previous = costs.get(size)
        costs[size] = seconds if previous is None else previous + COST_MEMORY * (seconds - previous)


@dataclass(slots=True)
class DraftLoad:
    """Measurements shared by a load; transition copies keep this one owner."""

    costs: dict[tuple[int, int], LaunchCosts] = field(default_factory=dict)
    right: list[float] | None = None
    look_run: int | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)


class _RoundCosts:
    def __init__(self, load: DraftLoad, context: int, sequences: int) -> None:
        self.sequences = sequences
        with load.lock:
            self.own = copy.deepcopy(load.costs.get((context, sequences), LaunchCosts()))
            self.lesser = [
                copy.deepcopy(cost)
                for (c, s), cost in load.costs.items()
                if c <= context and s <= sequences and (c, s) != (context, sequences)
            ]

    def seconds(self, table: str, size: int) -> float:
        exact = getattr(self.own, table).get(size)
        if exact is not None:
            return exact
        return max(
            (
                seconds
                for measured in (self.own, *self.lesser)
                for width, seconds in getattr(measured, table).items()
                if width <= size
            ),
            default=0.0,
        )

    def draft(self, generated: int) -> float:
        return self.seconds("draft", generated) if generated else 0.0

    def verify(self, rows: int) -> float:
        return self.seconds("verify", rows)

    def round(self, generated: int, verified: int) -> float:
        return max(self.draft(generated) + self.verify(self.sequences * (verified + 1)), 1e-9)


@dataclass(frozen=True, slots=True)
class DraftOutlook:
    load: DraftLoad
    context_class: int
    right: tuple[float, ...]
    draft_seconds: float


def share_rows(outlooks: tuple[DraftOutlook, ...], rows: int) -> tuple[int, ...] | None:
    """Give each request an anchor, then allocate the most valuable draft rows."""
    requests = len(outlooks)
    if requests < 2 or rows < requests:
        return None
    costs = _RoundCosts(outlooks[0].load, max(x.context_class for x in outlooks), requests)
    drafted = sum(x.draft_seconds for x in outlooks)
    proposals = sorted(
        (
            (right, depth, request)
            for request, outlook in enumerate(outlooks)
            for depth, right in enumerate(outlook.right)
        ),
        key=lambda x: (-x[0], x[1]),
    )
    best, rate = 0, -math.inf
    row_class = power_of_two(requests)
    while True:
        verified = min(max(min(row_class, rows) - requests, 0), len(proposals))
        tokens = requests + sum(x[0] for x in proposals[:verified])
        candidate = tokens / max(drafted + costs.verify(requests + verified), 1e-9)
        if candidate > rate:
            best, rate = verified, candidate
        if row_class >= rows or verified == len(proposals):
            break
        row_class *= 2
    shares = [1] * requests
    for _, _, request in proposals[:best]:
        shares[request] += 1
    return tuple(shares)


@dataclass(slots=True)
class RoundChoice:
    generated: int = 0
    verified: int = 0
    look: bool = False


@dataclass(slots=True)
class _Block:
    tokens: tuple[int, ...]
    matched: int = 0
    run_look: bool = False


class DraftController:
    def __init__(self, widest: int, load: DraftLoad | None = None) -> None:
        self.ladder = draft_ladder(widest)
        self.load = load if load is not None else DraftLoad()
        with self.load.lock:
            seeded = self.load.right is not None and len(self.load.right) == widest
            self._right = list(self.load.right or ()) if seeded else [1.0] * widest
            self.look_run = self.load.look_run or 1
        self.blocks_scored = [U32_MAX if seeded else 0] * widest
        self.blocks: list[_Block] = []
        self.look_seconds = 0.0
        self.accepted_run = 0
        self.run_look = False
        self.kept: tuple[int, float] | None = None
        self.timing_draft: int | None = None
        self.timing_target = False
        self._drafted: int | None = None
        self.context_class = 1
        self.sequences = 1
        self.choice = RoundChoice()

    def fork(self) -> DraftController:
        return copy.deepcopy(self, {id(self.load): self.load})

    def right(self) -> tuple[float, ...]:
        bound = 1.0
        result = []
        for right in self._right:
            bound = min(bound, right)
            result.append(bound)
        return tuple(result)

    def _costs(self) -> _RoundCosts:
        return _RoundCosts(self.load, self.context_class, self.sequences)

    def measured(self) -> LaunchCosts:
        return self._costs().own

    def outlook(self) -> DraftOutlook | None:
        if self._drafted is None:
            return None
        return DraftOutlook(
            self.load,
            self.context_class,
            self.right()[: self._drafted],
            self._costs().draft(self.choice.generated),
        )

    def _widths(self, limit: int) -> tuple[int, ...]:
        return (0, *(width for width in self.ladder if width <= limit))

    @staticmethod
    def _tokens(right: tuple[float, ...], verified: int) -> float:
        return 1.0 + sum(right[:verified])

    def _verified(
        self, right: tuple[float, ...], costs: _RoundCosts, generated: int, limit: int
    ) -> tuple[int, float]:
        return max(
            (
                (width, self._tokens(right, width) / costs.round(generated, width))
                for width in self._widths(min(limit, generated))
            ),
            key=lambda x: x[1],
        )

    def generate(self, limit: int, position: int) -> int:
        bounded(limit, "draft limit")
        bounded(position, "target position")
        self.context_class = power_of_two(position)
        costs, right = self._costs(), self.right()
        pairs = [
            (generated, *self._verified(right, costs, generated, limit))
            for generated in self._widths(limit)
        ]
        fastest, verified, rate = max(pairs, key=lambda x: x[2])
        wider = []
        for generated in self._widths(limit):
            if generated <= fastest:
                continue
            _, best = self._verified((1.0,) * len(right), costs, generated, limit)
            offered, _ = self._verified(right, costs, generated, limit)
            lost = costs.round(generated, offered) - self._tokens(right, offered) / rate
            wider.append((generated, max(best / rate - 1.0, 0.0), max(lost, 0.0)))
        gains = {width: gain for width, gain, _ in wider}
        if self.kept is not None:
            kept, earned = self.kept
        else:
            kept, earned = fastest, 0.0
        if kept > limit:
            kept = fastest
        elif kept > fastest and earned > 0:
            gain = gains.get(kept, 0.0)
            stake = gain / (1 + gain)
            self.kept = kept, earned - 1.0 / max(stake, LOOK_SHARE)
        elif kept == fastest:
            self.kept = kept, min(earned + 1.0, 1.0 / LOOK_SHARE)
        else:
            kept = fastest
            self.kept = fastest, 1.0
        looks = [x for x in wider if x[0] > kept and x[1] > 0]
        generated = kept
        self.run_look = False
        if looks:
            width, gain, lost = max(looks, key=lambda x: (x[1] / max(x[2], 1e-12), x[0]))
            self.look_seconds += LOOK_SHARE * min(gain, 1.0) * costs.round(fastest, verified)
            if self.look_seconds >= lost:
                self.look_seconds = 0.0
                generated = width
            elif self.accepted_run >= self.look_run:
                self.accepted_run = 0
                self.run_look = True
                generated = width
        else:
            self.look_seconds = 0.0
        self.choice = RoundChoice(generated, 0, generated != kept)
        self.timing_draft = generated or None
        self.timing_target = generated == 0
        return generated

    def drafted(self, generated: int, block: tuple[int, ...]) -> None:
        if not 0 <= len(block) <= generated <= len(self._right):
            raise ValueError("drafted block exceeds generated width")
        for depth in range(len(block) + 1, generated + 1):
            self._score(depth, False)
        if block:
            self.blocks.append(_Block(tuple(block), run_look=self.run_look))
        self._drafted = len(block)

    def verify(self, drafted: int, limit: int) -> int:
        count, _ = self._verified(
            self.right(), self._costs(), self.choice.generated, min(limit, drafted)
        )
        self.choice.verified = count
        self.timing_target = True
        self._drafted = None
        return count

    def _score(self, depth: int, right: bool) -> None:
        scored = min(self.blocks_scored[depth - 1] + 1, U32_MAX)
        self.blocks_scored[depth - 1] = scored
        weight = max(1.0 / scored, MEMORY)
        self._right[depth - 1] += weight * (float(right) - self._right[depth - 1])

    def committed(self, tokens: tuple[int, ...], *, publish: bool = True) -> None:
        self.accepted_run = (
            min(self.accepted_run + 1, U32_MAX)
            if self.choice.verified > 0 and len(tokens) == self.choice.verified + 1
            else 0
        )
        for token in tokens:
            remaining = []
            for block in self.blocks:
                if block.tokens[block.matched] == token:
                    block.matched += 1
                    self._score(block.matched, True)
                    if block.matched == len(block.tokens):
                        if block.run_look:
                            self.look_run = max(self.look_run // 2, 1)
                    else:
                        remaining.append(block)
                else:
                    for depth in range(block.matched + 1, len(block.tokens) + 1):
                        self._score(depth, False)
                    if block.run_look:
                        self.look_run = min(self.look_run * 2, U32_MAX)
            self.blocks = remaining
        if publish:
            self.publish()

    def publish(self) -> None:
        """Publish learning only after the enclosing logical transition commits."""
        with self.load.lock:
            self.load.right = self._right.copy()
            self.load.look_run = self.look_run

    def timed(self, launch: TimedLaunch, seconds: float, sequences: int, rows: int) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("launch duration must be finite and nonnegative seconds")
        bounded(sequences, "launch sequences", minimum=1)
        bounded(rows, "launch rows", minimum=1)
        if launch is TimedLaunch.DRAFT:
            size, self.timing_draft = self.timing_draft, None
        else:
            size = rows if self.timing_target else None
            self.timing_target = False
        if size is None:
            return
        self.sequences = sequences
        with self.load.lock:
            costs = self.load.costs.setdefault((self.context_class, sequences), LaunchCosts())
            LaunchCosts.record(
                costs.draft if launch is TimedLaunch.DRAFT else costs.verify, size, seconds
            )
