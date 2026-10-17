"""Pure geometry of target-conditioned drafter transactions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from ayaka.generation.contracts import bounded, checked_add


class DraftHistory(StrEnum):
    OWN = "own"
    TARGET = "target"


@dataclass(frozen=True, slots=True)
class DraftHeadCall:
    entry_rows: int
    generated: int
    position: int


@dataclass(frozen=True, slots=True)
class DraftRoundState:
    history: DraftHistory
    entry_bound: int
    position: int
    pending_rows: int
    open_row: bool

    def __post_init__(self) -> None:
        bounded(self.entry_bound, "drafter entry bound", minimum=1)
        bounded(self.position, "drafter position")
        bounded(self.pending_rows, "pending rows")
        if self.history is DraftHistory.TARGET and (self.pending_rows > 1 or self.open_row):
            raise ValueError("target-reading drafter has an invalid boundary")
        self.target_position()

    def target_position(self) -> int:
        if self.history is DraftHistory.TARGET:
            return self.position
        return checked_add(checked_add(self.position, self.pending_rows), int(self.open_row))

    def entered(self, rows: int) -> DraftRoundState:
        bounded(rows, "entered rows")
        return replace(
            self,
            position=checked_add(self.position, rows)
            if self.history is DraftHistory.OWN
            else self.position,
        )

    def proposing(self, generated: int) -> tuple[DraftHeadCall | None, DraftRoundState]:
        bounded(generated, "generated proposals")
        if generated == 0:
            return None, self
        if generated >= self.entry_bound or not 0 < self.pending_rows <= self.entry_bound:
            raise ValueError("drafter proposal does not fit its settled boundary")
        if self.open_row:
            raise ValueError("drafter proposal requires a closed boundary")
        call = DraftHeadCall(self.pending_rows, generated, self.position)
        # Remove entered pairs before incrementing position to avoid counting twice.
        successor = replace(self, pending_rows=0).entered(self.pending_rows)
        return call, successor

    def flushing(self, keep: int) -> tuple[DraftHeadCall | None, DraftRoundState]:
        bounded(keep, "kept rows")
        if self.history is DraftHistory.TARGET or self.pending_rows <= keep:
            return None, self
        return (
            DraftHeadCall(self.pending_rows - keep, 0, self.position),
            replace(self, pending_rows=keep),
        )

    def observed(self, accepted: int) -> DraftTargetTransition:
        bounded(accepted, "accepted drafter prefix", minimum=1)
        appended = (
            replace(self, position=checked_add(self.position, accepted), pending_rows=1)
            if self.history is DraftHistory.TARGET
            else replace(
                self,
                pending_rows=checked_add(
                    checked_add(self.pending_rows, int(self.open_row)), accepted
                ),
                open_row=False,
            )
        )
        if appended.history is DraftHistory.OWN and appended.pending_rows > self.entry_bound:
            call, remaining = appended.flushing(1)
            assert call is not None
            return DraftTargetTransition(appended, call, remaining.entered(call.entry_rows))
        return DraftTargetTransition(appended, None, appended)


@dataclass(frozen=True, slots=True)
class DraftTargetTransition:
    appended: DraftRoundState
    maintenance: DraftHeadCall | None
    successor: DraftRoundState
