"""Ordinary target-only generation, with no draft work or retained features."""

from __future__ import annotations

from ayaka.generation.contracts import (
    FeatureReader,
    FeatureRef,
    HeadOperation,
    HeadOutcome,
    SelectSpec,
)
from ayaka.generation.draft_control import DraftOutlook, TimedLaunch
from ayaka.generation.method import (
    MethodCheckpoint,
    MethodEffects,
    MethodKind,
    MethodRequirements,
    Propose,
    Verification,
)


class Plain:
    def identity(self) -> str:
        return "plain"

    def requires(self) -> MethodRequirements:
        return MethodRequirements()

    def priming_requires_successor(self) -> bool:
        return False

    def embeds_prompt_tokens(self) -> bool:
        return False

    def proposals(self) -> int:
        return 0

    def create(self, checkpoint: MethodCheckpoint | None = None) -> PlainState:
        if checkpoint is not None and checkpoint.kind is not MethodKind.PLAIN:
            raise ValueError("a drafter checkpoint cannot restore plain state")
        return PlainState()


class PlainState:
    def fork_transition(self) -> PlainState:
        return PlainState()

    def priming_position(self) -> int | None:
        return None

    def prime(
        self,
        request: str,
        tokens: tuple[int, ...],
        next_token: int | None,
        features: FeatureRef | None,
        draft_from: int,
        primed: int | None,
        reader: FeatureReader | None,
    ) -> MethodEffects:
        return MethodEffects()

    def propose(self, request: str, selects: tuple[SelectSpec, ...]) -> Propose:
        return Propose()

    def observe(
        self, request: str, verification: Verification, reader: FeatureReader | None
    ) -> MethodEffects:
        return MethodEffects()

    def reconcile(self, operation: HeadOperation, outcome: HeadOutcome) -> None:
        raise ValueError("plain generation cannot receive method operations")

    def timed(self, launch: TimedLaunch, seconds: float, sequences: int, rows: int) -> None:
        pass

    def outlook(self) -> DraftOutlook | None:
        return None

    def checkpoint(self) -> MethodCheckpoint:
        return MethodCheckpoint()

    def evict(self) -> None:
        pass

    def reclaimable(self) -> int:
        return 0

    def publish_learning(self) -> None:
        pass
