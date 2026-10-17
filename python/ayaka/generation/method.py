"""Method factories and request-local transactional drafter state contracts."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ayaka.generation.contracts import (
    Demand,
    FeatureReader,
    FeatureRef,
    FeatureRows,
    HeadOperation,
    HeadOutcome,
    SelectSpec,
    bounded,
)
from ayaka.generation.draft_control import DraftOutlook, TimedLaunch


class MethodKind(StrEnum):
    PLAIN = "plain"
    MTP = "mtp"
    DFLASH = "dflash"


@dataclass(frozen=True, slots=True)
class MethodChoice:
    kind: MethodKind = MethodKind.PLAIN
    proposals: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MethodKind):
            raise TypeError("method kind must be MethodKind")
        bounded(self.proposals, "method proposals", 255)
        if (self.kind is MethodKind.PLAIN) != (self.proposals == 0):
            raise ValueError("plain needs zero proposals; draft methods need 1..255")


@dataclass(frozen=True, slots=True)
class MethodRequirements:
    prefill_demand: Demand = Demand.NONE
    closing_demand: Demand = Demand.NONE
    verify_demand: Demand = Demand.NONE
    head: bool = False


@dataclass(frozen=True, slots=True)
class PendingRows:
    tokens: tuple[int, ...]
    features: FeatureRows


@dataclass(frozen=True, slots=True)
class OpenFeature:
    host: FeatureRows | None = None
    device: weakref.ReferenceType[FeatureRef] | None = None

    def __post_init__(self) -> None:
        if (self.host is None) == (self.device is None):
            raise ValueError("open feature must have exactly one host or weak device owner")

    def host_bytes(self) -> int:
        return 0 if self.host is None else len(self.host.data)

    def read(self, reader: FeatureReader | None) -> FeatureRows:
        if self.host is not None:
            return self.host
        feature = None if self.device is None else self.device()
        if feature is None or reader is None:
            raise ValueError("executor boundary feature is no longer resident")
        return reader.read(feature, 0, 1)


@dataclass(frozen=True, slots=True)
class DraftCheckpoint:
    position: int = 0
    pending: PendingRows | None = None
    open: OpenFeature | None = None

    def target_rows(self) -> int:
        return (
            self.position
            + (0 if self.pending is None else len(self.pending.tokens))
            + int(self.open is not None)
        )

    def retained_bytes(self) -> int:
        return (0 if self.pending is None else len(self.pending.features.data)) + (
            0 if self.open is None else self.open.host_bytes()
        )


@dataclass(frozen=True, slots=True)
class MethodCheckpoint:
    kind: MethodKind = MethodKind.PLAIN
    draft: DraftCheckpoint | None = None

    def __post_init__(self) -> None:
        if (self.kind is MethodKind.PLAIN) != (self.draft is None):
            raise ValueError("checkpoint kind disagrees with draft payload")

    def retained_bytes(self) -> int:
        return 0 if self.draft is None else self.draft.retained_bytes()


class MethodCheckpointError(ValueError):
    """Unresolved head work/proposals cannot be checkpointed."""


@dataclass(frozen=True, slots=True)
class Verification:
    inputs: tuple[int, ...]
    accepted: int
    next: int
    features: FeatureRef | None = None


@dataclass(frozen=True, slots=True)
class Propose:
    tokens: tuple[int, ...] = ()
    operations: tuple[HeadOperation, ...] = ()

    def __post_init__(self) -> None:
        if self.tokens and self.operations:
            raise ValueError("proposal cannot contain both tokens and pending operations")


@dataclass(frozen=True, slots=True)
class MethodEffects:
    operations: tuple[HeadOperation, ...] = ()


class MethodState(Protocol):
    def fork_transition(self) -> MethodState: ...
    def priming_position(self) -> int | None: ...
    def prime(
        self,
        request: str,
        tokens: tuple[int, ...],
        next_token: int | None,
        features: FeatureRef | None,
        draft_from: int,
        primed: int | None,
        reader: FeatureReader | None,
    ) -> MethodEffects: ...
    def propose(self, request: str, selects: tuple[SelectSpec, ...]) -> Propose: ...
    def observe(
        self, request: str, verification: Verification, reader: FeatureReader | None
    ) -> MethodEffects: ...
    def reconcile(self, operation: HeadOperation, outcome: HeadOutcome) -> None: ...
    def timed(self, launch: TimedLaunch, seconds: float, sequences: int, rows: int) -> None: ...
    def outlook(self) -> DraftOutlook | None: ...
    def checkpoint(self) -> MethodCheckpoint: ...
    def evict(self) -> None: ...
    def reclaimable(self) -> int: ...
    def publish_learning(self) -> None: ...


class Method(Protocol):
    def priming_requires_successor(self) -> bool: ...
    def embeds_prompt_tokens(self) -> bool: ...
    def identity(self) -> str: ...
    def requires(self) -> MethodRequirements: ...
    def proposals(self) -> int: ...
    def create(self, checkpoint: MethodCheckpoint | None = None) -> MethodState: ...
