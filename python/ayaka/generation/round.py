"""Immutable target rounds and their per-row sample reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ayaka.generation.acceptance import accept_prefix
from ayaka.generation.contracts import (
    Constraint,
    Demand,
    ForwardOperation,
    Priming,
    Sampling,
    SelectSpec,
    Shaping,
    WorkKind,
)
from ayaka.generation.method import MethodRequirements
from ayaka.generation.shaping import verification_selects

_PLAIN_REQUIREMENTS = MethodRequirements()
_DEFAULT_SHAPING = Shaping()


@dataclass(frozen=True, slots=True)
class RoundForward:
    kind: WorkKind
    tokens: tuple[int, ...]
    demand: Demand
    selects: tuple[SelectSpec, ...]
    committed: int
    prime: Priming | None = None

    def into_operation(
        self, request: str, position: int, conditioning: str | None = None
    ) -> ForwardOperation:
        operation = ForwardOperation(
            request,
            self.kind,
            self.tokens,
            position,
            self.demand,
            self.selects,
            self.committed,
            self.prime,
            conditioning,
        )
        operation.validate()
        return operation


class MethodUpdate(StrEnum):
    NONE = "none"
    PRIME = "prime"
    OBSERVE = "observe"


@dataclass(frozen=True, slots=True)
class RoundAcceptance:
    inputs: tuple[int, ...]
    emitted: tuple[int, ...]
    committed_rows: int
    proposed: int
    accepted_proposals: int
    method_update: MethodUpdate


@dataclass(frozen=True, slots=True)
class RoundState:
    forward: RoundForward
    mode: str
    emitted: tuple[int, ...] = ()
    proposal: tuple[int, ...] = ()

    @classmethod
    def progress(
        cls,
        kind: WorkKind,
        tokens: tuple[int, ...],
        emitted: tuple[int, ...] = (),
        select: SelectSpec | None = None,
        prime: Priming | None = None,
        requirements: MethodRequirements = _PLAIN_REQUIREMENTS,
    ) -> RoundState:
        if (
            not tokens
            or kind not in (WorkKind.PREFILL, WorkKind.REPLAY)
            or (kind is WorkKind.REPLAY and (emitted or select is not None))
            or (emitted and select is not None)
            or len(emitted) > 1
        ):
            raise ValueError("invalid prefill or replay round")
        demand = requirements.prefill_demand
        if select is not None or emitted:
            demand |= requirements.closing_demand
        if select is not None:
            demand |= Demand.SELECT
        forward = RoundForward(
            kind, tuple(tokens), demand, () if select is None else (select,), len(tokens), prime
        )
        forward.into_operation("validation", 0)
        return cls(forward, "progress", tuple(emitted))

    @staticmethod
    def proposal_limit(
        allowance: int, remaining: int, output_credit: int, method_is_plain: bool
    ) -> int:
        return 0 if method_is_plain else max(min(allowance, remaining, output_credit) - 1, 0)

    @classmethod
    def verification(
        cls,
        anchor: int,
        proposal: tuple[int, ...],
        proposal_limit: int,
        generated_position: int,
        accepted_history: tuple[int, ...],
        constraint: Constraint | None = None,
        sampling: Sampling = Sampling.GREEDY,
        shaping: Shaping = _DEFAULT_SHAPING,
        seed: int = 0,
        requirements: MethodRequirements = _PLAIN_REQUIREMENTS,
    ) -> RoundState:
        if len(proposal) > proposal_limit:
            raise ValueError("method proposal exceeds the round width limit")
        selects = verification_selects(
            generated_position, accepted_history, proposal, constraint, sampling, shaping, seed
        )
        return cls(
            RoundForward(
                WorkKind.VERIFY if proposal else WorkKind.DECODE,
                (anchor, *proposal),
                Demand.SELECT | requirements.verify_demand,
                selects,
                1,
            ),
            "verification",
            proposal=tuple(proposal),
        )

    @classmethod
    def forced(
        cls, anchor: int, forced: tuple[int, ...], requirements: MethodRequirements
    ) -> RoundState:
        if not forced:
            raise ValueError("forced round requires at least one accepted token")
        return cls(
            RoundForward(
                WorkKind.DECODE,
                (anchor, *forced[:-1]),
                requirements.verify_demand & Demand.FEATURES,
                (),
                len(forced),
            ),
            "forced",
            tuple(forced),
        )

    def publication_bound(self) -> int:
        return len(self.emitted) + len(self.forward.selects)

    def reconcile(self, samples: tuple[int, ...], stops: frozenset[int]) -> RoundAcceptance:
        if len(samples) != len(self.forward.selects):
            raise ValueError("round returned the wrong number of samples")
        if self.mode == "verification":
            accepted = accept_prefix(self.proposal, samples, stops)
            return RoundAcceptance(
                self.forward.tokens,
                (*self.proposal[:accepted], samples[accepted]),
                accepted + 1,
                len(self.proposal),
                accepted,
                MethodUpdate.OBSERVE,
            )
        return RoundAcceptance(
            self.forward.tokens,
            self.emitted + tuple(samples),
            len(self.forward.tokens),
            0,
            0,
            MethodUpdate.PRIME if self.mode == "progress" else MethodUpdate.OBSERVE,
        )
