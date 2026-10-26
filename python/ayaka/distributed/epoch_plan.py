"""Rank-coherent epoch plan selection with per-rank cost estimates (research R4).

Each rank proposes, for one epoch, the bounded set of candidate plans it could
run. Every candidate with the same ``plan_key`` has the same semantics on every
rank: the same requests, token counts and collective sequence. Ranks gather
all proposals and apply :func:`decide_epoch_plan`, a pure function of the
gathered bytes, so every rank reaches the same decision without another round.

The cost model only chooses between same-semantics plans: ``cost(plan) =
max_rank(compute + non-overlapped comm)``, ties broken by ``plan_key``. No rank
may drop a request from a shared forward on its own; a plan that is not ready
on every active rank is not a candidate, and when none is left every rank
aborts before submit. Idle ranks still join the chosen plan's collective
sequence with idle-shaped work.

This is decision-only: nothing in the scheduler or worker calls it. The
gather in :func:`gather_proposals` costs one broadcast per rank on the
existing control plane; measure that against the straggler wait it saves
before adding it to a hot path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ayaka.distributed.execution import DistributedControlPlane
from ayaka.utils.validation import require_int, require_text

__all__ = [
    "CandidatePlan",
    "EpochDecision",
    "EpochPlanStatus",
    "RankProposal",
    "decide_epoch_plan",
    "gather_proposals",
]

#: Candidate plans one rank may propose per epoch (bounded scoring cost).
MAX_CANDIDATES = 8
#: Encoded proposal size limit; a proposal is metadata, never tensors.
MAX_PROPOSAL_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """One same-semantics plan as seen by one rank."""

    plan_key: str
    #: Tokens this plan runs; identical on every rank for one ``plan_key``.
    tokens: int
    #: Whether this rank's leases (pages, COW, workspace) are ready for it.
    ready: bool
    #: This rank can replay a captured graph for this shape.
    graph_eligible: bool
    #: Measured-model estimate of rank-local compute, microseconds.
    compute_us: float
    #: Communication this rank cannot overlap with compute, microseconds.
    comm_us: float

    def __post_init__(self) -> None:
        require_text(self.plan_key, "plan_key")
        require_int(self.tokens, "tokens")
        for name in ("compute_us", "comm_us"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def cost_us(self) -> float:
        return self.compute_us + self.comm_us


@dataclass(frozen=True, slots=True)
class RankProposal:
    """Everything one rank contributes to an epoch decision."""

    rank: int
    epoch: int
    model_generation: int
    #: Digest of the rank's execution capability (plan identity, graph mode, layout).
    capability: str
    #: No local work; the rank joins the chosen plan with idle-shaped work.
    idle: bool
    candidates: tuple[CandidatePlan, ...] = ()

    def __post_init__(self) -> None:
        require_int(self.rank, "rank")
        require_int(self.epoch, "epoch")
        require_int(self.model_generation, "model_generation")
        require_text(self.capability, "capability")
        if len(self.candidates) > MAX_CANDIDATES:
            raise ValueError(f"a rank may propose at most {MAX_CANDIDATES} plans")
        keys = [candidate.plan_key for candidate in self.candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate plan keys must be unique")
        if self.idle and self.candidates:
            raise ValueError("an idle rank proposes no plans")

    def encode(self) -> str:
        text = json.dumps(
            {
                "rank": self.rank,
                "epoch": self.epoch,
                "model_generation": self.model_generation,
                "capability": self.capability,
                "idle": self.idle,
                "candidates": [
                    [c.plan_key, c.tokens, c.ready, c.graph_eligible, c.compute_us, c.comm_us]
                    for c in self.candidates
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(text.encode("utf-8")) > MAX_PROPOSAL_BYTES:
            raise ValueError("epoch proposal exceeds its byte budget")
        return text

    @classmethod
    def decode(cls, text: str) -> RankProposal:
        if len(text.encode("utf-8")) > MAX_PROPOSAL_BYTES:
            raise ValueError("epoch proposal exceeds its byte budget")
        payload: dict[str, Any] = json.loads(text)
        return cls(
            rank=payload["rank"],
            epoch=payload["epoch"],
            model_generation=payload["model_generation"],
            capability=payload["capability"],
            idle=payload["idle"],
            candidates=tuple(
                CandidatePlan(key, tokens, ready, graph, float(compute), float(comm))
                for key, tokens, ready, graph, compute, comm in payload["candidates"]
            ),
        )


class EpochPlanStatus(StrEnum):
    RUN = "run"
    #: Every rank is idle: no forward and no collective this epoch.
    IDLE = "idle"
    #: Agreed abort before submit; every rank rolls back its tentative leases.
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class EpochDecision:
    status: EpochPlanStatus
    epoch: int
    reason: str
    plan_key: str | None = None
    #: Graph replay only when every active rank is eligible for the shape.
    graph: bool = False
    idle_ranks: tuple[int, ...] = ()
    #: ``max_rank`` cost of the chosen plan, microseconds.
    cost_us: float = 0.0
    #: Predicted wait of each active rank for the slowest one, by rank.
    wait_us: tuple[tuple[int, float], ...] = ()


def _abort(epoch: int, reason: str) -> EpochDecision:
    return EpochDecision(EpochPlanStatus.ABORT, epoch, reason)


def decide_epoch_plan(proposals: Sequence[RankProposal], *, world_size: int) -> EpochDecision:
    """Choose the cheapest plan every active rank can run, or abort for all.

    Deterministic in the gathered proposals: their order does not matter and
    no rank-local state is read, so every rank decides identically.
    """
    require_int(world_size, "world_size", minimum=1)
    by_rank = {proposal.rank: proposal for proposal in proposals}
    epoch = min((proposal.epoch for proposal in proposals), default=0)
    if len(by_rank) != len(proposals) or set(by_rank) != set(range(world_size)):
        return _abort(epoch, "membership: need exactly one proposal per rank")
    ordered = [by_rank[rank] for rank in range(world_size)]
    for field_name in ("epoch", "model_generation", "capability"):
        if len({getattr(proposal, field_name) for proposal in ordered}) != 1:
            return _abort(epoch, f"ranks disagree on {field_name}")
    epoch = ordered[0].epoch
    active = [proposal for proposal in ordered if not proposal.idle]
    idle_ranks = tuple(proposal.rank for proposal in ordered if proposal.idle)
    if not active:
        return EpochDecision(EpochPlanStatus.IDLE, epoch, "every rank is idle")
    offers = [{c.plan_key: c for c in proposal.candidates} for proposal in active]
    common = sorted(set.intersection(*(set(offer) for offer in offers)))
    scored: list[tuple[float, str]] = []
    for key in common:
        views = [offer[key] for offer in offers]
        if len({view.tokens for view in views}) != 1:
            return _abort(epoch, f"plan {key!r} has different semantics across ranks")
        if all(view.ready for view in views):
            scored.append((max(view.cost_us for view in views), key))
    if not scored:
        return _abort(epoch, "no plan is ready on every active rank")
    cost, key = min(scored)
    views = [offer[key] for offer in offers]
    return EpochDecision(
        EpochPlanStatus.RUN,
        epoch,
        "cheapest plan ready on every active rank",
        plan_key=key,
        graph=all(view.graph_eligible for view in views),
        idle_ranks=idle_ranks,
        cost_us=cost,
        wait_us=tuple(
            (proposal.rank, cost - view.cost_us)
            for proposal, view in zip(active, views, strict=True)
        ),
    )


def gather_proposals(
    control: DistributedControlPlane,
    proposal: RankProposal,
    *,
    timeout_s: float | None = None,
) -> tuple[RankProposal, ...]:
    """Exchange proposals with one broadcast per rank, in rank order.

    Every rank issues the same collective sequence. A collective failure
    propagates; it never becomes a partial gather.
    """
    if proposal.rank != control.rank:
        raise ValueError("a rank may only propose for itself")
    own = proposal.encode()
    gathered = []
    for source in range(control.world_size):
        text = control.broadcast_text(
            own if source == control.rank else None, source_rank=source, timeout_s=timeout_s
        )
        gathered.append(RankProposal.decode(text))
    return tuple(gathered)
