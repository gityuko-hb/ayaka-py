"""R4: a rank-coherent epoch plan chosen by per-rank cost, decision-only.

Every rank gathers all proposals and applies the same pure decision, so the
group runs one plan or aborts together. The cost model only chooses between
plans every active rank can run with the same semantics; it never lets a rank
drop work from a shared forward.
"""

from __future__ import annotations

import itertools
import threading

import pytest
from ayaka.distributed.epoch_plan import (
    MAX_CANDIDATES,
    CandidatePlan,
    EpochPlanStatus,
    RankProposal,
    decide_epoch_plan,
    gather_proposals,
)
from ayaka.distributed.process_group import DistributedCollectiveTimeout
from fake_ranks import LoopbackGroup


def _plan(key: str, compute: float, *, tokens: int = 64, ready: bool = True, graph: bool = True):
    return CandidatePlan(key, tokens, ready, graph, compute_us=compute, comm_us=10.0)


def _rank(rank: int, *plans: CandidatePlan, **overrides) -> RankProposal:
    values = {
        "rank": rank,
        "epoch": 7,
        "model_generation": 1,
        "capability": "tp2-eager",
        "idle": not plans,
        "candidates": plans,
    } | overrides
    return RankProposal(**values)


def test_the_plan_with_the_smallest_slowest_rank_wins() -> None:
    # Rank 1 straggles on the large chunk; the small chunk is better for the group.
    decision = decide_epoch_plan(
        [
            _rank(0, _plan("chunk64", 100.0), _plan("chunk32", 70.0, tokens=32)),
            _rank(1, _plan("chunk64", 290.0), _plan("chunk32", 110.0, tokens=32)),
        ],
        world_size=2,
    )
    assert decision.status is EpochPlanStatus.RUN and decision.plan_key == "chunk32"
    assert decision.cost_us == 120.0
    assert decision.wait_us == ((0, 40.0), (1, 0.0))  # rank 0 waits for rank 1


def test_a_plan_not_ready_on_one_rank_is_never_chosen() -> None:
    decision = decide_epoch_plan(
        [
            _rank(0, _plan("chunk64", 50.0), _plan("chunk32", 90.0, tokens=32)),
            _rank(1, _plan("chunk64", 50.0, ready=False), _plan("chunk32", 90.0, tokens=32)),
        ],
        world_size=2,
    )
    assert decision.plan_key == "chunk32"  # slower, but runnable everywhere


def test_no_plan_ready_everywhere_aborts_every_rank_before_submit() -> None:
    decision = decide_epoch_plan(
        [_rank(0, _plan("a", 1.0)), _rank(1, _plan("a", 1.0, ready=False))], world_size=2
    )
    assert decision.status is EpochPlanStatus.ABORT
    assert decision.reason == "no plan is ready on every active rank"
    # Plans only one rank offers are not candidates either.
    disjoint = decide_epoch_plan(
        [_rank(0, _plan("a", 1.0)), _rank(1, _plan("b", 1.0))], world_size=2
    )
    assert disjoint.status is EpochPlanStatus.ABORT


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"epoch": 8}, "ranks disagree on epoch"),
        ({"model_generation": 2}, "ranks disagree on model_generation"),
        ({"capability": "tp2-graph"}, "ranks disagree on capability"),
    ],
)
def test_ranks_on_different_epochs_generations_or_capabilities_abort(override, reason) -> None:
    decision = decide_epoch_plan(
        [_rank(0, _plan("a", 1.0)), _rank(1, _plan("a", 1.0), **override)], world_size=2
    )
    assert decision.status is EpochPlanStatus.ABORT and decision.reason == reason


@pytest.mark.parametrize("ranks", [(0,), (0, 0), (0, 2), (0, 1, 1)])
def test_a_missing_or_duplicated_rank_aborts(ranks) -> None:
    decision = decide_epoch_plan([_rank(rank, _plan("a", 1.0)) for rank in ranks], world_size=2)
    assert decision.status is EpochPlanStatus.ABORT
    assert decision.reason.startswith("membership")


def test_the_same_key_with_different_semantics_aborts() -> None:
    decision = decide_epoch_plan(
        [_rank(0, _plan("a", 1.0, tokens=64)), _rank(1, _plan("a", 1.0, tokens=63))],
        world_size=2,
    )
    assert decision.status is EpochPlanStatus.ABORT
    assert "different semantics" in decision.reason


def test_idle_ranks_join_the_plan_and_an_all_idle_epoch_runs_nothing() -> None:
    some = decide_epoch_plan([_rank(0, _plan("a", 5.0)), _rank(1), _rank(2)], world_size=3)
    assert some.status is EpochPlanStatus.RUN and some.idle_ranks == (1, 2)
    assert some.wait_us == ((0, 0.0),)
    none = decide_epoch_plan([_rank(0), _rank(1)], world_size=2)
    assert none.status is EpochPlanStatus.IDLE and none.plan_key is None


def test_graph_replay_needs_every_active_rank_eligible() -> None:
    mixed = decide_epoch_plan(
        [_rank(0, _plan("a", 1.0, graph=True)), _rank(1, _plan("a", 1.0, graph=False))],
        world_size=2,
    )
    assert mixed.status is EpochPlanStatus.RUN and not mixed.graph
    uniform = decide_epoch_plan(
        [_rank(0, _plan("a", 1.0)), _rank(1, _plan("a", 1.0))], world_size=2
    )
    assert uniform.graph


def test_the_decision_ignores_proposal_order_and_breaks_ties_by_key() -> None:
    proposals = [
        _rank(0, _plan("b", 10.0), _plan("a", 10.0)),
        _rank(1, _plan("a", 10.0), _plan("b", 10.0)),
        _rank(2, _plan("b", 10.0), _plan("a", 10.0)),
    ]
    decisions = {
        decide_epoch_plan(list(order), world_size=3) for order in itertools.permutations(proposals)
    }
    assert len(decisions) == 1
    (decision,) = decisions
    assert decision.plan_key == "a"


def test_proposals_round_trip_exactly_and_stay_bounded() -> None:
    proposal = _rank(1, _plan("chunk32", 0.1 + 0.2, tokens=32), _plan("chunk64", 1e-7))
    assert RankProposal.decode(proposal.encode()) == proposal
    with pytest.raises(ValueError, match="at most"):
        _rank(0, *(_plan(f"p{i}", 1.0) for i in range(MAX_CANDIDATES + 1)))
    with pytest.raises(ValueError, match="unique"):
        _rank(0, _plan("a", 1.0), _plan("a", 2.0))
    with pytest.raises(ValueError, match="idle"):
        _rank(0, _plan("a", 1.0), idle=True)
    with pytest.raises(ValueError, match="byte budget"):
        _rank(0, _plan("x" * 20_000, 1.0)).encode()
    with pytest.raises(ValueError):
        _plan("a", float("nan"))


# ---------------------------------------------------------------------------
# Gather on fake ranks: every rank decides the same thing
# ---------------------------------------------------------------------------


def _decide_on_ranks(proposals, *, wedged: int | None = None, timeout_s: float = 5.0):
    group = LoopbackGroup(len(proposals), timeout_s=timeout_s)
    outcomes: list[object] = [None] * len(proposals)

    def run(rank: int) -> None:
        if rank == wedged:
            return  # never joins the gather
        try:
            gathered = gather_proposals(group.plane(rank), proposals[rank])
            outcomes[rank] = decide_epoch_plan(gathered, world_size=len(proposals))
        except DistributedCollectiveTimeout as exc:
            outcomes[rank] = exc

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(len(proposals))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    return outcomes


def test_every_rank_reaches_the_same_decision_from_the_gather() -> None:
    proposals = [
        _rank(0, _plan("chunk64", 100.0), _plan("chunk32", 70.0, tokens=32)),
        _rank(1, _plan("chunk64", 290.0), _plan("chunk32", 110.0, tokens=32)),
        _rank(2),
    ]
    outcomes = _decide_on_ranks(proposals)
    assert len(set(outcomes)) == 1
    assert outcomes[0] == decide_epoch_plan(proposals, world_size=3)


def test_one_unready_rank_makes_every_rank_abort() -> None:
    proposals = [_rank(0, _plan("a", 1.0)), _rank(1, _plan("a", 1.0, ready=False))]
    outcomes = _decide_on_ranks(proposals)
    assert [outcome.status for outcome in outcomes] == [EpochPlanStatus.ABORT] * 2  # type: ignore[union-attr]


def test_a_rank_that_never_joins_the_gather_times_out_instead_of_deciding() -> None:
    proposals = [_rank(0, _plan("a", 1.0)), _rank(1, _plan("a", 1.0))]
    outcomes = _decide_on_ranks(proposals, wedged=1, timeout_s=0.3)
    assert isinstance(outcomes[0], DistributedCollectiveTimeout)


def test_a_rank_cannot_propose_for_another() -> None:
    group = LoopbackGroup(2)
    with pytest.raises(ValueError, match="only propose for itself"):
        gather_proposals(group.plane(0), _rank(1, _plan("a", 1.0)))
