"""D01: resident engines step through the distributed protocol on fake ranks.

Each rank owns a real ``ResidentKVEngine`` whose worker is a
``DistributedStepWorker`` over loopback collectives (see ``fake_ranks``). The
source rank broadcasts every step, the others cross-check it against the step
their own scheduler staged, and nothing is published before the group's
success proof and commit acknowledgements.

Two failure classes must stay distinct:

* refused before launch (identity, plan or step mismatch): every rank rolls
  back, no rank samples, and the group stays usable and shuts down cleanly;
* failed after launch (execution error, stall, commit failure, diverged
  collective sequence): every rank fails closed, the step's tokens are not
  published anywhere, and no rank reports a clean shutdown.

CPU threads prove the host protocol only: no device collective, NCCL timing or
GPU memory reuse is exercised here.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from ayaka.distributed.process_group import DistributedStepError
from ayaka.exceptions import InvariantViolationError
from ayaka.worker.distributed import LocalStepHost
from ayaka.worker.lifecycle import WorkerState
from ayaka.worker.rank_local import RankLocalKVWorkerState
from fake_ranks import (
    LoopbackGroup,
    ScriptedLogits,
    build_engine,
    rank_local_epochs,
    requests,
    run_ranks,
)


@pytest.fixture(autouse=True)
def _epochs():
    with rank_local_epochs():
        yield


def _reference():
    logits = ScriptedLogits()
    (result,) = run_ranks([build_engine(logits=logits)], [requests()])
    assert result.closed and set(result.states.values()) == {"finished"}
    return result, logits


def _ranks(*, timeout_s: float = 5.0, step_timeout_s: float = 2.0, **per_rank):
    """Two fake ranks; ``per_rank`` maps a build option to a (rank0, rank1) pair."""
    group = LoopbackGroup(2, timeout_s=timeout_s)
    logits = [ScriptedLogits(), ScriptedLogits()]
    for rank, fault in enumerate(per_rank.pop("fault", (None, None))):
        logits[rank].fault = fault
    engines = [
        build_engine(
            group.plane(rank),
            logits=logits[rank],
            step_timeout_s=step_timeout_s,
            **{key: values[rank] for key, values in per_rank.items()},
        )
        for rank in range(2)
    ]
    return group, engines, logits


def _coordinator_state(engine) -> RankLocalKVWorkerState:
    return engine.worker.coordinator.worker.state


# ---------------------------------------------------------------------------
# Success: lockstep parity and coordinated shutdown
# ---------------------------------------------------------------------------


def test_two_ranks_match_the_single_rank_engine_and_shut_down_together() -> None:
    reference, reference_logits = _reference()
    group, engines, logits = _ranks()

    results = run_ranks(engines, [requests(), requests()])

    for rank, result in enumerate(results):
        assert result.step_error is None and result.close_error is None
        assert result.closed and engines[rank].worker.closed
        assert result.events == reference.events
        assert logits[rank].sampled == reference_logits.sampled
        assert set(result.states.values()) == {"finished"}
    # Every rank ran the same collective sequence.
    assert group.collectives[0] == group.collectives[1] > 0


# ---------------------------------------------------------------------------
# Refused before launch: agreed rollback, group stays usable
# ---------------------------------------------------------------------------


def _assert_refused_everywhere(results, engines, logits) -> None:
    for rank, result in enumerate(results):
        assert logits[rank].sampled == [], "a refused step must not launch on any rank"
        assert result.events == []
        assert set(result.states.values()) == {"failed"}
        assert _coordinator_state(engines[rank]) is not RankLocalKVWorkerState.FAILED
        assert result.close_error is None and result.closed  # clean group shutdown
    assert results[0].states == results[1].states


def test_a_rank_whose_scheduler_diverges_refuses_and_nobody_launches() -> None:
    _, engines, logits = _ranks()
    # Rank 1's last prompt is one token longer: its staged step disagrees.
    results = run_ranks(engines, [requests(), requests(extra=1)])
    _assert_refused_everywhere(results, engines, logits)


def test_a_rank_on_other_weights_refuses_every_step() -> None:
    _, engines, logits = _ranks(weights_revision=("w", "w-new"))
    results = run_ranks(engines, [requests(), requests()])
    _assert_refused_everywhere(results, engines, logits)


def test_a_stale_step_on_one_rank_is_refused_by_the_group() -> None:
    _, engines, logits = _ranks()
    # Rank 1 already retired later step ids (a replayed or duplicated epoch).
    engines[1].worker.coordinator.worker._last_step = 1_000
    results = run_ranks(engines, [requests(), requests()])
    _assert_refused_everywhere(results, engines, logits)


def test_an_unbound_host_refuses_and_a_bound_identity_never_changes() -> None:
    host = LocalStepHost(SimpleNamespace(), plan_identity=None)
    with pytest.raises(RuntimeError, match="no bound execution plan"):
        host.plan_identity()
    host.bind_plan_identity({"execution_plan_id": "a"})
    host.bind_plan_identity({"execution_plan_id": "a"})  # idempotent
    with pytest.raises(InvariantViolationError):
        host.bind_plan_identity({"execution_plan_id": "b"})
    assert host.plan_identity() == {"execution_plan_id": "a"}


def test_from_configs_binds_the_engine_plan_identity() -> None:
    _, engines, _ = _ranks()
    for engine in engines:
        identity = engine.worker.host.plan_identity()
        assert identity["execution_plan_id"] == engine._plan.execution.plan_id
        assert identity["weights_revision"] == "w"
    for engine in engines:  # never stepped: a plain local close is enough
        engine.worker.coordinator.worker.close()
        engine.worker.local.close()


# ---------------------------------------------------------------------------
# Failed after launch: fail-closed on every rank, nothing published
# ---------------------------------------------------------------------------


def _raise_at(call: int):
    def fault(index: int) -> None:
        if index == call:
            raise RuntimeError("injected device failure")

    return fault


def _stall_at(call: int, seconds: float):
    def fault(index: int) -> None:
        if index == call:
            time.sleep(seconds)

    return fault


def _assert_failed_closed(results, engines, *, published_tokens: int) -> None:
    for rank, result in enumerate(results):
        assert _coordinator_state(engines[rank]) is RankLocalKVWorkerState.FAILED
        assert engines[rank].worker.state is WorkerState.FAILED
        assert not result.closed, "a failed group must not report a clean shutdown"
        # A refused shutdown vote, or a collective timeout once the group broke.
        assert isinstance(result.close_error, DistributedStepError)
        assert set(result.states.values()) == {"failed"}
        # Nothing of the failed step (or later) reached any client.
        assert max((tokens for _, tokens, _ in result.events), default=0) == published_tokens
    assert results[0].events == results[1].events


def test_an_execution_failure_on_one_rank_fails_every_rank_closed() -> None:
    _, engines, logits = _ranks(fault=(None, _raise_at(2)))
    results = run_ranks(engines, [requests(), requests()])
    # Steps 0 and 1 (prefill + one decode) published 2 tokens per request.
    _assert_failed_closed(results, engines, published_tokens=2)
    assert len(logits[0].sampled) == 3  # rank 0 ran step 2; its tokens were dropped


def test_a_stalled_rank_times_out_the_group_instead_of_hanging() -> None:
    _, engines, _ = _ranks(
        timeout_s=0.5, step_timeout_s=0.5, fault=(None, _stall_at(1, seconds=1.5))
    )
    started = time.monotonic()
    results = run_ranks(engines, [requests(), requests()])
    assert time.monotonic() - started < 10.0
    _assert_failed_closed(results, engines, published_tokens=1)


def test_a_commit_acknowledgement_failure_publishes_nothing() -> None:
    _, engines, _ = _ranks()
    host = engines[1].worker.host
    original = host.commit
    calls = []

    def commit(envelope, result) -> None:
        calls.append(envelope.step_id)
        if len(calls) == 2:
            raise RuntimeError("injected commit failure")
        original(envelope, result)

    host.commit = commit
    results = run_ranks(engines, [requests(), requests()])
    _assert_failed_closed(results, engines, published_tokens=1)


def test_a_diverged_collective_sequence_fails_closed() -> None:
    _, engines, _ = _ranks(timeout_s=1.0, step_timeout_s=1.0)
    # Rank 0 has no work and goes straight to the shutdown vote while rank 1
    # waits for a step broadcast: the ranks disagree on the next collective.
    results = run_ranks(engines, [[], requests()])
    assert not results[0].closed and results[0].close_error is not None
    assert isinstance(results[0].close_error, DistributedStepError)
    assert _coordinator_state(engines[1]) is RankLocalKVWorkerState.FAILED
    assert results[1].events == [] and not results[1].closed


# ---------------------------------------------------------------------------
# Real collectives: two OS processes over Gloo
# ---------------------------------------------------------------------------


def _gloo_ranks(tmp_path, *fault_call: int) -> list[dict]:
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    tests = Path(__file__).parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tests.parent / "python"), str(tests)])
    store = tmp_path / "store"
    outputs = [tmp_path / f"rank{rank}.json" for rank in range(2)]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(tests / "fake_ranks.py"),
                str(rank),
                "2",
                str(store),
                str(outputs[rank]),
                *map(str, fault_call),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for rank in range(2)
    ]
    for process in processes:
        try:
            output, _ = process.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            for other in processes:
                other.kill()
            raise
        assert process.returncode == 0, output.decode()[-2000:]
    return [json.loads(path.read_text()) for path in outputs]


@pytest.mark.slow
def test_two_gloo_processes_match_the_single_rank_engine(tmp_path) -> None:
    reference, reference_logits = _reference()
    for rank in _gloo_ranks(tmp_path):
        assert rank["backend"] == "gloo"
        assert rank["closed"] and rank["close_error"] is None
        assert [tuple(event) for event in rank["events"]] == reference.events
        assert [(step, tuple(tokens)) for step, tokens in rank["sampled"]] == (
            reference_logits.sampled
        )


@pytest.mark.slow
def test_a_gloo_rank_failure_fails_both_processes_closed(tmp_path) -> None:
    for rank in _gloo_ranks(tmp_path, 2):
        assert not rank["closed"] and rank["close_error"] is not None
        assert set(rank["states"].values()) == {"failed"}
        assert max(tokens for _, tokens, _ in rank["events"]) == 2
