"""Fail-closed settlement under injected execution faults (research task C03, cases E08-E10).

Tier T1 of the research validation plan: the *real* ``CompletionCoordinator``,
``Executor`` state machine, ``LifecycleManager``/``RequestLifecycle`` and
``BatchStepPlan`` run on CPU; only the execution boundary is scripted -- fences
that are pending/ready/failing/throwing, an enqueue that fails part-way, and a
resource bundle that records and can fault every callback.  Nothing here proves
GPU safety (a CPU fence is not a CUDA event); it proves the host state machine
never frees, retries or publishes on uncertainty.

Invariants pinned below:

* resources are retired exactly once, only after a quiescent proof, and a host
  settlement fault retains them instead of retrying;
* cancellation suppresses publication but never short-circuits the drain proof;
* a duplicate or re-entrant outcome cannot commit or publish a second time;
* an unknown completion quarantines the ticket and stops all new admission.
"""

from __future__ import annotations

import dataclasses
import itertools
from types import SimpleNamespace

import pytest
import torch
from ayaka.executor.base import Executor
from ayaka.executor.completion import CompletionCoordinator, CompletionResult
from ayaka.executor.ticket import (
    ExecutionTicket,
    FenceResult,
    SampleOutputs,
    TerminalStatus,
    TicketState,
    WorkState,
)
from ayaka.handles import SequenceHandle
from ayaka.plan import SamplingPlan
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.request.machine import next_request_epoch
from ayaka.request.schema import Request, RequestId, StopCriteria
from ayaka.request.states import RequestState
from ayaka.sched.plan import BatchStepPlan, Phase, ScheduledSlice

PENDING = FenceResult(WorkState.PENDING)
DONE = FenceResult(WorkState.SUCCEEDED, quiescent=True)
FAILED_BUSY = FenceResult(WorkState.FAILED, quiescent=False, error="device lost")
FAILED_QUIET = FenceResult(WorkState.FAILED, quiescent=True, error="kernel fault")


class ScriptedFence:
    """Nonblocking fence whose answer the test controls; ``state`` may be an exception."""

    def __init__(self, state: object = PENDING) -> None:
        self.state = state
        self.queries = 0

    def query(self):
        self.queries += 1
        if isinstance(self.state, BaseException):
            raise self.state
        return self.state

    def release(self) -> None:
        self.state = DONE


class RecordingResources:
    """``ExecutionResources`` fake: records order, faults on demand, versions follow the plan."""

    generation = None

    def __init__(self, journal: list[str], faults: dict[str, BaseException] | None = None):
        self.journal = journal
        self.faults = faults or {}
        self.calls: list[str] = []
        self.retired: list[tuple[bool, tuple]] = []
        self.versions: tuple[int, ...] | None = None
        self.override: tuple[int, ...] | None = None

    def _hit(self, name: str) -> None:
        self.calls.append(name)
        self.journal.append(name)
        if name in self.faults:
            raise self.faults[name]

    def adopt(self, ticket_id, prepared) -> None:
        self.versions = tuple(s.expected_state_version + 1 for s in prepared.step.slices)
        self._hit("adopt")

    def mark_submitted(self, ticket_id) -> None:
        self._hit("mark_submitted")

    def commit(self, ticket_id, *, written=None):
        self._hit("commit")
        return self.override if self.override is not None else self.versions

    def retire(self, ticket_id, *, succeeded, discarded_sequences=()) -> None:
        self._hit("retire")
        self.retired.append((succeeded, tuple(discarded_sequences)))


class Backend(Executor):
    """Executor whose enqueue/drain behaviour is scripted per test."""

    def __init__(self, journal: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.journal = journal
        self.fences: list[ScriptedFence] = []
        self.enqueues = 0
        self.closes = 0
        self.enqueue_error: BaseException | None = None
        self.track_fence_on_enqueue = True
        self.fences_per_enqueue = 1
        self.provide_samples = True
        self.drain_error: BaseException | None = None
        self.drain_state: object = DONE
        self.drain_fences: list[ScriptedFence] = []

    def _enqueue(self, ticket: ExecutionTicket) -> None:
        self.enqueues += 1
        self.journal.append("enqueue")
        if self.track_fence_on_enqueue:
            for _ in range(self.fences_per_enqueue):
                fence = ScriptedFence()
                self.fences.append(fence)
                self.track_fence(ticket, fence)
        if self.provide_samples:
            rows = len(ticket.prepared.step.sampling_rows)
            self.set_samples(ticket, SampleOutputs(torch.arange(100, 100 + rows)))
        if self.enqueue_error is not None:
            raise self.enqueue_error

    def _begin_drain(self, ticket: ExecutionTicket) -> ScriptedFence:
        if self.drain_error is not None:
            raise self.drain_error
        fence = ScriptedFence(self.drain_state)
        self.drain_fences.append(fence)
        return fence

    def _close(self) -> None:
        self.closes += 1


class World:
    """Real owners wired together; ``admit``/``prepared`` build valid fixtures."""

    def __init__(self, **backend_kwargs) -> None:
        self.journal: list[str] = []
        self.requests = LifecycleManager()
        self.executor = Backend(self.journal, **backend_kwargs)
        self.executor.initialize()
        self.coordinator = CompletionCoordinator(self.executor, self.requests)
        self._handles = itertools.count(1)
        self._steps = itertools.count(0)

    def admit(self, request_id: str = "a", *, prompt: int = 3, max_tokens: int = 4):
        request = Request(
            request_id=RequestId(request_id),
            prompt_token_ids=tuple(range(1, prompt + 1)),
            stop=StopCriteria(max_tokens=max_tokens),
        )
        lifecycle = self.requests.create(request)
        self.requests.advance_to_queue(request_id)
        lifecycle.machine.on_admitted()
        lifecycle.bind_sequence(SequenceHandle(next(self._handles), 1), state_version=0)
        return lifecycle

    def prepared(self, *lifecycles: RequestLifecycle, step_id: int | None = None):
        """One slice per request: finish the prompt (with a sample) or decode one token."""
        slices, inputs, rows, offset = [], [], [], 0
        for lifecycle in lifecycles:
            snap = lifecycle.snapshot()
            remaining = snap.prompt_tokens - snap.computed_tokens
            scheduled = ScheduledSlice(
                lifecycle.request_id,
                snap.sequence_epoch,
                snap.state_version,
                snap.computed_tokens,
                remaining if remaining > 0 else 1,
                Phase.PREFILL if remaining > 0 else Phase.DECODE,
                True,
            )
            slices.append(scheduled)
            inputs.append(snap)
            rows.append(offset + scheduled.query_count - 1)
            offset += scheduled.query_count
        plan = BatchStepPlan(
            step_id=next(self._steps) if step_id is None else step_id,
            execution_plan_id="plan",
            slices=tuple(slices),
            inputs=tuple(inputs),
            padded_num_tokens=offset,
            sampling_rows=tuple(rows),
            sampling=SamplingPlan(num_rows=len(rows)),
        )
        return SimpleNamespace(step=plan)

    def adopt(self, *lifecycles: RequestLifecycle, faults=None):
        resources = RecordingResources(self.journal, faults)
        ticket = self.coordinator.adopt(self.prepared(*lifecycles), resources)
        return ticket, resources

    def launch(self, *lifecycles: RequestLifecycle, faults=None):
        ticket, resources = self.adopt(*lifecycles, faults=faults)
        self.coordinator.launch(ticket)
        return ticket, resources

    def release(self, index: int = -1) -> None:
        self.journal.append("fence_done")
        self.executor.fences[index].release()


def _only(results: tuple[CompletionResult, ...]) -> CompletionResult:
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# Exactly-once logical completion
# ---------------------------------------------------------------------------


def test_prefill_then_decode_settle_in_order_and_each_ticket_retires_once() -> None:
    world = World()
    lifecycle = world.admit(max_tokens=4)

    first, res1 = world.launch(lifecycle)
    assert world.coordinator.poll() == ()  # fence still pending: nothing may settle
    world.release()
    result = _only(world.coordinator.poll())
    assert result.status is TerminalStatus.SUCCEEDED
    assert [p.token_id for p in result.published] == [100]
    assert res1.calls == ["adopt", "mark_submitted", "commit", "retire"]
    assert res1.retired == [(True, ())]
    assert lifecycle.output_token_ids == (100,)
    assert world.executor.tickets == ()

    second, res2 = world.launch(lifecycle)
    world.release()
    result = _only(world.coordinator.poll())
    assert [p.token_id for p in result.published] == [100]
    assert lifecycle.output_token_ids == (100, 100)
    assert lifecycle.state is RequestState.DECODING
    assert res2.calls == ["adopt", "mark_submitted", "commit", "retire"]
    assert world.executor.tickets == ()
    assert first.id != second.id


def test_duplicate_delivery_of_a_settled_outcome_commits_and_publishes_nothing() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)
    world.release()
    (outcome,) = world.executor.poll_terminal()
    assert outcome is world.executor.poll_terminal()[0]  # redelivered until acknowledged

    assert world.coordinator.handle(outcome) is not None
    assert world.coordinator.handle(outcome) is None  # ticket already retired
    assert world.coordinator.poll() == ()
    assert resources.calls.count("commit") == 1
    assert resources.calls.count("retire") == 1
    assert lifecycle.output_token_ids == (100,)
    assert ticket.state is TicketState.RETIRED


def test_reentrant_settlement_of_the_same_ticket_is_ignored() -> None:
    world = World()
    lifecycle = world.admit()
    _, resources = world.launch(lifecycle)
    world.release()
    (outcome,) = world.executor.poll_terminal()
    inner: list[CompletionResult | None] = []
    original_commit = resources.commit

    def commit_and_reenter(ticket_id, *, written=None):
        inner.append(world.coordinator.handle(outcome))  # callback re-enters the coordinator
        return original_commit(ticket_id, written=written)

    resources.commit = commit_and_reenter  # type: ignore[method-assign]
    assert world.coordinator.handle(outcome) is not None
    assert inner == [None]
    assert resources.calls.count("commit") == 1


def test_stale_epoch_completion_is_ignored_but_physical_state_still_settles() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)
    sequence = ticket.prepared.step.inputs[0].sequence
    lifecycle.machine.sequence_epoch = next_request_epoch()  # recompute/rebind happened
    world.release()

    result = _only(world.coordinator.poll())
    assert result.published == ()
    assert result.ignored_requests == ("a",)
    assert lifecycle.output_token_ids == ()
    assert resources.calls.count("commit") == 1  # the forward really ran on the device
    assert resources.retired == [(True, (sequence,))]


def test_request_id_reuse_cannot_receive_an_old_completion() -> None:
    world = World()
    old = world.admit("a")
    _, resources = world.launch(old)
    old.token.cancel("client left")
    old.machine.on_cancelled()
    world.requests.retire_terminal(old)
    world.requests.forget("a")
    new = world.admit("a")  # same id, brand-new incarnation
    before = (new.state, new.output_token_ids, new.sequence_epoch)

    world.release()
    result = _only(world.coordinator.poll())
    assert result.ignored_requests == ("a",)
    assert result.published == ()
    assert (new.state, new.output_token_ids, new.sequence_epoch) == before
    assert resources.calls.count("retire") == 1


def test_independent_tickets_settle_out_of_order() -> None:
    world = World()
    a, b = world.admit("a"), world.admit("b")
    ticket_a, res_a = world.launch(a)
    ticket_b, res_b = world.launch(b)

    world.release(1)  # only the second flight finished
    result = _only(world.coordinator.poll())
    assert result.ticket_id == ticket_b.id
    assert b.output_token_ids == (100,) and a.output_token_ids == ()
    assert res_a.calls == ["adopt", "mark_submitted"]  # untouched while its fence is pending
    assert world.executor.tickets == (ticket_a,)

    world.release(0)
    assert _only(world.coordinator.poll()).ticket_id == ticket_a.id
    assert res_a.calls.count("retire") == res_b.calls.count("retire") == 1
    assert world.executor.tickets == ()


def test_a_request_has_at_most_one_unsettled_slice() -> None:
    world = World()
    lifecycle = world.admit()
    world.launch(lifecycle)
    second = RecordingResources(world.journal)
    with pytest.raises(ValueError, match="unfinished slice"):
        world.coordinator.adopt(world.prepared(lifecycle), second)
    assert second.calls == []  # preflight failure leaves ownership with the caller
    assert len(world.executor.tickets) == 1


# ---------------------------------------------------------------------------
# Admission preflight keeps ownership with the caller
# ---------------------------------------------------------------------------


def test_cancelled_request_is_rejected_before_any_ownership_transfer() -> None:
    world = World()
    lifecycle = world.admit()
    lifecycle.token.cancel("gone")
    resources = RecordingResources(world.journal)
    with pytest.raises(ValueError, match="terminal or cancelled"):
        world.coordinator.adopt(world.prepared(lifecycle), resources)
    assert resources.calls == []
    assert world.executor.tickets == ()


def test_stale_input_snapshot_is_rejected_before_ownership_transfer() -> None:
    world = World()
    lifecycle = world.admit()
    prepared = world.prepared(lifecycle)
    lifecycle.machine.num_computed_tokens = 1  # the request moved on after planning
    resources = RecordingResources(world.journal)
    with pytest.raises(ValueError):
        world.coordinator.adopt(prepared, resources)
    assert resources.calls == []
    assert world.executor.tickets == ()


def test_prepared_snapshot_that_differs_from_the_lifecycle_is_rejected() -> None:
    """Same shape and bounds, different content: only the equality check can catch it."""
    world = World()
    lifecycle = world.admit()
    prepared = world.prepared(lifecycle)
    forged = dataclasses.replace(prepared.step.inputs[0], known_tokens=(9, 9, 9))
    plan = dataclasses.replace(prepared.step, inputs=(forged,))
    resources = RecordingResources(world.journal)
    with pytest.raises(ValueError, match="snapshot is stale"):
        world.coordinator.adopt(SimpleNamespace(step=plan), resources)
    assert resources.calls == []
    assert world.executor.tickets == ()


def test_capacity_ceiling_refuses_adoption_without_taking_ownership() -> None:
    world = World(max_inflight=1)
    a, b = world.admit("a"), world.admit("b")
    world.launch(a)
    resources = RecordingResources(world.journal)
    with pytest.raises(RuntimeError, match="no submission capacity"):
        world.coordinator.adopt(world.prepared(b), resources)
    assert resources.calls == []


def test_resource_adopt_failure_leaves_no_ticket_and_the_step_id_reusable() -> None:
    world = World()
    lifecycle = world.admit()
    prepared = world.prepared(lifecycle)
    broken = RecordingResources(world.journal, {"adopt": RuntimeError("no workspace")})
    with pytest.raises(RuntimeError, match="no workspace"):
        world.coordinator.adopt(prepared, broken)
    assert world.executor.tickets == ()
    ticket = world.coordinator.adopt(prepared, RecordingResources(world.journal))
    assert ticket.state is TicketState.ADOPTED


# ---------------------------------------------------------------------------
# Cancellation (E08)
# ---------------------------------------------------------------------------


def test_cancel_after_launch_suppresses_publication_but_waits_for_the_fence() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)
    sequence = ticket.prepared.step.inputs[0].sequence
    lifecycle.token.cancel("client disconnect")

    assert world.coordinator.poll() == ()  # cancel alone is not a drain proof
    assert "retire" not in resources.calls
    world.release()
    result = _only(world.coordinator.poll())

    assert result.published == ()
    assert result.ignored_requests == ("a",)
    assert lifecycle.output_token_ids == ()
    assert lifecycle.state is RequestState.CANCELLED
    assert resources.retired == [(True, (sequence,))]
    assert world.journal.index("fence_done") < world.journal.index("retire")


def test_executor_cancel_drains_before_reporting_cancelled() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)

    world.executor.cancel(ticket, "shed load")
    assert ticket.state is TicketState.DRAINING
    assert world.coordinator.poll() == ()
    assert resources.calls == ["adopt", "mark_submitted"]

    world.release()
    result = _only(world.coordinator.poll())
    assert result.status is TerminalStatus.CANCELLED
    assert result.published == ()
    assert "commit" not in resources.calls  # no logical commit for cancelled work
    assert resources.retired[0][0] is False
    assert lifecycle.state is RequestState.CANCELLED


def test_cancel_between_adopt_and_launch_never_enqueues() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.adopt(lifecycle)
    lifecycle.token.cancel("gone before launch")

    world.coordinator.launch(ticket)
    assert world.executor.enqueues == 0
    assert "mark_submitted" not in resources.calls
    result = _only(world.coordinator.poll())
    assert result.status is TerminalStatus.CANCELLED
    assert resources.retired[0][0] is False
    assert world.executor.tickets == ()


def test_cancelling_one_member_leaves_its_batch_peer_intact() -> None:
    world = World()
    a, b = world.admit("a"), world.admit("b")
    ticket, resources = world.launch(a, b)
    a.token.cancel("gone")
    world.release()

    result = _only(world.coordinator.poll())
    assert [p.request_id for p in result.published] == ["b"]
    assert result.ignored_requests == ("a",)
    assert b.output_token_ids == (101,)  # sampling row 1 belongs to the peer
    assert a.output_token_ids == ()
    assert resources.calls.count("commit") == 1
    assert ticket.state is TicketState.RETIRED


def test_cancel_cannot_clear_a_quarantine() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, _ = world.launch(lifecycle)
    world.executor.quarantine(ticket, "unknown device state")
    world.executor.cancel(ticket, "operator")
    assert ticket.state is TicketState.QUARANTINED


# ---------------------------------------------------------------------------
# Enqueue / fence faults (E10)
# ---------------------------------------------------------------------------


def test_enqueue_failure_requires_a_whole_ticket_drain_proof() -> None:
    world = World()
    world.executor.enqueue_error = RuntimeError("copy launch failed")
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)  # the failure is observable, not raised

    assert ticket.state is TicketState.DRAINING
    world.executor.drain_state = PENDING
    assert world.coordinator.poll() == ()
    assert resources.calls == ["adopt", "mark_submitted"]  # nothing retired on a guess
    # The fence that WAS recorded would report success, but it covers only part
    # of what may have been submitted -- the drain proof is the only authority.
    world.executor.fences[0].release()
    assert world.coordinator.poll() == ()
    assert world.executor.fences[0].queries == 0

    world.executor.drain_fences[0].state = DONE
    result = _only(world.coordinator.poll())
    assert result.status is TerminalStatus.FAILED
    assert "launch failed" in result.error
    assert "commit" not in resources.calls
    assert resources.retired[0][0] is False
    assert lifecycle.state is RequestState.FAILED


def test_enqueue_that_records_no_fence_is_a_partial_launch() -> None:
    world = World()
    world.executor.track_fence_on_enqueue = False
    lifecycle = world.admit()
    ticket, _ = world.launch(lifecycle)
    assert ticket.state is TicketState.DRAINING
    assert "without any completion fence" in ticket.error


def test_non_exception_during_enqueue_still_leaves_a_drainable_ticket() -> None:
    world = World()
    world.executor.enqueue_error = KeyboardInterrupt()
    lifecycle = world.admit()
    ticket, _ = world.adopt(lifecycle)
    with pytest.raises(KeyboardInterrupt):
        world.coordinator.launch(ticket)
    assert ticket.state is TicketState.DRAINING  # retained for a drain, never dropped


def test_mark_submitted_failure_quarantines_and_never_enqueues() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.adopt(lifecycle, faults={"mark_submitted": RuntimeError("boom")})
    world.coordinator.launch(ticket)

    assert ticket.state is TicketState.QUARANTINED
    assert world.executor.enqueues == 0
    assert not world.executor.accepting_work
    assert not world.executor.has_submission_capacity()
    assert world.coordinator.poll() == ()
    assert "retire" not in resources.calls


def test_unprovable_drain_quarantines_and_retains_every_resource() -> None:
    world = World()
    world.executor.enqueue_error = RuntimeError("partial launch")
    world.executor.drain_error = RuntimeError("stream lost")
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)

    assert world.coordinator.poll() == ()
    assert ticket.state is TicketState.QUARANTINED
    assert world.executor.quarantined == (ticket,)
    assert not world.executor.accepting_work
    assert not world.executor.pending_completion  # quarantine is not "progress"
    assert resources.calls == ["adopt", "mark_submitted"]
    assert not lifecycle.is_terminal  # the request is not declared finished on a guess


def test_fence_query_exception_quarantines_and_recovers_only_through_a_drain_proof() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)
    world.executor.fences[0].state = RuntimeError("cuda event query failed")
    world.executor.drain_state = PENDING

    assert world.coordinator.poll() == ()
    assert ticket.state is TicketState.QUARANTINED
    assert world.coordinator.poll() == ()  # still no proof: stays quarantined
    assert "retire" not in resources.calls

    world.executor.drain_fences[0].state = DONE
    result = _only(world.coordinator.poll())
    assert result.status is TerminalStatus.FAILED  # never SUCCEEDED after an unknown query
    assert result.published == ()
    assert resources.retired[0][0] is False


def test_malformed_fence_answer_is_treated_as_unknown_completion() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, _ = world.launch(lifecycle)
    world.executor.fences[0].state = None  # not a FenceResult
    world.executor.drain_state = PENDING
    assert world.coordinator.poll() == ()
    assert ticket.state is TicketState.QUARANTINED
    assert "FenceResult" in ticket.error


def test_failed_fence_that_is_not_quiescent_quarantines() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)
    world.executor.fences[0].state = FAILED_BUSY
    assert world.coordinator.poll() == ()
    assert ticket.state is TicketState.QUARANTINED
    assert "retire" not in resources.calls


def test_quiescent_failure_fails_the_request_and_retires_unsuccessfully() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle)
    world.executor.fences[0].state = FAILED_QUIET

    result = _only(world.coordinator.poll())
    assert result.status is TerminalStatus.FAILED
    assert result.error == "kernel fault"
    assert result.published == ()
    assert lifecycle.state is RequestState.FAILED
    assert "commit" not in resources.calls
    assert resources.retired[0][0] is False
    assert ticket.state is TicketState.RETIRED


def test_success_needs_every_fence_quiescent() -> None:
    world = World()
    world.executor.fences_per_enqueue = 2  # e.g. a compute stream and a copy stream
    lifecycle = world.admit()
    world.launch(lifecycle)

    world.release(0)
    assert world.coordinator.poll() == ()  # the second consumer may still be reading
    world.release(1)
    assert _only(world.coordinator.poll()).status is TerminalStatus.SUCCEEDED


def test_success_without_sampling_results_is_a_failure_not_a_silent_success() -> None:
    world = World()
    world.executor.provide_samples = False
    lifecycle = world.admit()
    world.launch(lifecycle)
    world.release()

    result = _only(world.coordinator.poll())
    assert result.status is TerminalStatus.FAILED
    assert "sampling output missing" in result.error
    assert result.published == ()


# ---------------------------------------------------------------------------
# Host settlement faults: retain, never retry (E10)
# ---------------------------------------------------------------------------


def test_commit_fault_quarantines_as_host_failure_and_is_never_retried() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle, faults={"commit": RuntimeError("kv commit failed")})
    world.release()

    assert world.coordinator.poll() == ()
    assert ticket.state is TicketState.QUARANTINED
    assert "completion settlement failed" in ticket.error
    for _ in range(3):
        assert world.coordinator.poll() == ()
    assert resources.calls.count("commit") == 1  # no blind retry
    assert "retire" not in resources.calls
    assert lifecycle.output_token_ids == ()
    assert ticket in world.executor.tickets
    assert not world.executor.accepting_work


def test_retire_fault_quarantines_and_the_ticket_is_never_acknowledged() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle, faults={"retire": RuntimeError("release failed")})
    world.release()

    assert world.coordinator.poll() == ()
    assert ticket.state is TicketState.QUARANTINED
    assert ticket in world.executor.tickets
    assert resources.calls.count("retire") == 1
    assert world.coordinator.poll() == ()
    assert resources.calls.count("retire") == 1  # not re-run on later polls


def test_wrong_kv_versions_from_commit_are_a_fault_not_a_publication() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.adopt(lifecycle)
    resources.override = (99,)
    world.coordinator.launch(ticket)
    world.release()

    assert world.coordinator.poll() == ()
    assert ticket.state is TicketState.QUARANTINED
    assert "unexpected KV versions" in ticket.error
    assert lifecycle.output_token_ids == ()
    assert "retire" not in resources.calls


def test_non_exception_settlement_interrupt_quarantines_then_propagates() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.launch(lifecycle, faults={"commit": KeyboardInterrupt()})
    world.release()
    with pytest.raises(KeyboardInterrupt):
        world.coordinator.poll()
    assert ticket.state is TicketState.QUARANTINED
    assert "retire" not in resources.calls


def test_only_settled_terminal_work_can_be_acknowledged() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, _ = world.launch(lifecycle)

    with pytest.raises(ValueError, match="only settled terminal work"):
        world.executor.acknowledge_retired(ticket)  # still in flight
    world.release()
    (outcome,) = world.executor.poll_terminal()
    assert ticket.state is TicketState.COMPLETED and outcome.status is TerminalStatus.SUCCEEDED
    with pytest.raises(ValueError, match="only settled terminal work"):
        world.executor.acknowledge_retired(ticket)  # terminal, but resources not retired yet
    assert ticket in world.executor.tickets


# ---------------------------------------------------------------------------
# Quiescent shutdown
# ---------------------------------------------------------------------------


def test_shutdown_does_not_close_while_a_flight_is_unsettled() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, _ = world.launch(lifecycle)

    result = world.coordinator.shutdown()
    assert result.closed is False
    assert result.pending == (ticket.id,)
    assert world.executor.closes == 0

    world.release()
    result = world.coordinator.shutdown()
    assert result.closed is True
    assert len(result.completed) == 1
    assert world.executor.closes == 1
    assert world.coordinator.shutdown().closed is True
    assert world.executor.closes == 1  # idempotent


def test_a_quarantined_ticket_blocks_close_forever() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, _ = world.launch(lifecycle, faults={"commit": RuntimeError("boom")})
    world.release()
    for _ in range(3):
        result = world.coordinator.shutdown()
        assert result.closed is False
        assert result.pending == (ticket.id,)
    assert world.executor.closes == 0


def test_shutdown_cancels_adopted_work_instead_of_enqueueing_it() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, resources = world.adopt(lifecycle)

    result = world.coordinator.shutdown()
    assert world.executor.enqueues == 0
    assert result.closed is True
    assert [r.status for r in result.completed] == [TerminalStatus.CANCELLED]
    assert resources.retired[0][0] is False


def test_launch_after_stop_began_cancels_instead_of_enqueueing() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, _ = world.adopt(lifecycle)
    world.executor.shutdown()  # cancels the ADOPTED ticket and starts stopping
    world.coordinator.launch(ticket)  # a late launch is absorbed
    assert world.executor.enqueues == 0
    with pytest.raises(RuntimeError, match="stopping or closed"):
        world.executor.initialize()


# ---------------------------------------------------------------------------
# Diagnostics never touch fences
# ---------------------------------------------------------------------------


def test_snapshot_counts_quarantine_without_polling_a_fence() -> None:
    world = World()
    lifecycle = world.admit()
    ticket, _ = world.launch(lifecycle)
    world.executor.quarantine(ticket, "unknown device state")
    world.executor.quarantine(ticket, "still unknown")
    queries_before = world.executor.fences[0].queries

    snapshot = world.executor.snapshot_metrics()
    assert snapshot.quarantined_tickets == 1
    assert dict(snapshot.counters)["quarantine_entries"] == 1  # counted on entry only
    assert world.executor.fences[0].queries == queries_before
