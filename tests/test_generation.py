"""Rust transcript cases and transactional failure/replay contracts, CPU only."""

from __future__ import annotations

import gc
import subprocess
import sys
from dataclasses import dataclass, replace

import pytest
from ayaka.generation import (
    Controlled,
    DFlash,
    DraftBudget,
    DraftCheckpoint,
    DraftHistory,
    DraftRoundState,
    EndOfGeneration,
    FeatureRef,
    FeatureRows,
    FinishReason,
    Generation,
    GenerationSeed,
    HeadOutcome,
    InputLayout,
    InputSpan,
    MethodCheckpoint,
    MethodCheckpointError,
    MethodChoice,
    MethodKind,
    MethodRequirements,
    MethodWork,
    Mtp,
    Options,
    Plain,
    ReasoningBudget,
    RoundState,
    SelectedToken,
    SelectSpec,
    Shaping,
    StartedRound,
    Verification,
    WorkKind,
    accept_prefix,
    selection_history,
    verification_selects,
)
from ayaka.generation.generation import DraftStats
from hypothesis import given
from hypothesis import strategies as st


@dataclass(frozen=True)
class RejectThree:
    offset: int = 0

    def position(self):
        return self.offset

    def fork(self):
        return replace(self)

    def stage(self, tokens):
        if 3 in tokens:
            raise ValueError("rejected")
        return replace(self, offset=self.offset + len(tokens))

    def forced(self, limit):
        return ()

    def mask(self):
        return (self.offset,)


@dataclass(frozen=True)
class Forced:
    tokens: tuple[int, ...]
    offset: int = 0

    def position(self):
        return self.offset

    def fork(self):
        return replace(self)

    def stage(self, tokens):
        expected = self.tokens[self.offset : self.offset + len(tokens)]
        if tuple(tokens) != expected:
            raise ValueError("unexpected token")
        return replace(self, offset=self.offset + len(tokens))

    def forced(self, limit):
        return self.tokens[self.offset : self.offset + limit]

    def mask(self):
        return (1 << self.tokens[self.offset],) if self.offset < len(self.tokens) else (0,)


def options(**kwargs):
    return replace(Options(12, 16, 64, 32), **kwargs)


def generation(**kwargs):
    value = Generation((1, 2), InputLayout(2), options(**kwargs))
    value.resume_at()
    return value


def settle(value, samples, allowance=16):
    bound = value.publication_bound(allowance)
    round_ = value.start_round("r", allowance)
    assert isinstance(round_, StartedRound)
    assert round_.round_publication_bound() <= bound
    transition = round_.prepare_round_transition("r", tuple(samples))
    return round_.commit(transition)[0]


class Reader:
    def read(self, feature, start, count):
        assert 0 <= start <= start + count <= feature.rows
        return FeatureRows(bytes(range(start, start + count)), count, 1)


@pytest.mark.parametrize(
    "proposal,samples,expected",
    [
        ((1, 2, 3), (1, 2, 3, 4), 3),
        ((1, 2, 3), (1, 8, 3, 4), 1),
        ((1, 2, 3), (8, 2, 3, 4), 0),
        ((1, 9, 3), (1, 9, 3, 4), 1),
        ((), (), 0),
    ],
)
def test_rust_acceptance_table(proposal, samples, expected):
    assert accept_prefix(proposal, samples, {9}) == expected


def test_acceptance_rejects_missing_rows():
    with pytest.raises(ValueError):
        accept_prefix((1,), ())


@given(st.lists(st.integers(0, 31), max_size=20), st.lists(st.integers(0, 31), max_size=21))
def test_acceptance_is_the_longest_equal_nonstop_prefix(proposal, samples):
    samples += [0] * max(0, len(proposal) - len(samples))
    count = accept_prefix(proposal, samples, {9})
    assert proposal[:count] == samples[:count]
    assert 9 not in proposal[:count]
    assert count == len(proposal) or proposal[count] != samples[count] or proposal[count] == 9


@pytest.mark.parametrize("generated,verified", [(0, 0), (3, 1), (3, 3), (255, 255)])
def test_rust_draft_budget_valid(generated, verified):
    DraftBudget(generated, verified).validate(255)


@pytest.mark.parametrize("generated,verified", [(1, 2), (256, 1), (-1, 0), (True, 0)])
def test_draft_budget_construction_bounds(generated, verified):
    with pytest.raises((ValueError, TypeError)):
        DraftBudget(generated, verified)


@pytest.mark.parametrize("budget,cap", [(DraftBudget(3, 0), 3), (DraftBudget(3, 1), 2)])
def test_draft_budget_validation_bounds(budget, cap):
    with pytest.raises(ValueError):
        budget.validate(cap)


def test_rust_history_and_poisoned_previews():
    history = selection_history(tuple(range(70)), (70, 71))
    assert len(history) == 64 and history[0] == 8 and history[-2:] == (70, 71)
    constraint = RejectThree(1)
    selects = verification_selects(
        5, (1,), (2, 3, 4), constraint, shaping=Shaping(repetition_penalty=1.1), seed=7
    )
    assert [s.position for s in selects] == [5, 6, 7, 8]
    assert [s.mask for s in selects] == [(1,), (2,), (2,), (2,)]
    assert selects[-1].history[-3:] == (2, 3, 4)
    assert constraint.position() == 1
    with pytest.raises(ValueError):
        selection_history((1 << 31,))


def allowed(state):
    return tuple(i for i in range(32) if state.mask()[i // 32] & (1 << (i % 32)))


def controlled(budget=None, suppress=False, never=frozenset()):
    return Controlled.compose(
        None,
        EndOfGeneration.SUPPRESS if suppress else EndOfGeneration.STOP,
        frozenset({13}),
        never,
        budget,
        14,
    )


def test_uncontrolled_is_identity():
    assert controlled() is None
    inner = Forced((1,))
    assert (
        Controlled.compose(inner, EndOfGeneration.STOP, frozenset(), frozenset(), None, 14) is inner
    )


@pytest.mark.parametrize("suppress", [False, True])
def test_never_selected_tokens_are_suppressed(suppress):
    state = controlled(suppress=suppress, never=frozenset({3, 7}))
    assert allowed(state) == tuple(
        i for i in range(14) if i not in ({3, 7, 13} if suppress else {3, 7})
    )
    with pytest.raises(ValueError):
        state.stage((7,))
    assert state.position() == 0


@pytest.mark.parametrize(
    "spec,tokens,forced",
    [
        (ReasoningBudget(2, (10,), (11, 12), True), (1, 2), (11, 12)),
        (ReasoningBudget(1, (10,), (11, 12)), (1, 2, 10, 4), (11, 12)),
        (ReasoningBudget(2, (10,), (11, 12), True), (1, 11), (12,)),
        (ReasoningBudget(3, (10,), (11, 12), True), (1, 11, 12, 5, 6), ()),
    ],
)
def test_rust_reasoning_budget_transcripts(spec, tokens, forced):
    original = controlled(spec)
    state = original.stage(tokens)
    assert original.position() == 0
    assert state.forced(8) == forced
    if forced:
        assert allowed(state) == forced[:1]
        with pytest.raises(ValueError):
            state.stage((3,))
        closed = state.stage(forced)
        assert allowed(closed) == tuple(range(14))


def test_budget_truncates_grammar_forced_run_and_conflicts_fail_atomically():
    state = Controlled.compose(
        Forced((1, 2, 3, 4)),
        EndOfGeneration.STOP,
        frozenset(),
        frozenset(),
        ReasoningBudget(2, (10,), (11, 12), True),
        14,
    )
    assert state.forced(4) == (1, 2)
    successor = state.stage((1, 2))
    with pytest.raises(ValueError):
        successor.mask()
    with pytest.raises(ValueError):
        successor.forced(3)
    assert successor.position() == 2


@pytest.mark.parametrize(
    "spec", [lambda: ReasoningBudget(0, (1,), (2,)), lambda: ReasoningBudget(1, (), (2,))]
)
def test_invalid_reasoning_budgets(spec):
    with pytest.raises(ValueError):
        spec()


def test_budget_cannot_force_suppressed_or_oov_tag():
    with pytest.raises(ValueError):
        controlled(ReasoningBudget(1, (10,), (13,)), suppress=True)
    with pytest.raises(ValueError):
        controlled(ReasoningBudget(1, (10,), (99,)))


@pytest.mark.parametrize(
    "samples,accepted,emitted",
    [
        ((8, 9, 10, 11), 0, (8,)),
        ((3, 9, 10, 11), 1, (3, 9)),
        ((3, 4, 10, 11), 2, (3, 4, 10)),
        ((3, 4, 5, 11), 3, (3, 4, 5, 11)),
    ],
)
def test_rust_round_acceptance_table(samples, accepted, emitted):
    state = RoundState.verification(7, (3, 4, 5), 3, 10, (1, 2))
    result = state.reconcile(samples, frozenset())
    assert result.emitted == emitted
    assert result.accepted_proposals == accepted
    assert result.committed_rows == accepted + 1
    assert state.forward.kind is WorkKind.VERIFY


def test_round_stop_is_target_bonus_and_bad_samples_refused():
    state = RoundState.verification(7, (3, 9, 5), 3, 0, ())
    result = state.reconcile((3, 9, 5, 1), frozenset({9}))
    assert result.emitted == (3, 9) and result.accepted_proposals == 1
    with pytest.raises(ValueError):
        state.reconcile((3,), frozenset())
    forced = RoundState.forced(2, (4, 5), MethodRequirements())
    assert forced.forward.tokens == (2, 4) and forced.publication_bound() == 2
    assert forced.reconcile((), frozenset()).emitted == (4, 5)
    with pytest.raises(ValueError):
        forced.reconcile((1,), frozenset())


def test_prefill_decode_stop_usage_and_causal_reconciliation():
    value = generation(stop_tokens=frozenset({9}))
    assert value.publication_bound(1) == 0
    value = settle(value, (), allowance=1)
    assert value.resident_position == 1 and value.accepted_position == 2
    value = settle(value, (4,), allowance=1)
    assert value.generated == (4,) and value.resident_position == 2
    value = settle(value, (9,))
    assert value.finish_reason is FinishReason.STOP
    assert value.usage().completion_tokens == 2
    assert [token.token for token in value.take(8)] == [4]
    assert value.pending_reconciliation().start == 3
    round_ = value.start_reconciliation(8, "r")
    transition = round_.prepare_round_transition("r", ())
    value, _ = round_.commit(transition)
    assert value.resident_position == value.accepted_position == 4
    assert value.pending_reconciliation() is None


def test_preparation_discard_retry_and_single_use_identity():
    value = generation()
    round_ = value.start_round("r", 16)
    with pytest.raises(ValueError):
        round_.prepare_round_transition("r", (99,))
    assert value.generated == () and value.resident_position == 0
    transition = round_.prepare_round_transition("r", (4,))
    assert value.generated == ()
    with pytest.raises(ValueError):
        value.take(1)
    other = generation().start_round("r", 16)
    with pytest.raises(ValueError):
        other.commit(transition)
    value, _ = round_.commit(transition)
    assert value.generated == (4,)
    with pytest.raises(ValueError):
        round_.commit(transition)


def test_constraint_failure_does_not_advance_request():
    value = Generation((1,), InputLayout(1), options(), Forced((4, 5)))
    value.resume_at()
    round_ = value.start_round("r", 16)
    assert round_.round_forward().selects == ()
    with pytest.raises(ValueError):
        round_.prepare_round_transition("r", (3,))
    transition = round_.prepare_round_transition("r", ())
    assert value.constraint_position() == 0
    value, _ = round_.commit(transition)
    assert value.generated == (4,) and value.constraint_position() == 1
    round_ = value.start_round("r", 16)
    value, _ = round_.commit(round_.prepare_round_transition("r", ()))
    assert value.generated == (4, 5)


def test_output_backpressure_and_stable_indices():
    value = settle(generation(output_capacity=1), (4,))
    with pytest.raises(ValueError, match="output is full"):
        value.start_round("r", 16)
    assert value.take(1)[0].index == 0
    value = settle(value, (5,))
    assert value.take(1)[0].index == 1


@pytest.mark.parametrize("method", ["cancel", "fail", "evicted"])
def test_discarded_round_never_publishes(method):
    value = generation()
    round_ = value.start_round("r", 16)
    transition = round_.prepare_round_transition("r", (4,))
    value = getattr(round_, method)()
    assert value.generated == () and value.output_len == 0
    with pytest.raises(ValueError):
        round_.commit(transition)


def test_eviction_replays_resident_rows_without_resampling_accepted_successor():
    value = settle(generation(), (4,))
    value = settle(value, (5,))
    assert value.resident_position == 3 and value.accepted_position == 4
    value.evicted()
    value.resume_at()
    assert value.publication_bound(16) == 0
    value = settle(value, ())
    value = settle(value, ())
    assert value.generated == (4, 5) and value.resident_position == 3
    value = settle(value, (6,))
    assert value.generated == (4, 5, 6)


def test_prefix_resume_and_atomic_media_boundaries():
    layout = InputLayout(6, (InputSpan(1, 5, False),))
    assert layout.chunk_end(0, 6, 3) == 1
    with pytest.raises(ValueError):
        layout.chunk_end(1, 6, 3)
    value = Generation((1,) * 6, layout, options())
    with pytest.raises(ValueError):
        value.resume_at((2, MethodCheckpoint()))
    value.resume_at((5, MethodCheckpoint()))
    value = settle(value, (4,))
    assert value.detailed_usage().cached_tokens == 5
    with pytest.raises(ValueError):
        Generation(
            (1,) * 6,
            layout,
            options(method=MethodChoice(MethodKind.MTP, 2)),
            method_factory=Mtp("x", 2),
        )
    Generation(
        (1,) * 6,
        layout,
        options(method=MethodChoice(MethodKind.MTP, 2)),
        method_factory=Mtp.reading_target("x", 2),
    )


def test_limits_and_seed_consumption():
    value = generation(max_tokens=0)
    assert value.finish_reason is FinishReason.LENGTH
    with pytest.raises(ValueError):
        value.start_round("r", 16)
    value = settle(generation(max_tokens=1), (4,))
    assert value.finish_reason is FinishReason.LENGTH
    value = settle(generation(context_limit=2), (4,))
    assert value.finish_reason is FinishReason.CONTEXT
    seed = GenerationSeed((1,), InputLayout(1), options())
    seed.into_generation()
    with pytest.raises(ValueError):
        seed.into_generation()
    with pytest.raises(ValueError):
        DraftStats().record(1, 2)
    with pytest.raises(ValueError):
        DraftStats((1 << 64) - 1).record(1, 0)


@pytest.mark.parametrize("history", [DraftHistory.OWN, DraftHistory.TARGET])
def test_rust_draft_geometry(history):
    settled = DraftRoundState(history, 4, 4, 1, False)
    call, drafted = settled.proposing(3)
    assert call.entry_rows == 1 and call.generated == 3 and call.position == 4
    assert drafted.position == (5 if history is DraftHistory.OWN else 4)
    observed = drafted.observed(3)
    assert observed.maintenance is None
    assert observed.successor.target_position() == (8 if history is DraftHistory.OWN else 7)
    assert settled.proposing(0) == (None, settled)
    with pytest.raises(ValueError):
        settled.proposing(4)


def test_rust_maintenance_and_open_feature_geometry():
    settled = DraftRoundState(DraftHistory.OWN, 3, 1, 3, False)
    observed = settled.observed(1)
    assert observed.appended.pending_rows == 4
    assert observed.maintenance.entry_rows == 3
    assert observed.successor.position == 4 and observed.successor.pending_rows == 1
    assert observed.appended.target_position() == observed.successor.target_position()
    replayed = DraftRoundState(DraftHistory.OWN, 4, 2, 0, True)
    assert replayed.observed(1).successor.pending_rows == 2


@pytest.mark.parametrize("factory", [Mtp("x", 3), Mtp.reading_target("x", 3), DFlash("x", 3)])
def test_drafter_head_transaction_and_checkpoint(factory):
    state = factory.create()
    feature = FeatureRef("prompt", 2)
    effects = state.prime("r", (1, 2), 4, feature, 2, None, Reader())
    for operation in effects.operations:
        state.reconcile(operation, HeadOutcome())
    checkpoint = state.checkpoint()
    assert checkpoint.draft.target_rows() == 2
    if isinstance(factory, Mtp) and factory.reads_target:
        assert checkpoint.retained_bytes() == 0
    else:
        assert checkpoint.retained_bytes() == 1
    proposal = state.propose("r", (SelectSpec(),) * 3)
    assert len(proposal.operations) == 1
    with pytest.raises(MethodCheckpointError):
        state.checkpoint()
    operation = proposal.operations[0]
    with pytest.raises(ValueError):
        state.fork_transition().reconcile(replace(operation, request="wrong"), HeadOutcome())
    state.reconcile(operation, HeadOutcome((SelectedToken(5), SelectedToken(6), SelectedToken(7))))
    assert state.propose("r", (SelectSpec(),) * 3).tokens == (5, 6, 7)
    state.observe("r", Verification((4, 5, 6, 7), 2, 8, FeatureRef("verify", 4)), Reader())
    assert state.checkpoint().draft.target_rows() == 4


def test_weak_boundary_checkpoint_does_not_pin_output_arena():
    state = Mtp("x", 2).create()
    boundary = FeatureRef("boundary", 1)
    features = FeatureRef("prompt", 2, boundary)
    state.prime("r", (1, 2), None, features, 2, 1, Reader())
    checkpoint = state.checkpoint()
    assert checkpoint.retained_bytes() == 0
    assert checkpoint.draft.target_rows() == 2
    del boundary, features
    gc.collect()
    with pytest.raises(ValueError, match="no longer resident"):
        checkpoint.draft.open.read(Reader())


@pytest.mark.parametrize(
    "factory,kind",
    [(Mtp.reading_target("x", 3), MethodKind.MTP), (DFlash("x", 3), MethodKind.DFLASH)],
)
def test_generation_drafter_method_work_commit_and_verification(factory, kind):
    value = Generation(
        (1,), InputLayout(1), options(method=MethodChoice(kind, 3)), method_factory=factory
    )
    value.resume_at()
    round_ = value.start_round("r", 16)
    transition = round_.prepare_round_transition("r", (4,), FeatureRef("p", 1), Reader())
    value, _ = round_.commit(transition)
    work = value.start_round("r", 16)
    assert isinstance(work, MethodWork)
    operation = work.operations[0]
    outcome = HeadOutcome((SelectedToken(5), SelectedToken(6), SelectedToken(7)))
    transition = value.prepare_method_transition(operation, outcome)
    value.commit_method_transition(transition)
    with pytest.raises(ValueError, match="stale"):
        value.commit_method_transition(transition)
    round_ = value.start_round("r", 16)
    transition = round_.prepare_round_transition("r", (5, 6, 9, 8), FeatureRef("v", 4), Reader())
    assert transition.decision.accepted_rows == 3
    value, _ = round_.commit(transition)
    assert value.generated == (4, 5, 6, 9)
    assert value.detailed_usage().draft_n == 3
    assert value.detailed_usage().draft_n_accepted == 2


def test_core_import_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import ayaka.generation,sys; assert 'torch' not in sys.modules"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_plain_cannot_restore_draft_checkpoint():
    with pytest.raises(ValueError):
        Plain().create(MethodCheckpoint(MethodKind.MTP, DraftCheckpoint()))
    assert Plain().create().checkpoint() == MethodCheckpoint()
