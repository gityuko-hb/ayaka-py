"""Model-based conservation test for KV page ownership (research invariant_checker / C03, P0).

The research asks for "random reserve/adopt/commit/retire/cancel/transfer
sequences; state partition = total; charge vs physical bytes". This drives the
real ``RuntimeMemoryManager`` (no tensors needed) through random interleavings
of its whole lifecycle protocol -- tentative reservation, rollback, prepare,
abort, launch, (speculative partial) commit, retirement, in-flight failure with
a safe epoch, deferred release, preemption, truncation, fork, prefix caching,
release, eviction, clearing and epoch advances -- and after every step checks:

* ``assert_invariants()`` (allocator, prefix cache, sequence arena and the
  cross-component request-ref equality);
* free + reserved + live + deferred + permanent pages == total pages;
* open transactions / active leases / live sequences match the model;
* every sequence's committed length matches the model.

Teardown drains every step, releases everything and requires
``leak_report().clean``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ayaka.exceptions import InvalidHandleError
from ayaka.handles import PrefixHandle, SequenceHandle
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.pressure import PreemptionStatus
from ayaka.memory.state import ReleaseStatus
from ayaka.memory.transaction import ReservationFailure
from ayaka.prefix.identity import PrefixCacheContext
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

PAGE = 4
TOTAL_PAGES = 14
MAX_SEQUENCES = 5
MAX_TOKENS = 24
CONTEXT = PrefixCacheContext(model_id="tiny", cache_dtype="float32")


@dataclass
class Seq:
    handle: SequenceHandle
    tokens: list[int]
    phase: str = "idle"  # idle | reserved | prepared | inflight | committed
    tx: object = None
    reservation: object = None
    lease: object = None
    planned: int = 0
    min_commit: int | None = None
    release_requested: bool = False
    blocked_until: int = 0


class MemoryOwnershipMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.manager = RuntimeMemoryManager(
            total_pages=TOTAL_PAGES,
            page_size=PAGE,
            max_sequences=MAX_SEQUENCES,
            max_sequence_tokens=MAX_TOKENS,
        )
        self.seqs: list[Seq] = []
        self.prefixes: list[PrefixHandle] = []
        self.steps = itertools.count(0)
        self.token_ids = itertools.count(1_000)
        self.names = itertools.count(0)
        self.epoch = 0

    # -- helpers -----------------------------------------------------------

    def _ready(self, seq: Seq) -> bool:
        return (
            seq.phase == "idle"
            and not seq.release_requested
            and seq.blocked_until <= self.manager.current_epoch
        )

    def _pick(self, data, predicate):
        candidates = [s for s in self.seqs if predicate(s)]
        return data.draw(st.sampled_from(candidates)) if candidates else None

    def _finish_step(self, seq: Seq) -> None:
        seq.phase, seq.tx, seq.reservation, seq.lease = "idle", None, None, None
        seq.planned, seq.min_commit = 0, None
        if seq.release_requested:
            self.seqs.remove(seq)  # the deferred release ran when the step ended

    # -- lifecycle rules -----------------------------------------------------

    @rule()
    def create(self) -> None:
        if len(self.seqs) < MAX_SEQUENCES:
            self.seqs.append(Seq(self.manager.create_sequence(f"r{next(self.names)}"), []))

    @rule(
        data=st.data(),
        tokens=st.integers(1, 9),
        speculative=st.booleans(),
        outcome=st.sampled_from(
            ["commit", "commit", "commit", "abort", "rollback", "fail", "hold"]
        ),
    )
    def whole_step(self, data, tokens: int, speculative: bool, outcome: str) -> None:
        """One step end to end, so long histories (and full pages) actually form."""
        seq = self.reserve(data, tokens, speculative)
        if seq is None or outcome == "hold":
            return
        if outcome == "rollback":
            self.manager.rollback_transaction(seq.tx)
            self._finish_step(seq)
            return
        seq.lease = self.manager.prepare_step(seq.tx)
        seq.phase = "prepared"
        if outcome == "abort":
            self.manager.abort_prepared_step(seq.lease)
            self._finish_step(seq)
            return
        self.manager.mark_step_in_flight(seq.lease)
        seq.phase = "inflight"
        if outcome == "fail":
            safe = self.manager.current_epoch + data.draw(st.integers(0, 2))
            self.manager.fail_in_flight_step(seq.lease, safe_epoch=safe)
            seq.blocked_until = max(seq.blocked_until, safe)
            self._finish_step(seq)
            return
        self._commit(seq, data)
        self.manager.retire_step(seq.lease)
        self._finish_step(seq)

    @rule(data=st.data(), tokens=st.integers(1, 9), speculative=st.booleans())
    def reserve_only(self, data, tokens: int, speculative: bool) -> None:
        self.reserve(data, tokens, speculative)

    def reserve(self, data, tokens: int, speculative: bool) -> Seq | None:
        """Reserve for one ready sequence; returns it when the reservation held."""
        seq = self._pick(data, self._ready)
        if seq is None:
            return None
        tx = self.manager.begin_transaction(next(self.steps))
        floor = data.draw(st.integers(1, tokens)) if speculative else None
        result = self.manager.try_reserve(tx, seq.handle, tokens, min_commit_tokens=floor)
        if not result.ok:
            assert result.reason in (
                ReservationFailure.NO_CAPACITY,
                ReservationFailure.REQUEST_TOO_LARGE,
            ), result
            self.manager.rollback_transaction(tx)
            return None
        seq.phase, seq.tx, seq.reservation = "reserved", tx, result.handle
        seq.planned, seq.min_commit = tokens, floor
        return seq

    @rule(data=st.data())
    def rollback(self, data) -> None:
        seq = self._pick(data, lambda s: s.phase == "reserved")
        if seq is not None:
            self.manager.rollback_transaction(seq.tx)
            self._finish_step(seq)

    @rule(data=st.data())
    def prepare(self, data) -> None:
        seq = self._pick(data, lambda s: s.phase == "reserved" and not s.release_requested)
        if seq is not None:
            seq.lease = self.manager.prepare_step(seq.tx)
            seq.phase = "prepared"

    @rule(data=st.data())
    def abort_prepared(self, data) -> None:
        seq = self._pick(data, lambda s: s.phase == "prepared")
        if seq is not None:
            self.manager.abort_prepared_step(seq.lease)
            self._finish_step(seq)

    @rule(data=st.data())
    def launch(self, data) -> None:
        seq = self._pick(data, lambda s: s.phase == "prepared")
        if seq is not None:
            self.manager.mark_step_in_flight(seq.lease)
            seq.phase = "inflight"

    @rule(data=st.data())
    def commit(self, data) -> None:
        seq = self._pick(data, lambda s: s.phase == "inflight")
        if seq is not None:
            self._commit(seq, data)

    def _commit(self, seq: Seq, data) -> None:
        if seq.min_commit is not None:
            kept = data.draw(st.integers(seq.min_commit, seq.planned))
            self.manager.commit_step(seq.lease, written_tokens={seq.reservation: kept})
        else:
            kept = seq.planned
            self.manager.commit_step(seq.lease)
        seq.tokens.extend(next(self.token_ids) for _ in range(kept))
        seq.phase = "committed"

    @rule(data=st.data())
    def retire(self, data) -> None:
        seq = self._pick(data, lambda s: s.phase == "committed")
        if seq is not None:
            self.manager.retire_step(seq.lease)
            self._finish_step(seq)

    @rule(data=st.data(), delay=st.integers(0, 2))
    def fail_in_flight(self, data, delay: int) -> None:
        seq = self._pick(data, lambda s: s.phase == "inflight")
        if seq is None:
            return
        safe = self.manager.current_epoch + delay
        self.manager.fail_in_flight_step(seq.lease, safe_epoch=safe)
        seq.blocked_until = max(seq.blocked_until, safe)
        self._finish_step(seq)

    @rule(data=st.data(), go=st.booleans())
    def release(self, data, go: bool) -> None:
        seq = self._pick(data, lambda s: not s.release_requested)
        if seq is None or not (go or len(self.seqs) == MAX_SEQUENCES):
            return
        status = self.manager.release_sequence(seq.handle)
        if seq.phase == "idle":
            assert status is ReleaseStatus.RELEASED
            self.seqs.remove(seq)
        else:
            assert status is ReleaseStatus.DEFERRED
            seq.release_requested = True

    @rule(data=st.data())
    def preempt(self, data) -> None:
        seq = self._pick(data, self._ready)
        if seq is None:
            return
        before = len(seq.tokens)
        result = self.manager.preempt_sequence(seq.handle)
        if result.status is PreemptionStatus.RELEASED:
            assert result.released_tokens == before
            seq.tokens.clear()
        elif result.status is PreemptionStatus.NO_RESIDENT_STATE:
            assert before == 0
        else:
            assert self.manager.get_sequence(seq.handle).committed_tokens == before

    @rule(data=st.data())
    def truncate(self, data) -> None:
        seq = self._pick(data, lambda s: self._ready(s) and s.tokens)
        if seq is None:
            return
        keep = data.draw(st.integers(0, len(seq.tokens)))
        result = self.manager.truncate_sequence(seq.handle, keep)
        assert result.retained_tokens == keep
        del seq.tokens[keep:]

    @rule(data=st.data())
    def fork(self, data) -> None:
        if len(self.seqs) >= MAX_SEQUENCES:
            return
        source = self._pick(data, lambda s: self._ready(s) and s.tokens)
        if source is None:
            return
        destination = self.manager.create_sequence(f"r{next(self.names)}")
        assert self.manager.fork_sequence(source.handle, destination) == len(source.tokens)
        self.seqs.append(Seq(destination, list(source.tokens)))

    # -- prefix cache rules --------------------------------------------------

    @rule(data=st.data())
    def cache(self, data) -> None:
        seq = self._pick(data, lambda s: self._ready(s) and len(s.tokens) >= PAGE)
        if seq is None:
            return
        handle = self.manager.cache_prefix(seq.handle, seq.tokens, context=CONTEXT)
        if handle is not None and handle not in self.prefixes:
            self.prefixes.append(handle)

    @rule(data=st.data())
    def release_prefix(self, data) -> None:
        if not self.prefixes:
            return
        handle = data.draw(st.sampled_from(self.prefixes))
        self.prefixes.remove(handle)
        try:
            self.manager.release_prefix(handle)
        except InvalidHandleError:
            pass  # pruned earlier as an unshared ancestor or by eviction

    @rule(op=st.sampled_from(["advance", "advance", "evict", "clear"]), pages=st.integers(0, 6))
    def maintenance(self, op: str, pages: int) -> None:
        if op == "advance":
            self.advance_epoch()
        elif op == "evict":
            self.manager.evict_prefixes(pages)
        else:
            self.manager.clear_prefix_cache()
            self.prefixes.clear()

    def advance_epoch(self) -> None:
        self.epoch = max(self.epoch, self.manager.current_epoch) + 1
        self.manager.advance_epoch(self.epoch)

    # -- invariants ----------------------------------------------------------

    @invariant()
    def accounting_is_conserved(self) -> None:
        self.manager.assert_invariants()
        snap = self.manager.snapshot()
        assert (
            snap.free_pages
            + snap.reserved_pages
            + snap.live_pages
            + snap.deferred_free_pages
            + snap.permanent_pages
            == snap.total_pages
        ), snap
        phases = [s.phase for s in self.seqs]
        assert snap.open_transactions == phases.count("reserved"), snap
        assert snap.active_leases == sum(
            phases.count(p) for p in ("prepared", "inflight", "committed")
        ), snap
        assert snap.live_sequences == len(self.seqs), snap
        if not any(p in ("inflight", "committed") for p in phases):
            assert snap.inflight_pages == 0, snap
        for seq in self.seqs:
            assert self.manager.get_sequence(seq.handle).committed_tokens == len(seq.tokens)

    def teardown(self) -> None:
        for seq in list(self.seqs):
            if seq.phase == "reserved":
                self.manager.rollback_transaction(seq.tx)
            elif seq.phase == "prepared":
                self.manager.abort_prepared_step(seq.lease)
            elif seq.phase == "inflight":
                self.manager.commit_step(
                    seq.lease,
                    written_tokens=(
                        None if seq.min_commit is None else {seq.reservation: seq.min_commit}
                    ),
                )
                self.manager.retire_step(seq.lease)
            elif seq.phase == "committed":
                self.manager.retire_step(seq.lease)
            if seq in self.seqs and not seq.release_requested:
                assert self.manager.release_sequence(seq.handle) is ReleaseStatus.RELEASED
        self.manager.clear_prefix_cache()
        for _ in range(4):
            self.advance_epoch()
        self.manager.reclaim_deferred()
        report = self.manager.leak_report()
        assert report.clean, report


def test_partial_speculative_commit_keeps_invariants_before_retirement() -> None:
    """Regression found by the machine above.

    A speculative step that keeps fewer rows than a whole new page abandons that
    page; it stays RESERVED with only in-flight references until retirement.
    ``assert_invariants`` used to compare reserved pages with uncommitted
    reservations only and raised "reservation registry does not match
    reserved-page accounting" for this legal state.
    """
    manager = RuntimeMemoryManager(total_pages=14, page_size=4, max_sequences=2)
    seq = manager.create_sequence("a")
    tx = manager.begin_transaction(0)
    first = manager.try_reserve(tx, seq, 3)
    lease = manager.prepare_step(tx)
    manager.mark_step_in_flight(lease)
    manager.commit_step(lease)
    manager.retire_step(lease)

    tx = manager.begin_transaction(1)
    spec = manager.try_reserve(tx, seq, 6, min_commit_tokens=1)  # spans two new pages
    assert first.ok and spec.ok and spec.allocated_pages == 2
    lease = manager.prepare_step(tx)
    manager.mark_step_in_flight(lease)
    manager.commit_step(lease, written_tokens={spec.handle: 1})  # both new pages rejected

    manager.assert_invariants()  # COMMITTED, not yet retired
    assert manager.snapshot().reserved_pages == 2  # still protected by in-flight refs
    manager.retire_step(lease)
    manager.assert_invariants()
    assert manager.snapshot().reserved_pages == 0
    assert manager.get_sequence(seq).committed_tokens == 4


def test_grouped_partial_speculative_commit_keeps_invariants_before_retirement() -> None:
    """Same false positive in ``KVCacheGroupManager.assert_invariants``."""
    from ayaka.kvcache.grouped_manager import KVCacheGroupManager
    from ayaka.kvcache.groups import KVCacheGroup
    from ayaka.kvcache.retention.policy import FullRetention
    from ayaka.kvcache.storage.geometry import MHAStorageSpec

    spec = MHAStorageSpec(
        num_layers=1,
        num_kv_heads_local=1,
        head_dim=4,
        page_size=4,
        capacity_pages=14,
        dtype="float32",
    )
    manager = KVCacheGroupManager(
        cache_groups=(KVCacheGroup("full", (0,), spec, FullRetention()),),
        max_sequences=2,
        max_sequence_tokens=40,
    )
    seq = manager.create_sequence("a")
    tx = manager.begin_transaction(0)
    assert manager.try_reserve(tx, seq, 3).ok
    lease = manager.prepare_step(tx)
    manager.mark_step_in_flight(lease)
    manager.commit_step(lease)
    manager.retire_step(lease)

    tx = manager.begin_transaction(1)
    spec_res = manager.try_reserve(tx, seq, 6, min_commit_tokens=1)
    assert spec_res.ok and spec_res.allocated_pages == 2
    lease = manager.prepare_step(tx)
    manager.mark_step_in_flight(lease)
    manager.commit_step(lease, written_tokens={spec_res.handle: 1})
    manager.assert_invariants()  # used to raise "reservation accounting differs"
    manager.retire_step(lease)
    manager.assert_invariants()


MemoryOwnershipMachine.TestCase.settings = settings(
    max_examples=300, stateful_step_count=80, deadline=None
)
test_memory_ownership_is_conserved = MemoryOwnershipMachine.TestCase
