"""Grouped COW unshare under pressure (research M01).

A consumer that attached a canonical grouped boundary shares its partial tail
with the cache entry's all-group pins, so its next append copies the tail in
every group. ``privatize_prefix_tail`` evicts exactly the entries pinning those
tails when the consumer is their only request, so the append writes in place.
It is all-or-nothing across groups and refuses whenever another owner exists:
a second request, a pin outside the cache, an in-flight step, a busy sequence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from ayaka.kvcache.grouped_manager import KVCacheGroupManager
from ayaka.kvcache.groups import KVCacheGroup
from ayaka.kvcache.retention.policy import FullRetention
from ayaka.kvcache.storage.geometry import MHAStorageSpec
from ayaka.prefix.identity import PrefixCacheContext

PAGE = 4
CONTEXT = PrefixCacheContext(model_id="tiny", cache_dtype="float32")
TOKENS = tuple(range(100, 140))


def _manager(*groups: str, capacity: int = 14) -> KVCacheGroupManager:
    spec = MHAStorageSpec(
        num_layers=1,
        num_kv_heads_local=1,
        head_dim=4,
        page_size=PAGE,
        capacity_pages=capacity,
        dtype="float32",
    )
    return KVCacheGroupManager(
        cache_groups=tuple(
            KVCacheGroup(name, (layer,), spec, FullRetention())
            for layer, name in enumerate(groups or ("full",))
        ),
        max_sequences=4,
        max_sequence_tokens=40,
    )


class _Steps:
    def __init__(self, manager: KVCacheGroupManager) -> None:
        self.manager = manager
        self.step = 0

    def append(self, sequence, tokens: int) -> int:
        """Run one complete step; returns the pages the reservation allocated."""
        tx = self.manager.begin_transaction(self.step)
        self.step += 1
        reservation = self.manager.try_reserve(tx, sequence, tokens)
        assert reservation.ok, reservation
        lease = self.manager.prepare_step(tx)
        self.manager.mark_step_in_flight(lease)
        self.manager.commit_step(lease)
        self.manager.retire_step(lease)
        return reservation.allocated_pages

    def pages_needed(self, sequence, tokens: int) -> int:
        """Pages one append would allocate, without keeping the reservation."""
        tx = self.manager.begin_transaction(self.step)
        self.step += 1
        reservation = self.manager.try_reserve(tx, sequence, tokens)
        assert reservation.ok, reservation
        self.manager.rollback_transaction(tx)
        return reservation.allocated_pages


def _published(manager: KVCacheGroupManager, *boundaries: int, produced: int | None = None):
    """A retired producer publishes each boundary, then is released."""
    steps = _Steps(manager)
    producer = manager.create_sequence("producer")
    steps.append(producer, produced or max(boundaries))
    matches = [
        manager.cache_resume(producer, TOKENS[:boundary], context=CONTEXT)
        for boundary in boundaries
    ]
    manager.release_sequence(producer)
    assert all(match is not None for match in matches)
    return steps, matches


def _consumer(manager: KVCacheGroupManager, match, name: str = "consumer"):
    sequence = manager.create_sequence(name)
    manager.attach_resume(sequence, match)
    return sequence


def _tails(manager: KVCacheGroupManager, sequence):
    snapshot = manager.get_sequence(sequence)
    block = snapshot.committed_tokens // PAGE
    for name, table in snapshot.group_page_tables:
        (tail,) = (entry for entry in table if entry.logical_block == block)
        yield manager._runtime_by_name[name].allocator, tail


def _entries(manager: KVCacheGroupManager) -> int:
    return manager.prefix_cache.snapshot().terminal_entries


def test_a_sole_consumer_takes_its_tail_and_appends_without_a_copy() -> None:
    manager = _manager()
    steps, (match,) = _published(manager, 6)
    consumer = _consumer(manager, match)
    assert steps.pages_needed(consumer, 1) == 1  # the shared tail must be copied
    before = manager.get_sequence(consumer)

    assert manager.privatize_prefix_tail(consumer)

    after = manager.get_sequence(consumer)
    assert (after.committed_tokens, after.version) == (before.committed_tokens, before.version)
    assert after.group_page_tables == before.group_page_tables
    assert _entries(manager) == 0
    for allocator, tail in _tails(manager, consumer):
        assert allocator.get_meta(tail.page).can_mutate
    assert steps.pages_needed(consumer, 1) == 0
    assert steps.append(consumer, 2) == 0  # fills the private tail in place
    manager.assert_invariants()
    manager.prefix_cache.assert_invariants()
    assert not manager.privatize_prefix_tail(consumer)  # nothing left to unshare


def test_physical_validity_narrows_to_the_consumer_boundary() -> None:
    manager = _manager()
    # The producer wrote 7 tokens; the published boundary stops at 6.
    steps, (match,) = _published(manager, 6, produced=7)
    consumer = _consumer(manager, match)
    ((allocator, tail),) = _tails(manager, consumer)
    assert (allocator.get_meta(tail.page).valid_tokens, tail.valid_tokens) == (3, 2)

    assert manager.privatize_prefix_tail(consumer)

    assert allocator.get_meta(tail.page).valid_tokens == 2
    assert steps.pages_needed(consumer, 1) == 0
    manager.assert_invariants()


def test_an_unpinned_but_wider_tail_is_narrowed_without_evicting() -> None:
    manager = _manager()
    steps, (match,) = _published(manager, 6, produced=7)
    consumer = _consumer(manager, match)
    assert manager.prefix_cache.evict_entry(match.entry_id, safe_epoch=manager.current_epoch)
    ((allocator, tail),) = _tails(manager, consumer)
    assert allocator.get_meta(tail.page).can_mutate  # sole owner, yet 3 tokens valid
    assert steps.pages_needed(consumer, 1) == 1

    assert manager.privatize_prefix_tail(consumer)

    assert allocator.get_meta(tail.page).valid_tokens == 2
    assert steps.pages_needed(consumer, 1) == 0


def test_only_the_entries_pinning_the_tail_are_evicted() -> None:
    manager = _manager()
    # 4 tokens pins page 0 only; 5 and 6 both pin the partial page 1.
    _, (full_page, five, six) = _published(manager, 4, 5, 6)
    consumer = _consumer(manager, six)

    assert manager.privatize_prefix_tail(consumer)

    snapshot = manager.prefix_cache.snapshot()
    assert [handle.index for handle in snapshot.handles] == [full_page.entry_id]
    assert manager.prefix_cache.find_exact(TOKENS[:5], context=CONTEXT) is None
    assert five.entry_id != six.entry_id
    manager.assert_invariants()
    manager.prefix_cache.assert_invariants()


def test_a_second_consumer_keeps_the_tail_shared() -> None:
    manager = _manager()
    steps, (match,) = _published(manager, 6)
    first = _consumer(manager, match, "first")
    _consumer(manager, match, "second")

    assert not manager.privatize_prefix_tail(first)

    assert _entries(manager) == 1
    assert steps.pages_needed(first, 1) == 1
    manager.assert_invariants()


def test_a_pin_outside_the_cache_refuses_and_evicts_nothing() -> None:
    manager = _manager()
    _, (match,) = _published(manager, 6)
    consumer = _consumer(manager, match)
    foreign = manager.pin_prefix(consumer, 6)

    assert not manager.privatize_prefix_tail(consumer)
    assert _entries(manager) == 1

    manager.unpin_prefix(foreign)
    assert manager.privatize_prefix_tail(consumer)
    manager.assert_invariants()


def test_a_busy_sequence_is_refused() -> None:
    manager = _manager()
    _, (match,) = _published(manager, 6)
    consumer = _consumer(manager, match)
    tx = manager.begin_transaction(99)
    assert manager.try_reserve(tx, consumer, 1).ok

    assert not manager.privatize_prefix_tail(consumer)
    assert _entries(manager) == 1

    manager.rollback_transaction(tx)
    assert manager.privatize_prefix_tail(consumer)


def test_a_full_page_boundary_has_nothing_to_unshare() -> None:
    manager = _manager()
    _, (match,) = _published(manager, 8)
    consumer = _consumer(manager, match)

    assert not manager.privatize_prefix_tail(consumer)
    assert _entries(manager) == 1


def test_without_a_prefix_cache_nothing_is_privatized() -> None:
    manager = _manager()
    steps = _Steps(manager)
    sequence = manager.create_sequence("plain")
    steps.append(sequence, 6)
    assert not manager.privatize_prefix_tail(sequence)


def test_a_host_tier_refuses_before_touching_any_entry() -> None:
    manager = _manager()
    _, (match,) = _published(manager, 6)
    consumer = _consumer(manager, match)
    manager._tier = SimpleNamespace(enabled=True)  # type: ignore[assignment]
    try:
        assert not manager.privatize_prefix_tail(consumer)
    finally:
        manager._tier = None
    assert _entries(manager) == 1


@pytest.mark.parametrize("owner", ["pin", "request"])
def test_every_group_is_unshared_or_none_is(owner: str) -> None:
    manager = _manager("a", "b")
    steps, (match,) = _published(manager, 6)
    consumer = _consumer(manager, match)
    assert steps.pages_needed(consumer, 1) == 2  # one copy per group
    (_, tail_a), (allocator_b, tail_b) = _tails(manager, consumer)
    # An owner only group "b" can see.
    acquire, release = {
        "pin": (allocator_b.pin_page, allocator_b.unpin_page),
        "request": (allocator_b.acquire_request_ref, allocator_b.release_request_ref),
    }[owner]
    acquire(tail_b.page)

    assert not manager.privatize_prefix_tail(consumer)
    assert _entries(manager) == 1
    assert steps.pages_needed(consumer, 1) == 2

    release(tail_b.page, safe_epoch=manager.current_epoch)
    assert manager.privatize_prefix_tail(consumer)
    assert manager._runtime_by_name["a"].allocator.get_meta(tail_a.page).can_mutate
    assert allocator_b.get_meta(tail_b.page).can_mutate
    assert steps.pages_needed(consumer, 1) == 0
    manager.assert_invariants()
    manager.prefix_cache.assert_invariants()
