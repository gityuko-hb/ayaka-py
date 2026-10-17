"""Actual CPU coordinator/sampler integration (no mocked torch or parser)."""

from types import SimpleNamespace

import pytest
import torch
from ayaka.grammar import ByteTokenTable, Vocabulary
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sampling.params import SamplingParams


@pytest.fixture
def setup():
    table = ByteTokenTable(
        tuple(bytes([i]) for i in range(256)) + (b"",),
        frozenset({256}),
        lambda text: list(text.encode()),
    )
    vocab = Vocabulary(table, 264)
    coordinator = SamplingCoordinator(
        4, device=torch.device("cpu"), vocab_size=264, backend="reference"
    )
    return coordinator, vocab


def slices(*names):
    return [SimpleNamespace(request_id=name, sample_last_query=True) for name in names]


def publish(coordinator, name, token):
    coordinator.record_published([SimpleNamespace(request_id=name, token_id=token)])


def sample(coordinator, *names):
    plan = coordinator.plan_for(slices(*names))
    logits = torch.zeros(len(names), 264)
    logits[:, 99] = 100  # forbidden 'c' wins unless the real sampler applies the mask
    return coordinator.sample(logits, plan, force_reference=True).tolist()


def test_mask_sample_and_publication(setup):
    coordinator, vocabulary = setup
    coordinator.add("a", SamplingParams(temperature=0))
    producer = coordinator.attach_grammar("a", vocabulary, 'root ::= "ab"')
    assert sample(coordinator, "a") == [97]
    assert producer.committed_len() == 0
    # A discarded sample must leave the grammar unchanged.
    assert sample(coordinator, "a") == [97]
    publish(coordinator, "a", 97)
    assert sample(coordinator, "a") == [98]
    publish(coordinator, "a", 98)
    assert sample(coordinator, "a") == [256]
    publish(coordinator, "a", 256)
    assert producer.committed_len() == 3


def test_sparse_slots_and_reordered_packed_rows(setup):
    coordinator, vocabulary = setup
    for name in ["idle", "a", "b"]:
        coordinator.add(name, SamplingParams(temperature=0))
    coordinator.attach_grammar("a", vocabulary, 'root ::= "a"')
    coordinator.attach_grammar("b", vocabulary, 'root ::= "b"')
    assert sample(coordinator, "b", "idle", "a") == [98, 99, 97]
    assert sample(coordinator, "b") == [98]
    assert sample(coordinator, "a") == [97]


def test_release_reuses_slot_without_grammar_leak(setup):
    coordinator, vocabulary = setup
    slot = coordinator.add("a", SamplingParams(temperature=0))
    coordinator.attach_grammar("a", vocabulary, 'root ::= "a"')
    coordinator.release("a")
    assert coordinator.add("b", SamplingParams(temperature=0)) == slot
    assert sample(coordinator, "b") == [99]


def test_stale_plan_rejected_after_attach(setup):
    coordinator, vocabulary = setup
    coordinator.add("a", SamplingParams(temperature=0))
    old = coordinator.plan_for(slices("a"))
    coordinator.attach_grammar("a", vocabulary, 'root ::= "a"')
    with pytest.raises(ValueError, match="fresh plan_for"):
        coordinator.sample(torch.zeros(1, 264), old)


def test_invalid_publication_does_not_advance(setup):
    coordinator, vocabulary = setup
    coordinator.add("a", SamplingParams(temperature=0))
    producer = coordinator.attach_grammar("a", vocabulary, 'root ::= "a"')
    with pytest.raises(ValueError):
        publish(coordinator, "a", 99)
    assert producer.committed_len() == 0
    assert sample(coordinator, "a") == [97]


def test_multiple_producers_intersect_and_scratch_windows_do_not_overlap(setup):
    coordinator, vocabulary = setup
    coordinator.add("a", SamplingParams(temperature=0))
    coordinator.add("b", SamplingParams(temperature=0))
    first = coordinator.attach_grammar("a", vocabulary, "root ::= [ab]")
    coordinator.attach_grammar("b", vocabulary, 'root ::= "b"')
    from ayaka.grammar import compile
    from ayaka.grammar.adapter import GBNFMaskProducer

    second = GBNFMaskProducer(vocabulary.bind(compile('root ::= "a"').grammar))
    coordinator.planner.attach(coordinator.slot_of("a"), second)
    assert sample(coordinator, "a", "b") == [97, 98]
    _, schedule = coordinator._mask_schedule
    windows = [set(range(e.mask_row, e.mask_row + e.propose_step + 1)) for e in schedule.entries]
    assert all(not a & b for i, a in enumerate(windows) for b in windows[i + 1 :])
    assert first.committed_len() == 0


def test_tokenizer_adapter_uses_raw_bytes_and_no_special_insertion():
    class Tokenizer:
        all_special_ids = frozenset({2, 3})

        def get_decoded_vocab(self):
            return [b"a", b"b", b"<bos>", b""]

        def encode(self, text, *, add_special_tokens=True):
            assert not add_special_tokens
            return [{"a": 0, "b": 1}[c] for c in text]

    vocabulary = Vocabulary.from_tokenizer(Tokenizer(), 8, stop_tokens=[3])
    from ayaka.grammar import compile

    state = vocabulary.bind(compile('root ::= "ab"').grammar)
    assert not state.allows(2)
    assert state.advance([0, 1, 3]).stopped()
