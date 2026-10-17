"""Live CPU runner controls, mask bridge and speculative eligibility."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ayaka.generation import Controlled, EndOfGeneration, ReasoningBudget
from ayaka.generation.adapter import ConstraintProducer, GenerationControls
from ayaka.request.schema import Request, RequestId, StopCriteria
from ayaka.sampling.mask.producer import MaskRows
from ayaka.sampling.params import SamplingParams
from ayaka.speculative.acceptance.greedy import accept_greedy, greedy_accept
from ayaka.speculative.config import AcceptanceMethod
from ayaka.speculative.plan import SpecDisableReason
from ayaka.speculative.resources import request_disable_reason


def request(**kwargs):
    return Request(RequestId("r"), (1, 2), **kwargs)


def test_controls_advance_only_with_committed_publication():
    registry = GenerationControls(32)
    registry.register(request(reasoning_budget=ReasoningBudget(2, (10,), (11, 12), True)))
    before = registry.selection_mask("r", ())
    assert registry.selection_mask("r", ()) == before
    registry.selection_mask("r", (1,))
    assert registry.selection_mask("r", (1, 2)) == (1 << 11,)
    assert registry.selection_mask("r", (1, 2)) == (1 << 11,)
    assert registry.selection_mask("r", (1, 2, 11)) == (1 << 12,)
    with pytest.raises(ValueError, match="history changed"):
        registry.selection_mask("r", (1, 3))
    assert registry.selection_mask("r", (1, 2, 11, 12)) == (0xFFFFFFFF,)
    registry.forget("r")
    assert not registry.contains("r")


def test_mask_intersects_existing_bans_and_fails_when_impossible():
    registry = GenerationControls(32)
    registry.register(request(reasoning_budget=ReasoningBudget(1, (10,), (11,), True)))
    logits = torch.zeros(32)
    registry.apply("r", (1,), logits)
    assert logits.argmax().item() == 11 and torch.isfinite(logits).sum().item() == 1
    logits = torch.zeros(32)
    logits[11] = -torch.inf
    with pytest.raises(ValueError, match="no valid token"):
        registry.apply("r", (1,), logits)


def test_constraint_producer_previews_without_advancing_and_commits_atomically():
    state = Controlled.compose(
        None,
        EndOfGeneration.STOP,
        frozenset(),
        frozenset(),
        ReasoningBudget(1, (10,), (11, 12), True),
        32,
    )
    producer = ConstraintProducer(state)
    masks = np.zeros((4, 1), dtype=np.uint32)
    accepted = producer.emit((1, 7, 8), MaskRows(masks, 32))
    assert accepted == 1
    assert tuple(int(x) for x in masks[:, 0]) == (0xFFFFFFFF, 1 << 11, 1 << 11, 1 << 11)
    assert producer.committed_len() == 0
    with pytest.raises(ValueError):
        producer.commit((1, 7))
    assert producer.committed_len() == 0
    producer.commit((1, 11, 12))
    assert producer.committed_len() == 3


@pytest.mark.parametrize(
    "fields",
    [
        dict(reasoning_budget=ReasoningBudget(2, (10,), (11,), True)),
        dict(suppressed_token_ids=(1,)),
    ],
)
def test_native_verifier_disables_controls_it_cannot_preview(fields):
    req = request(sampling=SamplingParams(temperature=0), **fields)
    assert (
        request_disable_reason(req, acceptance=AcceptanceMethod.GREEDY) is SpecDisableReason.GRAMMAR
    )
    plain = replace(req, reasoning_budget=None, suppressed_token_ids=())
    assert request_disable_reason(plain, acceptance=AcceptanceMethod.GREEDY) is None


def test_native_acceptance_oracle_uses_shared_prefix_and_keeps_default_parity():
    assert accept_greedy((3, 9, 5), (3, 9, 5, 7), stops={9}).tokens == (3, 9)
    drafts = torch.tensor([[1, 2, 3], [1, 2, 3], [-1, -1, -1]])
    targets = torch.tensor([[1, 2, 3, 4], [1, 9, 3, 4], [7, 8, 9, 10]])
    accepted, final = greedy_accept(drafts, targets)
    for i in range(3):
        oracle = accept_greedy(tuple(drafts[i].tolist()), tuple(targets[i].tolist()))
        assert accepted[i].item() == oracle.accepted
        assert final[i].item() == oracle.tokens[-1]


def test_resident_engine_refuses_unsupported_runner_before_admission():
    from ayaka.runtime.resident import ResidentKVEngine

    engine = object.__new__(ResidentKVEngine)
    engine._closing = False
    engine.executor = SimpleNamespace(runner=object())
    with pytest.raises(ValueError, match="does not implement"):
        engine.submit(request(suppressed_token_ids=(0,)))


@pytest.mark.parametrize("lanes", [False, True])
def test_native_qwen3_cpu_generation_controls_end_to_end(tmp_path, lanes):
    """Real model, paged attention, sampler, scheduler, KV commit and publication.

    Zero weights make the unmasked greedy winner token 0. Suppression changes
    it to 1; an open two-token budget then forces 11 and 12, and releases it.
    No checkpoint download or GPU is needed.
    """
    from ayaka.configs.serving import ServingConfig
    from ayaka.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from ayaka.runtime.serving import ServingRuntime
    from ayaka.worker.execution_lane import ExecutionLaneConfig
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="t0", eos_token="t31")
        tokenizer.save_pretrained(tmp_path)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=32,
                max_position_embeddings=32,
            ),
            device="cpu",
            dtype=torch.float32,
            backend="torch",
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        runtime = ServingRuntime(
            model,
            tmp_path,
            config=ServingConfig(
                model="tiny",
                decode_graph=False,
                structured_outputs=False,
                execution_lanes=ExecutionLaneConfig(memory_bytes_per_lane=1 << 20)
                if lanes
                else None,
            ),
            pages=32,
            page_size=4,
            max_requests=2,
            max_inflight=2 if lanes else 1,
            batch_tokens=16,
            prefill_chunk=4,
            max_model_len=32,
            tokenizer_workers=1,
            backend="reference",
            activation_bytes=0,
            detect_hardware_snapshot=False,
        )
        try:
            plain = Request(
                RequestId("plain"),
                (1,) * 5,
                sampling=SamplingParams(temperature=0),
                stop=StopCriteria(max_tokens=6),
            )
            controlled = replace(
                plain,
                request_id=RequestId("controlled"),
                prompt_token_ids=(1, 2),
                reasoning_budget=ReasoningBudget(2, (10,), (11, 12), True),
                suppressed_token_ids=(0,),
            )
            lifecycles = [runtime.engine.submit(req) for req in (plain, controlled)]
            for _ in range(64):
                runtime.engine.step()
                if all(lc.is_terminal for lc in lifecycles):
                    break
            assert all(lc.is_terminal for lc in lifecycles)
            assert lifecycles[0].output_token_ids == (0,) * 6
            assert lifecycles[1].output_token_ids == (1, 1, 11, 12, 1, 1)
            assert not runtime.engine.executor.runner._generation_controls.contains("controlled")
        finally:
            runtime.close()
    finally:
        torch.set_num_threads(threads)


def test_legacy_dense_speculation_refuses_controls_before_forward():
    from ayaka.runner.speculative_runner import require_plain_greedy

    req = request(sampling=SamplingParams(temperature=0), suppressed_token_ids=(0,))
    with pytest.raises(ValueError, match="plain greedy"):
        require_plain_greedy(req)
    require_plain_greedy(req, allow_generation_controls=True)
