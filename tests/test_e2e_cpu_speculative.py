"""Speculative decoding alignment end to end on CPU (research task C02).

The SGLang snapshot had branches where accepted drafts, the token cap and a stop
token disagreed about which rows to publish and which KV rows to keep. Ayaka's
coordinator truncates accepted drafts at the output budget and the first token
stop *before* committing KV, so the contract to pin is:

* greedy speculative decoding is lossless: same text as plain decoding, alone,
  batched, under a stop id and under every ``max_tokens`` cut;
* drafts really were accepted at more than one position (accepted 0/1/many);
* committed KV rows equal published tokens, and every reserved verify row is
  either committed or reclaimed.

The model uses ``scale=1`` because its greedy continuations repeat enough for
the n-gram proposer to be right; the default ``scale=4`` model rarely is.
"""

from __future__ import annotations

import pytest
from ayaka.configs.serving import ServingConfig
from ayaka.speculative.config import SpeculativeDecodingConfig
from cpu_e2e import MODEL, CpuServing, build_model, prompt_of

PROMPTS = [prompt_of(*[(i * 11 + k * 5) % 120 for k in range(4 + i)]) for i in range(6)]
TOKENS = 32


def _speculative_config(max_draft_tokens: int = 4) -> ServingConfig:
    return ServingConfig(
        model=MODEL,
        structured_outputs=False,
        speculative_decoding=SpeculativeDecodingConfig(max_draft_tokens=max_draft_tokens),
    )


@pytest.fixture(scope="module")
def plain():
    with CpuServing(model=build_model(scale=1.0)) as stack:
        yield stack


@pytest.fixture(scope="module")
def speculative():
    with CpuServing(model=build_model(scale=1.0), config=_speculative_config()) as stack:
        yield stack


@pytest.fixture(scope="module")
def expected(plain: CpuServing) -> list[str]:
    return [plain.generate(p, max_tokens=TOKENS).text for p in PROMPTS]


def test_greedy_speculation_is_lossless_and_really_accepts_drafts(
    speculative: CpuServing, expected: list[str]
) -> None:
    metrics = speculative.service.engine.speculation.metrics
    before = metrics.draft_tokens_accepted
    assert [speculative.generate(p, max_tokens=TOKENS).text for p in PROMPTS] == expected
    assert metrics.draft_tokens_accepted > before
    positions = metrics.accepted_per_position
    assert positions[1] > 0 and sum(positions[2:]) > 0  # one and many accepted drafts
    assert metrics.draft_tokens_proposed > metrics.draft_tokens_accepted  # and rejections


def test_batched_speculation_is_lossless(speculative: CpuServing, expected: list[str]) -> None:
    results = speculative.generate_concurrently(PROMPTS[:4], max_tokens=TOKENS)
    assert [g.text for g in results] == expected[:4]
    speculative.assert_quiescent_accounting()


def test_committed_kv_rows_equal_published_tokens(speculative: CpuServing) -> None:
    metrics = speculative.service.engine.speculation.metrics
    speculative.generate(PROMPTS[0], max_tokens=TOKENS)
    assert metrics.kv_tokens_committed == metrics.published_tokens
    assert metrics.kv_tokens_reserved == metrics.kv_tokens_committed + metrics.kv_tokens_reclaimed
    speculative.assert_quiescent_accounting()


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5, 7, 11])
def test_output_budget_cuts_accepted_drafts_exactly(
    plain: CpuServing, speculative: CpuServing, max_tokens: int
) -> None:
    want = plain.generate(PROMPTS[0], max_tokens=max_tokens)
    got = speculative.generate(PROMPTS[0], max_tokens=max_tokens)
    assert (got.text, got.finish_reason, got.completion_tokens) == (
        want.text,
        want.finish_reason,
        want.completion_tokens,
    )


@pytest.mark.parametrize("position", [2, 5, 9])
def test_stop_id_inside_accepted_drafts_ends_generation_identically(
    plain: CpuServing, speculative: CpuServing, expected: list[str], position: int
) -> None:
    stop_id = int(expected[0].split()[position].removeprefix("w")) + 4
    want = plain.generate(PROMPTS[0], max_tokens=TOKENS, stop_token_ids=[stop_id])
    got = speculative.generate(PROMPTS[0], max_tokens=TOKENS, stop_token_ids=[stop_id])
    assert want.finish_reason == "stop"
    assert (got.text, got.finish_reason, got.completion_tokens) == (
        want.text,
        want.finish_reason,
        want.completion_tokens,
    )
    speculative.assert_quiescent_accounting()


def test_cancelled_speculative_stream_releases_every_reservation(
    speculative: CpuServing,
) -> None:
    import asyncio

    metrics = speculative.service.engine.speculation.metrics
    result = asyncio.run(speculative.agenerate(PROMPTS[1], max_tokens=200, cancel_after=4))
    assert result.finish_reason == "cancelled"
    assert metrics.kv_tokens_reserved == metrics.kv_tokens_committed + metrics.kv_tokens_reclaimed
    speculative.assert_quiescent_accounting()
