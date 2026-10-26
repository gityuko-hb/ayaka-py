"""Scheduler scenarios end to end on CPU (research tasks F01 / E01, R1 parity).

Every test drives the production ``ServingRuntime`` (``cpu_e2e.CpuServing``): real
service owner thread, engine, Continuous or Eager scheduler, KV allocator, prefix
cache, ``LocalWorker`` and completion coordinator, with a tiny seeded model on the
reference attention backend. Greedy decoding makes the text a deterministic
function of the weights and the prompt, so any scheduling, chunking, batching,
preemption or KV bug shows up as different text. Page accounting is checked
after each scenario.

Scope: host contracts and output parity. Nothing here measures GPU timing,
fairness under real load, or CUDA lifetime.
"""

from __future__ import annotations

import asyncio

import pytest
from ayaka.configs.serving import ServingConfig
from ayaka.serving.errors import OverloadedError
from ayaka.serving.events import GenerationFinished
from cpu_e2e import MODEL, CpuServing, Generation, prompt_of

PROMPTS = [prompt_of(*[(i * 11 + k * 5) % 120 for k in range(4 + i)]) for i in range(6)]
LONG = prompt_of(*[(i * 7) % 100 for i in range(70)])


@pytest.fixture(scope="module")
def serving():
    with CpuServing() as stack:
        yield stack


@pytest.fixture(scope="module")
def reference_texts(serving: CpuServing) -> list[str]:
    """Solo greedy continuations from the default Continuous stack."""
    return [serving.generate(p, max_tokens=16).text for p in PROMPTS]


def _texts(generations: list[Generation]) -> list[str]:
    return [g.text for g in generations]


# ---------------------------------------------------------------------------
# E01: one request through the whole stack
# ---------------------------------------------------------------------------


def test_one_request_reports_exact_usage_and_finish(serving: CpuServing) -> None:
    result = serving.generate(PROMPTS[2], max_tokens=12)
    assert isinstance(result.finish, GenerationFinished)
    assert result.finish_reason == "length"
    assert result.finish.usage.prompt_tokens == 6
    assert result.completion_tokens == 12  # sampled, including hidden special ids
    serving.assert_quiescent_accounting()


@pytest.mark.parametrize("index", range(len(PROMPTS)))
def test_paged_serving_matches_a_dense_forward_oracle(serving: CpuServing, index: int) -> None:
    """Paged KV + reference attention + scheduler == a cache-free dense forward."""
    oracle = serving.dense_greedy(PROMPTS[index], 16)
    served = serving.generate(PROMPTS[index], max_tokens=16)
    assert served.text.strip() == serving.decode(oracle)


def test_greedy_generation_is_deterministic(serving: CpuServing) -> None:
    first = serving.generate(PROMPTS[0], max_tokens=16).text
    assert all(serving.generate(PROMPTS[0], max_tokens=16).text == first for _ in range(2))
    assert len(set(first.split())) > 4  # the model really depends on context


def test_continuations_differ_by_prompt(reference_texts: list[str]) -> None:
    assert len(set(reference_texts)) == len(reference_texts)


def test_eager_reference_scheduler_matches_continuous() -> None:
    """The golden-output comparison the research asks for (same weights, greedy)."""
    options = {"max_model_len": 128, "batch_tokens": 128}
    with CpuServing(**options) as continuous:
        expected = [continuous.generate(p, max_tokens=16).text for p in PROMPTS]
    with CpuServing(scheduler_kind="eager", **options) as eager:
        assert [eager.generate(p, max_tokens=16).text for p in PROMPTS] == expected
        assert _texts(eager.generate_concurrently(PROMPTS[:4], max_tokens=16)) == expected[:4]
        eager.assert_quiescent_accounting()


def test_eager_refuses_contexts_larger_than_its_step_budget() -> None:
    """Whole-prompt prefill cannot split a context; it is refused, not truncated."""
    from ayaka.configs.base import ConfigError

    with pytest.raises(ConfigError, match="UNCHUNKED_MODEL_TOO_LARGE"):
        CpuServing(scheduler_kind="eager", max_model_len=256, batch_tokens=64)


def test_unknown_scheduler_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="scheduler_kind"):
        CpuServing(scheduler_kind="fastest")


# ---------------------------------------------------------------------------
# F01: Continuous features never change greedy output
# ---------------------------------------------------------------------------


def test_continuous_batching_does_not_change_outputs(
    serving: CpuServing, reference_texts: list[str]
) -> None:
    concurrent = serving.generate_concurrently(PROMPTS[:4], max_tokens=16)
    assert _texts(concurrent) == reference_texts[:4]
    assert {g.finish_reason for g in concurrent} == {"length"}
    serving.assert_quiescent_accounting()


@pytest.mark.parametrize("chunk", [4, 8, 64])
def test_chunked_prefill_matches_whole_prompt_prefill(chunk: int) -> None:
    with CpuServing(prefill_chunk=64) as whole:
        expected = whole.generate(LONG, max_tokens=8).text
    with CpuServing(prefill_chunk=chunk) as chunked:
        assert chunked.generate(LONG, max_tokens=8).text == expected
        # A long chunked prefill next to short requests still yields their text.
        mixed = chunked.generate_concurrently([LONG, *PROMPTS[:2]], max_tokens=8)
        assert mixed[0].text == expected
        chunked.assert_quiescent_accounting()


def test_prefix_cache_hit_reuses_tokens_without_changing_output(serving: CpuServing) -> None:
    prompt = prompt_of(*range(40, 80))  # 40 tokens: two full pages are shareable
    first = serving.generate(prompt, max_tokens=8)
    second = serving.generate(prompt, max_tokens=8)
    assert second.text == first.text
    assert isinstance(second.finish, GenerationFinished)
    assert second.finish.usage.cached_tokens >= 32
    assert serving.memory().cache_owned_pages > 0  # the prefix stayed cached, unowned
    serving.assert_quiescent_accounting()


def test_recompute_preemption_under_kv_pressure_preserves_outputs() -> None:
    options = {"max_model_len": 64, "batch_tokens": 64}
    with CpuServing(**options) as roomy:
        expected = [roomy.generate(p, max_tokens=16).text for p in PROMPTS[:4]]
    with CpuServing(pages=5, max_requests=4, **options) as tight:  # 4 usable pages
        results = tight.generate_concurrently(PROMPTS[:4], max_tokens=16)
        stats = tight.service.engine.scheduler.stats
        assert stats.preemptions > 0 and stats.recomputed_tokens > 0  # pressure was real
        assert _texts(results) == expected
        tight.assert_quiescent_accounting()


@pytest.mark.parametrize("inflight", [2, 4])
def test_multiple_inflight_tickets_preserve_outputs(
    inflight: int, reference_texts: list[str]
) -> None:
    """R1 baseline parity: more tickets in flight never change the published text."""
    with CpuServing(max_inflight=inflight) as stack:
        assert (
            _texts(stack.generate_concurrently(PROMPTS[:4], max_tokens=16)) == (reference_texts[:4])
        )
        stack.assert_quiescent_accounting()


# ---------------------------------------------------------------------------
# Stops, cancellation, deadlines, admission
# ---------------------------------------------------------------------------


def test_stop_string_ends_generation_before_the_stop_text(
    serving: CpuServing, reference_texts: list[str]
) -> None:
    words = reference_texts[0].split()
    stop = words[5]
    index = words.index(stop)  # first occurrence
    result = serving.generate(PROMPTS[0], max_tokens=16, stop=[stop])
    assert result.finish_reason == "stop"
    assert result.text.split() == words[:index]


def test_stop_token_id_ends_generation_on_that_token(
    serving: CpuServing, reference_texts: list[str]
) -> None:
    """Nothing is published after the first accepted stop id.

    A non-special stop token keeps its text (vLLM semantics); special ids such as
    EOS are hidden by detokenization.
    """
    words = reference_texts[1].split()
    stop_word = words[3]
    first = words.index(stop_word)
    token_id = int(stop_word.removeprefix("w")) + 4
    result = serving.generate(PROMPTS[1], max_tokens=16, stop_token_ids=[token_id])
    assert result.finish_reason == "stop"
    assert result.text.split() == words[: first + 1]
    assert result.completion_tokens == first + 1


def test_cancelling_one_stream_leaves_its_neighbours_intact(
    serving: CpuServing, reference_texts: list[str]
) -> None:
    async def scenario():
        return await asyncio.gather(
            serving.agenerate(PROMPTS[0], max_tokens=64, cancel_after=3),
            serving.agenerate(PROMPTS[1], max_tokens=16),
        )

    cancelled, other = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert isinstance(cancelled.finish, GenerationFinished)
    assert cancelled.finish_reason == "cancelled"
    assert reference_texts[0].startswith(cancelled.text.split(" w", 3)[0])
    assert other.text == reference_texts[1]
    serving.assert_quiescent_accounting()


def test_request_deadline_times_out_and_releases_its_pages(serving: CpuServing) -> None:
    result = serving.generate(PROMPTS[3], max_tokens=200, timeout=0.02)
    assert result.finish_reason == "timeout"
    assert 0 <= result.completion_tokens < 200
    serving.assert_quiescent_accounting()


def test_queue_timeout_refuses_a_request_that_waited_too_long() -> None:
    config = ServingConfig(
        model=MODEL, structured_outputs=False, default_queue_timeout_seconds=0.05
    )
    with CpuServing(config=config, max_requests=1, max_pending_requests=2) as stack:

        async def scenario():
            first = asyncio.ensure_future(stack.agenerate(PROMPTS[4], max_tokens=200))
            await asyncio.sleep(0.01)
            second = await stack.agenerate(PROMPTS[5], max_tokens=4)
            return await first, second

        first, second = asyncio.run(asyncio.wait_for(scenario(), 60))
        assert first.finish_reason == "length"
        assert second.finish_reason == "timeout"  # queue deadline, before any compute
        assert second.completion_tokens == 0
        stack.assert_quiescent_accounting()


def test_admission_bound_refuses_excess_requests_without_side_effects() -> None:
    with CpuServing(max_requests=2, max_pending_requests=2) as stack:

        async def scenario():
            return await asyncio.gather(
                *(stack.agenerate(p, max_tokens=24) for p in PROMPTS[:5]),
                return_exceptions=True,
            )

        outcomes = asyncio.run(asyncio.wait_for(scenario(), 60))
        refused = [o for o in outcomes if isinstance(o, OverloadedError)]
        served = [o for o in outcomes if isinstance(o, Generation)]
        assert refused and served and len(refused) + len(served) == 5
        assert {g.finish_reason for g in served} == {"length"}
        stack.assert_quiescent_accounting()
        assert stack.generate(PROMPTS[0], max_tokens=4).finish_reason == "length"
