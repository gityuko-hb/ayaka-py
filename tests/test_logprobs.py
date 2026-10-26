"""Generation logprobs from the sampler to the HTTP response (research A-03).

The completion coordinator already materialized typed logprob reports, but
nothing carried them past the engine, so HTTP refused ``logprobs``. Now each
published token's report rides its ``OutputEvent``; the service queues it, under
its own byte ceiling, just before that token's text, and the OpenAI adapters
render it. Everything below runs on the CPU serving stack (``cpu_e2e``):

* RAW values equal the ``log_softmax`` of a cache-free dense forward, for the
  sampled token and the top-k, whatever the temperature;
* one report per sampled token, hidden special tokens included, so the count
  equals ``usage.completion_tokens``;
* reports do not change under prefix-cache hits, chunked prefill, batching or a
  speculative stack (a logprob request is never speculated);
* stream and non-stream HTTP responses carry the same reports, in both shapes;
* the logprob byte ceiling fails only the overflowing stream and frees its pages;
* combinations that cannot be reported truthfully stay refused.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import torch
from ayaka.configs.serving import ServingConfig
from ayaka.serving.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.serving.events import GenerationFailed, TokenLogprobs
from ayaka.serving.prepare import GenerationSpec, RequestProcessor
from ayaka.speculative.config import SpeculativeDecodingConfig
from cpu_e2e import MODEL, SPECIALS, CpuServing, Generation, build_model, prompt_of

PROMPTS = [prompt_of(*[(i * 11 + k * 5) % 120 for k in range(4 + i)]) for i in range(6)]
#: Twelve greedy tokens of this prompt end in the hidden special ``<unk>``.
HIDDEN = PROMPTS[2]
LONG = prompt_of(*range(40, 80))
ATOL = 1e-4


@pytest.fixture(scope="module")
def serving():
    with CpuServing() as stack:
        yield stack


def _reports(generation: Generation) -> list[TokenLogprobs]:
    return [event for event in generation.events if isinstance(event, TokenLogprobs)]


def _assert_same_reports(left: list[TokenLogprobs], right: list[TokenLogprobs]) -> None:
    assert [r.token_id for r in left] == [r.token_id for r in right]
    for a, b in zip(left, right, strict=True):
        assert a.logprob == pytest.approx(b.logprob, abs=ATOL)
        assert [t for t, _ in a.top] == [t for t, _ in b.top]
        assert [v for _, v in a.top] == pytest.approx([v for _, v in b.top], abs=ATOL)


def _assert_matches_dense_oracle(
    stack: CpuServing, prompt: str, reports: list[TokenLogprobs], k: int
) -> None:
    """Every value is the model's log-softmax at that position, sampled path fed back."""
    sequence = stack.encode(prompt)
    with torch.inference_mode():
        for report in reports:
            logits = stack.model.forward_dense(torch.tensor(sequence))[-1].float()
            oracle = torch.log_softmax(logits, -1)
            assert report.logprob == pytest.approx(float(oracle[report.token_id]), abs=ATOL)
            assert len(report.top) == k
            for token, value in report.top:
                assert value == pytest.approx(float(oracle[token]), abs=ATOL)
            if k:
                expected = torch.topk(oracle, k).values.tolist()
                assert [v for _, v in report.top] == pytest.approx(expected, abs=ATOL)
            sequence.append(report.token_id)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_raw_logprobs_equal_the_dense_model_distribution(serving: CpuServing) -> None:
    generation = serving.generate(PROMPTS[0], max_tokens=12, logprobs=3)
    reports = _reports(generation)
    assert len(reports) == generation.completion_tokens == 12
    _assert_matches_dense_oracle(serving, PROMPTS[0], reports, k=3)
    assert all(report.top[0][0] == report.token_id for report in reports)  # greedy = top-1


def test_raw_logprobs_ignore_temperature_and_follow_the_sampled_tokens(
    serving: CpuServing,
) -> None:
    generation = serving.generate(PROMPTS[1], max_tokens=16, logprobs=2, temperature=1.5, seed=7)
    reports = _reports(generation)
    assert any(report.top[0][0] != report.token_id for report in reports)  # really sampled
    _assert_matches_dense_oracle(serving, PROMPTS[1], reports, k=2)


def test_one_report_per_sampled_token_including_hidden_specials(serving: CpuServing) -> None:
    generation = serving.generate(HIDDEN, max_tokens=12, logprobs=0)
    reports = _reports(generation)
    assert len(reports) == generation.completion_tokens == 12
    assert len(generation.text.split()) < 12  # some token has no visible text
    assert any(report.token_id < 4 for report in reports)  # a special id was reported
    assert all(report.top == () for report in reports)


def test_requests_without_logprobs_get_no_reports(serving: CpuServing) -> None:
    assert _reports(serving.generate(PROMPTS[0], max_tokens=6)) == []


# ---------------------------------------------------------------------------
# Scheduling features never change a report
# ---------------------------------------------------------------------------


def test_prefix_cache_hit_keeps_the_reports(serving: CpuServing) -> None:
    first = serving.generate(LONG, max_tokens=6, logprobs=2)
    second = serving.generate(LONG, max_tokens=6, logprobs=2)
    assert second.finish is not None and second.finish.usage.cached_tokens >= 32
    _assert_same_reports(_reports(first), _reports(second))


def test_chunked_prefill_keeps_the_reports(serving: CpuServing) -> None:
    prompt = prompt_of(*[(i * 7) % 100 for i in range(70)])
    expected = _reports(serving.generate(prompt, max_tokens=6, logprobs=2))
    with CpuServing(prefill_chunk=4) as chunked:
        _assert_same_reports(_reports(chunked.generate(prompt, max_tokens=6, logprobs=2)), expected)
        chunked.assert_quiescent_accounting()


def test_batched_requests_keep_their_own_reports(serving: CpuServing) -> None:
    solo = [_reports(serving.generate(p, max_tokens=8, logprobs=2)) for p in PROMPTS[:4]]
    batched = serving.generate_concurrently(PROMPTS[:4], max_tokens=8, logprobs=2)
    for expected, generation in zip(solo, batched, strict=True):
        _assert_same_reports(_reports(generation), expected)
    serving.assert_quiescent_accounting()


def test_a_batch_mixing_rows_with_and_without_logprobs_reports_each_correctly(
    serving: CpuServing,
) -> None:
    """Regression: the runner indexed its packed RAW snapshot by sampling row.

    A batch whose leading sampling rows asked for no logprobs read past the
    snapshot, and every request in that batch failed with "model execution
    failed".
    """
    asks = [None, 2, None, 1]
    solo = [
        serving.generate(p, max_tokens=8, logprobs=k)
        for p, k in zip(PROMPTS[:4], asks, strict=True)
    ]

    async def scenario():
        return await asyncio.gather(
            *(
                serving.agenerate(p, max_tokens=8, logprobs=k)
                for p, k in zip(PROMPTS[:4], asks, strict=True)
            )
        )

    batched = asyncio.run(asyncio.wait_for(scenario(), 60))
    for expected, got, k in zip(solo, batched, asks, strict=True):
        assert got.finish_reason == "length" and got.text == expected.text
        assert bool(_reports(got)) == (k is not None)
        _assert_same_reports(_reports(got), _reports(expected))
    serving.assert_quiescent_accounting()


def test_speculative_stack_reports_logprobs_without_speculating() -> None:
    config = ServingConfig(
        model=MODEL,
        structured_outputs=False,
        speculative_decoding=SpeculativeDecodingConfig(max_draft_tokens=4),
    )
    with CpuServing(model=build_model(scale=1.0)) as plain:
        expected = plain.generate(PROMPTS[0], max_tokens=24, logprobs=2)
    with CpuServing(model=build_model(scale=1.0), config=config) as speculative:
        metrics = speculative.service.engine.speculation.metrics
        before = metrics.draft_tokens_proposed
        got = speculative.generate(PROMPTS[0], max_tokens=24, logprobs=2)
        assert metrics.draft_tokens_proposed == before  # the logprob request was not speculated
        assert got.text == expected.text
        _assert_same_reports(_reports(got), _reports(expected))
        speculative.generate(PROMPTS[0], max_tokens=24)
        assert metrics.draft_tokens_proposed > before  # ... while plain requests are
        speculative.assert_quiescent_accounting()


# ---------------------------------------------------------------------------
# Bounds, overflow and cancellation
# ---------------------------------------------------------------------------


def test_logprob_byte_ceiling_fails_only_the_overflowing_stream() -> None:
    from ayaka.serving.service import LOGPROB_ENTRY_BYTES

    config = ServingConfig(
        model=MODEL,
        structured_outputs=False,
        max_stream_logprob_bytes=4 * LOGPROB_ENTRY_BYTES - 1,  # one k=3 report is 4 entries
    )
    with CpuServing(config=config) as stack:

        async def scenario():
            return await asyncio.gather(
                stack.agenerate(PROMPTS[0], max_tokens=8, logprobs=3),
                stack.agenerate(PROMPTS[1], max_tokens=8),
                stack.agenerate(PROMPTS[2], max_tokens=8, logprobs=2),  # 3 entries fit
            )

        overflowed, plain, fitting = asyncio.run(asyncio.wait_for(scenario(), 60))
        assert isinstance(overflowed.finish, GenerationFailed)
        assert "too slow" in overflowed.finish.message
        assert plain.finish_reason == "length" and fitting.finish_reason == "length"
        assert len(_reports(fitting)) == 8
        registry = stack.service.stats.registry
        sample = registry.get_sample_value(
            "ayaka_stream_overflows_total", {"reason": "logprob_bytes"}
        )
        assert sample == 1.0
        stack.assert_quiescent_accounting()


def test_cancelling_a_logprob_stream_releases_its_pages(serving: CpuServing) -> None:
    generation = asyncio.run(
        asyncio.wait_for(
            serving.agenerate(PROMPTS[1], max_tokens=200, logprobs=5, cancel_after=3), 60
        )
    )
    assert generation.finish_reason == "cancelled"
    assert _reports(generation)
    serving.assert_quiescent_accounting()


def _processor(serving: CpuServing, **config: object) -> RequestProcessor:
    assert serving.runtime.tokenizer is not None
    return RequestProcessor(
        serving.runtime.tokenizer, ServingConfig(model=MODEL, structured_outputs=False, **config)
    )


def test_counts_above_the_server_ceiling_are_refused_before_tokenizing(
    serving: CpuServing,
) -> None:
    spec = GenerationSpec(model=MODEL, prompt="w1", logprobs=3)
    with pytest.raises(InvalidRequestError, match="at most 2"):
        _processor(serving, max_logprobs=2).prepare(spec, "tenant")
    assert _processor(serving, max_logprobs=3).prepare(spec, "tenant").sampling.logprobs == 3


def test_tool_calls_and_reasoning_cannot_be_reported_and_are_refused(serving: CpuServing) -> None:
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    messages = [{"role": "user", "content": "w1"}]
    with_tools = GenerationSpec(model=MODEL, messages=messages, tools=[tool], logprobs=1)
    with pytest.raises(UnsupportedFeatureError, match="tool calls"):
        _processor(serving, tool_parser="hermes").prepare(with_tools, "tenant")
    chat = GenerationSpec(model=MODEL, messages=messages, logprobs=1)
    with pytest.raises(UnsupportedFeatureError, match="reasoning parser"):
        _processor(serving, reasoning_parser="think").prepare(chat, "tenant")
    completion = GenerationSpec(model=MODEL, prompt="w1", logprobs=1)  # nothing to parse
    assert _processor(serving, reasoning_parser="think").prepare(completion, "tenant")


# ---------------------------------------------------------------------------
# HTTP: OpenAI shapes, stream/non-stream parity
# ---------------------------------------------------------------------------


def _body(protocol: str, prompt: str, **extra: object) -> dict:
    body: dict = {"model": MODEL, "max_tokens": 12, "temperature": 0, **extra}
    if protocol == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    return body


def _logprob_fields(protocol: str) -> dict:
    return {"logprobs": True, "top_logprobs": 2} if protocol == "chat" else {"logprobs": 2}


def _path(protocol: str) -> str:
    return "/v1/chat/completions" if protocol == "chat" else "/v1/completions"


def _stream_chunks(client, protocol: str, body: dict) -> list[dict]:
    chunks = []
    with client.stream("POST", _path(protocol), json={**body, "stream": True}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunks.append(json.loads(line[len("data: ") :]))
    return chunks


def test_http_completion_logprobs_match_the_engine_reports(serving: CpuServing) -> None:
    reports = _reports(serving.generate(PROMPTS[0], max_tokens=12, logprobs=2))
    response = serving.http().post(
        _path("completion"), json=_body("completion", PROMPTS[0], logprobs=2)
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    logprobs = choice["logprobs"]
    assert len(logprobs["tokens"]) == response.json()["usage"]["completion_tokens"] == 12
    assert logprobs["token_logprobs"] == pytest.approx([r.logprob for r in reports], abs=ATOL)
    for token, alternatives in zip(logprobs["tokens"], logprobs["top_logprobs"], strict=True):
        assert token in alternatives and len(alternatives) <= 3  # top-2 plus the sampled token
    offsets = logprobs["text_offset"]
    assert offsets[0] == 0 and offsets == sorted(offsets)
    assert serving.decode([r.token_id for r in reports]) == choice["text"].strip()


def test_http_completion_logprobs_zero_still_reports_the_sampled_token(
    serving: CpuServing,
) -> None:
    body = _body("completion", PROMPTS[1], logprobs=0, temperature=1.5, seed=3)
    logprobs = serving.http().post(_path("completion"), json=body).json()["choices"][0]["logprobs"]
    for token, value, alternatives in zip(
        logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"], strict=True
    ):
        assert alternatives == {token: value}  # OpenAI: "up to logprobs + 1" entries


def test_http_chat_logprobs_have_the_openai_shape(serving: CpuServing) -> None:
    response = serving.http().post(
        _path("chat"), json=_body("chat", PROMPTS[0], logprobs=True, top_logprobs=2)
    )
    assert response.status_code == 200, response.text
    content = response.json()["choices"][0]["logprobs"]["content"]
    assert len(content) == 12
    reports = _reports(serving.generate(PROMPTS[0], max_tokens=12, logprobs=2))
    for entry, report in zip(content, reports, strict=True):
        assert set(entry) == {"token", "logprob", "bytes", "top_logprobs"}
        assert entry["bytes"] == list(entry["token"].encode())
        assert entry["logprob"] == pytest.approx(report.logprob, abs=ATOL)
        assert [alt["logprob"] for alt in entry["top_logprobs"]] == pytest.approx(
            [value for _, value in report.top], abs=ATOL
        )


def test_http_without_logprobs_reports_null(serving: CpuServing) -> None:
    response = serving.http().post(_path("chat"), json=_body("chat", PROMPTS[0]))
    assert response.json()["choices"][0]["logprobs"] is None


@pytest.mark.parametrize("protocol", ["completion", "chat"])
def test_http_stream_and_nonstream_carry_the_same_reports(
    serving: CpuServing, protocol: str
) -> None:
    client = serving.http()
    body = _body(protocol, HIDDEN, **_logprob_fields(protocol))
    whole = client.post(_path(protocol), json=body).json()["choices"][0]["logprobs"]
    chunks = _stream_chunks(client, protocol, body)
    carried = [c["choices"][0] for c in chunks if c["choices"] and c["choices"][0]["logprobs"]]
    # The second run hits the prefix cache, so floats agree to rounding only.
    if protocol == "chat":
        streamed = [entry for choice in carried for entry in choice["logprobs"]["content"]]
        assert len(streamed) == len(whole["content"]) == 12
        for got, want in zip(streamed, whole["content"], strict=True):
            assert (got["token"], got["bytes"]) == (want["token"], want["bytes"])
            assert got["logprob"] == pytest.approx(want["logprob"], abs=ATOL)
            assert [a["token"] for a in got["top_logprobs"]] == [
                a["token"] for a in want["top_logprobs"]
            ]
        trailing = carried[-1]["delta"]
    else:
        merged = {
            key: [v for choice in carried for v in choice["logprobs"][key]]
            for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset")
        }
        assert merged["tokens"] == whole["tokens"] and len(whole["tokens"]) == 12
        assert merged["text_offset"] == whole["text_offset"]
        assert merged["token_logprobs"] == pytest.approx(whole["token_logprobs"], abs=ATOL)
        for got, want in zip(merged["top_logprobs"], whole["top_logprobs"], strict=True):
            assert got == pytest.approx(want, abs=ATOL)
        trailing = carried[-1]["text"]
    # Each chunk with visible text carries the reports of the tokens behind it.
    for chunk in chunks:
        if not chunk["choices"]:
            continue
        choice = chunk["choices"][0]
        text = choice["text"] if protocol == "completion" else choice["delta"].get("content")
        if text:
            reported = choice["logprobs"]["content" if protocol == "chat" else "tokens"]
            assert reported, chunk
    # The last token is a hidden special: its report still arrives, under its
    # own name, in an empty-text chunk just before the finish chunk.
    last = whole["content"][-1]["token"] if protocol == "chat" else whole["tokens"][-1]
    assert last in SPECIALS
    assert not trailing
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize(
    ("protocol", "extra", "status", "code"),
    [
        ("chat", {"top_logprobs": 2}, 400, "invalid_request"),
        ("chat", {"logprobs": "yes"}, 400, "invalid_request"),
        ("chat", {"logprobs": True, "top_logprobs": 21}, 400, "invalid_request"),
        ("completion", {"logprobs": True}, 400, "invalid_request"),
        ("completion", {"top_logprobs": 2}, 400, "unsupported_feature"),
        ("completion", {"echo": True, "logprobs": 1}, 400, "unsupported_feature"),
    ],
)
def test_http_refuses_what_it_cannot_report(
    serving: CpuServing, protocol: str, extra: dict, status: int, code: str
) -> None:
    response = serving.http().post(_path(protocol), json=_body(protocol, PROMPTS[0], **extra))
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code


# ---------------------------------------------------------------------------
# Typed report contract
# ---------------------------------------------------------------------------


def test_token_logprobs_validate_their_fields() -> None:
    with pytest.raises(TypeError):
        TokenLogprobs(1, 0)  # type: ignore[arg-type]  # an int is not a logprob
    with pytest.raises(TypeError):
        TokenLogprobs(1, -1.0, [(2, -2.0)])  # type: ignore[arg-type]  # mutable
    with pytest.raises(TypeError):
        TokenLogprobs(1, -1.0, ((2, True),))  # type: ignore[arg-type]
    assert TokenLogprobs(1, -1.0, ((2, -2.0),)).top == ((2, -2.0),)


def test_a_report_for_another_token_is_refused_by_the_output_owner() -> None:
    from ayaka.request.schema import Request, RequestId
    from ayaka.runtime.output import OutputProcessor

    output = OutputProcessor()
    output.register(Request(request_id=RequestId("r"), prompt_token_ids=(1, 2)))
    with pytest.raises(ValueError, match="different token"):
        output.on_published(
            "r",
            5,
            sequence_epoch=1,
            generated_tokens=1,
            prompt_tokens=2,
            logprobs=TokenLogprobs(6, -0.1),
        )
    output.on_published(
        "r",
        5,
        sequence_epoch=1,
        generated_tokens=1,
        prompt_tokens=2,
        logprobs=TokenLogprobs(5, -0.1),
    )
    assert output.events("r")[0].logprobs == TokenLogprobs(5, -0.1)
