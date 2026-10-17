"""Exercise the real tokenizer, RequestProcessor, GenerationPipeline and handle.

Only the service's device execution is scripted; these tests do not load model
weights, create a scheduler, or make GPU performance claims.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ayaka.configs.serving import ServingConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.serving.events import ContentDelta, GenerationFinished, UsageUpdate
from ayaka.serving.generation import GenerationPipeline
from ayaka.serving.http import create_app
from ayaka.serving.prepare import GenerationSpec, RequestProcessor
from ayaka.serving.protocols import chat
from ayaka.serving.protocols.engine import AyakaHost, AyakaModels, to_spec
from ayaka.serving.service import GenerationHandle
from ayaka.tokenizers.contracts import EncodeOptions, TextInput, TokenizerCapabilities
from ayaka.tokenizers.factory import LoadedTokenizer
from ayaka.tokenizers.service import TokenizerOverloaded, TokenizerService
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.websockets import WebSocketDisconnect


class Tokenizer:
    chat_template = "{{ messages }}"
    eos_token_id = None

    def fingerprint(self):
        return "contract-tokenizer"

    def get_vocab(self):
        return {str(i): i for i in range(256)}

    def encode(self, text, *, add_special_tokens=True):
        return list(text.encode())

    def encode_batch(self, texts, *, add_special_tokens=True):
        return [self.encode(text) for text in texts]

    def apply_chat_template(self, messages, *, tools=None, **kwargs):
        value = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)
        return value + ("<assistant>" if kwargs.get("add_generation_prompt", True) else "")


class Stats:
    def __init__(self):
        self.writes = []
        self.rejections = []

    def observe_client_ttft(self, ingress_ns):
        self.writes.append(ingress_ns)

    def record_rejection(self, reason):
        self.rejections.append(reason)


class Service:
    ready = True

    def __init__(self, config):
        self.config, self.stats, self.handles = config, Stats(), []

    def submit(self, request, spec, *, timings):
        handle = GenerationHandle(request, spec, self.config)
        handle.ingress_ns = timings.ingress_ns
        handle.put(ContentDelta("real pipeline"))
        handle.put(GenerationFinished("stop", UsageUpdate(request.prompt_len, 2)), terminal=True)
        self.handles.append(handle)
        future = Future()
        future.set_result(handle)
        return future

    def close(self):
        return True


@pytest.fixture
def backend():
    tokenizer = TokenizerService(
        TokenizerConfig("contract", encode_pool_workers=1),
        max_model_len=4096,
        model_vocab_size=256,
        loaded=LoadedTokenizer(
            Tokenizer(), TokenizerCapabilities("hf", True, True, True, False, False)
        ),
    )
    config = ServingConfig(model="test-model", expose_metrics=False)
    processor = RequestProcessor(tokenizer, config)
    service = Service(config)
    try:
        yield service, processor
    finally:
        tokenizer.close()


def request(**extra):
    return {"model": "test-model", "messages": [{"role": "user", "content": "hello"}], **extra}


@pytest.mark.parametrize("stream", [False, True])
def test_real_pipeline_cleanup_and_client_ttft(backend, stream):
    service, processor = backend
    app = create_app(service, processor)
    with TestClient(app) as client:
        reply = client.post("/v1/chat/completions", json=request(stream=stream))
        assert reply.status_code == 200, reply.text
        assert "real pipeline" in reply.text
        assert app.state.pipeline.active == 0
    assert service.handles[0].cancelled.is_set()
    assert len(service.stats.writes) == 1
    assert service.handles[0].first_socket_write_ns > 0


def test_omitted_limit_is_remaining_context_and_cache_policy_is_preserved(backend):
    service, processor = backend
    adapted = chat.adapt(request(cache_prompt=False, timeout=2.0, min_tokens=2, stop_token_ids=[3]))
    assert adapted.request.controls.max_output_tokens is None
    prepared = processor.prepare(to_spec("test-model", adapted.request), "tenant")
    assert prepared.stop.max_tokens == processor.tokenizer.max_model_len - prepared.prompt_len
    assert not prepared.cache.prefix_cache and not prepared.cache.store_kv
    assert prepared.stop.min_tokens == 2 and prepared.stop.stop_token_ids == (3,)
    assert prepared.deadline_ns is not None


def test_host_count_and_template_use_the_generation_renderer_without_admission(backend):
    service, processor = backend
    pipeline = GenerationPipeline(service, processor)
    models = AyakaModels(pipeline, processor)
    input = chat.adapt(request(max_tokens=2)).request.input
    host = AyakaHost(models, "test-model")
    count = host.count(input)
    applied = host.apply_template(input)
    rendered = applied["prompt"]
    assert applied["generation_prompt"] == "<assistant>"
    assert count == len(rendered.encode())
    prepared = processor.prepare(
        to_spec("test-model", chat.adapt(request(max_tokens=2)).request), "tenant"
    )
    assert prepared.prompt_len == count
    assert pipeline.active == 0 and not service.handles
    assert all(value is None for value in host.properties()["template_capabilities"].values())


def test_host_count_reserves_zero_output_even_at_full_context(backend):
    _, processor = backend
    t = processor.tokenizer
    source = TextInput("x" * t.max_model_len)
    encoded = t.encode(source, EncodeOptions(max_output_tokens=0))
    assert len(encoded.token_ids) == t.max_model_len
    with pytest.raises(Exception, match="context"):
        processor.prepare(
            GenerationSpec(model="test-model", prompt=source.text, max_tokens=None), "t"
        )


def test_required_tool_prompt_is_identical_for_counting_and_generation(backend):
    service, processor = backend
    adapted = chat.adapt(
        request(
            tools=[
                {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
            ],
            tool_choice="required",
            max_tokens=32,
        )
    )
    models = AyakaModels(GenerationPipeline(service, processor), processor)
    count = AyakaHost(models, "test-model").count(adapted.request.input)
    prepared = processor.prepare(to_spec("test-model", adapted.request), "tenant")
    assert prepared.prompt_len == count


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning_effort": "high"},
        {"thinking_budget_tokens": 8},
        {"chat_template_kwargs": {"custom": 1}},
        {"response_format": {"type": "grammar", "grammar": 'root ::= "x"'}},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                    ],
                }
            ]
        },
    ],
)
def test_unsupported_backend_controls_are_errors_before_execution(backend, extra):
    service, processor = backend
    with TestClient(create_app(service, processor)) as client:
        reply = client.post("/v1/chat/completions", json=request(**extra))
        assert reply.status_code == 400
        assert reply.json()["error"]["code"] == "unsupported_feature"
    assert not service.handles


def test_anthropic_aliases_and_websockets_share_auth(backend):
    service, processor = backend
    config = replace(processor.config, api_keys=("secret",))
    processor.config = service.config = config
    with TestClient(create_app(service, processor)) as client:
        for path in ("/v1/messages", "/anthropic/v1/messages"):
            payload = request(max_tokens=2)
            assert client.post(path, json=payload).status_code == 401
            assert (
                client.post(
                    path,
                    json=payload,
                    headers={"x-api-key": "secret", "anthropic-version": "2023-06-01"},
                ).status_code
                == 200
            )
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect("/v1/responses"):
                pass
        assert denied.value.code == 1008
        with client.websocket_connect(
            "/v1/responses", headers={"Authorization": "Bearer secret"}
        ) as ws:
            ws.send_json({"type": "response.create", "model": "test-model", "input": "hello"})
            while (value := ws.receive_json())["type"] != "response.completed":
                assert value["type"] != "response.failed"
    assert service.handles and service.stats.rejections.count("auth") == 3
    assert all(handle.request.tenant_id != "anonymous" for handle in service.handles)


def test_legacy_completion_endpoint_remains_available(backend):
    service, processor = backend
    with TestClient(create_app(service, processor)) as client:
        reply = client.post("/v1/completions", json={"model": "test-model", "prompt": "hello"})
        assert reply.status_code == 200
        assert reply.json()["choices"][0]["text"] == "real pipeline"


def test_template_rendering_releases_its_bounded_charge_on_exception(backend):
    _, processor = backend
    tokenizer = processor.tokenizer
    input = processor.render_source(
        GenerationSpec(model="test-model", messages=request()["messages"])
    )
    original = tokenizer.tokenizer.apply_chat_template
    tokenizer.tokenizer.apply_chat_template = lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("bad")
    )
    try:
        with pytest.raises(ValueError, match="bad"):
            tokenizer.render_prompt(input)
        assert tokenizer.pending == (0, 0)
    finally:
        tokenizer.tokenizer.apply_chat_template = original


def test_header_write_failure_releases_a_prepared_binding(backend):
    from ayaka.serving.protocols.common import Journal
    from ayaka.serving.protocols.router import _GenerationResponse, _Session, _source_events
    from ayaka.serving.protocols.source import Admitted

    async def run():
        service, processor = backend
        pipeline = GenerationPipeline(service, processor)
        adapted = chat.adapt(request(stream=True))
        session = _Session()
        source = _source_events(AyakaModels(pipeline, processor), adapted, "t", False, session)
        assert isinstance(await anext(source), Admitted)
        assert pipeline.active == 1

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            raise OSError("disconnected during headers")

        body = chat.project(source, Journal("test-model"), adapted)
        response = _GenerationResponse(body, source=source, session=session)
        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert pipeline.active == 0
        assert service.handles[0].cancelled.is_set()

    asyncio.run(run())


def test_tokenizer_host_admission_is_bounded(backend):
    _, processor = backend
    tokenizer = processor.tokenizer
    tokenizer.config = replace(tokenizer.config, encode_max_pending_bytes=2)
    with pytest.raises(TokenizerOverloaded):
        tokenizer.render_prompt(TextInput("large"))
    assert tokenizer.pending == (0, 0)


def test_default_protocol_properties_do_not_invent_model_size(backend):
    service, processor = backend
    host = AyakaHost(AyakaModels(GenerationPipeline(service, processor), processor), "test-model")
    props = host.properties()
    assert "model_size_bytes" not in props
    assert props["default_generation_settings"]["n_ctx"] == 4096
    assert props["modalities"] == {"vision": False, "audio": False, "video": False}


@pytest.mark.parametrize("constrained", [False, True])
def test_length_cutoff_keeps_completed_calls_and_drops_partial_arguments(constrained):
    from ayaka.serving.events import ToolCallEnd
    from ayaka.serving.parsers import OutputParser

    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    parser = OutputParser(tools=tools, constrained_tools=constrained)
    text = (
        '{"tool_calls":[{"name":"f","arguments":{}},{"name":"f","arguments":{"x":"cut'
        if constrained
        else '<tool_call>{"name":"f","arguments":{}}</tool_call>'
        '<tool_call>{"name":"f","arguments":{"x":"cut'
    )
    events = parser.feed(text) + parser.finish(truncated=True)
    calls = [event for event in events if isinstance(event, ToolCallEnd)]
    assert len(calls) == 1 and calls[0].arguments == "{}"
    assert parser.finish(truncated=True) == []


@pytest.mark.parametrize("reason,expected", [("length", "length"), ("timeout", "timeout")])
def test_service_terminal_cause_takes_precedence_over_earlier_tools(backend, reason, expected):
    from ayaka.request.states import RequestState
    from ayaka.sched.outcome import FinishReason
    from ayaka.serving.service import ServingService

    service, processor = backend
    spec = GenerationSpec(model="test-model", messages=request()["messages"], max_tokens=2)
    prepared = processor.prepare(spec, "t")
    handle = GenerationHandle(prepared, spec, service.config)
    handle.parser.had_tools = True
    lifecycle = SimpleNamespace(
        request=prepared,
        is_terminal=True,
        state=RequestState.FINISHED,
        finish_reason=FinishReason(reason),
        output_token_ids=(1, 2),
        machine=SimpleNamespace(
            num_cached_tokens=0,
            timing=SimpleNamespace(
                first_scheduled_ns=0,
                first_token_ns=0,
                inter_token_intervals_ns=(),
                inter_token_percentiles_ns=lambda: (0, 0),
            ),
        ),
    )
    owner = ServingService.__new__(ServingService)
    owner._handles = {str(prepared.request_id): handle}
    owner._clock = time.monotonic_ns
    owner.stats = Mock()
    owner.engine = SimpleNamespace(
        requests=Mock(get=lambda _: lifecycle),
        events=lambda _: [],
        forget=Mock(),
        executor=SimpleNamespace(runner=SimpleNamespace(forget_request=Mock())),
    )
    owner._collect()
    terminal = asyncio.run(handle.next_event())
    assert isinstance(terminal, GenerationFinished) and terminal.finish_reason == expected
    assert not owner._handles
