"""CPU contract tests derived from the supplied Rust Serving regression cases."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing

import pytest
from ayaka.serving.events import (
    ContentDelta,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageUpdate,
)
from ayaka.serving.protocols import anthropic, chat, responses
from ayaka.serving.protocols.common import Journal
from ayaka.serving.protocols.error import ApiError
from ayaka.serving.protocols.media import decode_image_data_url
from ayaka.serving.protocols.router import create_app
from ayaka.serving.protocols.source import (
    Admitted,
    Completed,
    Failed,
    ModelLoadProgress,
    ModelLoadStage,
    Output,
)
from fastapi.testclient import TestClient

MODEL = "test-model"
ANTHROPIC = {"anthropic-version": "2023-06-01"}
PATHS = ("/v1/chat/completions", "/v1/responses", "/anthropic/v1/messages")


def payload(path=PATHS[0], **extra):
    base = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}
    if path == PATHS[1]:
        base = {"model": MODEL, "input": "hello"}
    elif path == PATHS[2] or path.startswith("/v1/messages"):
        base["max_tokens"] = 32
    return {**base, **extra}


def wire_events(body):
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]


class FakeHost:
    def __init__(self):
        self.counted = []

    def count(self, input):
        self.counted.append(input)
        return 7

    def apply_template(self, input):
        return {"prompt": "exact template", "template_fingerprint": "known"}

    def properties(self):
        return {"general_name": MODEL, "modalities": {"vision": True}}


class FakeModels:
    def __init__(self, events=None, *, refused=None, stalled=False, loading=False):
        self.events = (
            events
            if events is not None
            else [
                Admitted(7),
                Output(ContentDelta("hello")),
                Completed(UsageUpdate(7, 2, 3)),
            ]
        )
        self.refused, self.stalled, self.loading = refused, stalled, loading
        self.requests, self.invokes, self.closed, self.host_calls = [], 0, 0, 0
        self.host_artifacts = FakeHost()
        self.cancelled = False

    def check(self, model):
        if model != MODEL:
            raise ApiError("unknown model", status=404, code="model_not_found")

    async def invoke(self, model, progress=None, *, tenant="anonymous"):
        self.check(model)
        self.invokes += 1
        if progress is not None and self.loading:
            progress(ModelLoadProgress(ModelLoadStage.LOADING_WEIGHTS, 0.5))
            await asyncio.sleep(0.001)
        if self.refused:
            raise self.refused
        models = self

        class Invocation:
            closed = False

            async def generate(self, request):
                models.requests.append((request, tenant))
                try:
                    for event in models.events:
                        yield event
                    if models.stalled:
                        await asyncio.Event().wait()
                finally:
                    models.cancelled = True

            async def aclose(self):
                if not self.closed:
                    self.closed = True
                    models.closed += 1

        return Invocation()

    async def host(self, model):
        self.check(model)
        self.host_calls += 1
        return self.host_artifacts


def post(models, path=PATHS[0], **extra):
    with TestClient(create_app(models)) as client:
        return client.post(path, json=payload(path, **extra), headers=ANTHROPIC)


@pytest.mark.parametrize("path", PATHS)
def test_buffered_protocols_share_one_model_pipeline(path):
    models = FakeModels()
    reply = post(models, path)
    assert reply.status_code == 200, reply.text
    assert models.invokes == models.closed == 1
    assert len(models.requests) == 1
    assert reply.headers.get("x-request-id")
    value = reply.json()
    if path == PATHS[0]:
        assert value["choices"][0]["message"]["content"] == "hello"
        assert value["usage"]["total_tokens"] == 9
    elif path == PATHS[1]:
        assert value["output"][0]["content"][0]["text"] == "hello"
        assert value["usage"]["input_tokens_details"]["cached_tokens"] == 3
    else:
        assert value["content"][0]["text"] == "hello"
        assert value["usage"]["input_tokens"] == 4
        assert value["usage"]["cache_read_input_tokens"] == 3
        assert reply.headers["request-id"]


@pytest.mark.parametrize("path", PATHS)
def test_streaming_terminal_usage_and_protocol_framing(path):
    reply = post(FakeModels(), path, stream=True, stream_options={"include_usage": True})
    assert reply.status_code == 200, reply.text
    values = wire_events(reply.text)
    if path == PATHS[0]:
        assert reply.text.endswith("data: [DONE]\n\n")
        assert values[-1]["usage"]["total_tokens"] == 9
    elif path == PATHS[1]:
        assert values[-1]["type"] == "response.completed"
        assert [v["sequence_number"] for v in values] == list(range(len(values)))
    else:
        assert values[0]["type"] == "message_start"
        assert values[-1]["type"] == "message_stop"
        assert values[-2]["usage"]["input_tokens"] == 4


@pytest.mark.parametrize("path", PATHS)
def test_preflight_errors_use_http_status_and_native_envelopes(path):
    error = ApiError(
        "memory pressure; retry",
        status=503,
        code="memory_pressure",
        kind="server_error",
        retryable=True,
    )
    reply = post(FakeModels(refused=error), path, stream=True)
    assert reply.status_code == 503
    assert reply.headers["retry-after"] == "1"
    assert reply.json()["error"]["type"] == (
        "overloaded_error" if path == PATHS[2] else "server_error"
    )


@pytest.mark.parametrize("path", PATHS)
def test_errors_after_commit_have_no_success_terminal(path):
    models = FakeModels(
        [Admitted(7), Output(ContentDelta("partial")), Failed(ApiError.server("boom"))]
    )
    reply = post(models, path, stream=True)
    assert reply.status_code == 200
    assert "boom" in reply.text
    assert "[DONE]" not in reply.text
    assert '"type":"message_stop"' not in reply.text
    assert '"type":"response.completed"' not in reply.text
    assert models.closed == 1


@pytest.mark.parametrize("path", PATHS)
def test_abrupt_end_is_an_error_in_both_modes(path):
    models = FakeModels([Admitted(7), Output(ContentDelta("partial"))])
    reply = post(models, path)
    assert reply.status_code == 500
    reply = post(models, path, stream=True)
    assert "terminal event" in reply.text and "[DONE]" not in reply.text


@pytest.mark.parametrize("path", PATHS)
def test_unknown_model(path):
    reply = post(FakeModels(), path, model="missing")
    assert reply.status_code == 404


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", "1"),
        ("max_tokens", True),
        ("stream", 1),
        ("n", True),
        ("stream_options", []),
        ("tools", "wrong"),
        ("top_p", 0.0),
        ("top_k", -1),
        ("max_tokens", 0),
        ("temperature", 3.0),
        ("store", True),
        ("n", 2),
    ],
)
def test_known_request_fields_are_strict_before_model_binding(field, value):
    models = FakeModels()
    reply = post(models, **{field: value})
    assert reply.status_code == 400, reply.text
    assert models.invokes == 0


def test_unknown_object_fields_are_ignored_recursively():
    adapted = chat.adapt(
        payload(
            unknown={"x": 1},
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello", "unknown": True}],
                    "unknown": True,
                }
            ],
            stream_options={"include_usage": True, "unknown": True},
        )
    )
    assert adapted.include_usage
    assert adapted.request.input.messages[0]["content"][0]["text"] == "hello"


def test_omitted_limits_and_fresh_seeds():
    a = chat.adapt(payload())
    b = chat.adapt(payload())
    assert a.request.controls.max_output_tokens is None
    assert responses.adapt(payload(PATHS[1])).request.controls.max_output_tokens is None
    assert a.request.controls.sampling.seed != b.request.controls.sampling.seed
    assert chat.adapt(payload(seed=9)).request.controls.sampling.seed == 9


def test_equivalent_wire_requests_make_equal_canonical_input_and_controls():
    a = chat.adapt(payload(seed=4, max_tokens=20))
    b = responses.adapt(payload(PATHS[1], seed=4, max_output_tokens=20))
    c = anthropic.adapt(payload(PATHS[2], seed=4, max_tokens=20, temperature=0.8, top_p=0.95))
    assert a.request == b.request == c.request


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/a.png",
        "data:text/plain;base64,AA==",
        "data:image/png,AA==",
        "data:image/png;base64,!!!!",
        "data:image/png;base64,",
    ],
)
def test_image_policy(url):
    with pytest.raises(ApiError):
        decode_image_data_url(url)


def test_image_decoded_byte_ceiling():
    assert decode_image_data_url("data:image/png;base64,AQID", 3).data == b"\x01\x02\x03"
    with pytest.raises(ApiError):
        decode_image_data_url("data:image/png;base64,AQIDBA==", 3)


def tool_history():
    return [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]


@pytest.mark.parametrize(
    "messages",
    [
        tool_history()[:-1],
        tool_history()[:2] + [{"role": "user", "content": "interrupted"}] + tool_history()[2:],
        tool_history() + tool_history()[2:],
    ],
)
def test_incomplete_or_separated_tool_history_rejected(messages):
    with pytest.raises(ApiError):
        chat.adapt(payload(messages=messages))


def test_leading_systems_empty_assistant_and_complete_tool_history():
    a = chat.adapt(
        payload(
            messages=[
                {"role": "system", "content": "a"},
                {"role": "developer", "content": "b"},
                {"role": "assistant", "content": None},
                *tool_history(),
            ]
        )
    )
    assert a.request.input.messages[0] == {"role": "system", "content": "a\nb"}
    assert len(a.request.input.messages) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "ultra"},
        {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": True}},
        {"reasoning_effort": "none", "thinking_budget_tokens": 4},
        {"chat_template_kwargs": {"thinking_mode": "disabled"}, "thinking_budget_tokens": 4},
        {"max_tokens": 2, "max_completion_tokens": 3},
        {"stop": ["END", "END"]},
    ],
)
def test_conflicting_or_lossy_controls_rejected(kwargs):
    with pytest.raises(ApiError):
        chat.adapt(payload(**kwargs))


def test_reasoning_and_tolerant_extensions():
    adapted = chat.adapt(payload(reasoning_effort="off", ignore_eos="true", timings_per_token=1))
    assert adapted.request.input.reasoning == "none"
    assert not adapted.request.controls.ignore_eos
    assert not adapted.timings_per_token


def test_anthropic_attribution_and_thinking_roundtrip():
    adapted = anthropic.adapt(
        payload(
            PATHS[2],
            system="x-anthropic-billing-header: version=1; cch=abc12;real prompt",
            thinking={"type": "enabled", "budget_tokens": 8},
            output_config={"effort": "high"},
        )
    )
    assert adapted.request.input.messages[0]["content"] == "real prompt"
    assert adapted.request.controls.reasoning_budget == 8
    assert adapted.request.input.reasoning == "high"
    assert anthropic.project_attribution(
        "prefix x-anthropic-billing-header: cch=abc12;"
    ).startswith("prefix")


def test_anthropic_requires_protocol_version_and_native_alias():
    models = FakeModels()
    with TestClient(create_app(models)) as client:
        reply = client.post(PATHS[2], json=payload(PATHS[2]))
        assert reply.status_code == 400
        assert reply.json()["type"] == "error"
        reply = client.post("/v1/messages", json=payload(PATHS[2]), headers=ANTHROPIC)
        assert reply.status_code == 200


def test_anthropic_gateway_alias_is_projection_only():
    models = FakeModels()
    with TestClient(create_app(models)) as client:
        reply = client.post(
            PATHS[2],
            json=payload(PATHS[2]),
            headers={**ANTHROPIC, "Magnitude-Gateway-Model": "anthropic-local/test-model"},
        )
    assert reply.status_code == 200
    assert reply.json()["model"] == "anthropic-local/test-model"
    assert models.invokes == 1


def test_host_operations_never_bind_a_generation():
    models = FakeModels()
    with TestClient(create_app(models)) as client:
        assert client.post(
            "/anthropic/v1/messages/count_tokens", json=payload(PATHS[2]), headers=ANTHROPIC
        ).json() == {"input_tokens": 7}
        assert client.post("/api/v1/chat/templates/apply", json=payload()).status_code == 200
        assert client.post(f"/api/v1/models/{MODEL}/properties").status_code == 200
    assert models.invokes == 0 and models.host_calls == 3


def tool_events(*, cutoff=False):
    return [
        Admitted(7),
        Output(ReasoningDelta("because")),
        Output(ContentDelta("writing")),
        Output(ToolCallStart(0, "lookup", call_id="call-1")),
        Output(ToolCallArgumentsDelta(0, "{}")),
        Output(ToolCallEnd(0, "lookup", "{}", "call-1")),
        *(
            [
                Output(ToolCallStart(1, "write", call_id="call-2")),
                Output(ToolCallArgumentsDelta(1, '{"path":"unfinished')),
            ]
            if cutoff
            else []
        ),
        Completed(UsageUpdate(7, 12, 3, 2), "length" if cutoff else "tool_call"),
    ]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("cutoff", [False, True])
def test_tool_streams_and_buffered_output_preserve_cutoff_reason(path, cutoff):
    reply = post(FakeModels(tool_events(cutoff=cutoff)), path)
    assert reply.status_code == 200, reply.text
    value = reply.json()
    if path == PATHS[0]:
        choice = value["choices"][0]
        assert choice["finish_reason"] == ("length" if cutoff else "tool_calls")
        assert len(choice["message"]["tool_calls"]) == 1
    elif path == PATHS[1]:
        assert value["status"] == ("incomplete" if cutoff else "completed")
        assert sum(item["type"] == "function_call" for item in value["output"]) == 1
    else:
        assert value["stop_reason"] == ("max_tokens" if cutoff else "tool_use")
        assert sum(item["type"] == "tool_use" for item in value["content"]) == 1
    stream = post(FakeModels(tool_events(cutoff=cutoff)), path, stream=True)
    assert stream.status_code == 200
    assert "backend_error" not in stream.text
    if path == PATHS[1] and cutoff:
        values = wire_events(stream.text)
        done = [v for v in values if v["type"] == "response.output_item.done"]
        assert done[-1]["item"]["status"] == "incomplete"
        assert values[-1]["type"] == "response.incomplete"


@pytest.mark.parametrize("path", PATHS)
def test_empty_output_uses_native_empty_shape(path):
    reply = post(FakeModels([Admitted(1), Completed()]), path)
    assert reply.status_code == 200
    value = reply.json()
    if path == PATHS[0]:
        assert value["choices"][0]["message"]["content"] is None
    else:
        assert value["output" if path == PATHS[1] else "content"] == []


def test_responses_opaque_tools_projection_and_malformed_functions():
    request = payload(PATHS[1], tools=[{"type": "web_search", "opaque": True}])
    adapted = responses.adapt(request)
    assert not adapted.request.input.tools
    assert adapted.projection["tools"] == request["tools"]
    with pytest.raises(ApiError):
        responses.adapt(payload(PATHS[1], tools=[{"type": "function", "name": "bad"}]))


def test_responses_output_replays_with_null_codex_reasoning_fields():
    items = [
        {"type": "reasoning", "content": None, "summary": None},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        },
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": ""},
    ]
    adapted = responses.adapt(payload(PATHS[1], input=items))
    assert adapted.request.input.messages[0]["content"] == "done"


def test_progress_is_opt_in_and_can_report_binding_failure_inside_stream():
    models = FakeModels(refused=ApiError("not found", status=404), loading=True)
    with TestClient(create_app(models)) as client:
        reply = client.post(
            PATHS[0], json=payload(stream=True), headers={"Ayaka-Include-Progress": "true"}
        )
    assert reply.status_code == 200
    assert "model_loading" in reply.text
    assert "[DONE]" not in reply.text


def test_body_limit_and_invalid_json_are_protocol_errors():
    with TestClient(create_app(FakeModels(), max_request_bytes=128)) as client:
        assert client.post(PATHS[0], content=b"x" * 129).status_code == 413
        assert client.post(PATHS[0], content=b"[]").status_code == 400
        assert client.post(PATHS[0], content=b'{"temperature":NaN}').status_code == 400


def receive_terminal(ws):
    events = []
    while True:
        value = ws.receive_json()
        events.append(value)
        if value["type"] in (
            "response.completed",
            "response.incomplete",
            "response.failed",
            "error",
        ):
            return events


def test_websocket_latest_only_history_warmup_and_continuation():
    models = FakeModels()
    with TestClient(create_app(models)) as client, client.websocket_connect("/v1/responses") as ws:
        ws.send_json(
            {"type": "response.create", "model": MODEL, "input": "warmup", "generate": False}
        )
        first = receive_terminal(ws)[-1]["response"]["id"]
        assert models.invokes == 0
        ws.send_json(
            {
                "type": "response.create",
                "model": MODEL,
                "previous_response_id": first,
                "input": "next",
            }
        )
        second = receive_terminal(ws)[-1]["response"]["id"]
        assert second != first
        assert len(models.requests[-1][0].input.messages) == 2
        ws.send_json(
            {
                "type": "response.create",
                "model": MODEL,
                "previous_response_id": first,
                "input": "stale",
            }
        )
        assert receive_terminal(ws)[-1]["error"]["code"] == "previous_response_not_found"
        ws.send_bytes(
            json.dumps(
                {
                    "type": "response.create",
                    "model": MODEL,
                    "previous_response_id": second,
                    "input": "binary",
                }
            ).encode()
        )
        assert receive_terminal(ws)[-1]["type"] == "response.completed"
    assert models.closed == models.invokes


def test_failed_websocket_continuation_evicts_predecessor():
    history = responses.WebsocketHistory()
    history.record("resp_1", [{"role": "user", "content": "one"}], [])
    history.evict("resp_1")
    with pytest.raises(ApiError, match="not found"):
        history.resolve(
            {"type": "response.create", "previous_response_id": "resp_1", "input": "two"}
        )


def test_websocket_disconnect_cancels_a_stalled_generation():
    models = FakeModels([Admitted(1)], stalled=True)
    with TestClient(create_app(models)) as client:
        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json({"type": "response.create", "model": MODEL, "input": "hello"})
            assert ws.receive_json()["type"] == "response.created"
            assert ws.receive_json()["type"] == "response.in_progress"
    assert models.cancelled and models.closed == 1


def test_iterator_close_releases_binding():
    from ayaka.serving.protocols.router import _Session, _source_events

    async def run():
        models = FakeModels([Admitted(1)], stalled=True)
        async with aclosing(
            _source_events(models, chat.adapt(payload()), "tenant", False, _Session())
        ) as stream:
            assert isinstance(await anext(stream), Admitted)
        assert models.cancelled and models.closed == 1

    asyncio.run(run())


def test_malformed_output_journal_is_not_silently_accepted():
    journal = Journal(MODEL)
    with pytest.raises(ApiError, match="open tool call"):
        journal.observe(Output(ToolCallArgumentsDelta(0, "{}")))
    journal.observe(Output(ToolCallStart(0, "f", call_id="c")))
    with pytest.raises(ApiError, match="inside a tool call"):
        journal.finish(Completed())


@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_official_sdks_can_parse_buffered_and_streamed_results(protocol, stream):
    openai = pytest.importorskip("openai")
    anthropic_sdk = pytest.importorskip("anthropic")
    with TestClient(create_app(FakeModels())) as http:
        if protocol == "anthropic":
            sdk = anthropic_sdk.Anthropic(
                api_key="test",
                base_url="http://testserver/anthropic",
                http_client=http,
                max_retries=0,
            )
            value = sdk.messages.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=32,
                stream=stream,
            )
            if stream:
                assert list(value)[-1].type == "message_stop"
            else:
                assert value.content[0].text == "hello"
        else:
            sdk = openai.OpenAI(
                api_key="test", base_url="http://testserver/v1", http_client=http, max_retries=0
            )
            if protocol == "chat":
                value = sdk.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": "hello"}], stream=stream
                )
                if stream:
                    assert list(value)[-1].choices[0].finish_reason == "stop"
                else:
                    assert value.choices[0].message.content == "hello"
            else:
                value = sdk.responses.create(model=MODEL, input="hello", stream=stream)
                if stream:
                    assert list(value)[-1].type == "response.completed"
                else:
                    assert value.output_text == "hello"


def test_official_anthropic_sdk_count_tokens():
    anthropic_sdk = pytest.importorskip("anthropic")
    models = FakeModels()
    with TestClient(create_app(models)) as http:
        sdk = anthropic_sdk.Anthropic(
            api_key="test", base_url="http://testserver/anthropic", http_client=http, max_retries=0
        )
        counted = sdk.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": "hello"}]
        )
    assert counted.input_tokens == 7 and models.invokes == 0
