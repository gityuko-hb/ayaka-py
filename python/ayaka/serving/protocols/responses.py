"""OpenAI Responses over HTTP/SSE and connection-local WebSocket history."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Any

from pydantic import Field

from ayaka.serving.events import (
    ContentDelta,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallStart,
)

from . import chat
from .common import Journal, WireModel, array, dumps, effort, fresh_seed, obj, parse, string
from .error import ApiError
from .source import (
    AdaptedRequest,
    Admitted,
    ChatInput,
    Completed,
    Failed,
    GenerationControls,
    GenerationRequest,
    GenerationStream,
    Output,
    Progress,
    SamplingControls,
)


class ResponseCreateRequest(WireModel):
    model: str = Field(min_length=1)
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0, le=2**32 - 1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    stream: bool = False
    store: bool | None = None
    metadata: dict[str, Any] | None = None
    include: list[str] | None = None
    client_metadata: dict[str, Any] | None = None
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    truncation: str | None = None
    timeout: float | None = Field(default=None, gt=0)


def message_content(value: Any, *, images: bool = True) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    result = []
    for part in array(value, "content"):
        part = obj(part, "content part")
        if part.get("type") in ("input_text", "output_text"):
            result.append({"type": "text", "text": string(part.get("text"), "text")})
        elif part.get("type") == "input_image" and images:
            result.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": string(part.get("image_url"), "image_url"),
                    },
                }
            )
        else:
            raise ApiError("unsupported Responses content part")
    if images:
        return result
    return "".join(part["text"] for part in result)


def context(request: ResponseCreateRequest) -> tuple[dict[str, Any], ...]:
    items = (
        [{"role": "user", "content": request.input}]
        if isinstance(request.input, str)
        else request.input
    )
    messages: list[dict[str, Any]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    reasoning: list[str] = []
    i = 0

    def take_reasoning() -> str | None:
        text = "\n".join(reasoning)
        reasoning.clear()
        return text or None

    while i < len(items):
        item = obj(items[i], "input item")
        kind = item.get("type", "message")
        if kind == "reasoning":
            content = item.get("content") or item.get("summary") or []
            text = []
            for part in array(content, "reasoning content"):
                part = obj(part, "reasoning part")
                if part.get("type") not in ("summary_text", "reasoning_text"):
                    raise ApiError("invalid reasoning content part")
                text.append(string(part.get("text"), "reasoning text"))
            if text:
                reasoning.append("".join(text))
        elif kind == "message":
            role = item.get("role")
            if role in ("system", "developer") and reasoning:
                raise ApiError("system messages must precede reasoning entries")
            if role not in ("system", "developer", "user", "assistant"):
                raise ApiError("unsupported Responses role")
            if role == "user" and reasoning:
                messages.append(
                    {"role": "assistant", "content": None, "reasoning_content": take_reasoning()}
                )
            message: dict[str, Any] = {
                "role": role,
                "content": message_content(item.get("content"), images=role == "user"),
            }
            if role == "assistant":
                message["reasoning_content"] = take_reasoning()
            messages.append(message)
        elif kind == "function_call":
            calls = []
            while i < len(items) and items[i].get("type") == "function_call":
                call = items[i]
                calls.append(
                    {
                        "id": string(call.get("call_id"), "call_id", nonempty=True),
                        "type": "function",
                        "function": {
                            "name": string(call.get("name"), "function name", nonempty=True),
                            "arguments": string(call.get("arguments"), "function arguments"),
                        },
                    }
                )
                i += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": calls,
                    "reasoning_content": take_reasoning(),
                }
            )
            while i < len(items) and items[i].get("type") == "function_call_output":
                result = items[i]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.get("call_id"),
                        "content": message_content(result.get("output")),
                    }
                )
                i += 1
            continue
        elif kind == "function_call_output":
            raise ApiError("function_call_output has no preceding function_call")
        else:
            raise ApiError("unsupported Responses input item")
        i += 1
    if reasoning:
        messages.append(
            {"role": "assistant", "content": None, "reasoning_content": take_reasoning()}
        )
    return chat.conversation(messages)


def adapt(body: dict[str, Any]) -> AdaptedRequest:
    request: ResponseCreateRequest = parse(ResponseCreateRequest, body)
    if request.store:
        raise ApiError("store is not supported by this local runtime")
    if request.previous_response_id is not None:
        raise ApiError(
            "previous_response_id is only supported within a WebSocket connection",
            param="previous_response_id",
        )
    if request.truncation not in (None, "disabled"):
        raise ApiError("automatic truncation is not supported")
    definitions = []
    for tool in request.tools or []:
        kind = string(tool.get("type"), "tool type", nonempty=True)
        if kind == "function":
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                        **({"strict": tool["strict"]} if "strict" in tool else {}),
                    },
                }
            )
        # Other declarations are projection-only, never executable by the engine.
    choice = request.tool_choice
    if isinstance(choice, dict):
        if choice.get("type") != "function":
            raise ApiError("unsupported Responses tool_choice")
        choice = {"type": "function", "function": {"name": choice.get("name")}}
    tools, choice = chat.tools_and_choice(definitions, choice)
    fmt = obj((request.text or {}).get("format", {"type": "text"}), "text format")
    output = chat.output_format(
        {"type": "json_schema", "json_schema": fmt} if fmt.get("type") == "json_schema" else fmt
    )
    reasoning = request.reasoning or {}
    normalized_effort = effort(reasoning.get("effort"))
    if reasoning.get("summary") is not None:
        string(reasoning["summary"], "reasoning summary")
    projection = {
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "parallel_tool_calls": True
        if request.parallel_tool_calls is None
        else request.parallel_tool_calls,
        "previous_response_id": None,
        "reasoning": {"effort": reasoning.get("effort"), "summary": reasoning.get("summary")},
        "store": False,
        "temperature": request.temperature,
        "text": {"format": fmt},
        "tool_choice": request.tool_choice or "auto",
        "tools": request.tools or [],
        "top_p": request.top_p,
        "truncation": "disabled",
        "metadata": request.metadata or {},
    }
    return AdaptedRequest(
        request.model,
        GenerationRequest(
            ChatInput(
                context(request),
                tools,
                choice,
                projection["parallel_tool_calls"],
                normalized_effort,
                output,
            ),
            GenerationControls(
                request.max_output_tokens,
                SamplingControls(
                    temperature=0.8 if request.temperature is None else request.temperature,
                    top_p=0.95 if request.top_p is None else request.top_p,
                    seed=fresh_seed(request.seed),
                ),
                timeout=request.timeout,
            ),
        ),
        stream=request.stream,
        projection=projection,
    )


def item(journal: Journal, key: str | int) -> dict[str, Any]:
    if key == "text":
        return {
            "id": "msg_" + journal.id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": journal.text, "annotations": []}],
        }
    if key == "reasoning":
        return {
            "id": "rs_" + journal.id,
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": journal.reasoning}],
        }
    assert isinstance(key, int)
    call = journal.calls[key]
    return {
        "id": "fc_" + call.id,
        "type": "function_call",
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments,
        "status": "completed" if call.finished else "incomplete",
    }


def response(journal: Journal, adapted: AdaptedRequest, *, ordered: bool = False) -> dict[str, Any]:
    completion = journal.completion
    if completion is None:
        raise ApiError.server("generation ended without a terminal event")
    keys = (
        journal.order
        if ordered
        else (
            (["reasoning"] if journal.reasoning else [])
            + (["text"] if journal.text else [])
            + sorted(journal.calls)
        )
    )
    output = [
        item(journal, key)
        for key in keys
        if not isinstance(key, int) or journal.calls[key].finished
    ]
    usage = completion.usage
    return {
        **base(journal, adapted),
        "status": "incomplete" if completion.finish_reason == "length" else "completed",
        "output": output,
        "incomplete_details": {"reason": "max_output_tokens"}
        if completion.finish_reason == "length"
        else None,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            "input_tokens_details": {"cached_tokens": usage.cached_tokens},
            "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        },
    }


def base(journal: Journal, adapted: AdaptedRequest) -> dict[str, Any]:
    return {
        "id": journal.id,
        "object": "response",
        "created_at": journal.created,
        "status": "in_progress",
        "model": journal.model,
        "output": [],
        "error": None,
        "incomplete_details": None,
        "usage": None,
        **adapted.projection,
    }


async def project(
    events: GenerationStream,
    journal: Journal,
    adapted: AdaptedRequest,
    *,
    progress: bool = False,
) -> AsyncGenerator[dict[str, Any]]:
    sequence = 0
    slots: dict[str | int, int] = {}

    def emit(kind, **value):
        nonlocal sequence
        event = {"type": kind, "sequence_number": sequence, **deepcopy(value)}
        sequence += 1
        return event

    yield emit("response.created", response=base(journal, adapted))
    yield emit("response.in_progress", response=base(journal, adapted))
    try:
        async for event in events:
            if isinstance(event, Failed):
                raise event.error
            if isinstance(event, Progress):
                if progress:
                    yield emit(
                        "response.magnitude_progress", response_id=journal.id, progress=event.value
                    )
            elif isinstance(event, Admitted):
                journal.prompt_tokens = event.prompt_tokens
            elif isinstance(event, Output):
                journal.observe(event)
                value = event.event
                if isinstance(value, (ContentDelta, ReasoningDelta, ToolCallStart)):
                    key = (
                        "text"
                        if isinstance(value, ContentDelta)
                        else ("reasoning" if isinstance(value, ReasoningDelta) else value.index)
                    )
                    if key not in slots:
                        index = slots[key] = len(slots)
                        opened = item(journal, key)
                        opened["status"] = "in_progress"
                        if key in ("text", "reasoning"):
                            opened["content" if key == "text" else "summary"] = []
                        else:
                            opened["arguments"] = ""
                        yield emit("response.output_item.added", output_index=index, item=opened)
                        if key == "text":
                            yield emit(
                                "response.content_part.added",
                                item_id=opened["id"],
                                output_index=index,
                                content_index=0,
                                part={"type": "output_text", "text": "", "annotations": []},
                            )
                        elif key == "reasoning":
                            yield emit(
                                "response.reasoning_summary_part.added",
                                item_id=opened["id"],
                                output_index=index,
                                summary_index=0,
                                part={"type": "summary_text", "text": ""},
                            )
                    index = slots[key]
                    if isinstance(value, (ContentDelta, ReasoningDelta)):
                        common = {
                            "item_id": item(journal, key)["id"],
                            "output_index": index,
                            "delta": value.text,
                        }
                        yield emit(
                            "response.output_text.delta"
                            if key == "text"
                            else "response.reasoning_summary_text.delta",
                            **common,
                            **(
                                {"content_index": 0, "logprobs": []}
                                if key == "text"
                                else {"summary_index": 0}
                            ),
                        )
                elif isinstance(value, ToolCallArgumentsDelta):
                    yield emit(
                        "response.function_call_arguments.delta",
                        item_id=item(journal, value.index)["id"],
                        output_index=slots[value.index],
                        delta=value.fragment,
                    )
            elif isinstance(event, Completed):
                journal.finish(event)
                for key, index in slots.items():
                    closed = item(journal, key)
                    common = {"item_id": closed["id"], "output_index": index}
                    if key == "text":
                        part = closed["content"][0]
                        yield emit(
                            "response.output_text.done",
                            **common,
                            content_index=0,
                            text=part["text"],
                            logprobs=[],
                        )
                        yield emit(
                            "response.content_part.done", **common, content_index=0, part=part
                        )
                    elif key == "reasoning":
                        part = closed["summary"][0]
                        yield emit(
                            "response.reasoning_summary_text.done",
                            **common,
                            summary_index=0,
                            text=part["text"],
                        )
                        yield emit(
                            "response.reasoning_summary_part.done",
                            **common,
                            summary_index=0,
                            part=part,
                        )
                    elif closed["status"] == "completed":
                        yield emit(
                            "response.function_call_arguments.done",
                            **common,
                            arguments=closed["arguments"],
                            name=closed["name"],
                        )
                    yield emit("response.output_item.done", output_index=index, item=closed)
                yield emit(
                    "response.incomplete"
                    if event.finish_reason == "length"
                    else "response.completed",
                    response=response(journal, adapted, ordered=True),
                )
                return
        raise ApiError.server("generation ended without a terminal event")
    except ApiError as error:
        yield emit(
            "response.failed",
            response={**base(journal, adapted), "status": "failed", "error": error.body()},
        )


class WebsocketHistory:
    """Retain only the last successful response; no cross-connection storage."""

    def __init__(self, max_bytes: int = 32 << 20) -> None:
        self.latest: tuple[str, list[dict[str, Any]]] | None = None
        self.max_bytes = max_bytes

    def resolve(self, value: Any) -> tuple[dict[str, Any], list[dict[str, Any]], str | None, bool]:
        request = deepcopy(obj(value, "WebSocket message"))
        if request.pop("type", None) != "response.create":
            raise ApiError("WebSocket message type must be response.create")
        generate = request.pop("generate", True)
        if type(generate) is not bool:
            raise ApiError("generate must be a boolean")
        previous = request.pop("previous_response_id", None)
        if previous is not None:
            string(previous, "previous_response_id")
            if self.latest is None or self.latest[0] != previous:
                raise ApiError(
                    f"Previous response with id '{previous}' not found.",
                    code="previous_response_not_found",
                    param="previous_response_id",
                )
        source = request.pop("input", None)
        if isinstance(source, str):
            source = [{"role": "user", "content": source}]
        else:
            source = array(source, "input")
        history = deepcopy(self.latest[1]) if previous is not None and self.latest else []
        history.extend(source)
        request["input"], request["stream"] = history, True
        if len(dumps(request).encode()) > self.max_bytes:
            raise ApiError("accumulated WebSocket input exceeds the request byte limit", status=413)
        return request, history, previous, generate

    def record(
        self, identity: str, input: list[dict[str, Any]], output: list[dict[str, Any]]
    ) -> None:
        history = deepcopy(input + output)
        self.latest = (
            (identity, history) if len(dumps(history).encode()) <= self.max_bytes else None
        )

    def evict(self, previous: str | None) -> None:
        if previous is not None and self.latest and previous == self.latest[0]:
            self.latest = None
