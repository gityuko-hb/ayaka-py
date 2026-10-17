"""Anthropic Messages/count_tokens, including replayable thinking and tools."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field

from ayaka.serving.events import (
    ContentDelta,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallStart,
)

from . import chat
from .common import (
    Journal,
    WireModel,
    array,
    boolean,
    dumps,
    effort,
    fresh_seed,
    obj,
    parse,
    sse,
    string,
)
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
    SamplingControls,
)

LOCAL_THINKING_SIGNATURE = "magnitude-local-v1"
SUPPORTED_VERSION = "2023-06-01"


class MessagesRequest(WireModel):
    model: str = Field(min_length=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = Field(gt=0, le=2**32 - 1)
    stop_sequences: list[str] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0, le=2**32 - 1)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    timeout: float | None = Field(default=None, gt=0)


def validate_version(value: str | None) -> None:
    if value != SUPPORTED_VERSION:
        raise ApiError("anthropic-version must be 2023-06-01", param="anthropic-version")


def project_attribution(text: str) -> str:
    """Remove only the recognized leading Claude Code billing metadata."""
    sentinel = b"x-anthropic-billing-header:"
    raw = text.encode()
    if raw.startswith(sentinel):
        start = raw.find(b"cch=", len(sentinel))
        end = start + 4 + 5
        if start >= 0 and raw[end : end + 1] == b";":
            try:
                return raw[end + 1 :].decode()
            except UnicodeDecodeError:
                pass
    return text


def blocks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return [obj(block, "content block") for block in array(value, "content")]


def user_parts(value: Any, *, empty_result: bool = False) -> list[dict[str, Any]]:
    if value is None and empty_result:
        return []
    result = []
    for block in blocks(value):
        kind = block.get("type")
        if kind == "text":
            result.append({"type": "text", "text": string(block.get("text"), "text")})
        elif kind == "image":
            source = obj(block.get("source"), "image source")
            if source.get("type") != "base64":
                raise ApiError("network image URLs are not supported; use a base64 image source")
            media_type = string(source.get("media_type"), "media_type")
            data = string(source.get("data"), "image data")
            result.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }
            )
        else:
            raise ApiError("invalid content block for a user or tool result")
    return result


def context(request: MessagesRequest) -> tuple[dict[str, Any], ...]:
    messages = []
    if request.system is not None:
        texts = []
        for block in blocks(request.system):
            if block.get("type") != "text":
                raise ApiError("system blocks must be text")
            texts.append(string(block.get("text"), "system text"))
        if texts:
            texts[0] = project_attribution(texts[0])
        text = "\n".join(text for text in texts if text)
        if text:
            messages.append({"role": "system", "content": text})
    for message in request.messages:
        role, value = message.get("role"), message.get("content")
        if role in ("user", "system"):
            pieces, results, trailing = [], [], False
            for block in blocks(value):
                if block.get("type") == "tool_result" and role == "user":
                    if trailing:
                        raise ApiError("tool_result blocks must precede other user content")
                    if "is_error" in block:
                        boolean(block["is_error"], "is_error")
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": string(
                                block.get("tool_use_id"), "tool_use_id", nonempty=True
                            ),
                            "content": user_parts(block.get("content"), empty_result=True),
                        }
                    )
                else:
                    trailing = True
                    pieces.extend(user_parts([block]))
            messages.extend(results)
            if pieces or not results:
                # System-role messages are positioned user entries in this dialect.
                messages.append({"role": "user", "content": pieces})
        elif role == "assistant":
            text, thinking, calls = [], [], []
            for block in blocks(value):
                kind = block.get("type")
                if kind == "text":
                    text.append(string(block.get("text"), "assistant text"))
                elif kind == "thinking":
                    thinking.append(string(block.get("thinking"), "assistant thinking"))
                    if block.get("signature") is not None:
                        string(block["signature"], "signature")
                elif kind == "tool_use":
                    calls.append(
                        {
                            "id": string(block.get("id"), "tool_use id", nonempty=True),
                            "type": "function",
                            "function": {
                                "name": string(block.get("name"), "tool_use name", nonempty=True),
                                "arguments": dumps(obj(block.get("input"), "tool_use input")),
                            },
                        }
                    )
                else:
                    raise ApiError("invalid content block for an assistant message")
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(text) or None,
                    "reasoning_content": "".join(thinking) or None,
                    "tool_calls": calls,
                }
            )
        else:
            raise ApiError("invalid Anthropic message role")
    return chat.conversation(messages)


def adapt(body: dict[str, Any], *, count_only: bool = False) -> AdaptedRequest:
    request: MessagesRequest = parse(
        MessagesRequest, {**body, **({"max_tokens": 1} if count_only else {})}
    )
    if request.top_k is not None and not count_only:
        raise ApiError("top_k is not supported by this local Anthropic runtime")
    tools = []
    for tool in request.tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema"),
                },
            }
        )
    choice = request.tool_choice or {"type": "auto"}
    kind = choice.get("type")
    no_tools = kind == "none"
    parallel = not boolean(
        choice.get("disable_parallel_tool_use", False), "disable_parallel_tool_use"
    )
    if kind == "tool":
        tool_choice: str | dict[str, Any] = {
            "type": "function",
            "function": {"name": choice.get("name")},
        }
    elif kind in ("auto", "none", "any"):
        tool_choice = {"any": "required"}.get(kind, kind)
    else:
        raise ApiError("invalid Anthropic tool_choice")
    tools, tool_choice = chat.tools_and_choice(tools, tool_choice)
    thinking = request.thinking
    reasoning = effort((request.output_config or {}).get("effort"))
    if reasoning == "none":
        raise ApiError("output_config.effort must select an enabled reasoning behavior")
    budget = None
    if thinking is not None:
        kind = thinking.get("type")
        if kind == "disabled":
            if reasoning is not None:
                raise ApiError("output_config.effort cannot be used when thinking is disabled")
            reasoning = "none"
        elif kind == "enabled":
            budget = thinking.get("budget_tokens")
            if type(budget) is not int or not 0 < budget < 2**32:
                raise ApiError("thinking budget_tokens must be positive")
            reasoning = reasoning or "enabled"
        elif kind != "adaptive":
            raise ApiError("invalid thinking type")
    if any(not stop for stop in request.stop_sequences):
        raise ApiError("stop sequences must not be empty")
    return AdaptedRequest(
        request.model,
        GenerationRequest(
            ChatInput(
                context(request), tools, tool_choice, parallel if not no_tools else False, reasoning
            ),
            GenerationControls(
                request.max_tokens,
                SamplingControls(
                    temperature=1.0 if request.temperature is None else request.temperature,
                    top_p=1.0 if request.top_p is None else request.top_p,
                    seed=fresh_seed(request.seed),
                ),
                tuple(request.stop_sequences),
                reasoning_budget=budget,
                timeout=request.timeout,
            ),
        ),
        request.stream,
    )


def usage(completion: Completed) -> dict[str, int]:
    value = completion.usage
    return {
        "input_tokens": value.prompt_tokens - value.cached_tokens,
        "output_tokens": value.completion_tokens,
        "cache_read_input_tokens": value.cached_tokens,
        "cache_creation_input_tokens": 0,
    }


def stop(completion: Completed) -> tuple[str, str | None]:
    if completion.finish_reason == "length":
        return "max_tokens", None
    if completion.finish_reason == "tool_call":
        return "tool_use", None
    if completion.stop_sequence is not None:
        return "stop_sequence", completion.stop_sequence
    return "end_turn", None


def response(journal: Journal) -> dict[str, Any]:
    if journal.completion is None:
        raise ApiError.server("generation ended without a terminal event")
    content = []
    if journal.reasoning:
        content.append(
            {
                "type": "thinking",
                "thinking": journal.reasoning,
                "signature": LOCAL_THINKING_SIGNATURE,
            }
        )
    if journal.text:
        content.append({"type": "text", "text": journal.text})
    content.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": json_object_for_output(call.arguments),
        }
        for call in journal.complete_calls()
    )
    reason, sequence = stop(journal.completion)
    return {
        "id": journal.id,
        "type": "message",
        "role": "assistant",
        "model": journal.model,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": sequence,
        "usage": usage(journal.completion),
    }


def json_object_for_output(arguments: str) -> dict[str, Any]:
    from .common import json_object

    return json_object(arguments, "tool arguments")


async def project(events: GenerationStream, journal: Journal) -> AsyncGenerator[str]:
    slots: dict[str | int, int] = {}
    started = False

    def emit(kind: str, **data):
        return sse({"type": kind, **data}, kind)

    try:
        async for event in events:
            if isinstance(event, Failed):
                raise event.error
            if isinstance(event, Admitted):
                journal.prompt_tokens = event.prompt_tokens
            if isinstance(event, (Output, Completed)) and not started:
                started = True
                yield emit(
                    "message_start",
                    message={
                        "id": journal.id,
                        "type": "message",
                        "role": "assistant",
                        "model": journal.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": journal.prompt_tokens,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                )
            if isinstance(event, Output):
                journal.observe(event)
                value = event.event
                if isinstance(value, (ContentDelta, ReasoningDelta, ToolCallStart)):
                    key = (
                        "text"
                        if isinstance(value, ContentDelta)
                        else ("thinking" if isinstance(value, ReasoningDelta) else value.index)
                    )
                    if key not in slots:
                        slots[key] = len(slots)
                        block = (
                            {
                                "type": "tool_use",
                                "id": value.call_id,
                                "name": value.name,
                                "input": {},
                            }
                            if isinstance(value, ToolCallStart)
                            else {"type": key, key: ""}
                        )
                        if key == "thinking":
                            block["signature"] = ""
                        yield emit("content_block_start", index=slots[key], content_block=block)
                    if isinstance(value, (ContentDelta, ReasoningDelta)):
                        yield emit(
                            "content_block_delta",
                            index=slots[key],
                            delta={"type": f"{key}_delta", key: value.text},
                        )
                elif isinstance(value, ToolCallArgumentsDelta):
                    yield emit(
                        "content_block_delta",
                        index=slots[value.index],
                        delta={"type": "input_json_delta", "partial_json": value.fragment},
                    )
            elif isinstance(event, Completed):
                journal.finish(event)
                if "thinking" in slots:
                    yield emit(
                        "content_block_delta",
                        index=slots["thinking"],
                        delta={"type": "signature_delta", "signature": LOCAL_THINKING_SIGNATURE},
                    )
                for index in range(len(slots)):
                    yield emit("content_block_stop", index=index)
                reason, sequence = stop(event)
                yield emit(
                    "message_delta",
                    delta={"stop_reason": reason, "stop_sequence": sequence},
                    usage=usage(event),
                )
                yield emit("message_stop")
                return
        raise ApiError.server("generation ended without a terminal event")
    except ApiError as error:
        yield sse(error.anthropic(journal.id), "error")
