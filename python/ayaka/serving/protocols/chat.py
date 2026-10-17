"""OpenAI Chat Completions request adapter and response projector."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import Field

from ayaka.serving.events import (
    ContentDelta,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
)

from .common import (
    Journal,
    WireModel,
    array,
    boolean,
    effort,
    fresh_seed,
    json_object,
    obj,
    parse,
    schema,
    sse,
    string,
)
from .error import ApiError
from .media import decode_image_data_url
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


class ChatCompletionRequest(WireModel):
    model: str = Field(min_length=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0, le=2**32 - 1)
    max_completion_tokens: int | None = Field(default=None, gt=0, le=2**32 - 1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0, le=1 << 24)
    min_p: float | None = Field(default=None, ge=0, le=1)
    repetition_penalty: float | None = Field(default=None, gt=0)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    n: int = 1
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    store: bool | None = None
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = Field(default=None, gt=0, le=2**32 - 1)
    response_format: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    cache_prompt: bool = True
    # Match Rust's deliberately tolerant booleans for these two extension fields.
    ignore_eos: Any = False
    timings_per_token: Any = False
    # Existing Ayaka extensions retained across the protocol migration.
    timeout: float | None = Field(default=None, gt=0)
    min_tokens: int = Field(default=0, ge=0)
    stop_token_ids: list[int] = Field(default_factory=list)


def parts(content: Any, *, images: bool = True) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    result = []
    for part in array(content, "content"):
        part = obj(part, "content part")
        if part.get("type") == "text":
            text = string(part.get("text"), "text")
            if text:
                result.append({"type": "text", "text": text})
        elif part.get("type") == "image_url" and images:
            image_url = obj(part.get("image_url"), "image_url")
            image = decode_image_data_url(string(image_url.get("url"), "image_url.url"))
            result.append({"type": "image", "image": image})
        else:
            raise ApiError("unsupported content part")
    return result


def conversation(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Merge leading instructions, validate and order complete tool exchanges."""
    result: list[dict[str, Any]] = []
    instructions = []
    i = 0
    while i < len(messages) and messages[i].get("role") in ("system", "developer"):
        content = string(messages[i].get("content"), "system content")
        if content:
            instructions.append(content)
        i += 1
    if instructions:
        result.append({"role": "system", "content": "\n".join(instructions)})
    entries = 0
    while i < len(messages):
        message = obj(messages[i], "message")
        role = message.get("role")
        i += 1
        if role == "user":
            result.append({"role": "user", "content": parts(message.get("content"))})
            entries += 1
        elif role == "assistant":
            content = message.get("content")
            reasoning = message.get("reasoning_content")
            if content is not None:
                content = string(content, "assistant content") or None
            if reasoning is not None:
                reasoning = string(reasoning, "reasoning_content") or None
            raw_calls = array(message.get("tool_calls", []), "tool_calls")
            calls, ids = [], set()
            for raw in raw_calls:
                raw = obj(raw, "tool call")
                if raw.get("type") != "function":
                    raise ApiError("tool call type must be function")
                identity = string(raw.get("id"), "tool call id", nonempty=True)
                if identity in ids:
                    raise ApiError("duplicate assistant tool-call id")
                ids.add(identity)
                fn = obj(raw.get("function"), "function")
                calls.append(
                    {
                        "id": identity,
                        "type": "function",
                        "function": {
                            "name": string(fn.get("name"), "function name", nonempty=True),
                            "arguments": json_object(fn.get("arguments"), "function arguments"),
                        },
                    }
                )
            if content is None and reasoning is None and not calls:
                continue
            item: dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning:
                item["reasoning_content"] = reasoning
            if calls:
                item["tool_calls"] = calls
            result.append(item)
            entries += 1
            if calls:
                results = {}
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool = messages[i]
                    identity = string(tool.get("tool_call_id"), "tool_call_id", nonempty=True)
                    if identity in results:
                        raise ApiError("duplicate tool result")
                    results[identity] = parts(tool.get("content"))
                    i += 1
                if set(results) != ids:
                    raise ApiError(
                        "tool results must immediately complete every preceding tool call"
                    )
                for call in calls:
                    result.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": results[call["id"]]}
                    )
        elif role in ("system", "developer"):
            raise ApiError("system and developer messages must precede conversation entries")
        elif role == "tool":
            raise ApiError("tool results must immediately follow assistant tool calls")
        else:
            raise ApiError("unsupported message role")
    if not entries:
        raise ApiError("conversation must contain at least one nonempty conversation entry")
    return tuple(result)


def tools_and_choice(
    tools: list[dict[str, Any]],
    choice: str | dict[str, Any] | None,
) -> tuple[tuple[dict[str, Any], ...], str | dict[str, Any]]:
    names, definitions = set(), []
    for tool in tools:
        if tool.get("type") != "function":
            raise ApiError("only function tools are executable")
        fn = obj(tool.get("function"), "function")
        name = string(fn.get("name"), "tool function name", nonempty=True)
        if name in names:
            raise ApiError("duplicate tool function name")
        names.add(name)
        parameters = schema(fn.get("parameters"), "function parameters")
        definition: dict[str, Any] = {"name": name, "parameters": parameters}
        if fn.get("description") is not None:
            definition["description"] = string(fn["description"], "function description")
        if "strict" in fn:
            boolean(fn["strict"], "function strict")
        definitions.append({"type": "function", "function": definition})
    choice = "auto" if choice is None else choice
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise ApiError("unsupported tool_choice")
        if choice == "required" and not names:
            raise ApiError("tool_choice required requires tools")
    else:
        kind = choice.get("type")
        if kind == "function":
            fn = obj(choice.get("function"), "tool_choice function")
            name = string(fn.get("name"), "tool_choice name", nonempty=True)
            if name not in names:
                raise ApiError("tool_choice references an undefined tool")
            choice = {"type": "function", "function": {"name": name}}
        elif kind == "allowed_tools":
            allowed = obj(choice.get("allowed_tools"), "allowed_tools")
            if allowed.get("mode") not in ("auto", "required"):
                raise ApiError("invalid allowed_tools mode")
            selected = []
            for tool in array(allowed.get("tools"), "allowed_tools.tools"):
                tool = obj(tool, "allowed tool")
                name = obj(tool.get("function"), "allowed function").get("name")
                if tool.get("type") != "function" or not isinstance(name, str) or name not in names:
                    raise ApiError("allowed_tools references an undefined tool")
                if name in selected:
                    raise ApiError("duplicate allowed tool")
                selected.append(name)
            if not selected:
                raise ApiError("allowed_tools requires at least one tool")
            choice = {"type": "allowed_tools", "names": selected, "mode": allowed["mode"]}
        else:
            raise ApiError("invalid tool_choice")
    return tuple(definitions), choice


def output_format(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {"type": "text"}
    kind = value.get("type")
    if kind in ("text", "json_object"):
        return {"type": kind}
    if kind == "grammar":
        return {"type": kind, "grammar": string(value.get("grammar"), "grammar", nonempty=True)}
    if kind == "json_schema":
        fmt = obj(value.get("json_schema"), "json_schema")
        return {
            "type": kind,
            "json_schema": {
                "name": string(fmt.get("name"), "json_schema name", nonempty=True),
                "schema": schema(fmt.get("schema"), "json_schema schema"),
                "strict": boolean(fmt.get("strict", False), "json_schema strict"),
            },
        }
    raise ApiError("unsupported response_format")


def adapt(body: dict[str, Any]) -> AdaptedRequest:
    request: ChatCompletionRequest = parse(ChatCompletionRequest, body)
    if request.store:
        raise ApiError("store is not supported by this local runtime")
    if request.n != 1:
        raise ApiError("only one choice (n = 1) is supported")
    if request.max_tokens is not None and request.max_completion_tokens is not None:
        raise ApiError("max_tokens and max_completion_tokens cannot both be set")
    messages = conversation(request.messages)
    tools, choice = tools_and_choice(request.tools or [], request.tool_choice)
    args = request.chat_template_kwargs or {}
    if any(not key for key in args):
        raise ApiError("chat_template_kwargs keys must not be empty")
    reasoning = effort(request.reasoning_effort)
    controls = {
        "enable_thinking",
        "thinking",
        "thinking_mode",
        "reasoning_effort",
        "thinking_budget",
    }
    if reasoning is not None and controls.intersection(args):
        raise ApiError("reasoning_effort conflicts with reasoning controls in chat_template_kwargs")
    disabled = (
        reasoning == "none"
        or args.get("enable_thinking") is False
        or args.get("thinking") is False
        or args.get("thinking_mode") in ("chat", "disabled")
        or args.get("reasoning_effort") in ("none", "off", "disabled")
    )
    if request.thinking_budget_tokens is not None and disabled:
        raise ApiError("thinking_budget_tokens cannot be used when reasoning is disabled")
    stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
    if any(not stop for stop in stops) or len(set(stops)) != len(stops):
        raise ApiError("stop sequences must be nonempty and unique")
    options = request.stream_options or {}
    if any(type(token) is not int or token < 0 for token in request.stop_token_ids):
        raise ApiError("stop_token_ids must be nonnegative integers")
    limit = request.max_completion_tokens or request.max_tokens
    if limit is not None and request.min_tokens > limit:
        raise ApiError("min_tokens exceeds max_tokens")
    include_usage = options.get("include_usage")
    if include_usage is not None:
        boolean(include_usage, "stream_options.include_usage")
    return AdaptedRequest(
        model=request.model,
        request=GenerationRequest(
            ChatInput(
                messages,
                tools,
                choice,
                True if request.parallel_tool_calls is None else request.parallel_tool_calls,
                reasoning,
                output_format(request.response_format),
                args,
            ),
            GenerationControls(
                request.max_completion_tokens or request.max_tokens,
                SamplingControls(
                    0.8 if request.temperature is None else request.temperature,
                    0.95 if request.top_p is None else request.top_p,
                    request.top_k or 0,
                    request.min_p or 0.0,
                    1.0 if request.repetition_penalty is None else request.repetition_penalty,
                    request.presence_penalty or 0.0,
                    request.frequency_penalty or 0.0,
                    fresh_seed(request.seed),
                ),
                tuple(stops),
                request.ignore_eos is True,
                request.thinking_budget_tokens,
                request.cache_prompt,
                request.timeout,
                request.min_tokens,
                tuple(request.stop_token_ids),
            ),
        ),
        stream=request.stream,
        include_usage=bool(include_usage),
        timings_per_token=request.timings_per_token is True,
    )


def response(journal: Journal) -> dict[str, Any]:
    completion = journal.completion
    if completion is None:
        raise ApiError.server("generation ended without a terminal event")
    message: dict[str, Any] = {"role": "assistant", "content": journal.text or None}
    if journal.reasoning:
        message["reasoning_content"] = journal.reasoning
    calls = journal.complete_calls()
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for call in calls
        ]
    result = {
        "id": journal.id,
        "object": "chat.completion",
        "created": journal.created,
        "model": journal.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls"
                if completion.finish_reason == "tool_call"
                else completion.finish_reason,
            }
        ],
        "usage": journal.usage(),
    }
    if completion.timings is not None:
        result["timings"] = completion.timings
    return result


async def project(
    events: GenerationStream,
    journal: Journal,
    adapted: AdaptedRequest,
    *,
    progress: bool = False,
) -> AsyncGenerator[str]:
    def chunk(delta=None, finish=None, **extra):
        return sse(
            {
                "id": journal.id,
                "object": "chat.completion.chunk",
                "created": journal.created,
                "model": journal.model,
                "choices": []
                if delta is None and finish is None
                else [
                    {
                        "index": 0,
                        "delta": delta or {},
                        "finish_reason": finish,
                    }
                ],
                **extra,
            }
        )

    started = False
    try:
        async for event in events:
            if isinstance(event, Progress):
                if progress:
                    yield chunk(progress=event.value)
            elif isinstance(event, Admitted):
                journal.prompt_tokens = event.prompt_tokens
            elif isinstance(event, Failed):
                raise event.error
            elif isinstance(event, Output):
                journal.observe(event)
                if not started:
                    yield chunk(
                        {"role": "assistant", "content": ""},
                        **({"timings": event.timings} if event.timings is not None else {}),
                    )
                    started = True
                output = event.event
                delta = None
                if isinstance(output, ContentDelta):
                    delta = {"content": output.text}
                elif isinstance(output, ReasoningDelta):
                    delta = {"reasoning_content": output.text}
                elif isinstance(output, ToolCallStart):
                    delta = {
                        "tool_calls": [
                            {
                                "index": output.index,
                                "id": output.call_id,
                                "type": "function",
                                "function": {"name": output.name, "arguments": ""},
                            }
                        ]
                    }
                elif isinstance(output, ToolCallArgumentsDelta):
                    delta = {
                        "tool_calls": [
                            {"index": output.index, "function": {"arguments": output.fragment}}
                        ]
                    }
                elif isinstance(output, ToolCallEnd):
                    continue
                if delta is not None:
                    yield chunk(
                        delta,
                        **(
                            {"timings": event.timings}
                            if adapted.timings_per_token and event.timings is not None
                            else {}
                        ),
                    )
            elif isinstance(event, Completed):
                journal.finish(event)
                if not started:
                    yield chunk({"role": "assistant", "content": ""})
                reason = "tool_calls" if event.finish_reason == "tool_call" else event.finish_reason
                timings = {"timings": event.timings} if event.timings is not None else {}
                yield chunk({}, reason, **({} if adapted.include_usage else timings))
                if adapted.include_usage:
                    yield chunk(usage=journal.usage(), **timings)
                yield sse("[DONE]")
                return
        raise ApiError.server("generation ended without a terminal event")
    except ApiError as error:
        yield sse({"error": error.body()})
