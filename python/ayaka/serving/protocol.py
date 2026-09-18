"""Wire adapters: transport fields and serialization, without engine access."""

from __future__ import annotations

import json
import time
from uuid import uuid4

from pydantic import ValidationError

from ayaka.serving.errors import (
    InvalidRequestError,
    RequestCancelledError,
    ServingError,
    UnsupportedFeatureError,
)
from ayaka.serving.events import (
    ContentDelta,
    GenerationFailed,
    GenerationFinished,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
)
from ayaka.serving.prepare import GenerationSpec, text_content


def normalize(body: dict, protocol: str) -> GenerationSpec:
    data = dict(body)
    if protocol == "anthropic":
        allowed = {
            "model",
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
        }
        _fields(data, allowed)
        if "max_tokens" not in data:
            raise InvalidRequestError("max_tokens is required")
        data.pop("metadata", None)
        system = data.pop("system", None)
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": text_content(system)})
        for message in data.pop("messages", []):
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                raise InvalidRequestError("invalid Anthropic message")
            content = message.get("content")
            if isinstance(content, str):
                messages.append({"role": message["role"], "content": content})
                continue
            if not isinstance(content, list):
                raise InvalidRequestError("message content must be text or blocks")
            text, calls, results = [], [], []
            for block in content:
                if not isinstance(block, dict):
                    raise InvalidRequestError("content blocks must be objects")
                kind = block.get("type")
                if kind == "text":
                    text.append(text_content([block]))
                elif kind == "tool_use" and message["role"] == "assistant":
                    calls.append(
                        {
                            "type": "function",
                            "id": block.get("id"),
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                elif kind == "tool_result" and message["role"] == "user":
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": text_content(block.get("content", "")),
                        }
                    )
                else:
                    raise UnsupportedFeatureError("unsupported Anthropic content block")
            messages.extend(results)
            if text or calls:
                item = {"role": message["role"], "content": "".join(text)}
                if calls:
                    item["tool_calls"] = calls
                messages.append(item)
        data["messages"] = messages
        if "stop_sequences" in data:
            data["stop"] = data.pop("stop_sequences")
        if "tools" in data:
            data["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object"}),
                    },
                }
                for t in data["tools"]
            ]
        choice = data.pop("tool_choice", {"type": "auto"})
        if not isinstance(choice, dict) or set(choice) - {
            "type",
            "name",
            "disable_parallel_tool_use",
        }:
            raise InvalidRequestError("invalid tool_choice")
        kind = choice.get("type")
        if not isinstance(kind, str):
            raise InvalidRequestError("tool_choice requires a type")
        if choice.get("disable_parallel_tool_use"):
            raise UnsupportedFeatureError("disable_parallel_tool_use is not supported")
        data["tool_choice"] = (
            {"type": "function", "function": {"name": choice.get("name")}}
            if kind == "tool"
            else {"any": "required", "auto": "auto", "none": "none"}.get(kind, "")
        )
    elif protocol == "responses":
        allowed = {
            "model",
            "input",
            "instructions",
            "max_output_tokens",
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
            "text",
            "store",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "background",
        }
        _fields(data, allowed)
        for field in ("previous_response_id", "background", "store"):
            if data.pop(field, None):
                raise UnsupportedFeatureError(f"{field} is not supported; Responses is stateless")
        data.pop("metadata", None)
        if data.pop("parallel_tool_calls", True) is False:
            # Required-tool grammar can be bounded, but auto parser cannot enforce this.
            raise UnsupportedFeatureError("parallel_tool_calls=false is not supported")
        source = data.pop("input", None)
        messages = []
        instructions = data.pop("instructions", None)
        if instructions is not None:
            messages.append({"role": "system", "content": text_content(instructions)})
        if isinstance(source, str):
            messages.append({"role": "user", "content": source})
        elif isinstance(source, list):
            for item in source:
                if not isinstance(item, dict):
                    raise InvalidRequestError("Responses input items must be objects")
                kind = item.get("type", "message")
                if kind == "message":
                    messages.append({"role": item.get("role"), "content": item.get("content")})
                elif kind == "function_call":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": item.get("call_id"),
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name"),
                                        "arguments": item.get("arguments", "{}"),
                                    },
                                }
                            ],
                        }
                    )
                elif kind == "function_call_output":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item.get("call_id"),
                            "content": text_content(item.get("output")),
                        }
                    )
                else:
                    raise UnsupportedFeatureError("unsupported Responses input item")
        else:
            raise InvalidRequestError("input must be text or a list of messages")
        data["messages"] = messages
        if "max_output_tokens" in data:
            data["max_tokens"] = data.pop("max_output_tokens")
        if "tools" in data:
            tools = data["tools"]
            if any(t.get("type") != "function" for t in tools):
                raise UnsupportedFeatureError("only function tools are supported")
            data["tools"] = [
                {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
                for t in tools
            ]
        if isinstance(data.get("tool_choice"), dict):
            data["tool_choice"] = {
                "type": "function",
                "function": {"name": data["tool_choice"].get("name")},
            }
        fmt = data.pop("text", {}).get("format")
        if fmt:
            data["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {k: v for k, v in fmt.items() if k != "type"},
                }
                if fmt.get("type") == "json_schema"
                else fmt
            )
    else:
        allowed = set(GenerationSpec.model_fields) - {"include_usage"}
        allowed |= {
            "n",
            "max_completion_tokens",
            "stream_options",
            "user",
            "logprobs",
            "top_logprobs",
            "echo",
            "suffix",
            "best_of",
            "parallel_tool_calls",
        }
        _fields(data, allowed)
        if data.pop("n", 1) != 1 or data.pop("best_of", 1) != 1:
            raise UnsupportedFeatureError("only one completion per request is supported")
        for field in ("logprobs", "top_logprobs", "echo", "suffix"):
            value = data.pop(field, None)
            if value is not None and value is not False:
                raise UnsupportedFeatureError(f"{field} is not supported by this serving runner")
        if data.pop("parallel_tool_calls", True) is False:
            raise UnsupportedFeatureError("parallel_tool_calls=false is not supported")
        data.pop("user", None)
        if "max_completion_tokens" in data:
            if "max_tokens" in data:
                raise InvalidRequestError("provide only one output token limit")
            data["max_tokens"] = data.pop("max_completion_tokens")
        opts = data.pop("stream_options", None)
        if opts is not None:
            if not isinstance(opts, dict) or set(opts) - {"include_usage"}:
                raise InvalidRequestError("unsupported stream_options")
            data["include_usage"] = opts.get("include_usage", False)
        if protocol == "completion" and "messages" in data:
            raise InvalidRequestError("completions requires prompt")
        if protocol == "chat" and "prompt" in data:
            raise InvalidRequestError("chat completions requires messages")
    try:
        return GenerationSpec.model_validate(data)
    except ValidationError as exc:
        error = exc.errors(include_input=False)[0]
        raise InvalidRequestError(error["msg"], param=".".join(map(str, error["loc"]))) from exc


def _fields(data, allowed):
    unknown = set(data) - allowed
    if unknown:
        raise UnsupportedFeatureError(f"unsupported field: {sorted(unknown)[0]}")


def sse(value, event=None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return (prefix + "data: " + data + "\n\n").encode()


def usage_dict(usage):
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
    }


class Result:
    def __init__(self, model: str):
        self.id, self.model, self.created = "gen_" + uuid4().hex, model, int(time.time())
        self.text, self.reasoning, self.calls = "", "", {}
        self.finished = None

    def add(self, event):
        if isinstance(event, ContentDelta):
            self.text += event.text
        elif isinstance(event, ReasoningDelta):
            self.reasoning += event.text
        elif isinstance(event, ToolCallStart):
            self.calls[event.index] = {
                "id": event.call_id,
                "type": "function",
                "function": {"name": event.name, "arguments": ""},
            }
        elif isinstance(event, ToolCallArgumentsDelta):
            self.calls[event.index]["function"]["arguments"] += event.fragment
        elif isinstance(event, GenerationFinished):
            if event.finish_reason == "cancelled":
                raise RequestCancelledError("generation cancelled")
            self.finished = event
        elif isinstance(event, GenerationFailed):
            raise ServingError(event.message)

    def response(self, protocol: str):
        assert self.finished is not None
        usage, reason = self.finished.usage, self.finished.finish_reason
        if protocol == "anthropic":
            content = []
            if self.reasoning:
                content.append({"type": "thinking", "thinking": self.reasoning})
            if self.text:
                content.append({"type": "text", "text": self.text})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "input": json.loads(call["function"]["arguments"]),
                }
                for call in self.calls.values()
            )
            return {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": content,
                "stop_reason": anthropic_stop(reason),
                "stop_sequence": None,
                "usage": anthropic_usage(usage),
            }
        if protocol == "responses":
            output = []
            if self.reasoning:
                output.append(
                    {
                        "id": "rs_" + self.id,
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": self.reasoning}],
                    }
                )
            if self.text or not self.calls:
                output.append(
                    {
                        "id": "msg_" + self.id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": self.text, "annotations": []}],
                    }
                )
            output.extend(
                {
                    "id": "fc_" + call["id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call["id"],
                    **call["function"],
                }
                for call in self.calls.values()
            )
            return {
                "id": self.id,
                "object": "response",
                "created_at": self.created,
                "model": self.model,
                "status": "incomplete" if reason == "length" else "completed",
                "error": None,
                "incomplete_details": (
                    {"reason": "max_output_tokens"} if reason == "length" else None
                ),
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "store": False,
                "usage": {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                    "input_tokens_details": {"cached_tokens": usage.cached_tokens},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        message = {"role": "assistant", "content": self.text or None}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.calls:
            message["tool_calls"] = list(self.calls.values())
        choice = (
            {"text": self.text, "logprobs": None}
            if protocol == "completion"
            else {"message": message, "logprobs": None}
        )
        return {
            "id": self.id,
            "object": "text_completion" if protocol == "completion" else "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    **choice,
                    "finish_reason": "tool_calls" if reason == "tool_call" else reason,
                }
            ],
            "usage": usage_dict(usage),
        }


def anthropic_stop(reason):
    return {"tool_call": "tool_use", "length": "max_tokens"}.get(reason, "end_turn")


def anthropic_usage(usage):
    return {
        "input_tokens": usage.prompt_tokens - usage.cached_tokens,
        "output_tokens": usage.completion_tokens,
        "cache_read_input_tokens": usage.cached_tokens,
        "cache_creation_input_tokens": 0,
    }


def _wire_event(event):
    if isinstance(event, GenerationFinished) and event.finish_reason == "cancelled":
        return GenerationFailed("request_cancelled", "generation cancelled")
    return event


async def openai_stream(events, result: Result, spec, protocol):
    def chunk(delta=None, finish=None, usage=None):
        choice = (
            {"index": 0, "text": delta or "", "logprobs": None, "finish_reason": finish}
            if protocol == "completion"
            else {"index": 0, "delta": delta or {}, "logprobs": None, "finish_reason": finish}
        )
        return sse(
            {
                "id": result.id,
                "object": "text_completion"
                if protocol == "completion"
                else "chat.completion.chunk",
                "created": result.created,
                "model": result.model,
                "choices": [] if usage is not None else [choice],
                **({"usage": usage} if usage is not None else {}),
            }
        )

    if protocol == "chat":
        yield chunk({"role": "assistant", "content": ""})
    async for event in events:
        event = _wire_event(event)
        if isinstance(event, GenerationFailed):
            yield sse(
                {"error": {"message": event.message, "type": "server_error", "code": event.code}}
            )
            break
        result.add(event)
        if isinstance(event, ContentDelta):
            yield chunk(event.text if protocol == "completion" else {"content": event.text})
        elif isinstance(event, ReasoningDelta) and protocol == "chat":
            yield chunk({"reasoning_content": event.text})
        elif isinstance(event, ToolCallStart):
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": event.index,
                            "id": event.call_id,
                            "type": "function",
                            "function": {"name": event.name, "arguments": ""},
                        }
                    ]
                }
            )
        elif isinstance(event, ToolCallArgumentsDelta):
            yield chunk(
                {"tool_calls": [{"index": event.index, "function": {"arguments": event.fragment}}]}
            )
        elif isinstance(event, GenerationFinished):
            yield chunk(
                finish="tool_calls" if event.finish_reason == "tool_call" else event.finish_reason
            )
            if spec.include_usage:
                yield chunk(usage=usage_dict(event.usage))
    yield sse("[DONE]")


async def anthropic_stream(events, result: Result, prompt_tokens: int):
    yield sse(
        {
            "type": "message_start",
            "message": {
                "id": result.id,
                "type": "message",
                "role": "assistant",
                "model": result.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
            },
        },
        "message_start",
    )
    index, mode, open_block = -1, None, False
    async for event in events:
        event = _wire_event(event)
        if isinstance(event, GenerationFailed):
            yield sse(
                {"type": "error", "error": {"type": "api_error", "message": event.message}}, "error"
            )
            return
        result.add(event)
        if isinstance(event, (ContentDelta, ReasoningDelta, ToolCallStart)):
            target = (
                "text"
                if isinstance(event, ContentDelta)
                else ("thinking" if isinstance(event, ReasoningDelta) else "tool_use")
            )
            if target != mode or isinstance(event, ToolCallStart):
                if open_block:
                    yield sse({"type": "content_block_stop", "index": index}, "content_block_stop")
                index += 1
                mode, open_block = target, True
                if isinstance(event, ToolCallStart):
                    block = {
                        "type": "tool_use",
                        "id": event.call_id,
                        "name": event.name,
                        "input": {},
                    }
                else:
                    block = {"type": target, target: ""}
                yield sse(
                    {"type": "content_block_start", "index": index, "content_block": block},
                    "content_block_start",
                )
            if isinstance(event, (ContentDelta, ReasoningDelta)):
                yield sse(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": target + "_delta",
                            target: event.text,
                        },
                    },
                    "content_block_delta",
                )
        elif isinstance(event, ToolCallArgumentsDelta):
            yield sse(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": event.fragment,
                    },
                },
                "content_block_delta",
            )
        elif isinstance(event, ToolCallEnd):
            yield sse({"type": "content_block_stop", "index": index}, "content_block_stop")
            mode, open_block = None, False
        elif isinstance(event, GenerationFinished):
            if open_block:
                yield sse({"type": "content_block_stop", "index": index}, "content_block_stop")
            yield sse(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": anthropic_stop(event.finish_reason),
                        "stop_sequence": None,
                    },
                    "usage": anthropic_usage(event.usage),
                },
                "message_delta",
            )
            yield sse({"type": "message_stop"}, "message_stop")


async def responses_stream(events, result: Result):
    sequence = 0
    items: list[dict] = []
    slots: dict[str, int] = {}

    def emit(kind, **data):
        nonlocal sequence
        value = {"type": kind, "sequence_number": sequence, **data}
        sequence += 1
        return sse(value, kind)

    initial = {
        "id": result.id,
        "object": "response",
        "created_at": result.created,
        "model": result.model,
        "status": "in_progress",
        "output": [],
        "error": None,
        "incomplete_details": None,
        "usage": None,
    }
    yield emit("response.created", response=initial)
    yield emit("response.in_progress", response=initial)
    async for event in events:
        event = _wire_event(event)
        if isinstance(event, GenerationFailed):
            yield emit(
                "response.failed",
                response={
                    **initial,
                    "status": "failed",
                    "error": {"code": event.code, "message": event.message},
                },
            )
            return
        result.add(event)
        if isinstance(event, (ContentDelta, ReasoningDelta, ToolCallStart)):
            key = (
                "text"
                if isinstance(event, ContentDelta)
                else ("reasoning" if isinstance(event, ReasoningDelta) else f"tool_{event.index}")
            )
            if key not in slots:
                index = len(items)
                slots[key] = index
                item: dict
                if key == "text":
                    item = {
                        "id": "msg_" + result.id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                elif key == "reasoning":
                    item = {"id": "rs_" + result.id, "type": "reasoning", "summary": []}
                else:
                    assert isinstance(event, ToolCallStart) and event.call_id is not None
                    item = {
                        "id": "fc_" + event.call_id,
                        "type": "function_call",
                        "call_id": event.call_id,
                        "name": event.name,
                        "arguments": "",
                        "status": "in_progress",
                    }
                items.append(item)
                yield emit("response.output_item.added", output_index=index, item=dict(item))
                if key == "text":
                    part = {"type": "output_text", "text": "", "annotations": []}
                    item["content"].append(part)
                    yield emit(
                        "response.content_part.added",
                        item_id=item["id"],
                        output_index=index,
                        content_index=0,
                        part=dict(part),
                    )
                elif key == "reasoning":
                    part = {"type": "summary_text", "text": ""}
                    item["summary"].append(part)
                    yield emit(
                        "response.reasoning_summary_part.added",
                        item_id=item["id"],
                        output_index=index,
                        summary_index=0,
                        part=dict(part),
                    )
            index = slots[key]
            item = items[index]
            if key == "text":
                assert isinstance(event, ContentDelta)
                item["content"][0]["text"] += event.text
                yield emit(
                    "response.output_text.delta",
                    item_id=item["id"],
                    output_index=index,
                    content_index=0,
                    delta=event.text,
                    logprobs=[],
                )
            elif key == "reasoning":
                assert isinstance(event, ReasoningDelta)
                item["summary"][0]["text"] += event.text
                yield emit(
                    "response.reasoning_summary_text.delta",
                    item_id=item["id"],
                    output_index=index,
                    summary_index=0,
                    delta=event.text,
                )
        elif isinstance(event, ToolCallArgumentsDelta):
            index = slots[f"tool_{event.index}"]
            item = items[index]
            item["arguments"] += event.fragment
            yield emit(
                "response.function_call_arguments.delta",
                item_id=item["id"],
                output_index=index,
                delta=event.fragment,
            )
        elif isinstance(event, GenerationFinished):
            for index, item in enumerate(items):
                common = {"item_id": item["id"], "output_index": index}
                if item["type"] == "message":
                    part = item["content"][0]
                    yield emit(
                        "response.output_text.done",
                        **common,
                        content_index=0,
                        text=part["text"],
                        logprobs=[],
                    )
                    yield emit("response.content_part.done", **common, content_index=0, part=part)
                elif item["type"] == "reasoning":
                    part = item["summary"][0]
                    yield emit(
                        "response.reasoning_summary_text.done",
                        **common,
                        summary_index=0,
                        text=part["text"],
                    )
                    yield emit(
                        "response.reasoning_summary_part.done", **common, summary_index=0, part=part
                    )
                else:
                    yield emit(
                        "response.function_call_arguments.done",
                        **common,
                        arguments=item["arguments"],
                        name=item["name"],
                    )
                if "status" in item:
                    item["status"] = "completed"
                yield emit("response.output_item.done", output_index=index, item=item)
            final = result.response("responses")
            final["output"] = items
            kind = (
                "response.incomplete" if event.finish_reason == "length" else "response.completed"
            )
            yield emit(kind, response=final)
