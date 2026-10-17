"""Shared strict validation and semantic output journal."""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, ValidationError

from ayaka.serving.events import (
    ContentDelta,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
)

from .error import ApiError
from .source import Completed, Output


class WireModel(BaseModel):
    """Compatibility fields are ignored; known fields keep strict JSON types."""

    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)


def parse(model: type[WireModel], body: dict[str, Any]) -> Any:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        param = ".".join(str(x) for x in first["loc"])
        raise ApiError(f"{param}: {first['msg']}", param=param or None) from exc


def obj(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ApiError(f"{field} must be an object", param=field)
    return value


def array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ApiError(f"{field} must be an array", param=field)
    return value


def string(value: Any, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ApiError(f"{field} must be {'a nonempty' if nonempty else 'a'} string", param=field)
    return value


def boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ApiError(f"{field} must be a boolean", param=field)
    return value


def json_object(value: Any, field: str) -> dict[str, Any]:
    try:
        value = json.loads(string(value, field), parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise ApiError(f"{field} must contain a JSON object", param=field) from exc
    return obj(value, field)


def invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def schema(value: Any, field: str) -> dict[str, Any]:
    value = obj(value, field)
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise ApiError(f"invalid JSON Schema in {field}", param=field) from exc
    return value


def effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = string(value, "reasoning_effort").strip().lower()
    normalized = {"off": "none", "disabled": "none"}.get(normalized, normalized)
    if normalized not in ("none", "minimal", "low", "medium", "high", "xhigh"):
        raise ApiError(f"unsupported reasoning effort: {value}", param="reasoning_effort")
    return normalized


def fresh_seed(value: int | None) -> int:
    return value if value is not None else secrets.randbits(63)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sse(value: Any, event: str | None = None) -> str:
    data = value if isinstance(value, str) else dumps(value)
    return (f"event: {event}\n" if event else "") + f"data: {data}\n\n"


@dataclass(slots=True)
class ToolOutput:
    id: str
    name: str
    arguments: str = ""
    finished: bool = False


class Journal:
    """One accumulator for buffered output and all stream projectors.

    Completed calls survive a length cutoff; unfinished calls are excluded
    from the final semantic result. Malformed streams never imply success.
    """

    def __init__(self, model: str, prefix: str = "chatcmpl_") -> None:
        self.model, self.id, self.created = model, prefix + uuid4().hex, int(time.time())
        self.text, self.reasoning = "", ""
        self.calls: dict[int, ToolOutput] = {}
        self.completion: Completed | None = None
        self.prompt_tokens = 0
        self.order: list[str | int] = []

    def observe(self, output: Output) -> None:
        if self.completion is not None:
            raise ApiError.server("output arrived after completion")
        event = output.event
        if isinstance(event, ContentDelta):
            if "text" not in self.order:
                self.order.append("text")
            self.text += event.text
        elif isinstance(event, ReasoningDelta):
            if "reasoning" not in self.order:
                self.order.append("reasoning")
            self.reasoning += event.text
        elif isinstance(event, ToolCallStart):
            if event.index in self.calls or not event.name or not event.call_id:
                raise ApiError.server("duplicate or malformed tool call start")
            if any(call.id == event.call_id for call in self.calls.values()):
                raise ApiError.server("duplicate tool call identity")
            if not event.args_prefix_stable:
                raise ApiError.server("tool arguments must have a stable streaming prefix")
            self.calls[event.index] = ToolOutput(event.call_id, event.name)
            self.order.append(event.index)
        elif isinstance(event, (ToolCallArgumentsDelta, ToolCallEnd)):
            call = self.calls.get(event.index)
            if call is None or call.finished:
                raise ApiError.server("tool input arrived outside an open tool call")
            if isinstance(event, ToolCallArgumentsDelta):
                call.arguments += event.fragment
            else:
                try:
                    final = json_object(event.arguments, "tool arguments")
                    streamed = json_object(call.arguments, "streamed tool arguments")
                except ApiError as exc:
                    raise ApiError.server(
                        "completed tool arguments must contain a JSON object"
                    ) from exc
                if final != streamed or event.name != call.name:
                    raise ApiError.server("completed tool arguments differ from streamed arguments")
                if event.call_id is not None and event.call_id != call.id:
                    raise ApiError.server("tool call identity changed")
                call.finished = True
        else:
            raise ApiError.server("unexpected event in semantic output")

    def finish(self, completion: Completed) -> None:
        if self.completion is not None:
            raise ApiError.server("duplicate terminal event")
        if completion.finish_reason not in ("stop", "length", "tool_call"):
            raise ApiError.server("invalid generation termination")
        if completion.finish_reason != "length" and any(
            not c.finished for c in self.calls.values()
        ):
            raise ApiError.server("generation ended inside a tool call")
        usage = completion.usage
        if (
            min(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cached_tokens,
                usage.reasoning_tokens,
            )
            < 0
            or usage.cached_tokens > usage.prompt_tokens
        ):
            raise ApiError.server("invalid token usage")
        self.completion = completion

    def usage(self) -> dict[str, Any]:
        if self.completion is None:
            raise ApiError.server("generation ended without a terminal event")
        usage = self.completion.usage
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        }

    def complete_calls(self) -> list[ToolOutput]:
        return [self.calls[i] for i in sorted(self.calls) if self.calls[i].finished]
