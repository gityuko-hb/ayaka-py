"""Validated transport-neutral generation specifications and Request conversion."""

from __future__ import annotations

import hashlib
import json
import time
from uuid import uuid4

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ayaka.configs.serving import ServingConfig
from ayaka.request.cancel import CancellationToken
from ayaka.request.input import ConstraintSpec
from ayaka.request.schema import CacheHints, Request, RequestId, StopCriteria
from ayaka.sampling.params import SamplingParams
from ayaka.serving.errors import (
    ContextLengthExceededError,
    InvalidRequestError,
    ModelNotFoundError,
    OverloadedError,
    UnsupportedFeatureError,
)
from ayaka.tokenizers.contracts import EncodeOptions, TemplateInput, TextInput, TokenIdsInput
from ayaka.tokenizers.service import TokenizerOverloaded, TokenizerService


class GenerationSpec(BaseModel):
    """Every protocol maps into this model before touching the engine."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: str
    prompt: str | list[int] | None = None
    messages: list[dict] | None = None
    max_tokens: int = Field(default=128, ge=1)
    min_tokens: int = Field(default=0, ge=0)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = -1
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, gt=0)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stop: str | list[str] | None = None
    stop_token_ids: list[int] = Field(default_factory=list)
    ignore_eos: bool = False
    stream: bool = False
    include_usage: bool = False
    response_format: dict | None = None
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict = "auto"

    @model_validator(mode="after")
    def check_values(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("exactly one of prompt or messages is required")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens exceeds max_tokens")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or positive")
        stops = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if len(stops) > 16 or any(not stop for stop in stops):
            raise ValueError("stop must contain at most 16 nonempty strings")
        if any(token < 0 for token in self.stop_token_ids):
            raise ValueError("stop token IDs must be nonnegative")
        if self.tools and self.response_format:
            raise ValueError("tools and response_format cannot be combined")
        if self.prompt is not None and self.tools:
            raise ValueError("tools require chat messages")
        if isinstance(self.tool_choice, str) and self.tool_choice not in (
            "auto",
            "none",
            "required",
        ):
            raise ValueError("unsupported tool_choice")
        if self.tool_choice not in ("auto", "none") and not self.tools:
            raise ValueError("tool_choice requires tools")
        return self


def text_content(content, *, allow_none=False) -> str:
    if content is None and allow_none:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise InvalidRequestError("content must be text or a list of text parts")
    pieces = []
    for part in content:
        if not isinstance(part, dict):
            raise InvalidRequestError("content parts must be objects")
        if part.get("type") not in ("text", "input_text", "output_text"):
            raise UnsupportedFeatureError("this serving model supports text content only")
        if not isinstance(part.get("text"), str):
            raise InvalidRequestError("text part requires a text string")
        pieces.append(part["text"])
    return "".join(pieces)


def checked_schema(schema) -> dict:
    if not isinstance(schema, dict):
        raise InvalidRequestError("JSON schema must be an object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise InvalidRequestError("invalid JSON schema") from exc
    return schema


def checked_tools(tools: list[dict]) -> list[dict]:
    names = set()
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise UnsupportedFeatureError("only function tools are supported")
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise InvalidRequestError("tool names must be nonempty and unique")
        names.add(name)
        checked_schema(function.get("parameters", {"type": "object"}))
    return tools


def constraint_for(spec: GenerationSpec) -> ConstraintSpec | None:
    tools = checked_tools(spec.tools)
    if tools and spec.tool_choice not in ("auto", "none"):
        selected = tools
        if isinstance(spec.tool_choice, dict):
            choice = spec.tool_choice
            name = choice.get("function", {}).get("name")
            if choice.get("type") != "function" or not isinstance(name, str):
                raise InvalidRequestError("invalid named tool_choice")
            selected = [t for t in tools if t["function"]["name"] == name]
            if not selected:
                raise InvalidRequestError("tool_choice names an unknown tool")
        variants = []
        for tool in selected:
            function = tool["function"]
            variants.append(
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": function["name"]},
                        "arguments": function.get("parameters", {"type": "object"}),
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                }
            )
        schema = {
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "items": {"anyOf": variants},
                    "minItems": 1,
                    "maxItems": 1,
                }
            },
            "required": ["tool_calls"],
            "additionalProperties": False,
        }
        return ConstraintSpec("tool_calls", json.dumps(schema))
    fmt = spec.response_format
    if fmt is None or fmt == {"type": "text"}:
        return None
    if fmt == {"type": "json_object"}:
        return ConstraintSpec("json_object")
    if fmt.get("type") == "json_schema" and isinstance(fmt.get("json_schema"), dict):
        schema = checked_schema(fmt["json_schema"].get("schema"))
        return ConstraintSpec("json_schema", json.dumps(schema))
    raise UnsupportedFeatureError("unsupported response_format")


def chat_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")
    result = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise InvalidRequestError("unsupported message role")
        if set(message) - {
            "role",
            "content",
            "name",
            "tool_calls",
            "tool_call_id",
            "reasoning_content",
        }:
            raise UnsupportedFeatureError("unsupported message field")
        item = dict(message)
        item["content"] = text_content(item.get("content"), allow_none=role == "assistant")
        if role == "tool" and not isinstance(item.get("tool_call_id"), str):
            raise InvalidRequestError("tool messages require tool_call_id")
        if "tool_calls" in item:
            if role != "assistant" or not isinstance(item["tool_calls"], list):
                raise InvalidRequestError("tool_calls require an assistant message")
            calls = []
            for call in item["tool_calls"]:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise InvalidRequestError("invalid tool call history")
                fn = dict(call["function"])
                if not isinstance(fn.get("name"), str) or not isinstance(call.get("id"), str):
                    raise InvalidRequestError("tool call history requires id and name")
                args = fn.get("arguments", "{}")
                try:
                    fn["arguments"] = json.loads(args) if isinstance(args, str) else args
                except ValueError as exc:
                    raise InvalidRequestError("tool arguments must be valid JSON") from exc
                if not isinstance(fn["arguments"], dict):
                    raise InvalidRequestError("tool arguments must be an object")
                calls.append({**call, "function": fn})
            item["tool_calls"] = calls
        result.append(item)
    return result


class RequestProcessor:
    """Tokenization stays in its bounded service before scheduler admission."""

    def __init__(self, tokenizer: TokenizerService, config: ServingConfig):
        self.tokenizer, self.config = tokenizer, config

    def prepare(
        self, spec: GenerationSpec, tenant: str, cancellation: CancellationToken | None = None
    ) -> Request:
        if spec.model != self.config.model:
            raise ModelNotFoundError(f"unknown model: {spec.model}", param="model")
        constraint = constraint_for(spec)
        if constraint is not None and not self.config.structured_outputs:
            raise UnsupportedFeatureError("structured decoding is disabled")
        if spec.tools and spec.tool_choice == "auto" and self.config.tool_parser == "none":
            raise UnsupportedFeatureError("automatic tools require a configured tool parser")
        if spec.messages is not None:
            messages = chat_messages(spec.messages)
            if constraint is not None and constraint.kind == "tool_calls":
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": 'Return only JSON: {"tool_calls":[{"name":"function_name",'
                        '"arguments":{}}]}. Choose from the supplied functions.',
                    },
                )
            source = TemplateInput(
                json.dumps(messages, ensure_ascii=False),
                json.dumps(spec.tools if spec.tool_choice != "none" else []),
            )
        else:
            source = (
                TextInput(spec.prompt)
                if isinstance(spec.prompt, str)
                else TokenIdsInput(tuple(spec.prompt or ()))
            )
        namespace = hashlib.sha256(tenant.encode()).hexdigest()
        try:
            encoded = self.tokenizer.encode(
                source,
                EncodeOptions(max_output_tokens=spec.max_tokens, cache_namespace=namespace),
                cancellation=cancellation,
            )
        except TokenizerOverloaded as exc:
            raise OverloadedError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            if "context" in str(exc) or "reservation" in str(exc) or "budget" in str(exc):
                raise ContextLengthExceededError(str(exc)) from exc
            raise InvalidRequestError(str(exc)) from exc
        if any(t >= self.tokenizer.model_vocab_size for t in spec.stop_token_ids):
            raise InvalidRequestError("stop token ID exceeds model vocabulary")
        stops = (spec.stop,) if isinstance(spec.stop, str) else tuple(spec.stop or ())
        return Request(
            RequestId("req_" + uuid4().hex),
            tuple(encoded.token_ids),
            sampling=SamplingParams(
                temperature=spec.temperature,
                top_p=spec.top_p,
                top_k=spec.top_k,
                min_p=spec.min_p,
                repetition_penalty=spec.repetition_penalty,
                frequency_penalty=spec.frequency_penalty,
                presence_penalty=spec.presence_penalty,
                seed=spec.seed,
            ),
            stop=StopCriteria(
                max_tokens=spec.max_tokens,
                min_tokens=spec.min_tokens,
                stop_strings=stops,
                stop_token_ids=tuple(spec.stop_token_ids),
                ignore_eos=spec.ignore_eos,
            ),
            cache=CacheHints(cache_salt=namespace),
            tenant_id=tenant,
            constraint=constraint,
            arrival_ns=time.monotonic_ns(),
        )
