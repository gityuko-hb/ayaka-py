"""Request-local incremental parsers; syntax is selected explicitly by the server."""

from __future__ import annotations

import json
from uuid import uuid4

from jsonschema import Draft202012Validator, ValidationError

from ayaka.serving.errors import ParserError
from ayaka.serving.events import (
    ContentDelta,
    ReasoningDelta,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
)


class OutputParser:
    """Retain only an incomplete delimiter or one incomplete tool payload.

    Tool arguments are emitted after the closing delimiter, so escaped strings,
    arbitrary chunk boundaries and malformed JSON can never create revised deltas.
    A length-limited unfinished tool is an explicit error.
    """

    def __init__(self, *, reasoning=False, tools=(), constrained_tools=False):
        self.reasoning = reasoning
        self.tools = {t["function"]["name"]: t["function"] for t in tools}
        self.constrained_tools = constrained_tools
        self.mode = "content"
        self.buffer = ""
        self.index = 0
        self.had_tools = False

    def _call(self, value):
        if not isinstance(value, dict) or value.get("name") not in self.tools:
            raise ParserError("model emitted an unknown or malformed tool")
        name, arguments = value["name"], value.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise ParserError("model emitted invalid tool arguments") from exc
        if not isinstance(arguments, dict):
            raise ParserError("tool arguments must be a JSON object")
        try:
            Draft202012Validator(self.tools[name].get("parameters", {"type": "object"})).validate(
                arguments
            )
        except ValidationError as exc:
            raise ParserError("model tool arguments do not match the function schema") from exc
        rendered = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        index, call_id = self.index, "call_" + uuid4().hex
        self.index += 1
        self.had_tools = True
        return [
            ToolCallStart(index, name, call_id=call_id),
            ToolCallArgumentsDelta(index, rendered),
            ToolCallEnd(index, name, rendered, call_id),
        ]

    def feed(self, text: str):
        self.buffer += text
        if self.constrained_tools:
            return []
        events = []
        while self.buffer:
            if self.mode == "tool":
                pos = self.buffer.find("</tool_call>")
                if pos < 0:
                    break
                try:
                    value = json.loads(self.buffer[:pos])
                except ValueError as exc:
                    raise ParserError("model emitted invalid tool JSON") from exc
                events.extend(self._call(value))
                self.buffer = self.buffer[pos + len("</tool_call>") :]
                self.mode = "content"
                continue
            tags = {"</think>": "content"} if self.mode == "reasoning" else {}
            if self.mode == "content":
                if self.reasoning:
                    tags["<think>"] = "reasoning"
                if self.tools:
                    tags["<tool_call>"] = "tool"
            matches = [(self.buffer.find(tag), tag) for tag in tags if tag in self.buffer]
            if matches:
                pos, tag = min(matches)
                if pos:
                    cls = ReasoningDelta if self.mode == "reasoning" else ContentDelta
                    events.append(cls(self.buffer[:pos]))
                self.buffer = self.buffer[pos + len(tag) :]
                self.mode = tags[tag]
                continue
            keep = max(
                (
                    size
                    for tag in tags
                    for size in range(1, len(tag))
                    if self.buffer.endswith(tag[:size])
                ),
                default=0,
            )
            emit = self.buffer[:-keep] if keep else self.buffer
            self.buffer = self.buffer[-keep:] if keep else ""
            if emit:
                cls = ReasoningDelta if self.mode == "reasoning" else ContentDelta
                events.append(cls(emit))
            break
        return events

    def finish(self):
        if self.constrained_tools:
            try:
                value = json.loads(self.buffer)
                calls = value["tool_calls"]
                if not isinstance(calls, list) or not calls:
                    raise ValueError("missing tool calls")
            except (ValueError, KeyError, TypeError) as exc:
                raise ParserError("incomplete or invalid constrained tool output") from exc
            events = []
            for call in calls:
                events.extend(self._call(call))
            self.buffer = ""
            return events
        if self.mode == "tool":
            raise ParserError("incomplete tool call at end of generation")
        tail, self.buffer = self.buffer, ""
        if not tail:
            return []
        return [ReasoningDelta(tail) if self.mode == "reasoning" else ContentDelta(tail)]
