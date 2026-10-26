from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ayaka.request.stream import TokenLogprobs

__all__ = [
    "ContentDelta",
    "GenerationFailed",
    "GenerationFinished",
    "ReasoningDelta",
    "ServingEvent",
    "TokenLogprobs",
    "ToolCallArgumentsDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "UsageUpdate",
]


@dataclass(frozen=True, slots=True)
class ContentDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    index: int
    name: str
    args_prefix_stable: bool = True
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallArgumentsDelta:
    index: int
    fragment: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    index: int
    name: str
    arguments: str
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True, slots=True)
class GenerationFinished:
    finish_reason: Literal["stop", "length", "tool_call", "cancelled", "timeout"]
    usage: UsageUpdate = UsageUpdate()


@dataclass(frozen=True, slots=True)
class GenerationFailed:
    code: str
    message: str
    retryable: bool = False


#: ``TokenLogprobs`` reports one sampled token and is queued just before the
#: text events that token produced; only requests that asked for logprobs get it.
type ServingEvent = (
    ContentDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallArgumentsDelta
    | ToolCallEnd
    | TokenLogprobs
    | GenerationFinished
    | GenerationFailed
)
