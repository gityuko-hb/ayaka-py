"""Canonical request and model-source ports corresponding to Rust source.rs.

No device, scheduler, tokenizer, or HTTP implementation is imported here.
Host operations do not acquire a generation binding. Invocation.aclose and
the generation iterator's aclose must cancel work and release its binding.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ayaka.serving.events import ServingEvent, UsageUpdate

from .error import ApiError


@dataclass(frozen=True, slots=True)
class ChatInput:
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | dict[str, Any] = "auto"
    parallel_tool_calls: bool = True
    reasoning: str | None = None
    output: dict[str, Any] = field(default_factory=lambda: {"type": "text"})
    template_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SamplingControls:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationControls:
    max_output_tokens: int | None = None
    sampling: SamplingControls = SamplingControls()
    stops: tuple[str, ...] = ()
    ignore_eos: bool = False
    reasoning_budget: int | None = None
    cache_prompt: bool = True
    timeout: float | None = None
    min_tokens: int = 0
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    input: ChatInput
    controls: GenerationControls = GenerationControls()


@dataclass(frozen=True, slots=True)
class AdaptedRequest:
    model: str
    request: GenerationRequest
    stream: bool = False
    include_usage: bool = False
    timings_per_token: bool = False
    projection: dict[str, Any] = field(default_factory=dict)


class ModelLoadStage(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    OPTIMIZING = "optimizing"
    LOADING_WEIGHTS = "loading_weights"
    FINALIZING = "finalizing"


@dataclass(frozen=True, slots=True)
class ModelLoadProgress:
    stage: ModelLoadStage
    fraction: float


type LoadProgress = Callable[[ModelLoadProgress], None]


@dataclass(frozen=True, slots=True)
class Progress:
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Admitted:
    prompt_tokens: int


@dataclass(frozen=True, slots=True)
class Output:
    event: ServingEvent
    timings: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Completed:
    usage: UsageUpdate = UsageUpdate()
    finish_reason: str = "stop"
    stop_sequence: str | None = None
    timings: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Failed:
    error: ApiError


type GenerationEvent = Progress | Admitted | Output | Completed | Failed
type GenerationStream = AsyncGenerator[GenerationEvent]


class ModelInvocation(Protocol):
    """A single-use generation binding, held through terminal output or cancellation."""

    def generate(self, request: GenerationRequest) -> GenerationStream: ...
    async def aclose(self) -> None: ...


class HostChat(Protocol):
    """Blocking host work; called on a worker thread, never under model admission."""

    def count(self, input: ChatInput) -> int: ...
    def apply_template(self, input: ChatInput) -> dict[str, Any]: ...
    def properties(self) -> dict[str, Any]: ...


class ServedModels(Protocol):
    async def invoke(
        self,
        model: str,
        progress: LoadProgress | None = None,
        *,
        tenant: str = "anonymous",
    ) -> ModelInvocation: ...
    async def host(self, model: str) -> HostChat: ...
