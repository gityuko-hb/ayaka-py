"""Incremental detokenization and stop evaluation for published samples.

One state per request incarnation. Token-stop masking happens before sampling;
text stops are evaluated after the token is decoded, in the order fixed by
``ayaka.request.stream``. This owner never commits KV or retires resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.configs.base import ConfigError
from ayaka.configs.tokenizer import DetokenizeParams
from ayaka.request.schema import Request
from ayaka.request.stream import OutputEvent, ResolvedStopPolicy, TokenUsage
from ayaka.sched.outcome import FinishReason

if TYPE_CHECKING:
    from ayaka.tokenizers.detokenizer import IncrementalDetokenizer
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["FinishDecision", "OutputProcessor"]


@dataclass(frozen=True, slots=True)
class FinishDecision:
    """A completed generation the engine must report to the lifecycle manager."""

    request_id: str
    reason: FinishReason
    detail: str = ""


@dataclass(slots=True)
class _OutputState:
    policy: ResolvedStopPolicy
    prompt_tokens: int
    decoder: IncrementalDetokenizer | None
    text: str = ""
    events: list[OutputEvent] = field(default_factory=list)


class OutputProcessor:
    """Detokenize published samples, enforce stop policy and keep text history.

    Without a tokenizer only token-ID stops and length limits are available;
    advertising that limitation is the caller's job (see ``text_stops_supported``).
    """

    def __init__(
        self,
        tokenizer: TokenizerService | None = None,
        *,
        eos_token_ids: tuple[int, ...] = (),
    ) -> None:
        self._tokenizer = tokenizer
        self._eos_token_ids = tuple(eos_token_ids)
        self._states: dict[str, _OutputState] = {}

    @property
    def text_stops_supported(self) -> bool:
        return self._tokenizer is not None

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_token_ids

    def register(self, request: Request) -> None:
        request_id = str(request.request_id)
        if request_id in self._states:
            raise ValueError(f"request {request_id!r} already registered")
        if request.stop.stop_strings and self._tokenizer is None:
            raise ConfigError(
                "request.stop.stop_strings",
                "UNSUPPORTED_STOP",
                "text stop strings require a tokenizer-backed output owner",
            )
        policy = ResolvedStopPolicy.resolve(request.stop, eos_token_ids=self._eos_token_ids)
        decoder = None
        if self._tokenizer is not None:
            params = DetokenizeParams(
                accumulate_text=True,
                stop=request.stop.stop_strings,
                include_stop_str_in_output=request.stop.include_stop_str_in_output,
            )
            decoder = self._tokenizer.new_detokenizer(
                params, prompt_token_ids=request.prompt_token_ids
            )
        self._states[request_id] = _OutputState(
            policy=policy,
            prompt_tokens=request.prompt_len,
            decoder=decoder,
        )

    def forget(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def masked_ids(self, request_id: str, *, generated_tokens: int) -> frozenset[int]:
        return self._state(request_id).policy.masked_token_ids(generated_tokens)

    def on_published(
        self,
        request_id: str,
        token_id: int,
        *,
        sequence_epoch: int,
        generated_tokens: int,
        prompt_tokens: int,
    ) -> FinishDecision | None:
        """Decode one published sample and evaluate its stop policy exactly once."""
        state = self._state(request_id)
        before = generated_tokens - 1
        delta = ""
        stop_matched = None
        if state.decoder is not None:
            update = state.decoder.update(
                [token_id], check_stops=state.policy.text_stops_enabled(before)
            )
            delta = update.delta
            stop_matched = update.stop_matched
        reason = state.policy.evaluate(
            token_id, generated_tokens=generated_tokens, stop_matched=stop_matched
        )
        if state.decoder is not None and reason is not None:
            delta += state.decoder.finish().delta
        state.text += delta
        state.events.append(
            OutputEvent(
                request_id=request_id,
                sequence_epoch=sequence_epoch,
                event_index=len(state.events),
                delta=delta,
                usage=TokenUsage(prompt_tokens, generated_tokens),
                finish_reason=reason,
            )
        )
        if reason is None:
            return None
        return FinishDecision(request_id, reason, "stop policy matched")

    def text(self, request_id: str) -> str:
        return self._state(request_id).text

    def events(self, request_id: str) -> tuple[OutputEvent, ...]:
        return tuple(self._state(request_id).events)

    def _state(self, request_id: str) -> _OutputState:
        state = self._states.get(request_id)
        if state is None:
            raise KeyError(f"request {request_id!r} has no output state")
        return state
