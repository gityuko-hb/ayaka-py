"""P0 stream values and termination policy.

TokenEvent is transport-neutral data from the completion owner, NOT proof of GPU
completion. P2's output owner must reject stale/duplicate events, evaluate this
policy, then ask LifecycleManager to finish; it must never recommit sample IDs.
EOS/stop IDs are masked by the sampler before min_tokens. Text matching begins
only when text_stops_enabled() becomes true; matches cannot span that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.configs.tokenizer import require_bool
from ayaka.request.schema import StopCriteria
from ayaka.sched.outcome import FinishReason
from ayaka.utils.validation import require_frozen, require_int, require_text

STREAM_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    sequence_epoch: int
    token_start: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        require_frozen(self, "token event")
        require_text(self.request_id, "request_id")
        require_int(self.sequence_epoch, "sequence_epoch", minimum=1)
        require_int(self.token_start, "token_start")
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise ValueError("token_ids must be a non-empty tuple")
        for token in self.token_ids:
            require_int(token, "token id")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Sampled tokens include hidden EOS/stop IDs; visible text is not re-encoded."""

    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        require_int(self.prompt_tokens, "prompt_tokens")
        require_int(self.completion_tokens, "completion_tokens")


@dataclass(frozen=True, slots=True)
class OutputEvent:
    """Append-only output. event_index is per request incarnation, starting at 0."""

    request_id: str
    sequence_epoch: int
    event_index: int
    delta: str
    usage: TokenUsage
    finish_reason: FinishReason | None = None

    def __post_init__(self) -> None:
        require_frozen(self, "output event")
        require_text(self.request_id, "request_id")
        require_int(self.sequence_epoch, "sequence_epoch", minimum=1)
        require_int(self.event_index, "event_index")
        if type(self.delta) is not str or not isinstance(self.usage, TokenUsage):
            raise TypeError("output needs text delta and TokenUsage")
        if self.finish_reason is not None and not isinstance(self.finish_reason, FinishReason):
            raise TypeError("finish_reason must be FinishReason")


@dataclass(frozen=True, slots=True)
class ResolvedStopPolicy:
    """Token/string stop wins over LENGTH on the same token; explicit IDs win EOS.

    Cancellation/error are lifecycle decisions, not ordinary generation stops.
    This pure policy performs no sampling, publication, KV commit or retirement.
    """

    max_tokens: int
    min_tokens: int = 0
    stop_token_ids: tuple[int, ...] = ()
    eos_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False

    def __post_init__(self) -> None:
        require_frozen(self, "stop policy")
        require_int(self.max_tokens, "max_tokens", minimum=1)
        require_int(self.min_tokens, "min_tokens")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens exceeds max_tokens")
        require_bool(self.ignore_eos, "ignore_eos")
        require_bool(self.include_stop_str_in_output, "include_stop_str_in_output")
        for name in ("stop_token_ids", "eos_token_ids", "stop_strings"):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be tuple")
        for token in (*self.stop_token_ids, *self.eos_token_ids):
            require_int(token, "stop token id")
        if any(type(s) is not str or not s for s in self.stop_strings):
            raise ValueError("stop strings must be non-empty strings")

    @classmethod
    def resolve(
        cls, criteria: StopCriteria, *, eos_token_ids: tuple[int, ...] = ()
    ) -> ResolvedStopPolicy:
        """Snapshot validated criteria without retaining mutable aliases."""
        if not isinstance(criteria, StopCriteria):
            raise TypeError("criteria must be StopCriteria")
        return cls(
            max_tokens=criteria.max_tokens,
            min_tokens=criteria.min_tokens,
            stop_token_ids=criteria.stop_token_ids,
            eos_token_ids=eos_token_ids,
            stop_strings=criteria.stop_strings,
            ignore_eos=criteria.ignore_eos,
            include_stop_str_in_output=criteria.include_stop_str_in_output,
        )

    def masked_token_ids(self, generated_tokens: int) -> frozenset[int]:
        """IDs forbidden BEFORE sampling the next token until min_tokens is reached."""
        require_int(generated_tokens, "generated_tokens")
        if generated_tokens >= self.min_tokens:
            return frozenset()
        return frozenset(self.stop_token_ids + (() if self.ignore_eos else self.eos_token_ids))

    def text_stops_enabled(self, generated_tokens: int) -> bool:
        """Whether a token about to be decoded may start/complete a text stop."""
        require_int(generated_tokens, "generated_tokens")
        return generated_tokens >= self.min_tokens

    def evaluate(
        self, token_id: int, *, generated_tokens: int, stop_matched: str | None = None
    ) -> FinishReason | None:
        """Evaluate AFTER sampling; count includes token_id.

        The min_tokens guard must agree with the sampler's pre-sample mask.
        Text-stop matching uses the count before this token.
        """
        require_int(token_id, "token_id")
        require_int(generated_tokens, "generated_tokens", minimum=1)
        if generated_tokens > self.max_tokens:
            raise ValueError("generation exceeded its token limit")
        if stop_matched is not None and stop_matched not in self.stop_strings:
            raise ValueError("unknown text stop")
        if generated_tokens > self.min_tokens:
            if token_id in self.stop_token_ids or stop_matched is not None:
                return FinishReason.STOP
            if not self.ignore_eos and token_id in self.eos_token_ids:
                return FinishReason.EOS
        if generated_tokens == self.max_tokens:
            return FinishReason.LENGTH
        return None
