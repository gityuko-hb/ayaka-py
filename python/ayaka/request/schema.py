from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

from ayaka.sampling.params import SamplingParams
from ayaka.utils.validation import require_frozen, require_int, require_text

RequestId = NewType("RequestId", str)
SessionId = NewType("SessionId", str)


@dataclass(frozen=True, slots=True)
class StopCriteria:
    max_tokens: int = 128
    min_tokens: int = 0
    stop_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0 <= self.min_tokens <= self.max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens]")


@dataclass(frozen=True, slots=True)
class CacheHints:
    """Prefix-cache controls.

    ``cache_salt`` isolates the hash chain: two requests with identical prompts
    but different salts can never share a block.  This is the tenancy boundary —
    without it a shared prefix index leaks content across users.
    """

    cache_salt: str | None = None
    prefix_cache: bool = True
    store_kv: bool = True
    session_id: SessionId | None = None


@dataclass(frozen=True, slots=True)
class Request:
    """The Request IR itself."""

    request_id: RequestId
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    stop: StopCriteria = field(default_factory=StopCriteria)
    cache: CacheHints = field(default_factory=CacheHints)

    # Admission / scheduling inputs.  Higher priority is served first; the
    # scheduler breaks ties on arrival_ns so ordering is total and stable.
    priority: int = 0
    arrival_ns: int = 0
    deadline_ns: int | None = None

    # Set at the API edge, carried unchanged to
    # every span the request produces.
    trace_id: str | None = None

    def __post_init__(self) -> None:
        require_frozen(self, "request")
        require_text(self.request_id, "request_id")
        for token in self.prompt_token_ids:
            require_int(token, "prompt token id")
        if not self.prompt_token_ids:
            raise ValueError(f"{self.request_id}: empty prompt")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def max_total_len(self) -> int:
        """Upper bound on sequence length — what the memory planner reserves
        against, and what admission control tests against ``max_model_len``."""
        return self.prompt_len + self.stop.max_tokens
