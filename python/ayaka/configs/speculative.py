"""Opt-in greedy speculative serving configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class SpeculativeConfig:
    """Opt-in greedy serving; stochastic acceptance is a separate reference API."""

    width: int = 4
    draft_backend: str = "eager"
    verify_backend: str = "eager"
    token_buckets: tuple[int, ...] = ()
    memory_bytes: int = 256 << 20
    max_captures: int = 16
    capture_seconds: int = 120

    def __post_init__(self) -> None:
        require_int(self.width, "speculative width", minimum=1)
        require_int(self.memory_bytes, "speculative memory_bytes", minimum=1)
        require_int(self.max_captures, "max_captures", minimum=1)
        require_int(self.capture_seconds, "capture_seconds", minimum=1)
        if self.width > 16:
            raise ValueError("linear speculative width is limited to 16")
        if any(b not in ("eager", "full") for b in (self.draft_backend, self.verify_backend)):
            raise ValueError("speculative roles support eager or full only")
        for size in self.token_buckets:
            require_int(size, "speculative token bucket", minimum=1)
        if tuple(sorted(set(self.token_buckets))) != self.token_buckets:
            raise ValueError("speculative token buckets must be sorted and unique")
        graphs = (self.draft_backend != "eager") + (self.verify_backend != "eager")
        if graphs and (
            not self.token_buckets or graphs * len(self.token_buckets) > self.max_captures
        ):
            raise ValueError("speculative capture count exceeds its bounded registry")
        if not graphs and self.token_buckets:
            raise ValueError("eager speculative roles do not use token buckets")
