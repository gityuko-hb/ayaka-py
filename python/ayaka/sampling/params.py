from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """What the sampler does with the logits.  Pure description — no tensors.

    ``seed=None`` means "draw from the engine RNG"; a concrete int makes the
    request bitwise reproducible only when it also lands alone in its batch,
    because reduction order in a batched kernel is not stable across shapes.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    n: int = 1
    logprobs: int | None = None  # top-k logprobs to return, None = off
    prompt_logprobs: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (off) or >= 1")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.n < 1:
            raise ValueError("n must be >= 1")

    @property
    def is_greedy(self) -> bool:
        """Greedy path skips the whole sort/scan pipeline — argmax only."""
        return self.temperature == 0.0 or self.top_k == 1
