"""Sampling configuration parameters and admission validation.

Defines the `SamplingParams` descriptor capturing token generation policies,
sampling constraints, penalties, and logprob reporting requirements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ayaka.sampling.logprobs import LogprobMode

TOP_K_DISABLED: Final[int] = -1

#: Magnitude limit for OpenAI-compatible logit_bias values; validated at admission.
LOGIT_BIAS_LIMIT: Final[float] = 100.0


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Sampling configuration parameters for generation and logprob evaluation.

    Encapsulates decoding hyperparameters (e.g., temperature, top-p, top-k,
    min-p, penalties) and reporting options (logprobs, support capture) as a pure
    descriptor without tensor allocations.

    Attributes:
        temperature: Softmax temperature scaling factor. Non-negative float.
        top_p: Nucleus sampling cumulative probability threshold in (0, 1].
        top_k: Top-k filtering threshold. Positive integer or -1 (disabled).
        min_p: Minimum probability threshold relative to the argmax probability in [0, 1].
        repetition_penalty: Multiplicative discount for repeated tokens.
        frequency_penalty: Linear penalty proportional to token frequency.
        presence_penalty: Flat linear penalty for token presence.
        seed: Optional 64-bit random seed for pseudo-random generation. None draws
            from the engine RNG.
        n: Number of output sequences to generate per prompt.
        logprobs: Number of top-k token logprobs to return per decoding step. None disables.
        prompt_logprobs: Number of top-k prompt token logprobs to return. None disables.
        logprob_mode: Distribution mode for returned logprobs (`RAW` or `SAMPLING`).
        return_sampling_support: Whether to return the filtered positive support distribution.
        token_ids_logprobs: Explicit tuple of token ids to score logprobs for.
        logit_bias: Mapping or sequence of (token_id, bias) pairs added to raw logits.

    Notes:
        Logprob requests (`logprobs`, `prompt_logprobs`, `return_sampling_support`)
        are reporting configurations only and never alter the sampled token output.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_DISABLED  # -1 = disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    n: int = 1
    logprobs: int | None = None  # top-k logprobs to return, None = off
    prompt_logprobs: int | None = None
    logprob_mode: LogprobMode = LogprobMode.RAW
    return_sampling_support: bool = False
    token_ids_logprobs: tuple[int, ...] | None = None
    logit_bias: Mapping[int, float] | tuple[tuple[int, float], ...] | None = None

    def __post_init__(self) -> None:
        # Validate temperature bounds.
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        # Validate top-p nucleus threshold in (0, 1].
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        # Validate top-k cutoff bounds.
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (off) or >= 1")
        # Validate min-p probability cutoff in [0, 1].
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        # Validate candidate sequence count.
        if self.n < 1:
            raise ValueError("n must be >= 1")
        # Validate generation logprobs count.
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError("logprobs must be None or >= 0")
        # Validate prompt logprobs count.
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError("prompt_logprobs must be None or >= 0")
        # Normalize logprob_mode string representations into LogprobMode enum.
        if not isinstance(self.logprob_mode, LogprobMode):
            if isinstance(self.logprob_mode, str):
                object.__setattr__(self, "logprob_mode", LogprobMode(self.logprob_mode))
            else:
                raise TypeError("logprob_mode must be LogprobMode")
        # Validate and canonicalize explicit token_ids_logprobs.
        if self.token_ids_logprobs is not None:
            if isinstance(self.token_ids_logprobs, (list, tuple)):
                ids = tuple(int(token) for token in self.token_ids_logprobs)
            else:
                raise TypeError("token_ids_logprobs must be a sequence of ints or None")
            if any(token < 0 for token in ids):
                raise ValueError("token_ids_logprobs tokens must be >= 0")
            object.__setattr__(self, "token_ids_logprobs", ids)
        # Validate and canonicalize logit_bias mapping into sorted tuples.
        raw_bias: Any = self.logit_bias
        if raw_bias is not None and not isinstance(raw_bias, tuple):
            if isinstance(raw_bias, Mapping):
                items = tuple(
                    sorted((int(token), float(value)) for token, value in raw_bias.items())
                )
            else:
                raise TypeError("logit_bias must be a Mapping[int, float] or None")
            for token, value in items:
                if token < 0:
                    raise ValueError("logit_bias token ids must be >= 0")
                if not -LOGIT_BIAS_LIMIT <= value <= LOGIT_BIAS_LIMIT:
                    raise ValueError(
                        f"logit_bias value {value} out of range "
                        f"[-{LOGIT_BIAS_LIMIT:g}, {LOGIT_BIAS_LIMIT:g}]"
                    )
            object.__setattr__(self, "logit_bias", items)

    @property
    def is_greedy(self) -> bool:
        """Indicate whether the configuration dictates a deterministic argmax draw.

        Returns:
            True if temperature is 0.0 or top_k is 1, bypassing stochastic filtering.
        """
        return self.temperature == 0.0 or self.top_k == 1
