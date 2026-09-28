"""Linear-chain speculation payload and separate greedy/stochastic acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from ayaka.utils.validation import require_int, require_text


class SpeculativeRole(StrEnum):
    DRAFT = "draft"
    VERIFY = "verify"
    DRAFT_EXTEND = "draft_extend"


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


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionBatch:
    """One linear candidate chain, with explicit validity and logical positions."""

    draft_identity: str
    target_identity: str
    prefix: tuple[int, ...]
    candidates: tuple[int, ...]
    vocab_size: int

    def __post_init__(self) -> None:
        require_text(self.draft_identity, "draft identity")
        require_text(self.target_identity, "target identity")
        require_int(self.vocab_size, "vocab_size", minimum=1)
        if not self.prefix or len(self.candidates) > 16:
            raise ValueError("invalid speculative prefix/width")
        for token in (*self.prefix, *self.candidates):
            require_int(token, "token id")
            if token >= self.vocab_size:
                raise ValueError("candidate token is outside the shared vocabulary")

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(range(len(self.prefix) - 1, len(self.prefix) + len(self.candidates)))

    @property
    def query_offsets(self) -> tuple[int, int]:
        return (0, len(self.candidates) + 1)

    @property
    def valid(self) -> tuple[bool, ...]:
        return (True,) * (len(self.candidates) + 1)


@dataclass(frozen=True, slots=True)
class Acceptance:
    tokens: tuple[int, ...]
    accepted: int


def accept_greedy(candidates: tuple[int, ...], target_tokens: tuple[int, ...]) -> Acceptance:
    """Accept the longest equal prefix, then emit target correction or bonus."""
    if len(target_tokens) != len(candidates) + 1:
        raise ValueError("verification needs candidate rows plus a bonus row")
    accepted = 0
    for draft, target in zip(candidates, target_tokens, strict=False):
        if draft != target:
            break
        accepted += 1
    return Acceptance((*candidates[:accepted], target_tokens[accepted]), accepted)


def accept_stochastic(
    candidates: tuple[int, ...],
    proposal: torch.Tensor,
    target: torch.Tensor,
    *,
    generator: torch.Generator,
) -> Acceptance:
    """Reference rejection sampling (Leviathan et al., ICML 2023, Algorithm 1).

    CPU normalized probabilities, request-owned RNG. On first rejection draw
    from normalized positive (p-q); after full acceptance draw a target bonus.
    This function is not wired to the single-token serving sampler/RNG ledger.
    """
    width = len(candidates)
    if (
        proposal.device.type != "cpu"
        or target.device.type != "cpu"
        or proposal.ndim != 2
        or target.shape != (width + 1, proposal.shape[1])
        or proposal.shape[0] != width
        or proposal.shape[1] < 1
    ):
        raise ValueError("expected CPU q[K,V], p[K+1,V]")
    for probabilities in (proposal, target):
        if (
            not probabilities.is_floating_point()
            or not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not torch.allclose(
                probabilities.sum(-1),
                torch.ones(probabilities.shape[0], dtype=probabilities.dtype),
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise ValueError("acceptance requires normalized finite probabilities")
    for i, token in enumerate(candidates):
        require_int(token, "candidate")
        if token >= proposal.shape[1] or proposal[i, token] <= 0:
            raise ValueError("candidate must have positive proposal probability")
    for i, token in enumerate(candidates):
        ratio = min(1.0, float(target[i, token] / proposal[i, token]))
        if float(torch.rand((), generator=generator)) >= ratio:
            residual = (target[i] - proposal[i]).clamp_min(0)
            sampled = int(torch.multinomial(residual, 1, generator=generator))
            return Acceptance((*candidates[:i], sampled), i)
    bonus = int(torch.multinomial(target[-1], 1, generator=generator))
    return Acceptance((*candidates, bonus), width)
