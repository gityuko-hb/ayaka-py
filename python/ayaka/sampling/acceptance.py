"""Greedy and stochastic speculative acceptance oracles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ayaka.utils.validation import require_int


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
