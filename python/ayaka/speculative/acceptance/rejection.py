"""Speculative rejection sampling (Leviathan et al., ICML 2023, Algorithm 1).

``accept_stochastic`` is the CPU golden oracle with an explicit generator. The
batched device form lives beside it so both share one definition of the
residual distribution, including the degenerate case.
"""

from __future__ import annotations

import torch

from ayaka.speculative.acceptance.greedy import Acceptance
from ayaka.utils.validation import require_int

__all__ = ["accept_stochastic", "residual_distribution"]


def residual_distribution(target: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    """Normalized ``max(p - q, 0)`` per row, falling back to ``p`` when it is zero.

    The residual sums to zero only when ``p <= q`` everywhere, i.e. ``p == q``
    for normalized rows; rejection then has probability zero, so any choice is
    distributionally exact. Falling back to ``p`` (never to a fixed token)
    keeps numerically degenerate rows unbiased.
    """
    if target.shape != proposal.shape:
        raise ValueError("target and proposal distributions must share a shape")
    residual = (target - proposal).clamp_min(0)
    mass = residual.sum(dim=-1, keepdim=True)
    safe = torch.where(
        mass > 0, residual / mass.clamp_min(torch.finfo(residual.dtype).tiny), target
    )
    return safe


def accept_stochastic(
    candidates: tuple[int, ...],
    proposal: torch.Tensor,
    target: torch.Tensor,
    *,
    generator: torch.Generator,
) -> Acceptance:
    """Reference rejection sampling over CPU probabilities.

    ``proposal`` is ``q[K, V]`` (the drafter's distribution per draft row) and
    ``target`` is ``p[K + 1, V]``. Draft ``i`` is accepted with probability
    ``min(1, p_i(x_i) / q_i(x_i))``; the first rejection samples from the
    normalized residual, and full acceptance samples a bonus from ``p_K``.
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
            residual = residual_distribution(target[i], proposal[i])
            sampled = int(torch.multinomial(residual, 1, generator=generator))
            return Acceptance((*candidates[:i], sampled), i)
    bonus = int(torch.multinomial(target[-1], 1, generator=generator))
    return Acceptance((*candidates, bonus), width)
