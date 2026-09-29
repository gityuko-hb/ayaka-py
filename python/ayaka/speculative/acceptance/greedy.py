"""Greedy speculative acceptance: host oracle and batched device form."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Acceptance", "accept_greedy", "greedy_accept"]


@dataclass(frozen=True, slots=True)
class Acceptance:
    """Emitted tokens (accepted drafts + one target token) and the draft count kept."""

    tokens: tuple[int, ...]
    accepted: int


def accept_greedy(candidates: tuple[int, ...], target_tokens: tuple[int, ...]) -> Acceptance:
    """Accept the longest equal prefix, then emit target correction or bonus.

    This is the golden oracle for :func:`greedy_accept`; it never touches a
    device.
    """
    if len(target_tokens) != len(candidates) + 1:
        raise ValueError("verification needs candidate rows plus a bonus row")
    accepted = 0
    for draft, target in zip(candidates, target_tokens, strict=False):
        if draft != target:
            break
        accepted += 1
    return Acceptance((*candidates[:accepted], target_tokens[accepted]), accepted)


def greedy_accept(drafts: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Longest matching prefix per row, entirely on ``targets.device``.

    Args:
        drafts: ``[R, K]`` draft ids, right-padded with a value that is never
            a token id (``-1``), so a padded column always rejects.
        targets: ``[R, K + 1]`` target token per verification row; column
            ``i`` is the target's choice after ``i`` accepted drafts.

    Returns:
        ``(accepted [R] int64, final [R])``: accepted draft count and the target
        token emitted after them (correction on rejection, bonus after full
        acceptance). No host synchronization occurs.
    """
    if drafts.dim() != 2 or targets.dim() != 2:
        raise ValueError("drafts must be [R, K] and targets [R, K + 1]")
    rows, width = drafts.shape
    if targets.shape != (rows, width + 1):
        raise ValueError("targets must have exactly one more column than drafts")
    if drafts.device != targets.device:
        raise ValueError("drafts and targets must share a device")
    if width == 0:
        accepted = torch.zeros(rows, dtype=torch.long, device=targets.device)
        return accepted, targets[:, 0]
    matches = drafts.to(targets.dtype) == targets[:, :width]
    accepted = torch.cumprod(matches.to(torch.long), dim=1).sum(dim=1)
    final = targets.gather(1, accepted.unsqueeze(1)).squeeze(1)
    return accepted, final
