"""Pre-sample token bans shared by the sampling runners.

Token-level bans (EOS and stop token ids before ``min_tokens``) must be enforced
before the draw. The current applier writes ``-inf`` directly into the logits
rows; the :class:`BanApplier` protocol exists so a future Tier-1 mask applier
(``ayaka.sampling.mask``) can replace the fill without touching the runners.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from ayaka.sched.plan import BatchStepPlan

__all__ = ["BanApplier", "BansProvider", "LogitFillBanApplier"]


@runtime_checkable
class BanApplier(Protocol):
    """Apply per-row banned token ids to packed ``[rows, vocab]`` logits."""

    def apply(self, logits: torch.Tensor, bans: Sequence[frozenset[int]]) -> None: ...


@runtime_checkable
class BansProvider(Protocol):
    """Per-step source of pre-sample bans in packed sampling order."""

    def __call__(self, step: BatchStepPlan) -> tuple[frozenset[int], ...]: ...


class LogitFillBanApplier:
    """In-place ``-inf`` fill of banned token ids, one row at a time."""

    __slots__ = ()

    def apply(self, logits: torch.Tensor, bans: Sequence[frozenset[int]]) -> None:
        if logits.dim() != 2:
            raise ValueError("bans apply to a 2-D [rows, vocab] logits tensor")
        if len(bans) != logits.size(0):
            raise ValueError(f"got {len(bans)} ban sets for {logits.size(0)} logits rows")
        vocab = logits.size(1)
        for row, ids in enumerate(bans):
            if not ids:
                continue
            blocked = [token for token in sorted(ids) if 0 <= token < vocab]
            if not blocked:
                continue
            index = torch.tensor(blocked, dtype=torch.long, device=logits.device)
            logits[row].index_fill_(0, index, float("-inf"))
