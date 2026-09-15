"""LogitsPlan + LogitsProcessor: hidden rows -> model logits (M1).

Responsibility boundary kept across the whole feature:
  the scheduler decides token positions; ``LogitsPlan`` decides which hidden
  rows need projection; the ``LogitsProcessor`` only produces logits; the
  ``LogprobProcessor`` only observes the distribution; only the ``Sampler``
  transforms it to pick tokens.

Projection happens BEFORE the LM head: with a plan of R rows, the head runs on
``[R, hidden]`` instead of the whole batch. The head is reached only through
the model's public interface (``logits_from_hidden``) — a quantized head must
flow through the ``QuantizationTarget.LM_HEAD`` contract/lifecycle, never a
private weight matmul here.

Model-owned normalization (logit scale, final softcap) belongs to this
processor because it is part of "what the model's logits are": the raw
logprob semantics are defined AFTER these transforms and BEFORE any sampling
transform. Accumulation stays FP32. Tensor-parallel sharded vocabulary is out
of scope for M1 (single device); the output is the full vocabulary for every
projection row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ayaka.utils.validation import require_int

__all__ = ["LogitsPlan", "LogitsProcessor"]


@dataclass(frozen=True, slots=True)
class LogitsPlan:
    """Which hidden rows need projection, decided before the forward pass.

    ``projection_rows`` indexes the packed hidden layout the runner holds.
    Sampling rows are the plan's PREFIX: the sampler consumes rows
    ``[0, sampling_row_count)`` in packed sampling order; logprob report rows
    (M2 prompt scoring) may extend beyond the prefix.
    """

    projection_rows: tuple[int, ...]
    sampling_row_count: int

    def __post_init__(self) -> None:
        if type(self.projection_rows) is not tuple:
            raise TypeError("projection_rows must be a tuple")
        require_int(self.sampling_row_count, "sampling_row_count")
        seen: set[int] = set()
        for row in self.projection_rows:
            require_int(row, "projection row")
            if row in seen:
                raise ValueError("projection rows must be unique")
            seen.add(row)
        if self.sampling_row_count > len(self.projection_rows):
            raise ValueError(
                f"sampling_row_count {self.sampling_row_count} exceeds "
                f"{len(self.projection_rows)} projection rows"
            )

    @property
    def num_rows(self) -> int:
        return len(self.projection_rows)


class LogitsProcessor:
    """Gathers hidden rows per plan, runs the LM head, applies model transforms."""

    __slots__ = ("_logit_scale", "_model", "_softcap")

    def __init__(
        self,
        model,
        *,
        logit_scale: float = 1.0,
        final_logit_softcapping: float | None = None,
    ) -> None:
        if logit_scale <= 0.0:
            raise ValueError("logit_scale must be > 0")
        if final_logit_softcapping is not None and final_logit_softcapping <= 0.0:
            raise ValueError("final_logit_softcapping must be > 0 when set")
        self._model = model
        self._logit_scale = float(logit_scale)
        self._softcap = final_logit_softcapping

    def __call__(self, hidden: torch.Tensor, plan: LogitsPlan) -> torch.Tensor:
        """Project exactly the plan's rows; returns ``[num_rows, vocab]``."""
        if hidden.dim() != 2:
            raise ValueError(f"hidden must be 2-D, got {tuple(hidden.shape)}")
        if plan.num_rows == 0:
            raise ValueError("logits plan has no rows")
        if plan.num_rows > hidden.size(0):
            raise ValueError(f"plan needs {plan.num_rows} rows but hidden has {hidden.size(0)}")
        rows = torch.tensor(plan.projection_rows, dtype=torch.long, device=hidden.device)
        projected = hidden.index_select(0, rows)
        logits = self._model.logits_from_hidden(projected)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits_from_hidden must return a torch.Tensor")
        if logits.size(0) != plan.num_rows:
            raise ValueError(
                f"LM head produced {logits.size(0)} rows for {plan.num_rows} projection rows"
            )
        return self._normalize(logits)

    def _normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """Model-owned transforms; FP32 accumulation for the softcap path."""
        out = logits
        if self._softcap is not None:
            cap = self._softcap
            out = cap * torch.tanh(out.to(torch.float32) / cap)
        if self._logit_scale != 1.0:
            out = out * self._logit_scale
        return out
