"""Sampling planner and sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ayaka.plan import SamplingPlan
from ayaka.sampling.mask.arena import MaskHandle
from ayaka.sampling.mask.pipeline import MaskEntry
from ayaka.sampling.mask.producer import MaskProducer
from ayaka.sampling.metadata import SamplingMetadata
from ayaka.sampling.ops import penalties as _pen
from ayaka.sampling.ops.bitmask import apply_allow_bitmask_
from ayaka.sampling.ops.topk_topp import softmax_stats_scaled, topk_topp_sample
from ayaka.utils.import_utils import CapabilityError

__all__ = ["MaskSchedule", "Sampler", "SamplerOutput", "SamplingPlanner"]


@dataclass(frozen=True, slots=True)
class MaskSchedule:
    entries: tuple[MaskEntry, ...] = ()

    @property
    def num_rows(self) -> int:
        return sum(e.propose_step + 1 for e in self.entries)

    def matches(self, plan: SamplingPlan) -> bool:
        return self.num_rows == plan.num_mask_rows

    def require_match(self, plan: SamplingPlan) -> None:
        if not self.matches(plan):
            raise ValueError(
                f"mask schedule has {self.num_rows} rows but the plan declares "
                f"{plan.num_mask_rows}; the arena would be sized for the wrong shape"
            )


@dataclass(slots=True)
class SamplerOutput:
    token_ids: torch.Tensor
    stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


class SamplingPlanner:
    __slots__ = ("_producers", "max_batch_size")

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        self.max_batch_size = max_batch_size
        self._producers: dict[int, MaskProducer] = {}

    def attach(self, slot: int, producer: MaskProducer) -> None:
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._producers[slot] = producer

    def detach(self, slot: int) -> None:
        self._producers.pop(slot, None)

    def move(self, src: int, dst: int) -> None:
        if src in self._producers:
            self._producers[dst] = self._producers.pop(src)

    def build(
        self,
        md: SamplingMetadata,
        *,
        drafts: dict[int, tuple[int, ...]] | None = None,
        commits: dict[int, tuple[int, ...]] | None = None,
        custom_ops: tuple[str, ...] = (),
    ) -> tuple[SamplingPlan, MaskSchedule]:
        if custom_ops:
            raise CapabilityError(
                "sampling_custom_ops",
                detail=(
                    f"plan requests custom ops {custom_ops}, but the sampling "
                    "custom-op registry has not been ported"
                ),
                remedy="drop custom_ops from the plan, or port ayaka.sampling.custom",
            )
        drafts = drafts or {}
        commits = commits or {}
        entries: list[MaskEntry] = []
        mask_row = 0
        for slot in sorted(self._producers):
            if slot >= md.n_active:
                continue
            draft = drafts.get(slot, ())
            entries.append(
                MaskEntry(
                    producer=self._producers[slot],
                    logits_row=slot,
                    mask_row=mask_row,
                    stream_idx=slot,
                    propose_step=len(draft),
                    draft=draft,
                    commit_tokens=commits.get(slot, ()),
                )
            )
            mask_row += len(draft) + 1

        schedule = MaskSchedule(entries=tuple(entries))
        plan = SamplingPlan(
            num_rows=md.n_active,
            num_mask_rows=schedule.num_rows,
            all_greedy=md.all_greedy,
            any_penalty=md.any_penalty,
            custom_ops=custom_ops,
        )
        schedule.require_match(plan)
        return plan, schedule


class Sampler:
    __slots__ = ("need_stats", "penalty_state")

    def __init__(
        self,
        penalty_state: _pen.PenaltyState | None = None,
        need_stats: bool = False,
    ) -> None:
        self.penalty_state = penalty_state
        self.need_stats = need_stats

    def __call__(
        self,
        logits: torch.Tensor,
        md: SamplingMetadata,
        plan: SamplingPlan,
        mask: MaskHandle | None = None,
        *,
        force_reference: bool = False,
    ) -> SamplerOutput:
        """Canonical sampling order, the single runtime semantics (M0).

        raw logits
          -> penalties (rep/freq/pres, in-place)
          -> allowed-token / grammar mask (in-place, NEG_INF)
          -> temperature scaling (exactly once, before stats and filtering)
          -> sampling-distribution stats
          -> greedy argmax | top-k/top-p/min-p + sample
          -> per-row greedy override for mixed batches

        Greedy is NOT an early exit before penalty/mask: those transforms can
        change argmax. Greedy only skips the stochastic filter/sample stage.
        Fail-closed validation happens before any in-place mutation.
        """
        if logits.size(0) != plan.num_rows:
            raise ValueError(f"logits has {logits.size(0)} rows, plan declares {plan.num_rows}")
        if (mask is not None) != plan.any_mask:
            raise ValueError(
                f"mask={'present' if mask is not None else 'None'} disagrees with "
                f"plan.any_mask={plan.any_mask}"
            )
        if plan.custom_ops:
            raise CapabilityError(
                "sampling_custom_ops",
                detail=(
                    f"plan requires custom ops {plan.custom_ops}, but the sampling "
                    "custom-op registry has not been ported"
                ),
                remedy="drop custom_ops from the plan, or port ayaka.sampling.custom",
            )

        if plan.any_penalty and self.penalty_state is not None:
            _pen.apply_penalties_(logits, md, self.penalty_state)

        if mask is not None:
            apply_allow_bitmask_(logits, mask.masks, mask.row_indices, mask.vocab_size)

        temperature = md.active("temperature")
        scaled = logits / temperature.clamp_min(1e-6).unsqueeze(1)

        stats = softmax_stats_scaled(scaled) if self.need_stats else None

        if plan.all_greedy:
            return SamplerOutput(token_ids=scaled.argmax(dim=-1), stats=stats)

        tok = topk_topp_sample(
            scaled,
            md.active("top_k"),
            md.active("top_p"),
            md.active("min_p"),
            md.active("seed"),
            md.active("offset"),
            force_reference=force_reference,
        )
        # Mixed batches need per-row greedy: a temperature-0 row must be argmax
        # (first maximal index), not a softmax draw among tied maxima at 1e-6.
        # vLLM splits the same way (is_greedy per request). RNG offsets still
        # advance one per row per step, so reproducibility is unaffected.
        greedy_rows = (temperature == 0.0) | (md.active("top_k") == 1)
        if bool(greedy_rows.any()):
            tok = torch.where(greedy_rows, scaled.argmax(dim=-1).to(tok.dtype), tok)
        return SamplerOutput(token_ids=tok, stats=stats)

    forward = __call__
