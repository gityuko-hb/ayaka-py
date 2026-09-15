"""Sampling planner and sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ayaka.caps import Cap
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
    """Các entry mask của step — MỘT entry per (slot, producer).

    A4b — một slot có thể mang NHIỀU producer: entry đầu (thứ tự khai báo)
    là primary (window chính, ghi row_indices), các entry sau là scratch
    (emit riêng rồi intersect AND vào window chính).
    """

    entries: tuple[MaskEntry, ...] = ()

    @property
    def num_rows(self) -> int:
        return sum(e.propose_step + 1 for e in self.entries if not e.scratch)

    @property
    def num_scratch_rows(self) -> int:
        return sum(e.propose_step + 1 for e in self.entries if e.scratch)

    @property
    def caps(self) -> Cap:
        """Aggregate theo AND: capability chỉ đúng khi MỌI producer có."""
        return _aggregate_caps(self.entries)

    def matches(self, plan: SamplingPlan) -> bool:
        return self.num_rows == plan.num_mask_rows

    def require_match(self, plan: SamplingPlan) -> None:
        if not self.matches(plan):
            raise ValueError(
                f"mask schedule has {self.num_rows} rows but the plan declares "
                f"{plan.num_mask_rows}; the arena would be sized for the wrong shape"
            )


def _aggregate_caps(entries: tuple[MaskEntry, ...]) -> Cap:
    caps = Cap.ARGMAX_INVARIANT | Cap.SPEC_VERIFIABLE | Cap.COMMUTATIVE
    for entry in entries:
        producer_caps = getattr(entry.producer, "caps", Cap.NONE)
        caps &= producer_caps
    return caps


@dataclass(slots=True)
class SamplerOutput:
    token_ids: torch.Tensor
    stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


class SamplingPlanner:
    """Attach NHIỀU producer per slot (A4b); build freeze plan + schedule."""

    __slots__ = ("_producers", "max_batch_size")

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        self.max_batch_size = max_batch_size
        self._producers: dict[int, list[MaskProducer]] = {}

    def attach(self, slot: int, producer: MaskProducer) -> None:
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._producers.setdefault(slot, []).append(producer)

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
        drafts = drafts or {}
        commits = commits or {}

        # Tier-2: resolve custom ops qua registry — op chưa đăng ký là lỗi
        # capability (không còn raise vô điều kiện với MỌI custom op).
        if custom_ops:
            from ayaka.kernel.ops import registered_ops

            registry = registered_ops()
            resolved_caps: list[Cap] = []
            for op_name in custom_ops:
                handle = registry.get(op_name)
                if handle is None:
                    raise CapabilityError(
                        "sampling_custom_ops",
                        detail=(
                            f"plan requests custom op {op_name!r} but it is "
                            "not registered in ayaka.kernel.ops"
                        ),
                        remedy=(
                            f"đăng ký {op_name} qua ayaka.kernel.ops.custom_op "
                            "(caps= nên khai rõ CUDAGRAPH_SAFE nếu op capture "
                            "được), hoặc drop custom_ops khỏi plan"
                        ),
                    )
                resolved_caps.append(handle.caps)
            custom_ops_caps: tuple[Cap, ...] = tuple(resolved_caps)
        else:
            custom_ops_caps = ()

        entries: list[MaskEntry] = []
        mask_row = 0
        scratch_row = 0  # cấp sau toàn bộ primary block
        argmax_invariant = True
        for slot in sorted(self._producers):
            if slot >= md.n_active:
                continue
            draft = drafts.get(slot, ())
            if draft and len(draft) > 0:
                # Tier-1: draft của producer thiếu SPEC_VERIFIABLE → từ chối.
                for producer in self._producers[slot]:
                    caps = getattr(producer, "caps", Cap.NONE)
                    if not caps & Cap.SPEC_VERIFIABLE:
                        raise CapabilityError(
                            "sampling_speculative_draft",
                            detail=(
                                f"slot {slot} có draft {len(draft)} token nhưng "
                                "producer thiếu Cap.SPEC_VERIFIABLE"
                            ),
                            remedy=(
                                "drop draft cho slot này hoặc dùng producer "
                                "hỗ trợ speculative verification"
                            ),
                        )
            producers = self._producers[slot]
            commit_tokens = commits.get(slot, ())
            for order, producer in enumerate(producers):
                span = len(draft) + 1
                if order == 0:
                    entry = MaskEntry(
                        producer=producer,
                        logits_row=slot,
                        mask_row=mask_row,
                        stream_idx=slot,
                        propose_step=len(draft),
                        draft=draft,
                        commit_tokens=commit_tokens,
                    )
                    mask_row += span
                else:
                    entry = MaskEntry(
                        producer=producer,
                        logits_row=slot,
                        mask_row=(mask_row + scratch_row),
                        stream_idx=slot,
                        propose_step=len(draft),
                        draft=draft,
                        commit_tokens=commit_tokens,
                        scratch=True,
                    )
                    scratch_row += span
                entries.append(entry)
                if not getattr(producer, "caps", Cap.NONE) & Cap.ARGMAX_INVARIANT:
                    argmax_invariant = False

        schedule = MaskSchedule(entries=tuple(entries))
        plan = SamplingPlan(
            num_rows=md.n_active,
            num_mask_rows=schedule.num_rows,
            all_greedy=md.all_greedy,
            any_penalty=md.any_penalty,
            custom_ops=custom_ops,
            custom_ops_caps=custom_ops_caps,
            argmax_invariant=argmax_invariant and bool(self._producers),
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
             [A4 fast-path: mọi producer ARGMAX_INVARIANT + all_greedy →
              argmax trên logits CHƯA mask; winner bị bitmask chặn → fallback
              đường apply-mask đầy đủ]
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
                    "custom-op executor has not been ported"
                ),
                remedy="drop custom_ops from the plan, or port ayaka.sampling.custom",
            )

        if plan.any_penalty and self.penalty_state is not None:
            _pen.apply_penalties_(logits, md, self.penalty_state)

        if mask is not None:
            fast_path = plan.argmax_invariant and plan.all_greedy and not self.need_stats
            if fast_path:
                winner = _greedy_winner_with_mask_check(logits, mask)
                if winner is not None:
                    # Mọi winner đều allowed → mask không đổi argmax ⇒ bỏ
                    # qua apply O(n×V). RNG offset vẫn do caller advance.
                    return SamplerOutput(token_ids=winner, stats=None)
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


def _greedy_winner_with_mask_check(logits: torch.Tensor, mask: MaskHandle) -> torch.Tensor | None:
    """Argmax unmasked + kiểm winner trong bitmask; None nếu có row vi phạm
    (caller fallback apply-mask). argmax trên logits GỐC — scaling temperature
    không đổi argmax (clamp 1e-6 chỉ chạm temperature == 0 → chia constant)."""
    winner = logits.argmax(dim=-1)
    masks = mask.masks
    rows = mask.row_indices.to(torch.long)
    if masks.device != winner.device:
        masks = masks.to(winner.device)
        rows = rows.to(winner.device)
    word = torch.div(winner, 32, rounding_mode="floor")
    bit = (winner % 32).to(torch.int32)
    sel = masks.index_select(0, rows).gather(1, word.unsqueeze(1)).squeeze(1)
    allowed = ((sel.to(torch.int32) >> bit) & 1).to(torch.bool)
    if bool(allowed.all()):
        return winner
    return None
