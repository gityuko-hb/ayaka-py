"""Engine-facing sampling coordinator (K06 P0).

Bridges the scheduler's per-step plan (packed slices, one row per sampling
request) to the persistent slot store in ``SamplingMetadata``:

  * the scheduler owns request lifecycle and calls ``add``/``release`` on
    admission/terminal settlement;
  * each step calls ``plan_for(slices)`` to pin the active-row map -- stable
    slot indices in packed sampling order -- and to freeze the ``SamplingPlan``
    *before* the forward pass;
  * the runner calls ``flush`` then ``sample`` on the logits rows; results stay
    device-resident until the completion boundary materializes them;
  * after ``CompletionCoordinator`` commits output tokens, the scheduler calls
    ``record_published`` so penalty state sees prompt + output tokens (vLLM
    semantics) without optimistic recording or rollback.

Slot indices are stable for the life of a request: compaction is forbidden
because it would rewrite the penalty/RNG state of live rows that are not
sampled in the current step (chunked prefill, mixed PREFILL+DECODE).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch

from ayaka.plan import SamplingPlan
from ayaka.sampling.logprobs import MODE_ORDINALS, LogprobMode
from ayaka.sampling.metadata import SamplingMetadata
from ayaka.sampling.ops.penalties import PenaltyState
from ayaka.sampling.params import SamplingParams
from ayaka.sampling.plan import Sampler, SamplingPlanner

__all__ = ["SamplingCoordinator"]


class _SamplingSlice(Protocol):
    """Structural view of ``ayaka.sched.plan.ScheduledSlice`` (no sched import)."""

    @property
    def request_id(self) -> str: ...

    @property
    def sample_last_query(self) -> bool: ...


class _PublishedToken(Protocol):
    """Structural view of ``ayaka.executor.completion.PublishedSample``."""

    @property
    def request_id(self) -> str: ...

    @property
    def token_id(self) -> int: ...


def _needs_penalty(params: SamplingParams) -> bool:
    return (
        params.repetition_penalty != 1.0
        or params.frequency_penalty != 0.0
        or params.presence_penalty != 0.0
    )


class SamplingCoordinator:
    """Owns slots, penalty state, the planner and the sampler for one engine."""

    __slots__ = (
        "_free_slots",
        "_next_slot",
        "_request_to_slot",
        "max_batch_size",
        "md",
        "penalty_state",
        "planner",
        "sampler",
    )

    def __init__(
        self,
        max_batch_size: int,
        *,
        device: torch.device | None = None,
        vocab_size: int | None = None,
        penalty_state: PenaltyState | None = None,
        need_stats: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        self.max_batch_size = max_batch_size
        self.md = SamplingMetadata(max_batch_size, device=device)
        self.penalty_state = penalty_state or PenaltyState(
            max_batch_size,
            vocab_size=vocab_size,
            device=self.md.device,
        )
        self.planner = SamplingPlanner(max_batch_size)
        self.sampler = Sampler(penalty_state=self.penalty_state, need_stats=need_stats)
        self._request_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0

    # ------------------------------------------------------------------
    # Slot registry (called by the scheduler)
    # ------------------------------------------------------------------

    def add(
        self,
        request_id: str,
        params: SamplingParams,
        prompt_token_ids: Sequence[int] = (),
        *,
        request_index: int = 0,
    ) -> int:
        """Register a request, write its columns, and count prompt penalties.

        ``request_index`` only feeds ``derive_seed`` when the request has no
        explicit seed; callers pass a stable per-request ordinal so rows never
        share an implicit RNG stream.
        """
        if request_id in self._request_to_slot:
            raise ValueError(
                f"request {request_id!r} already owns sampling slot "
                f"{self._request_to_slot[request_id]}"
            )
        slot = self._allocate()
        self._request_to_slot[request_id] = slot
        self.md.write_slot(slot, params, request_index=request_index)
        if _needs_penalty(params):
            self.penalty_state.record(slot, [int(token) for token in prompt_token_ids])
        return slot

    def release(self, request_id: str) -> int | None:
        """Drop a request and reset its slot; missing ids are a caller bug."""
        slot = self._request_to_slot.pop(request_id, None)
        if slot is None:
            return None
        self.penalty_state.reset(slot)
        self.md.reset_slot(slot)
        self._free_slots.append(slot)
        return slot

    def slot_of(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._request_to_slot

    def _allocate(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot >= self.max_batch_size:
            raise RuntimeError(
                f"sampling slot capacity {self.max_batch_size} exhausted; the "
                "scheduler must bound admission by max_batch_size"
            )
        slot = self._next_slot
        self._next_slot += 1
        return slot

    # ------------------------------------------------------------------
    # Per-step planning / device I/O (called by the runner)
    # ------------------------------------------------------------------

    def plan_for(self, slices: Sequence[_SamplingSlice]) -> SamplingPlan:
        """Pin active rows (packed sampling order) and freeze the plan."""
        rows = [self._slot_for(slice_.request_id) for slice_ in slices if slice_.sample_last_query]
        self.md.set_active_rows(rows)
        plan, _schedule = self.planner.build(self.md)
        if plan.num_rows != len(rows):
            raise RuntimeError(
                f"sampling plan has {plan.num_rows} rows but {len(rows)} slices sample"
            )
        return plan

    def flush(self, stream: Any = None) -> int:
        """Mirror dirty columns and the active-row map to the device."""
        return self.md.flush(stream)

    def sample(
        self,
        logits: torch.Tensor,
        plan: SamplingPlan,
        *,
        force_reference: bool = False,
    ) -> torch.Tensor:
        """Sample one token per active row; advances each row's RNG offset.

        Returns device-resident token ids in packed sampling order. The
        completion boundary materializes host values; nothing here syncs.
        """
        if logits.size(0) != plan.num_rows:
            raise ValueError(
                f"logits has {logits.size(0)} rows but the plan declares {plan.num_rows}; "
                "the runner must gather logits into packed sampling order"
            )
        output = self.sampler(logits, self.md, plan, None, force_reference=force_reference)
        self.md.step_rng()
        return output.token_ids

    def record_published(self, published: Sequence[_PublishedToken]) -> None:
        """Count committed output tokens for penalties (prompt counted at add)."""
        for item in published:
            slot = self._request_to_slot.get(item.request_id)
            if slot is None:
                raise KeyError(
                    f"published token for {item.request_id!r} but the request no longer "
                    "owns a sampling slot; record before terminal settlement"
                )
            self.penalty_state.record(slot, (int(item.token_id),))

    def logprob_k_for(self, request_id: str) -> int | None:
        """Generation-logprobs top-k for one request (host staging); None = off."""
        slot = self._request_to_slot.get(request_id)
        if slot is None:
            raise KeyError(f"request {request_id!r} has no sampling slot")
        value = int(self.md.host_scalar("logprobs_k", slot))
        return None if value < 0 else value

    def logprob_mode_for(self, request_id: str) -> LogprobMode:
        """Logprob reporting mode for one request (host staging)."""
        slot = self._request_to_slot.get(request_id)
        if slot is None:
            raise KeyError(f"request {request_id!r} has no sampling slot")
        ordinal = int(self.md.host_scalar("logprob_mode", slot))
        for mode, value in MODE_ORDINALS.items():
            if value == ordinal:
                return mode
        raise ValueError(f"unknown logprob_mode ordinal {ordinal} for {request_id!r}")

    def _slot_for(self, request_id: str) -> int:
        slot = self._request_to_slot.get(request_id)
        if slot is None:
            raise KeyError(
                f"request {request_id!r} has no sampling slot; the scheduler must "
                "call add() before it can appear in a step"
            )
        return slot
