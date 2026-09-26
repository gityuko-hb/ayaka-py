"""Engine-facing sampling coordinator and state manager.

Bridges the scheduler's per-step plan (packed slices, one row per sampling
request) to the persistent slot store in `SamplingMetadata`:

  * The scheduler owns request lifecycles and calls `add`/`release` on
    admission and terminal settlement.
  * Each step calls `plan_for(slices)` to pin the active-row map (stable
    slot indices in packed sampling order) and to freeze the `SamplingPlan`
    before the model forward pass.
  * The runner calls `flush` then `sample` on the logits rows; sampled outputs
    remain device-resident until the completion boundary materializes them.
  * After `CompletionCoordinator` commits output tokens, the scheduler calls
    `record_published` so penalty state accounts for prompt and generated tokens
    (matching vLLM semantics) without optimistic recording or rollback.

Notes:
    Slot indices are stable for the lifetime of a request: compaction is forbidden
    because it would rewrite penalty and RNG states of live rows that are not
    sampled in the current step (such as during chunked prefill or mixed prefill/decode).

RNG policy (explicit contract, no speculative rollback):
    * The counter-based RNG state is one offset per request slot; every sampled
      row advances its own offset exactly once per sampling step, greedy rows
      included, and no operation rewinds an offset.
    * Cancellation, transient prepare failure, dropped candidate plans and
      discarded samples never rewind RNG state. A request that is re-admitted
      after a terminal outcome gets a fresh slot/seed from its new ordinal.
    * ``release`` resets the slot (penalties, bias, ids logprobs, RNG offset),
      so a reused slot cannot leak state into the next request.
    * Penalty history is recorded only from prompt admission and from committed
      token publication (``record_published``); recompute never re-records.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Protocol

import torch

from ayaka.distributed.device import CommOpType
from ayaka.plan import SamplingPlan
from ayaka.sampling.logprobs import MODE_ORDINALS, LogprobMode
from ayaka.sampling.metadata import SamplingMetadata
from ayaka.sampling.ops.penalties import BiasState, PenaltyState
from ayaka.sampling.ops.sampling import (
    SamplingSupportTensors,
    build_support_output,
    filter_probs,
    greedy_support_output,
)
from ayaka.sampling.params import SamplingParams
from ayaka.sampling.plan import SamplingPlanner, create_sampler
from ayaka.sampling.trace import trace_sampler
from ayaka.utils.import_utils import CapabilityError

__all__ = ["SamplingCoordinator"]


def _SYNC_TOKEN_IDS_ACROSS_TP_ENV() -> bool:
    return os.environ.get("AYAKA_SYNC_TOKEN_IDS_ACROSS_TP", "0").lower() in (
        "1",
        "true",
        "yes",
    )


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
        "_comm",
        "_custom_ops",
        "_free_slots",
        "_ids_logprobs",
        "_next_slot",
        "_request_to_slot",
        "_sampling_support_max_tokens",
        "_sync_group",
        "_sync_statuses",
        "bias_state",
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
        custom_ops: tuple[str, ...] = (),
        sampling_support_max_tokens: int = 512,
        sync_statuses: Callable[[torch.Tensor], torch.Tensor] | None = None,
        comm: Any | None = None,
        sync_group: Any | None = None,
        backend: str | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if sampling_support_max_tokens < 1:
            raise ValueError("sampling_support_max_tokens must be >= 1")
        self.max_batch_size = max_batch_size
        self._sampling_support_max_tokens = sampling_support_max_tokens
        self._sync_statuses = sync_statuses
        self._comm = comm
        self._sync_group = sync_group
        self._custom_ops = tuple(custom_ops)
        if self._custom_ops:
            from ayaka.kernel.ops import registered_ops

            registry = registered_ops()
            missing = [name for name in self._custom_ops if name not in registry]
            if missing:
                raise CapabilityError(
                    "sampling_custom_ops",
                    detail=f"custom ops not registered: {missing}",
                    remedy="register them via ayaka.kernel.ops.custom_op before "
                    "constructing the coordinator",
                )
        self.md = SamplingMetadata(max_batch_size, device=device)
        self.penalty_state = penalty_state or PenaltyState(
            max_batch_size,
            vocab_size=vocab_size,
            device=self.md.device,
        )
        self.planner = SamplingPlanner(max_batch_size)

        self.bias_state = BiasState(max_batch_size, vocab_size=vocab_size, device=self.md.device)
        self.sampler = create_sampler(
            backend,
            penalty_state=self.penalty_state,
            need_stats=need_stats,
            bias_state=self.bias_state,
        )
        self._request_to_slot: dict[str, int] = {}
        self._ids_logprobs: dict[str, tuple[int, ...]] = {}
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
        if params.logit_bias:
            self.bias_state.set(slot, dict(params.logit_bias))
        if params.token_ids_logprobs is not None:
            self._ids_logprobs[request_id] = tuple(params.token_ids_logprobs)
        return slot

    def release(self, request_id: str) -> int | None:
        """Drop a request and reset its slot; missing ids are a caller bug.

        RNG offsets, penalty/bias state, ids-logprobs and any attached mask
        producers are reset, so a released slot cannot leak per-request state
        into the next occupant. No other request's RNG stream is touched.
        """
        slot = self._request_to_slot.pop(request_id, None)
        if slot is None:
            return None
        self.penalty_state.reset(slot)
        self.bias_state.reset(slot)
        self._ids_logprobs.pop(request_id, None)
        self.planner.detach(slot)
        self.md.reset_slot(slot)
        self._free_slots.append(slot)
        return slot

    def slot_of(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    @property
    def custom_ops(self) -> tuple[str, ...]:
        """Sequence of custom operator identifiers applied to every sampling step."""
        return self._custom_ops

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
        """Pin active rows (packed sampling order) and freeze the plan.

        The active-row map is mirrored to the metadata device before returning.
        Sampling runs on the same engine thread right after planning, so a plan
        that left its rows dirty could be read against a stale row map. Flushing
        here keeps the device mirror and the frozen plan consistent even when a
        candidate plan is rebuilt (pressure shrink, transient retry) after a
        previous flush.
        """
        rows = [self._slot_for(slice_.request_id) for slice_ in slices if slice_.sample_last_query]
        self.md.set_active_rows(rows)
        try:
            plan, _schedule = self.planner.build(self.md, custom_ops=self._custom_ops)
            if plan.num_rows != len(rows):
                raise RuntimeError(
                    f"sampling plan has {plan.num_rows} rows but {len(rows)} slices sample"
                )
        finally:
            # Mirror even when the plan build fails: the row map must never stay
            # dirty past the planning boundary.
            self.flush()
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
        return self.sample_with_support(logits, plan, force_reference=force_reference)[0]

    def sample_with_support(
        self,
        logits: torch.Tensor,
        plan: SamplingPlan,
        *,
        force_reference: bool = False,
    ) -> tuple[torch.Tensor, SamplingSupportTensors | None]:
        """Sample one token per active row and capture distribution support for opt-in rows.

        Support capture is reporting-only: it executes after the sampling draw, does not
        consume RNG state, and never alters sampled token output. Returns None if no
        active row requested support capture in this step.

        Args:
            logits: Gathered logits tensor of shape `[plan.num_rows, vocab_size]`.
            plan: Frozen sampling plan declaring active batch requirements.
            force_reference: Whether to force reference kernel implementations.

        Returns:
            Tuple `(token_ids, support)` where `token_ids` contains sampled output tokens
            and `support` contains captured distribution support tensors or None.

        Raises:
            ValueError: If logits row count does not match `plan.num_rows`.
        """
        if logits.size(0) != plan.num_rows:
            raise ValueError(
                f"logits has {logits.size(0)} rows but the plan declares {plan.num_rows}; "
                "the runner must gather logits into packed sampling order"
            )
        trace_sampler("coordinator_flush_done", rows=plan.num_rows)
        output = self.sampler(logits, self.md, plan, None, force_reference=force_reference)
        self.md.step_rng()
        trace_sampler("coordinator_rng_stepped")
        support = self._capture_support(logits, output.token_ids, plan)
        return output.token_ids, support

    def _capture_support(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        plan: SamplingPlan,
    ) -> SamplingSupportTensors | None:
        """Reconstruct filtered token distribution in token-id space for opt-in rows.

        Only executes when `any_support_capture` is flagged in host staging to avoid
        unnecessary device synchronization. Logits at this stage are post-penalties and
        post-mask, but pre-temperature; temperature scaling is reapplied here to match
        the exact distribution evaluated during sampling.

        Args:
            logits: Logits tensor prior to temperature scaling.
            token_ids: Sampled token ids output from the sampler.
            plan: Frozen sampling plan for the current step.

        Returns:
            SamplingSupportTensors container for opt-in rows, or None.
        """
        if not self.md.any_support_capture:
            return None
        rows_opt = self.md.active("return_support").nonzero(as_tuple=True)[0]
        if rows_opt.numel() == 0:
            return None
        tok_opt = token_ids.index_select(0, rows_opt).to(torch.long)
        if plan.all_greedy:
            return greedy_support_output(tok_opt, rows_opt)

        temperature = self.md.active("temperature").index_select(0, rows_opt)
        scaled = logits.index_select(0, rows_opt) / temperature.clamp_min(1e-6).unsqueeze(1)
        top_k = self.md.active("top_k").index_select(0, rows_opt)
        top_p = self.md.active("top_p").index_select(0, rows_opt)
        min_p = self.md.active("min_p").index_select(0, rows_opt)

        # Greedy rows (temperature == 0; top_k == 1 naturally collapses to 1 survivor
        # in the filter, requiring no special branch): action space is restricted to the winner.
        greedy_rows = temperature == 0.0
        sp, si = filter_probs(scaled, top_k, top_p, min_p)
        by_id = torch.zeros_like(sp).scatter_(1, si, sp)
        winner = torch.zeros_like(by_id).scatter_(1, scaled.argmax(dim=-1, keepdim=True), 1.0)
        weights = torch.where(greedy_rows.unsqueeze(1), winner, by_id)
        support = build_support_output(
            weights,
            tok_opt,
            max_tokens=self._sampling_support_max_tokens,
            row_indices=rows_opt,
        )
        if self._sync_statuses is not None:
            support = replace(support, statuses=self._sync_statuses(support.statuses))
        return support

    def _sync_token_ids(
        self,
        token_ids: torch.Tensor,
        *,
        grammar_active: bool = False,
    ) -> None:
        """Synchronize sampled token ids across tensor-parallel ranks via MIN all-reduce.

        Mirrors SGLang semantics: by default, tensor-parallel token id synchronization is
        bypassed to save an all-reduce collective, relying on deterministic LM-head matmuls
        and sampling kernels. Synchronization is enabled via `AYAKA_SYNC_TOKEN_IDS_ACROSS_TP`
        or when structured grammar constraints are active to prevent TP rank desynchronization.

        Args:
            token_ids: Tensor-resident token ids modified in place via collective all-reduce.
            grammar_active: Whether grammar constraints are actively enforced in this step.
        """
        if self._comm is None or self._sync_group is None:
            return
        if not (_SYNC_TOKEN_IDS_ACROSS_TP_ENV() or grammar_active):
            return
        self._comm.all_reduce(token_ids, self._sync_group, op=CommOpType.MIN)

    def sync_token_ids(
        self,
        token_ids: torch.Tensor,
        *,
        grammar_active: bool = False,
    ) -> torch.Tensor:
        """Synchronize token ids across TP ranks if requested or required by active grammar.

        Args:
            token_ids: Tensor of sampled token ids.
            grammar_active: Whether grammar constraints are active for this step.

        Returns:
            Synchronized token ids tensor.
        """
        self._sync_token_ids(token_ids, grammar_active=grammar_active)
        return token_ids

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

    def token_ids_logprobs_for(self, request_id: str) -> tuple[int, ...] | None:
        """Return explicit token ids configured for logprob reporting, or None if disabled.

        Args:
            request_id: Request identifier string.

        Returns:
            Tuple of target token ids or None if not configured.
        """
        return self._ids_logprobs.get(request_id)

    def _slot_for(self, request_id: str) -> int:
        slot = self._request_to_slot.get(request_id)
        if slot is None:
            raise KeyError(
                f"request {request_id!r} has no sampling slot; the scheduler must "
                "call add() before it can appear in a step"
            )
        return slot
