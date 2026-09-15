"""Sampling planner and sampler."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ayaka.caps import Cap
from ayaka.plan import SamplingPlan
from ayaka.sampling.executor import apply_custom_ops
from ayaka.sampling.mask.arena import MaskHandle
from ayaka.sampling.mask.pipeline import MaskEntry
from ayaka.sampling.mask.producer import MaskProducer
from ayaka.sampling.metadata import SamplingMetadata
from ayaka.sampling.ops import penalties as _pen
from ayaka.sampling.ops.penalties import PenaltyState, apply_bias_, apply_penalties_
from ayaka.sampling.ops.sampling import (
    apply_allow_bitmask_,
    gumbel_sample,
    softmax_stats_scaled,
    topk_first_hint,
    topk_topp_sample,
)
from ayaka.utils.import_utils import CapabilityError

__all__ = ["MaskSchedule", "Sampler", "SamplerOutput", "SamplingPlanner"]


def _trace_enabled() -> bool:
    """Check whether end-to-end sampler tracing is enabled in the environment."""
    return os.environ.get("AYAKA_TRACE_SAMPLER_E2E", "0").lower() in ("1", "true", "yes")


def _rank_prefix() -> str:
    """Construct distributed rank prefix string for formatted trace output."""
    try:
        from ayaka.distributed.env import local_rank, rank, world_size

        return f"ws={world_size()} rank={rank()} local={local_rank()}"
    except Exception:  # pragma: no cover - defensive, prefix only
        return "rank=unknown"


def trace_sampler(stage: str, **fields: Any) -> None:
    """Emit a structured trace log entry for a sampling pipeline stage.

    When `AYAKA_TRACE_SAMPLER_E2E` is enabled, formats and prints the stage name,
    distributed rank information, and key-value fields to stdout with immediate flush.
    Acts as a no-op when tracing is disabled.

    Args:
        stage: Identifier string of the current sampling execution stage.
        **fields: Arbitrary key-value attributes associated with the stage event.
    """
    # Fast exit when tracing is disabled.
    if not _trace_enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"AYAKA_TRACE_SAMPLER {_rank_prefix()} stage={stage}{suffix}", flush=True)


@dataclass(frozen=True, slots=True)
class MaskSchedule:
    """Schedule of mask entries to materialize for a sampling step.

    A single slot may be bound to multiple mask producers: the initial entry
    (in registration order) serves as the primary window recorded in `row_indices`,
    while subsequent entries represent scratch space emitted independently and
    intersected via bitwise AND into the primary window.

    Attributes:
        entries: Ordered sequence of mask entries scheduled for the step.
    """

    entries: tuple[MaskEntry, ...] = ()

    @property
    def num_rows(self) -> int:
        """Return total primary mask rows across all non-scratch scheduled entries."""
        return sum(e.propose_step + 1 for e in self.entries if not e.scratch)

    @property
    def num_scratch_rows(self) -> int:
        """Return total scratch mask rows scheduled across secondary producers."""
        return sum(e.propose_step + 1 for e in self.entries if e.scratch)

    @property
    def caps(self) -> Cap:
        """Aggregate producer capabilities via bitwise AND across all entries."""
        return _aggregate_caps(self.entries)

    def matches(self, plan: SamplingPlan) -> bool:
        """Check whether the schedule row count matches the declared plan row count."""
        return self.num_rows == plan.num_mask_rows

    def require_match(self, plan: SamplingPlan) -> None:
        """Assert that the schedule row count matches the plan; raise otherwise."""
        if not self.matches(plan):
            raise ValueError(
                f"mask schedule has {self.num_rows} rows but the plan declares "
                f"{plan.num_mask_rows}; the arena would be sized for the wrong shape"
            )


def _aggregate_caps(entries: tuple[MaskEntry, ...]) -> Cap:
    """Compute aggregate capability flags by intersecting all producer capabilities."""
    caps = Cap.ARGMAX_INVARIANT | Cap.SPEC_VERIFIABLE | Cap.COMMUTATIVE
    for entry in entries:
        producer_caps = getattr(entry.producer, "caps", Cap.NONE)
        caps &= producer_caps
    return caps


@dataclass(slots=True)
class SamplerOutput:
    """Output container produced by the sampling execution pipeline.

    Attributes:
        token_ids: Tensor of sampled token ids of shape `[batch_size]`.
        stats: Optional tuple `(max_prob, entropy, exp_entropy)` if requested.
    """

    token_ids: torch.Tensor
    stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


class SamplingPlanner:
    """Plans mask scheduling, custom operator validation, and sampling plan freezing.

    Supports binding multiple mask producers per slot and freezing deterministic
    execution plans prior to launching the forward pass.
    """

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
        """Build and freeze the sampling execution plan and mask schedule for a step.

        Validates requested custom operators against the registry, enforces speculative
        decoding capability invariants on attached mask producers, and constructs
        immutable `SamplingPlan` and `MaskSchedule` instances.

        Args:
            md: Sampling metadata for the current engine state.
            drafts: Optional mapping of slot indices to speculative draft token sequences.
            commits: Optional mapping of slot indices to committed token sequences.
            custom_ops: Names of registered custom operators to execute in this step.

        Returns:
            A tuple of `(plan, schedule)` frozen for the upcoming forward pass.

        Raises:
            CapabilityError: If a custom op is unregistered or if a draft is provided
                for a producer lacking `Cap.SPEC_VERIFIABLE`.
        """
        drafts = drafts or {}
        commits = commits or {}

        # Resolve custom operators via the registry; unregistered operators raise a CapabilityError.
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
        scratch_row = 0  # Secondary entries allocated after the primary block.
        argmax_invariant = True
        for slot in sorted(self._producers):
            if slot >= md.n_active:
                continue
            draft = drafts.get(slot, ())
            if draft and len(draft) > 0:
                # Reject speculative draft tokens if any attached producer lacks SPEC_VERIFIABLE.
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
            any_bias=md.any_bias,
            custom_ops=custom_ops,
            custom_ops_caps=custom_ops_caps,
            argmax_invariant=argmax_invariant and bool(self._producers),
        )
        schedule.require_match(plan)
        return plan, schedule


class Sampler:
    __slots__ = ("bias_state", "need_stats", "penalty_state")

    def __init__(
        self,
        penalty_state: _pen.PenaltyState | None = None,
        need_stats: bool = False,
        bias_state: Any | None = None,
    ) -> None:
        self.penalty_state = penalty_state
        self.need_stats = need_stats
        self.bias_state = bias_state

    def __call__(
        self,
        logits: torch.Tensor,
        md: SamplingMetadata,
        plan: SamplingPlan,
        mask: MaskHandle | None = None,
        *,
        force_reference: bool = False,
    ) -> SamplerOutput:
        """Execute the canonical sampling stage pipeline on logits.

        Stage Execution Order:
            1. Custom operators: Applied sequentially (in-place or functional rebind).
            2. Penalties: Repetition, frequency, and presence penalties applied in place.
            3. Allowed-token / grammar bitmask: Applied in place (unmasked values -> -inf).
               Fast path: When all producers are ARGMAX_INVARIANT, the batch is all greedy,
               and no statistics are requested, evaluate unmasked argmax. If the candidate
               is allowed by the bitmask, bypass full mask tensor application.
            4. Temperature scaling: Scaled exactly once prior to statistics and filtering.
            5. Sampling statistics: Softmax distribution statistics computed on scaled logits.
            6. Draw: Greedy argmax or stochastic top-k / top-p / min-p selection.
            7. Per-row greedy override: Enforce deterministic argmax for temperature=0 rows
               in mixed batches.

        Notes:
            Greedy decoding does NOT exit early before custom operators, penalties,
            or masking, as those transforms can alter the argmax index. Greedy execution
            only bypasses stochastic filtering and random drawing stages.

        Args:
            logits: Model logits tensor of shape `[batch_size, vocab_size]`.
            md: Sampling metadata container.
            plan: Frozen sampling plan declaring stage requirements.
            mask: Optional handle containing bitmasks and row indices.
            force_reference: Whether to force CPU reference kernel implementations.

        Returns:
            SamplerOutput containing sampled token ids and optional distribution statistics.

        Raises:
            ValueError: If logits rows or mask presence disagree with the frozen plan.
        """
        if logits.size(0) != plan.num_rows:
            raise ValueError(f"logits has {logits.size(0)} rows, plan declares {plan.num_rows}")
        if (mask is not None) != plan.any_mask:
            raise ValueError(
                f"mask={'present' if mask is not None else 'None'} disagrees with "
                f"plan.any_mask={plan.any_mask}"
            )
        trace_sampler(
            "forward_enter",
            rows=logits.size(0),
            vocab=logits.size(1),
            all_greedy=plan.all_greedy,
            any_mask=plan.any_mask,
            any_penalty=plan.any_penalty,
            custom_ops=plan.custom_ops or None,
        )
        if plan.custom_ops:
            logits = apply_custom_ops(logits, plan)
            trace_sampler("custom_ops_done", ops=plan.custom_ops)

        if plan.any_bias and self.bias_state is not None:
            apply_bias_(logits, md, self.bias_state)
            trace_sampler("bias_done")

        if plan.any_penalty and self.penalty_state is not None:
            _pen.apply_penalties_(logits, md, self.penalty_state)
            trace_sampler("penalties_done")

        if mask is not None:
            fast_path = plan.argmax_invariant and plan.all_greedy and not self.need_stats
            if fast_path:
                winner = _greedy_winner_with_mask_check(logits, mask)
                if winner is not None:
                    # All greedy winners are permitted by mask; skip O(N x V) bitmask application.
                    # RNG offset advancement remains the caller's responsibility.
                    trace_sampler("mask_fast_path_hit")
                    return SamplerOutput(token_ids=winner, stats=None)
                trace_sampler("mask_fast_path_miss")
            apply_allow_bitmask_(logits, mask.masks, mask.row_indices, mask.vocab_size)
            trace_sampler("mask_applied")

        temperature = md.active("temperature")
        scaled = logits / temperature.clamp_min(1e-6).unsqueeze(1)

        stats = softmax_stats_scaled(scaled) if self.need_stats else None

        if plan.all_greedy:
            trace_sampler("greedy_returned", rows=logits.size(0))
            return SamplerOutput(token_ids=scaled.argmax(dim=-1), stats=stats)

        host_top_k = md.staging("top_k")
        allow_topk_first = topk_first_hint(host_top_k, scaled.size(1))
        tok = topk_topp_sample(
            scaled,
            md.active("top_k"),
            md.active("top_p"),
            md.active("min_p"),
            md.active("seed"),
            md.active("offset"),
            force_reference=force_reference,
            allow_topk_first=allow_topk_first,
        )
        # Mixed batches need per-row greedy: a temperature-0 row must be argmax
        # (first maximal index), not a softmax draw among tied maxima at 1e-6.
        # vLLM splits the same way (is_greedy per request). RNG offsets still
        # advance one per row per step, so reproducibility is unaffected.
        greedy_rows = (temperature == 0.0) | (md.active("top_k") == 1)
        if bool(greedy_rows.any()):
            tok = torch.where(greedy_rows, scaled.argmax(dim=-1).to(tok.dtype), tok)
        trace_sampler("forward_returned", dtype=str(tok.dtype))
        return SamplerOutput(token_ids=tok, stats=stats)

    forward = __call__


def _greedy_winner_with_mask_check(logits: torch.Tensor, mask: MaskHandle) -> torch.Tensor | None:
    """Evaluate unmasked argmax and verify winner compliance against bitmask.

    Performs argmax directly on unmasked logits. If all selected tokens are allowed
    by the mask bitmask, returns the winning token ids. If any row is rejected,
    returns None to trigger the full mask application fallback path.

    Args:
        logits: Raw unmasked logits tensor.
        mask: Mask handle containing bitmasks and row mapping indices.

    Returns:
        Tensor of shape `[batch_size]` containing valid argmax token ids if all comply,
        or None if any token violates the mask constraint.
    """
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


BUILT_IN_SAMPLER_BACKENDS: frozenset[str] = frozenset({"triton", "reference"})
_FACTORIES: dict[str, Callable[[], Sampler]] = {}

logger = logging.getLogger(__name__)


def register_sampler_backend(backend: str, factory: Callable[[], Sampler]) -> None:
    """Register a custom sampler factory for a backend identifier.

    The factory callable must adhere to a zero-argument convention and return an
    object satisfying the `Sampler` subclass contract. If a backend with the given
    identifier is already registered, the existing factory is overwritten with a
    warning, matching SGLang registration semantics.

    Args:
        backend: Unique string identifier for the backend.
        factory: Zero-argument callable returning a `Sampler`-compatible instance.

    Raises:
        ValueError: If `backend` is an empty string.
    """
    # Enforce non-empty backend identifier.
    if not backend:
        raise ValueError("backend must be a non-empty string")
    # Log warning when overwriting an existing backend factory.
    if backend in _FACTORIES:
        logger.warning("Overriding existing sampler factory for backend '%s'", backend)
    _FACTORIES[backend] = factory


def registered_sampler_backends() -> tuple[str, ...]:
    """Return all currently registered custom sampler backend names.

    Returns:
        Sorted tuple of registered custom backend identifiers, excluding
        built-in backend names.
    """
    return tuple(sorted(_FACTORIES))


def create_sampler(
    backend: str | None = None,
    *,
    penalty_state: Any | None = None,
    need_stats: bool = False,
    bias_state: Any | None = None,
) -> Sampler:
    """Instantiate a sampler for the specified backend identifier.

    When `backend` is `None` or matches a built-in backend (`triton`, `reference`),
    the canonical `Sampler` is instantiated with the provided execution states.
    When a custom backend is requested, its registered zero-argument factory is
    invoked; custom factories are responsible for their own state wiring.

    Args:
        backend: Optional backend identifier. Defaults to None (canonical Sampler).
        penalty_state: Optional state container tracking repetition/frequency penalties.
        need_stats: Whether to compute sampling distribution statistics.
        bias_state: Optional state container tracking logit biases.

    Returns:
        Instantiated `Sampler` instance.

    Raises:
        TypeError: If a custom backend factory returns an object that does not
            subclass `Sampler`.
        ValueError: If `backend` is neither registered nor a recognized built-in.
    """
    from ayaka.sampling.plan import Sampler

    # Dispatch to custom factory if registered.
    if backend in _FACTORIES:
        sampler = _FACTORIES[backend]()
        # Assert subclass contract for custom sampler instances.
        if not isinstance(sampler, Sampler):
            raise TypeError(f"Sampler factory for backend '{backend}' must return a Sampler")
        return sampler
    # Default and built-in backends resolve to the canonical Sampler implementation.
    if backend is None or backend in BUILT_IN_SAMPLER_BACKENDS:
        return Sampler(penalty_state=penalty_state, need_stats=need_stats, bias_state=bias_state)
    # Reject unknown backend identifiers fail-closed.
    raise ValueError(
        f"Unknown sampling backend {backend!r}. Register it via register_sampler_backend()."
    )


_SAMPLING_STAGES = ("penalty", "bitmask", "temperature", "stats", "filter_sample")


def run_sampling_pipeline(
    logits: torch.Tensor,
    md: SamplingMetadata,
    penalty_state: PenaltyState,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    *,
    sampler: str = "inverse_cdf",  # "inverse_cdf" | "gumbel"
    compute_stats: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Execute the canonical sampling pipeline stages on logits in specification order.

    Applies penalties and grammar/allowed-token masks in place to `logits`. If the
    caller requires preservation of the original raw logits (e.g., for raw logprob
    computation), `logits` must be cloned prior to invoking this function.

    Args:
        logits: Float tensor of raw model logits of shape `[batch_size, vocab_size]`.
            Modified in place by penalties and bitmask stages.
        md: Sampling metadata container holding column parameters and active rows.
        penalty_state: State tracking per-slot token frequencies and penalty occurrences.
        temperature: Tensor of shape `[batch_size]` containing temperature scaling values.
        top_k: Tensor of shape `[batch_size]` specifying top-k filtering thresholds.
        top_p: Tensor of shape `[batch_size]` specifying top-p cumulative thresholds.
        min_p: Tensor of shape `[batch_size]` specifying min-p probability cutoffs.
        seed: Tensor of 64-bit random seeds per batch row.
        offset: Tensor of 64-bit RNG step offsets per batch row.
        mask: Optional packed 32-bit bitmask tensor defining allowed token sets.
        row_indices: Optional mapping of batch rows to corresponding mask rows.
        sampler: Sampling algorithm identifier, either `"inverse_cdf"` or `"gumbel"`.
        compute_stats: Whether to compute and return distribution statistics.

    Returns:
        A tuple `(token_ids, stats)` where `token_ids` is the sampled token tensor
        and `stats` is an optional tuple `(max_prob, entropy, exp_entropy)` or `None`.

    Raises:
        AssertionError: If `mask` is provided without `row_indices`.
        ValueError: If `sampler` is not recognized.
    """
    # Stage 1: Apply frequency, presence, and repetition penalties in place.
    apply_penalties_(logits, md, penalty_state)

    # Stage 2: Apply allowed-token bitmask constraints in place.
    if mask is not None:
        assert row_indices is not None, "mask không row_indices không rõ ràng buộc row nào"
        apply_allow_bitmask_(logits, mask, row_indices, logits.size(1))

    # Stage 3: Scale logits by temperature exactly once (clamped to prevent division by zero).
    scaled = logits / temperature.clamp_min(1e-6).unsqueeze(1)

    # Stage 4: Compute softmax statistics on temperature-scaled distribution if requested.
    stats = softmax_stats_scaled(scaled) if compute_stats else None

    # Stage 5: Stochastic filter and sampling draw.
    if sampler == "gumbel":
        tok = gumbel_sample(scaled, top_k, top_p, min_p, seed, offset)
    elif sampler == "inverse_cdf":
        tok = topk_topp_sample(scaled, top_k, top_p, min_p, seed, offset)
    else:
        raise ValueError(f"sampler không hợp lệ: {sampler!r}")

    return tok, stats
