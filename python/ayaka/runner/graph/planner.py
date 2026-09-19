"""Decode-graph eligibility, routing and generation validation (R08).

This is the single place a decode step becomes ``GraphMode.REPLAY``. It runs on
the engine thread from ``KVStepRuntime.prepare`` (before KV reservation), and
``should_replay`` re-checks the same conditions immediately before the runner
enqueues, so a resource generation that moved between prepare and enqueue can
never replay a stale graph.

What is deliberately NOT part of the decision: block-table values, sequence
lengths, computed lengths and prefix hits. Those are staged into backing the
capture already bound; changing them does not invalidate the graph.

A generation move does not silently fall back forever: ``invalidate`` latches
the pool invalid, every further replay is refused, and recapturing under a new
resource generation is the only path back to REPLAY.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ayaka.plan import EMPTY_GRAPH_PLAN, GraphMode, GraphPlan
from ayaka.runner.graph.graph import GraphCapabilityError
from ayaka.runner.graph.identity import graph_identity_key
from ayaka.runner.graph.runner import pad_to_bucket
from ayaka.types import AttentionCudaGraphSupport, ForwardMode

if TYPE_CHECKING:
    from ayaka.memory.capacity import ResourceGeneration
    from ayaka.sched.plan import BatchStepPlan

logger = logging.getLogger(__name__)

__all__ = ["DecodeGraphPlanner", "GraphFallbackReason", "GraphStats"]


class GraphFallbackReason(StrEnum):
    """Finite reason set for one step leaving the graph path.

    Labels are bounded and request-free, so they are safe as metric labels.
    """

    DISABLED = "disabled"
    NOT_PURE_DECODE = "not_pure_decode"
    SAMPLING_UNCAPTURABLE = "sampling_uncapturable"
    PROMPT_LOGPROBS = "prompt_logprobs"
    EAGER_ONLY_REQUEST = "eager_only_request"
    BACKEND_UNSUPPORTED = "backend_unsupported"
    NO_TOKENS = "no_tokens"
    BUCKET_CEILING = "bucket_ceiling"
    MISSING_BUCKET = "missing_bucket"
    GENERATION_MISMATCH = "generation_mismatch"
    INVALIDATED = "invalidated"
    WORKSPACE_GROWTH = "workspace_growth"


@dataclass(frozen=True, slots=True)
class GraphStats:
    """Observability snapshot; counters are monotonic per planner instance."""

    hits: int
    misses: int
    captures: int
    invalidations: int
    eager_fallbacks: int
    workspace_growth: int
    captured_buckets: tuple[int, ...]
    invalidated: bool
    last_reason: str

    def as_dict(self) -> dict[str, int | str | tuple[int, ...] | bool]:
        """Metric-named view; reason is a finite label, never a request id."""
        return {
            "graph_hit": self.hits,
            "graph_miss": self.misses,
            "graph_capture": self.captures,
            "graph_invalidation": self.invalidations,
            "workspace_growth": self.workspace_growth,
            "eager_fallback": self.eager_fallbacks,
            "graph_captured_buckets": self.captured_buckets,
            "graph_invalidated": self.invalidated,
            "graph_reason": self.last_reason,
        }


class DecodeGraphPlanner:
    """Route pure-decode steps onto captured buckets, or fall back with a reason.

    ``captured`` must report the buckets that currently have a captured graph
    for the live resource generation; the pool owns that state and calls
    ``note_capture`` when a bucket becomes replayable.
    """

    def __init__(
        self,
        *,
        buckets: tuple[int, ...],
        support: AttentionCudaGraphSupport,
        backend: str,
        dtype: str,
        generation: Callable[[], ResourceGeneration | None],
        captured: Callable[[], frozenset[int]],
        eager_only_reason: Callable[[BatchStepPlan], str | None] | None = None,
        enabled: bool = True,
        metrics: Any | None = None,
    ) -> None:
        if not buckets or tuple(sorted(buckets)) != tuple(buckets):
            raise ValueError("buckets must be a non-empty ascending tuple")
        if len(set(buckets)) != len(buckets):
            raise ValueError("buckets must be deduplicated")
        if not isinstance(support, AttentionCudaGraphSupport):
            raise TypeError("support must be an AttentionCudaGraphSupport")
        if not backend or not dtype:
            raise ValueError("backend and dtype must be non-empty")
        self._buckets = tuple(buckets)
        self._support = support
        self._backend = backend
        self._dtype = dtype
        self._generation = generation
        self._captured = captured
        self._eager_only_reason = eager_only_reason
        self._enabled = bool(enabled)
        self._metrics = metrics
        self._captured_generation: ResourceGeneration | None = None
        self._invalidated = False
        self._hits = 0
        self._misses = 0
        self._captures = 0
        self._invalidations = 0
        self._eager_fallbacks = 0
        self._workspace_growth = 0
        self._last_reason = ""

    # ── capture-side state ────────────────────────────────────────────────

    def note_capture(self, buckets: tuple[int, ...] = ()) -> None:
        """Record that a capture session completed under the current generation.

        The route decision compares the live generation against
        ``_captured_generation``; a capture against any other generation would
        produce graphs whose pointers belong to a replaced owner.
        """
        if buckets:
            for bucket in buckets:
                if bucket not in self._buckets:
                    raise ValueError(f"captured bucket {bucket} is not in the planner's bucket list")
                if bucket < 1:
                    raise ValueError("captured buckets must be positive")
        self._captured_generation = self._generation()
        self._captures += 1
        self._invalidated = False
        self._bump("graph_capture")

    @property
    def captured_generation(self) -> ResourceGeneration | None:
        return self._captured_generation

    def captured_buckets(self) -> frozenset[int]:
        return frozenset(self._captured())

    @property
    def invalidated(self) -> bool:
        return self._invalidated

    @property
    def buckets(self) -> tuple[int, ...]:
        return self._buckets

    # ── routing ───────────────────────────────────────────────────────────

    def plan(self, step: BatchStepPlan) -> GraphPlan:
        """Decide REPLAY or eager for one step, with a bounded reason on fallback."""
        if step.graph.mode is GraphMode.REPLAY:
            raise ValueError("step already carries a graph plan")
        if not self._enabled:
            return self._fallback(GraphFallbackReason.DISABLED)
        if not step.is_pure_decode:
            return self._fallback(GraphFallbackReason.NOT_PURE_DECODE)
        if not step.sampling.graph_capturable:
            return self._fallback(GraphFallbackReason.SAMPLING_UNCAPTURABLE)
        if step.prompt_logprobs:
            return self._fallback(GraphFallbackReason.PROMPT_LOGPROBS)
        if self._eager_only_reason is not None:
            reason = self._eager_only_reason(step)
            if reason:
                return self._fallback(GraphFallbackReason.EAGER_ONLY_REQUEST, detail=reason)
        if self._support < AttentionCudaGraphSupport.PURE_DECODE:
            return self._fallback(GraphFallbackReason.BACKEND_UNSUPPORTED)
        if self._invalidated:
            return self._fallback(GraphFallbackReason.INVALIDATED)
        if not self._generation_current():
            self.invalidate(GraphFallbackReason.GENERATION_MISMATCH)
            return self._fallback(GraphFallbackReason.GENERATION_MISMATCH)
        raw = step.padded_num_tokens
        if raw < 1:
            return self._fallback(GraphFallbackReason.NO_TOKENS)
        if raw > self._buckets[-1]:
            return self._fallback(GraphFallbackReason.BUCKET_CEILING)
        bucket = pad_to_bucket(raw, self._buckets)
        if bucket not in self._captured():
            return self._fallback(GraphFallbackReason.MISSING_BUCKET)
        return GraphPlan(
            mode=GraphMode.REPLAY,
            bucket=bucket,
            graph_key=self.graph_key(bucket),
        )

    def validate(self, step: BatchStepPlan) -> None:
        """Re-check a plan before any reservation/admission commits to it.

        Raises:
            GraphCapabilityError: when a REPLAY plan disagrees with the live
                planner state. This is a routing bug, not a fallback case —
                ``plan`` already produced EAGER for every legitimate refusal.
        """
        if step.graph.mode is not GraphMode.REPLAY:
            return
        if not step.is_pure_decode or not step.sampling.graph_capturable:
            raise GraphCapabilityError(
                "graph replay plan disagrees with pure-decode/capturable sampling"
            )
        if not self._enabled or self._invalidated or not self._generation_current():
            raise GraphCapabilityError(
                "graph replay plan belongs to a disabled/invalidated/replaced generation"
            )
        bucket = pad_to_bucket(step.padded_num_tokens, self._buckets)
        if bucket != step.graph.bucket:
            raise GraphCapabilityError(
                f"graph bucket {step.graph.bucket} disagrees with the step's padded "
                f"size {step.padded_num_tokens} -> {bucket}"
            )
        if step.graph.bucket not in self._captured():
            raise GraphCapabilityError(
                f"graph bucket {step.graph.bucket} has no captured instance"
            )
        expected = self.graph_key(step.graph.bucket)
        if step.graph.graph_key != expected:
            raise GraphCapabilityError(
                "graph key disagrees with the live resource generation (routing bug)"
            )

    def should_replay(self, step: BatchStepPlan) -> bool:
        """Execute-time gate: the last line of defense before ``replay()``.

        Returns False (and records a bounded fallback reason) when the graph
        became unusable between prepare and enqueue. The caller must run the
        eager path for this step instead — never retry after a replay enqueue.
        """
        if step.graph.mode is not GraphMode.REPLAY:
            return False
        if not self._enabled:
            return self._refuse_execution(GraphFallbackReason.DISABLED)
        if self._invalidated:
            return self._refuse_execution(GraphFallbackReason.INVALIDATED)
        if not self._generation_current():
            self.invalidate(GraphFallbackReason.GENERATION_MISMATCH)
            return self._refuse_execution(GraphFallbackReason.GENERATION_MISMATCH)
        if step.graph.bucket not in self._captured():
            return self._refuse_execution(GraphFallbackReason.MISSING_BUCKET)
        return True

    def note_replay(self, bucket: int) -> None:
        """Record one successful replay; only real replays count as hits."""
        if bucket not in self._buckets:
            raise ValueError(f"replayed bucket {bucket} is not a configured bucket")
        self._hits += 1
        self._last_reason = ""
        self._bump("graph_hit")

    # ── invalidation ──────────────────────────────────────────────────────

    def invalidate(self, reason: GraphFallbackReason | str) -> None:
        """Latch every captured graph unusable until a new capture session."""
        label = str(reason)
        if self._invalidated and self._last_reason == label:
            return
        self._invalidated = True
        self._last_reason = label
        self._invalidations += 1
        self._bump("graph_invalidation")
        logger.info("decode graphs invalidated: %s", label)

    def note_workspace_growth(self) -> None:
        """A step grew the shared workspace; captured pointers may have moved."""
        self._workspace_growth += 1
        self._bump("workspace_growth")
        self.invalidate(GraphFallbackReason.WORKSPACE_GROWTH)

    # ── introspection ─────────────────────────────────────────────────────

    def graph_key(self, bucket: int) -> str:
        generation = self._captured_generation or self._generation()
        if generation is None:
            raise GraphCapabilityError("no live resource generation to key a decode graph")
        return graph_identity_key(
            generation=generation,
            bucket=bucket,
            mode=ForwardMode.DECODE,
            dtype=self._dtype,
            backend=self._backend,
        )

    def stats(self) -> GraphStats:
        return GraphStats(
            hits=self._hits,
            misses=self._misses,
            captures=self._captures,
            invalidations=self._invalidations,
            eager_fallbacks=self._eager_fallbacks,
            workspace_growth=self._workspace_growth,
            captured_buckets=tuple(sorted(self._captured())),
            invalidated=self._invalidated,
            last_reason=self._last_reason,
        )

    def attach_metrics(self, metrics: Any | None) -> None:
        """Bind the engine's bounded counter sink once the executor exists."""
        self._metrics = metrics

    # ── internals ─────────────────────────────────────────────────────────

    def _generation_current(self) -> bool:
        live = self._generation()
        return live is not None and live == self._captured_generation

    def _fallback(self, reason: GraphFallbackReason, *, detail: str = "") -> GraphPlan:
        self._misses += 1
        self._last_reason = str(reason)
        self._bump("graph_miss")
        if detail:
            logger.debug("decode graph eager fallback: %s (%s)", reason, detail)
        return EMPTY_GRAPH_PLAN

    def _refuse_execution(self, reason: GraphFallbackReason) -> bool:
        self._eager_fallbacks += 1
        self._last_reason = str(reason)
        self._bump("eager_fallback")
        return False

    def _bump(self, name: str) -> None:
        if self._metrics is not None:
            self._metrics.increment(name)
