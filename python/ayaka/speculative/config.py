"""Validated configuration for the production speculative decoding subsystem.

The dense draft-model reference path keeps its own
:class:`ayaka.configs.speculative.SpeculativeConfig`; this module configures the
scheduler-aware subsystem. There is deliberately no default draft length: the
break-even point depends on model, hardware and batch size, so an operator must
state ``max_draft_tokens`` (and, for a batch schedule, every range).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from ayaka.speculative.mode import Certification, SpeculativeMode, mode_capabilities
from ayaka.utils.validation import require_int

__all__ = [
    "MAX_DRAFT_TOKENS",
    "AcceptanceMethod",
    "NGramConfig",
    "NGramSelection",
    "PolicyConfig",
    "PolicyKind",
    "SpeculativeDecodingConfig",
]

#: Upper bound on one request's draft length in a step. Verification rows,
#: device acceptance scratch and per-position metrics are sized from it.
MAX_DRAFT_TOKENS = 64


class NGramSelection(StrEnum):
    """Which earlier occurrence of the matched suffix supplies the draft.

    Among occurrences with the longest match: ``OLDEST`` takes the earliest
    (the vLLM/TensorRT-LLM convention), ``MOST_RECENT`` the latest, and
    ``LONGEST`` the latest occurrence that still has a full draft-length
    continuation, falling back to the earliest.
    """

    OLDEST = "oldest"
    MOST_RECENT = "most_recent"
    LONGEST = "longest"


class PolicyKind(StrEnum):
    FIXED = "fixed"
    BATCH_SCHEDULE = "batch_schedule"
    ADAPTIVE = "adaptive"


class AcceptanceMethod(StrEnum):
    """``GREEDY`` is exact for greedy requests; ``REJECTION`` is distributional."""

    GREEDY = "greedy"
    REJECTION = "rejection"


@dataclass(frozen=True, slots=True, kw_only=True)
class NGramConfig:
    """N-gram proposer settings.

    Attributes:
        min_matching_ngram_size: Shortest suffix pattern that may match.
        max_matching_ngram_size: Longest suffix pattern searched.
        selection: Occurrence choice among the longest matches.
        max_scan_occurrences: Bound on candidate occurrences inspected per
            lookup (the most recent ones); keeps host cost independent of
            context length.
        public_pool: Also propose continuations observed in other requests of
            the same isolation namespace (tenant + cache salt). Off by default:
            acceptance timing can reveal whether another request produced a
            sequence, so sharing is an explicit operator decision.
        keep_all: Keep several continuations per public pattern instead of
            only the newest.
        public_pool_max_patterns: Global bound on public patterns (LRU).
        public_pool_namespace_max_patterns: Per-namespace bound, so one tenant
            cannot evict every other tenant's entries.
        public_pool_max_continuations: Continuations kept per pattern when
            ``keep_all`` is set.
    """

    min_matching_ngram_size: int = 2
    max_matching_ngram_size: int = 4
    selection: NGramSelection = NGramSelection.OLDEST
    max_scan_occurrences: int = 64
    public_pool: bool = False
    keep_all: bool = False
    public_pool_max_patterns: int = 1 << 16
    public_pool_namespace_max_patterns: int = 1 << 14
    public_pool_max_continuations: int = 4

    def __post_init__(self) -> None:
        require_int(self.min_matching_ngram_size, "ngram.min_matching_ngram_size", minimum=1)
        require_int(self.max_matching_ngram_size, "ngram.max_matching_ngram_size", minimum=1)
        if self.min_matching_ngram_size > self.max_matching_ngram_size:
            raise ValueError("ngram.min_matching_ngram_size exceeds max_matching_ngram_size")
        if self.max_matching_ngram_size > 32:
            raise ValueError("ngram.max_matching_ngram_size is limited to 32")
        if not isinstance(self.selection, NGramSelection):
            raise TypeError("ngram.selection must be NGramSelection")
        require_int(self.max_scan_occurrences, "ngram.max_scan_occurrences", minimum=1)
        for name in ("public_pool", "keep_all"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"ngram.{name} must be bool")
        require_int(self.public_pool_max_patterns, "ngram.public_pool_max_patterns", minimum=1)
        require_int(
            self.public_pool_namespace_max_patterns,
            "ngram.public_pool_namespace_max_patterns",
            minimum=1,
        )
        require_int(
            self.public_pool_max_continuations, "ngram.public_pool_max_continuations", minimum=1
        )
        if self.public_pool_namespace_max_patterns > self.public_pool_max_patterns:
            raise ValueError("per-namespace public pool bound exceeds the global bound")


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyConfig:
    """How many draft tokens a step may request.

    ``FIXED`` uses ``fixed_k`` (default: the configured maximum).
    ``BATCH_SCHEDULE`` maps the decode batch size through inclusive
    ``(start, end, k)`` ranges that must start at 1 and not overlap; batch sizes
    past the last range use ``k=0``. ``ADAPTIVE`` starts from the schedule (or
    the maximum) and moves among ``candidate_k`` using measured acceptance and
    step cost, with hysteresis and periodic re-probing; it never disables
    speculation permanently.
    """

    kind: PolicyKind = PolicyKind.FIXED
    fixed_k: int | None = None
    batch_schedule: tuple[tuple[int, int, int], ...] = ()
    candidate_k: tuple[int, ...] = ()
    ewma_alpha: float = 0.2
    warmup_steps: int = 8
    update_interval: int = 4
    hysteresis: float = 0.05
    probe_interval: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PolicyKind):
            raise TypeError("policy.kind must be PolicyKind")
        if self.fixed_k is not None:
            require_int(self.fixed_k, "policy.fixed_k")
        if type(self.batch_schedule) is not tuple:
            raise TypeError("policy.batch_schedule must be a tuple")
        previous_end = 0
        for entry in self.batch_schedule:
            if type(entry) is not tuple or len(entry) != 3:
                raise ValueError("policy.batch_schedule entries are (start, end, k) tuples")
            start, end, k = entry
            require_int(start, "batch_schedule start", minimum=1)
            require_int(end, "batch_schedule end", minimum=start)
            require_int(k, "batch_schedule k")
            if start != previous_end + 1:
                raise ValueError("policy.batch_schedule ranges must start at 1 and be contiguous")
            previous_end = end
        if self.kind is PolicyKind.BATCH_SCHEDULE and not self.batch_schedule:
            raise ValueError("a batch-schedule policy needs at least one range")
        if type(self.candidate_k) is not tuple:
            raise TypeError("policy.candidate_k must be a tuple")
        for k in self.candidate_k:
            require_int(k, "policy.candidate_k")
        if tuple(sorted(set(self.candidate_k))) != self.candidate_k:
            raise ValueError("policy.candidate_k must be sorted and unique")
        if self.kind is PolicyKind.ADAPTIVE and len(self.candidate_k) < 2:
            raise ValueError("an adaptive policy needs at least two candidate draft lengths")
        if (
            isinstance(self.ewma_alpha, bool)
            or not isinstance(self.ewma_alpha, (int, float))
            or not math.isfinite(self.ewma_alpha)
            or not 0.0 < self.ewma_alpha <= 1.0
        ):
            raise ValueError("policy.ewma_alpha must be in (0, 1]")
        require_int(self.warmup_steps, "policy.warmup_steps")
        require_int(self.update_interval, "policy.update_interval", minimum=1)
        require_int(self.probe_interval, "policy.probe_interval", minimum=1)
        if (
            isinstance(self.hysteresis, bool)
            or not isinstance(self.hysteresis, (int, float))
            or not math.isfinite(self.hysteresis)
            or not 0.0 <= self.hysteresis < 1.0
        ):
            raise ValueError("policy.hysteresis must be in [0, 1)")

    def validate_against(self, max_draft_tokens: int) -> None:
        """Every draft length the policy can choose must fit the configured maximum."""
        if self.fixed_k is not None and self.fixed_k > max_draft_tokens:
            raise ValueError("policy.fixed_k exceeds max_draft_tokens")
        if any(k > max_draft_tokens for _, _, k in self.batch_schedule):
            raise ValueError("policy.batch_schedule k exceeds max_draft_tokens")
        if any(k > max_draft_tokens for k in self.candidate_k):
            raise ValueError("policy.candidate_k exceeds max_draft_tokens")


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeculativeDecodingConfig:
    """Opt-in production speculative decoding.

    Attributes:
        max_draft_tokens: Largest draft length any request may verify in one
            step. Required; sizes verification rows and metrics.
        mode: Speculative method. ``AUTO`` resolves to the strongest production
            mode available at runtime construction.
        ngram: N-gram proposer settings (used by ``NGRAM``).
        policy: Draft-length policy.
        acceptance: Acceptance algorithm. ``REJECTION`` requires a mode that
            declares advanced sampling support.
        max_draft_tokens_per_step: Optional cap on all draft rows of one step,
            protecting prefill budget under wide batches.
        allow_experimental: Required to run a mode certified ``EXPERIMENTAL``.
    """

    max_draft_tokens: int
    mode: SpeculativeMode = SpeculativeMode.NGRAM
    ngram: NGramConfig = field(default_factory=NGramConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    acceptance: AcceptanceMethod = AcceptanceMethod.GREEDY
    max_draft_tokens_per_step: int | None = None
    allow_experimental: bool = False

    def __post_init__(self) -> None:
        require_int(self.max_draft_tokens, "max_draft_tokens", minimum=1)
        if self.max_draft_tokens > MAX_DRAFT_TOKENS:
            raise ValueError(f"max_draft_tokens is limited to {MAX_DRAFT_TOKENS}")
        if not isinstance(self.mode, SpeculativeMode):
            raise TypeError("mode must be SpeculativeMode")
        if self.mode is SpeculativeMode.NONE:
            raise ValueError("disable speculation by omitting the config, not with mode=none")
        if not isinstance(self.ngram, NGramConfig):
            raise TypeError("ngram must be NGramConfig")
        if not isinstance(self.policy, PolicyConfig):
            raise TypeError("policy must be PolicyConfig")
        self.policy.validate_against(self.max_draft_tokens)
        if not isinstance(self.acceptance, AcceptanceMethod):
            raise TypeError("acceptance must be AcceptanceMethod")
        if self.max_draft_tokens_per_step is not None:
            require_int(self.max_draft_tokens_per_step, "max_draft_tokens_per_step", minimum=1)
        if type(self.allow_experimental) is not bool:
            raise TypeError("allow_experimental must be bool")
        if self.mode is not SpeculativeMode.AUTO:
            self.require_runnable(self.mode)

    def require_runnable(self, mode: SpeculativeMode) -> None:
        """Refuse a mode this build cannot run under this configuration.

        Raises:
            ValueError: If the mode is unavailable, experimental without
                opt-in, or the acceptance method exceeds its capabilities.
        """
        capabilities = mode_capabilities(mode)
        if capabilities.certification is Certification.UNAVAILABLE:
            raise ValueError(f"speculative mode {mode.value!r} is not available in this build")
        if capabilities.certification is Certification.EXPERIMENTAL and not self.allow_experimental:
            raise ValueError(
                f"speculative mode {mode.value!r} is experimental; set allow_experimental=True"
            )
        if self.acceptance is AcceptanceMethod.REJECTION and not (
            capabilities.supports_advanced_sampling
        ):
            raise ValueError(
                f"speculative mode {mode.value!r} does not implement rejection sampling"
            )

    @property
    def initial_k(self) -> int:
        """Draft length before any batch or feedback information exists."""
        if self.policy.fixed_k is not None:
            return self.policy.fixed_k
        return self.max_draft_tokens
