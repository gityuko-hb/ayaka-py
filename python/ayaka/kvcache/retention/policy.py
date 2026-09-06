from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final

#: Ceiling on the exhaustive scan the default page bound falls back to.
#: 64k lengths is roughly 50 ms; beyond that a policy must state its own bound
#: rather than making bring-up quietly take seconds per layer.
DEFAULT_MAX_SCAN_TOKENS: Final[int] = 65_536

def _validate_layer_id(layer_id: int) -> None:
    """Reject a non-integer or negative layer id."""
    if not isinstance(layer_id, int) or isinstance(layer_id, bool):
        raise TypeError("layer_id must be an integer")
    if layer_id < 0:
        raise ValueError("layer_id must be non-negative")

def _validate_sequence_length(sequence_length: int) -> None:
    """Reject a non-integer or negative sequence length."""
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
        raise TypeError("sequence_length must be an integer")
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")

def _validate_page_size(page_size: int) -> None:
    if not isinstance(page_size, int) or isinstance(page_size, bool):
        raise TypeError("page_size must be an integer")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

def _pages_for(start: int, stop: int, page_size: int) -> int:
    """Pages an interval touches. A partially retained page stays readable."""
    if start >= stop:
        return 0
    return (stop + page_size - 1) // page_size - start // page_size

class RetentionPolicy(ABC):
    """The half-open logical token interval a layer must keep readable."""

    __slots__ = ()

    policy_id: ClassVar[str]
    """Stable identity used in :attr:`compatibility_key`."""

    @abstractmethod
    def required_token_range(self, layer_id: int, sequence_length: int) -> range:
        """Committed positions attention may read after a step of this length.

        Must be contiguous (``step == 1``) and within ``[0, sequence_length)``.
        """

    @property
    @abstractmethod
    def compatibility_key(self) -> tuple[object, ...]:
        """Identity of policies whose retention behaviour is identical."""

    def window_start(self, layer_id: int, sequence_length: int) -> int:
        """First readable position. Cheaper than building the whole range."""
        return self.required_token_range(layer_id, sequence_length).start

    def suffix_span(self, layer_id: int) -> int | float | None:
        """Retained window expressed as a suffix length, when it is one.

        Returns ``math.inf`` for unbounded retention, a positive integer for a
        fixed suffix window, or ``None`` when the policy cannot be expressed
        that way. The three-state return exists so a caller can vectorize --
        ``start = clamp(length - span, min=0)`` -- and fall back cleanly rather
        than branching on concrete policy classes.
        """
        _validate_layer_id(layer_id)
        return None

    def max_retained_pages(
        self,
        *,
        layer_id: int,
        max_sequence_tokens: int,
        page_size: int,
        max_scan_tokens: int = DEFAULT_MAX_SCAN_TOKENS,
    ) -> int:
        """Exact upper bound on retained pages across every length up to the max.

        The default walks every length, which is exact for an arbitrary policy
        and O(max_sequence_tokens). Built-in policies override it with closed
        form. A custom policy that needs long contexts should do the same:
        beyond ``max_scan_tokens`` this raises rather than spend seconds per
        layer at bring-up with no indication why.

        Raises:
            ValueError: if the scan would exceed ``max_scan_tokens``.
        """
        _validate_layer_id(layer_id)
        _validate_sequence_length(max_sequence_tokens)
        _validate_page_size(page_size)
        if max_sequence_tokens == 0:
            return 0
        if max_sequence_tokens > max_scan_tokens:
            raise ValueError(
                f"{type(self).__name__} has no closed-form page bound, and scanning "
                f"{max_sequence_tokens} lengths exceeds the {max_scan_tokens} limit. "
                "Override max_retained_pages() on the policy."
            )
        ceil = (max_sequence_tokens + page_size - 1) // page_size
        best = 0
        for length in range(1, max_sequence_tokens + 1):
            token_range = self.required_token_range(layer_id, length)
            best = max(best, _pages_for(token_range.start, token_range.stop, page_size))
            if best >= ceil:
                # Cannot grow further; stop instead of walking the remainder.
                return ceil
        return best

@dataclass(frozen=True, slots=True)
class FullRetention(RetentionPolicy):
    """Keep every committed token for full causal attention.

    The live interval is always ``[0, sequence_length)``, which makes this the
    default for MHA/GQA cache groups.

    >>> FullRetention().required_token_range(0, 5)
    range(0, 5)
    >>> FullRetention().suffix_span(0)
    inf
    """

    policy_id: ClassVar[str] = "full"

    def required_token_range(self, layer_id: int, sequence_length: int) -> range:
        _validate_layer_id(layer_id)
        _validate_sequence_length(sequence_length)
        return range(0, sequence_length)

    def window_start(self, layer_id: int, sequence_length: int) -> int:
        _validate_layer_id(layer_id)
        _validate_sequence_length(sequence_length)
        return 0

    def suffix_span(self, layer_id: int) -> float:
        _validate_layer_id(layer_id)
        return math.inf

    def max_retained_pages(
        self,
        *,
        layer_id: int,
        max_sequence_tokens: int,
        page_size: int,
        max_scan_tokens: int = DEFAULT_MAX_SCAN_TOKENS,
    ) -> int:
        _validate_layer_id(layer_id)
        _validate_sequence_length(max_sequence_tokens)
        _validate_page_size(page_size)
        return (max_sequence_tokens + page_size - 1) // page_size

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (self.policy_id,)

@dataclass(frozen=True, slots=True)
class SlidingWindowRetention(RetentionPolicy):
    """Keep only the most recent ``window_size`` committed tokens.

    Positions ``[max(0, sequence_length - window_size), sequence_length)`` stay
    readable; older pages become eligible for release after a step completes,
    subject to the safe-epoch deferred-free rule.

    >>> SlidingWindowRetention(window_size=4).required_token_range(0, 10)
    range(6, 10)
    >>> SlidingWindowRetention(window_size=4).required_token_range(0, 2)
    range(0, 2)
    """
    policy_id: ClassVar[str] = "sliding_window"

    window_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise TypeError("window_size must be an integer")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

    def required_token_range(self, layer_id: int, sequence_length: int) -> range:
        _validate_layer_id(layer_id)
        _validate_sequence_length(sequence_length)
        # Clamp to zero for short sequences; the range stays half-open and
        # empty when sequence_length is zero.
        return range(max(0, sequence_length - self.window_size), sequence_length)

    def window_start(self, layer_id: int, sequence_length: int) -> int:
        _validate_layer_id(layer_id)
        _validate_sequence_length(sequence_length)
        return max(0, sequence_length - self.window_size)

    def suffix_span(self, layer_id: int) -> int:
        _validate_layer_id(layer_id)
        return self.window_size

    def max_retained_pages(
        self,
        *,
        layer_id: int,
        max_sequence_tokens: int,
        page_size: int,
        max_scan_tokens: int = DEFAULT_MAX_SCAN_TOKENS,
    ) -> int:
        """Closed form, O(1), verified against a brute-force scan in the tests.

        An interval of ``W`` tokens starting at an arbitrary offset touches at
        most ``floor((W - 2) / P) + 2`` pages -- worst case is a start at
        ``P - 1``, which wastes one page at each end. That is what
        ``(W + 2P - 2) // P`` computes. The result is then capped by the pages
        the whole context occupies, which binds when the sequence is shorter
        than the window.

        This bound is worth being careful about in both directions: too small
        and a live page gets reused; too large and fewer sequences fit for no
        reason. ``tests/test_retention.py`` brute-forces it against the
        definition over a grid of windows, page sizes and contexts, so an edit
        here cannot pass without also passing the oracle.
        """
        _validate_layer_id(layer_id)
        _validate_sequence_length(max_sequence_tokens)
        _validate_page_size(page_size)
        if max_sequence_tokens == 0:
            return 0
        retained = min(self.window_size, max_sequence_tokens)
        worst_alignment = (retained + 2 * page_size - 2) // page_size
        total_pages = (max_sequence_tokens + page_size - 1) // page_size
        return min(worst_alignment, total_pages)

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (self.policy_id, self.window_size)

@dataclass(frozen=True, slots=True)
class HybridRetention(RetentionPolicy):
    """Per-layer retention selection, used only while constructing cache groups.

    Cache groups require one resolved policy, so this is resolved into
    per-group policies by the caller and rejected by ``KVLayerConfig`` and
    ``KVCacheGroup``.
    """

    policy_id: ClassVar[str] = "hybrid"

    layer_policies: tuple[RetentionPolicy, ...]

    def __post_init__(self) -> None:
        policies = tuple(self.layer_policies)
        if not policies:
            raise ValueError("HybridRetention requires at least one layer policy")
        if any(not isinstance(policy, RetentionPolicy) for policy in policies):
            raise TypeError("layer_policies must be RetentionPolicy instances")
        if any(isinstance(policy, HybridRetention) for policy in policies):
            raise ValueError("nested HybridRetention policies are not supported")
        object.__setattr__(self, "layer_policies", policies)

    def policy_for_layer(self, layer_id: int) -> RetentionPolicy:
        """Resolve the policy for one layer.

        Raises:
            IndexError: if ``layer_id`` falls outside the policy tuple.
        """
        _validate_layer_id(layer_id)
        if layer_id >= len(self.layer_policies):
            raise IndexError(
                f"layer_id {layer_id} is outside HybridRetention "
                f"({len(self.layer_policies)} layers)"
            )
        return self.layer_policies[layer_id]

    # Every method below delegates *without* re-validating: policy_for_layer
    # checks the layer id, and the resolved policy checks the length. The
    # previous version validated in all three places on every call.

    def required_token_range(self, layer_id: int, sequence_length: int) -> range:
        return self.policy_for_layer(layer_id).required_token_range(layer_id, sequence_length)

    def window_start(self, layer_id: int, sequence_length: int) -> int:
        return self.policy_for_layer(layer_id).window_start(layer_id, sequence_length)

    def suffix_span(self, layer_id: int) -> int | float | None:
        return self.policy_for_layer(layer_id).suffix_span(layer_id)

    def max_retained_pages(
        self,
        *,
        layer_id: int,
        max_sequence_tokens: int,
        page_size: int,
        max_scan_tokens: int = DEFAULT_MAX_SCAN_TOKENS,
    ) -> int:
        return self.policy_for_layer(layer_id).max_retained_pages(
            layer_id=layer_id,
            max_sequence_tokens=max_sequence_tokens,
            page_size=page_size,
            max_scan_tokens=max_scan_tokens,
        )

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.policy_id,
            tuple(policy.compatibility_key for policy in self.layer_policies),
        )
