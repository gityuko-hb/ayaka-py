from __future__ import annotations

from ayaka.kvcache.retention.policy import DEFAULT_MAX_SCAN_TOKENS, RetentionPolicy


def retained_page_range(
    policy: RetentionPolicy,
    *,
    layer_id: int,
    sequence_length: int,
    page_size: int,
) -> range:
    """Every logical page intersecting the live token interval.

    Because a partially retained page must stay readable, both boundary pages
    of the interval are included.

    Args:
        policy: Retention policy to evaluate.
        layer_id: Model layer the policy applies to.
        sequence_length: Committed sequence length after the step.
        page_size: Tokens per page.

    Returns:
        A half-open range of logical block indexes, empty when nothing is live.

    Raises:
        TypeError: if ``policy`` is not a ``RetentionPolicy`` or ``page_size``
            is not an integer.
        ValueError: if the policy returns a non-contiguous range, or one
            outside the committed tokens.
    """
    if not isinstance(policy, RetentionPolicy):
        raise TypeError(f"policy must be a RetentionPolicy, got {type(policy).__name__}")
    if not isinstance(page_size, int) or isinstance(page_size, bool):
        raise TypeError("page_size must be an integer")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    # layer_id and sequence_length are validated by the policy itself; checking
    # them again here cost a second pair of isinstance calls on every prune.
    token_range = policy.required_token_range(layer_id, sequence_length)
    if token_range.step != 1:
        raise ValueError("retention ranges must be contiguous")
    if token_range.start < 0 or token_range.stop > sequence_length:
        raise ValueError("retention range must stay within committed tokens")
    if token_range.start >= token_range.stop:
        return range(0, 0)
    # Floor the start and ceil the stop so both boundary pages are covered.
    return range(
        token_range.start // page_size,
        (token_range.stop + page_size - 1) // page_size,
    )

def maximum_retained_pages(
    policy: RetentionPolicy,
    *,
    layer_id: int,
    max_sequence_tokens: int,
    page_size: int,
    max_scan_tokens: int = DEFAULT_MAX_SCAN_TOKENS,
) -> int:
    """Exact bound on retained pages across every length up to the maximum.

    Delegates to the policy. It used to be an ``isinstance`` ladder over the
    three built-ins with an exhaustive scan as the fallback, which meant a
    third-party policy always paid O(max_sequence_tokens) -- about 25 ms per
    layer at a 32k context, and once per layer, so seconds at bring-up for a
    hybrid model. A policy now answers for itself, and the built-ins answer in
    closed form.

    Args:
        policy: Retention policy to bound.
        layer_id: Model layer the policy applies to.
        max_sequence_tokens: Hard per-sequence length limit.
        page_size: Tokens per page.
        max_scan_tokens: Guard for the default scan; ignored by policies with a
            closed form.

    Raises:
        TypeError: if ``policy`` is not a ``RetentionPolicy``.
        ValueError: if a scanning policy is asked for a context above the guard.
    """
    if not isinstance(policy, RetentionPolicy):
        raise TypeError(f"policy must be a RetentionPolicy, got {type(policy).__name__}")
    return policy.max_retained_pages(
        layer_id=layer_id,
        max_sequence_tokens=max_sequence_tokens,
        page_size=page_size,
        max_scan_tokens=max_scan_tokens,
    )

def window_starts(
    policy: RetentionPolicy,
    positions: tuple[int, ...],
    *,
    layer_id: int = 0,
) -> tuple[int, ...]:
    """First readable position for each query, given its logical position.

    A query at position ``p`` sees the interval the policy returns for a
    sequence of length ``p + 1``.

    Uses :meth:`~.policy.RetentionPolicy.suffix_span` when the policy is a plain
    suffix window, which turns the whole computation into arithmetic; otherwise
    it calls the policy once per position. The distinction matters on a prefill
    of several thousand queries, where the per-position path is thousands of
    Python calls that a single clamp replaces.
    """
    if not isinstance(policy, RetentionPolicy):
        raise TypeError(f"policy must be a RetentionPolicy, got {type(policy).__name__}")
    span = policy.suffix_span(layer_id)
    if span is None:
        return tuple(policy.window_start(layer_id, position + 1) for position in positions)
    if span == float("inf"):
        return (0,) * len(positions)
    return tuple(max(0, position + 1 - int(span)) for position in positions)
