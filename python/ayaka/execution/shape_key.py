"""Decode shape policy; resource identity remains in runner.graph.identity."""

from bisect import bisect_left
from dataclasses import dataclass

from ayaka.runner.graph.identity import GraphIdentity
from ayaka.utils.validation import require_int

__all__ = ["DecodeShape", "GraphIdentity", "select_bucket", "validate_buckets"]


def validate_buckets(buckets: tuple[int, ...]) -> None:
    """Reject coercion, duplicates and reordering of an operator's choices."""
    if not isinstance(buckets, tuple) or not buckets:
        raise ValueError("buckets must be a non-empty tuple")
    for bucket in buckets:
        require_int(bucket, "bucket", minimum=1)
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("buckets must be ascending and deduplicated")


def select_bucket(requests: int, buckets: tuple[int, ...]) -> int | None:
    """Return the ceil bucket; idle and over-ceiling invocations have no graph."""
    require_int(requests, "requests", minimum=0)
    validate_buckets(buckets)
    if requests == 0 or requests > buckets[-1]:
        return None
    return buckets[bisect_left(buckets, requests)]


@dataclass(frozen=True, slots=True)
class DecodeShape:
    bucket: int
    output_mode: str = "logits"

    def __post_init__(self) -> None:
        require_int(self.bucket, "bucket", minimum=1)
        if self.output_mode != "logits":
            raise ValueError("EP0 decode graphs only produce logits")
