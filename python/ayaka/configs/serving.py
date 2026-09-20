"""Server policy, separate from scheduler and model configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ServingConfig:
    model: str = "ayaka"
    api_keys: tuple[str, ...] = ()
    max_concurrent_requests: int = 0
    max_request_bytes: int = 4 << 20
    max_stream_events: int = 128
    max_pending_commands: int = 256
    reasoning_parser: Literal["none", "think"] = "none"
    tool_parser: Literal["none", "hermes"] = "none"
    structured_outputs: bool = True
    expose_metrics: bool = True
    decode_graph: bool = False
    graph_buckets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.model or any(not key for key in self.api_keys):
            raise ValueError("model and configured API keys must be nonempty")
        if self.max_concurrent_requests < 0:
            raise ValueError("max_concurrent_requests must be nonnegative")
        for name in ("max_request_bytes", "max_stream_events", "max_pending_commands"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.reasoning_parser not in ("none", "think"):
            raise ValueError("unknown reasoning parser")
        if self.tool_parser not in ("none", "hermes"):
            raise ValueError("unknown tool parser")
        if self.graph_buckets is not None:
            buckets = tuple(self.graph_buckets)
            if not buckets or any(
                not isinstance(bucket, int) or isinstance(bucket, bool) or bucket < 1
                for bucket in buckets
            ):
                raise ValueError("graph_buckets must contain positive integers")
            if tuple(sorted(set(buckets))) != buckets:
                raise ValueError("graph_buckets must be ascending and deduplicated")
