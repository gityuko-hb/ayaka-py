"""Server policy, separate from scheduler and model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ayaka.configs.phase import RequestedOverrides
    from ayaka.configs.speculative import SpeculativeConfig
    from ayaka.lora.binding import LoRAConfig
    from ayaka.worker.execution_lane import ExecutionLaneConfig


@dataclass(frozen=True, slots=True)
class SloTargets:
    """Optional p95 performance targets; never a correctness gate.

    A measured p95 above its target records ``FAIL_PERFORMANCE`` in the SLO
    summary.  Thresholds are frozen per hardware/model/workload by the caller;
    no default assumes a GPU speed.
    """

    queue_latency_p95_ms: float | None = None
    tokenizer_p95_ms: float | None = None
    service_ttft_p95_ms: float | None = None
    itl_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    e2e_p95_ms: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "queue_latency_p95_ms",
            "tokenizer_p95_ms",
            "service_ttft_p95_ms",
            "itl_p95_ms",
            "tpot_p95_ms",
            "e2e_p95_ms",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be positive or None")


@dataclass(frozen=True, slots=True)
class ServingConfig:
    model: str = "ayaka"
    api_keys: tuple[str, ...] = ()
    max_concurrent_requests: int = 0
    max_admitted_requests: int = 0
    max_request_bytes: int = 4 << 20
    max_request_chunks: int = 4096
    max_stream_events: int = 128
    max_stream_bytes: int = 1 << 20
    max_pending_commands: int = 256
    #: Dedicated control-channel budget; 0 reuses ``max_pending_commands``.
    max_pending_controls: int = 0
    max_parser_bytes: int = 1 << 20
    #: Bounded SLO history: the most recent settled requests behind /stats.
    slo_history_size: int = 128
    default_request_timeout_seconds: float | None = None
    default_queue_timeout_seconds: float | None = None
    slo_targets: SloTargets | None = None
    reasoning_parser: Literal["none", "think"] = "none"
    tool_parser: Literal["none", "hermes"] = "none"
    structured_outputs: bool = True
    expose_metrics: bool = True
    decode_graph: bool | None = None
    graph_buckets: tuple[int, ...] | None = None
    execution: RequestedOverrides | None = None
    lora: LoRAConfig | None = None
    speculative: SpeculativeConfig | None = None
    execution_lanes: ExecutionLaneConfig | None = None

    def __post_init__(self) -> None:
        from ayaka.configs.phase import RequestedOverrides
        from ayaka.configs.speculative import SpeculativeConfig
        from ayaka.lora.binding import LoRAConfig
        from ayaka.worker.execution_lane import ExecutionLaneConfig

        if self.lora is not None and not isinstance(self.lora, LoRAConfig):
            raise TypeError("lora must be LoRAConfig or None")
        if self.speculative is not None and not isinstance(self.speculative, SpeculativeConfig):
            raise TypeError("speculative must be SpeculativeConfig or None")
        if self.speculative is not None and self.lora is not None:
            raise ValueError("speculative x LoRA is not certified")
        if self.execution_lanes is not None:
            if not isinstance(self.execution_lanes, ExecutionLaneConfig):
                raise TypeError("execution_lanes must be ExecutionLaneConfig or None")
            if self.speculative is not None or self.lora is not None:
                raise ValueError("PDMux x LoRA/speculative is not certified")

        if self.decode_graph is not None and type(self.decode_graph) is not bool:
            raise TypeError("decode_graph must be boolean or None")
        if self.execution is not None and not isinstance(self.execution, RequestedOverrides):
            raise TypeError("execution must be RequestedOverrides or None")
        if not self.model or any(not key for key in self.api_keys):
            raise ValueError("model and configured API keys must be nonempty")
        if self.max_concurrent_requests < 0:
            raise ValueError("max_concurrent_requests must be nonnegative")
        if self.max_admitted_requests < 0:
            raise ValueError("max_admitted_requests must be nonnegative")
        if self.max_pending_controls < 0:
            raise ValueError("max_pending_controls must be nonnegative")
        if self.default_request_timeout_seconds is not None and (
            not isinstance(self.default_request_timeout_seconds, (int, float))
            or isinstance(self.default_request_timeout_seconds, bool)
            or self.default_request_timeout_seconds <= 0
        ):
            raise ValueError("default_request_timeout_seconds must be positive or None")
        if self.default_queue_timeout_seconds is not None and (
            not isinstance(self.default_queue_timeout_seconds, (int, float))
            or isinstance(self.default_queue_timeout_seconds, bool)
            or self.default_queue_timeout_seconds <= 0
        ):
            raise ValueError("default_queue_timeout_seconds must be positive or None")
        if self.slo_targets is not None and not isinstance(self.slo_targets, SloTargets):
            raise TypeError("slo_targets must be SloTargets or None")
        for name in (
            "max_request_bytes",
            "max_request_chunks",
            "max_stream_events",
            "max_stream_bytes",
            "max_pending_commands",
            "max_parser_bytes",
            "slo_history_size",
        ):
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
