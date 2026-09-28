"""Lane-local stream incarnation and immutable resource ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import torch

from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class ExecutionLaneConfig:
    """Two dedicated phase lanes; mixed steps use prefill, CPU resolves serialized.

    Each lane has one stream and separate runner/backend/graph state. Flight
    buffers stay with the existing ticket allocator. Submitted work never waits
    in a second host queue, so finite scheduler prefill chunks cannot starve here.
    """

    count: int = 2
    memory_bytes_per_lane: int = 256 << 20
    max_inflight_per_lane: int = 2
    policy: str = "concurrent"
    max_decode_burst: int = 8

    def __post_init__(self) -> None:
        require_int(self.count, "lane count", minimum=1)
        if self.count != 2:
            raise ValueError("PDMux currently requires exactly two phase lanes")
        require_int(self.memory_bytes_per_lane, "lane memory quota", minimum=1)
        require_int(self.max_inflight_per_lane, "lane max_inflight", minimum=1)
        require_int(self.max_decode_burst, "max_decode_burst", minimum=1)
        if self.policy not in ("concurrent", "serialized"):
            raise ValueError("lane policy must be concurrent or serialized")


class ExecutionLane:
    def __init__(self, name: str, runner: Any, config: ExecutionLaneConfig) -> None:
        self.name, self.runner, self.config = name, runner, config
        self.generation = uuid4().hex
        self.stream = (
            torch.cuda.Stream(device=runner.device, priority=-1 if name == "decode" else 0)
            if runner.device.type == "cuda"
            else None
        )
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(runner.device))
        self.submitted = self.active = self.dependency_waits = 0

    @property
    def identity(self) -> tuple:
        return (
            self.name,
            self.generation,
            id(self.runner),
            None if self.stream is None else self.stream.cuda_stream,
        )
