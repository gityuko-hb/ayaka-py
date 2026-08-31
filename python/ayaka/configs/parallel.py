from __future__ import annotations

import enum
from dataclasses import dataclass

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.configs.model import ArchitectureConfig


class AllToAllBackend(enum.StrEnum):
    """Backend implementations for All-to-All collective communication primitives.

    Used by distributed runtimes to route tokens during Mixture-of-Experts (MoE)
    expert dispatch/combine phases and DeepSpeed Ulysses sequence-parallel transposes.
    """

    AUTO = "auto"
    """Automatically selects the optimal backend based on hardware topology and library availability."""

    NCCL = "nccl"
    """NVIDIA Collective Communications Library.
    Standard collective baseline with cross-platform stability over NVLink and InfiniBand/RoCE."""

    NVSHMEM = "nvshmem"
    """NVIDIA OpenSHMEM Partitioned Global Address Space (PGAS) runtime.
    Provides low-latency one-sided memory transfers for fine-grained MoE routing."""

    DEEPEP = "deepep"
    """DeepSeek Expert Parallelism communication library.
    Highly optimized for MoE dispatch/combine with fine-grained SM-to-SM overlapping."""

    @property
    def is_moe_specialized(self) -> bool:
        """Whether this backend provides specialized kernel optimizations for MoE token routing."""
        return self in (AllToAllBackend.DEEPEP, AllToAllBackend.NVSHMEM)

    @property
    def uses_pgas(self) -> bool:
        """Whether this backend utilizes a Partitioned Global Address Space memory model."""
        return self is AllToAllBackend.NVSHMEM

    @property
    def supports_computation_overlap(self) -> bool:
        """Whether the backend natively supports fine-grained communication-computation pipelining."""
        return self in (AllToAllBackend.DEEPEP, AllToAllBackend.NVSHMEM)