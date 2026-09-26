"""Memory policy and deterministic per-device budget planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from ayaka.configs.base import ConfigError, ConfigMixin

__all__ = [
    "DeviceMemoryPlan",
    "HostMemoryLimits",
    "MemoryConfig",
    "MemoryProfile",
    "plan_device_memory",
]


@dataclass(frozen=True, slots=True)
class HostMemoryLimits(ConfigMixin):
    """Operator ceilings for pinned host memory, resolved by machine facts.

    Config-time numbers only: the effective limit also depends on cgroup,
    ``MemAvailable``, PSI and NUMA facts that are only knowable at bootstrap.
    :class:`ayaka.memory.host.host_policy.HostMemoryPolicy` is the single
    authority that combines both.
    """

    pinned_max_bytes: int | None = None
    pinned_max_ratio: float = 0.25
    min_available_bytes: int = 4 << 30
    min_available_ratio: float = 0.10
    #: Whether a refused pinned host mirror may degrade to pageable memory.
    #: ``False`` turns the pin ceiling into a hard startup refusal.
    allow_pageable_fallback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.allow_pageable_fallback, bool):
            raise ConfigError(
                "memory.host.allow_pageable_fallback",
                "HOST_FALLBACK_INVALID",
                "allow_pageable_fallback must be a bool",
            )
        if self.pinned_max_bytes is not None and self.pinned_max_bytes < 0:
            raise ConfigError(
                "memory.host.pinned_max_bytes",
                "PINNED_LIMIT_NEGATIVE",
                "pinned memory limit must be non-negative",
            )
        if not 0 < self.pinned_max_ratio <= 1:
            raise ConfigError(
                "memory.host.pinned_max_ratio",
                "PINNED_RATIO_INVALID",
                "pinned memory ratio must be in (0, 1]",
            )
        if self.min_available_bytes < 0 or not 0 <= self.min_available_ratio < 1:
            raise ConfigError(
                "memory.host",
                "HOST_RESERVE_INVALID",
                "host reserve bytes must be non-negative and ratio must be in [0, 1)",
            )


@dataclass(frozen=True, slots=True)
class MemoryConfig(ConfigMixin):
    gpu_memory_utilization: float = 0.90
    kv_cache_memory_bytes: int | None = None
    gpu_reserved_bytes: int = 0
    activation_reserve_bytes: int = 0
    host: HostMemoryLimits = field(default_factory=HostMemoryLimits)

    def __post_init__(self) -> None:
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ConfigError(
                "memory.gpu_memory_utilization",
                "GPU_UTILIZATION_INVALID",
                "gpu_memory_utilization must be in (0, 1]",
            )
        for name in ("gpu_reserved_bytes", "activation_reserve_bytes"):
            if getattr(self, name) < 0:
                raise ConfigError(
                    f"memory.{name}", "MEMORY_RESERVE_NEGATIVE", f"{name} must be >= 0"
                )
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes < 1:
            raise ConfigError(
                "memory.kv_cache_memory_bytes",
                "KV_MEMORY_OVERRIDE_INVALID",
                "explicit KV cache memory must be positive",
            )


@dataclass(frozen=True, slots=True)
class MemoryProfile(ConfigMixin):
    """Measured non-KV allocations; zeros mean an optimistic preflight profile.

    ``activation_bytes`` is the measured bootstrap activation peak;
    ``activation_reserve_bytes`` on :class:`MemoryConfig` is additional
    headroom. Both may be set on purpose and each is subtracted exactly once.
    """

    weights_bytes: int = 0
    activation_bytes: int = 0
    graph_pool_bytes: int = 0
    kernel_workspace_bytes: int = 0
    collective_bytes: int = 0
    runtime_bytes: int = 0
    allocator_fragmentation_bytes: int = 0
    measured: bool = False
    #: Persistent per-flight runner metadata token input, attention
    #: metadata and sampling-row buffers. Reserved exactly once at bootstrap.
    runner_buffer_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "weights_bytes",
            "activation_bytes",
            "graph_pool_bytes",
            "kernel_workspace_bytes",
            "collective_bytes",
            "runtime_bytes",
            "allocator_fragmentation_bytes",
            "runner_buffer_bytes",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(
                    f"memory.profile.{name}", "MEMORY_PROFILE_NEGATIVE", f"{name} must be >= 0"
                )

    @property
    def non_kv_bytes(self) -> int:
        return sum(
            (
                self.weights_bytes,
                self.activation_bytes,
                self.graph_pool_bytes,
                self.kernel_workspace_bytes,
                self.collective_bytes,
                self.runtime_bytes,
                self.allocator_fragmentation_bytes,
                self.runner_buffer_bytes,
            )
        )


@dataclass(frozen=True, slots=True)
class DeviceMemoryPlan(ConfigMixin):
    total_bytes: int
    policy_budget_bytes: int
    non_kv_bytes: int
    fixed_reserve_bytes: int
    kv_cache_bytes: int
    measured: bool

    @property
    def optimistic(self) -> bool:
        return not self.measured


def plan_device_memory(
    total_bytes: int,
    config: MemoryConfig,
    profile: MemoryProfile | None = None,
) -> DeviceMemoryPlan:
    if total_bytes < 1:
        raise ConfigError(
            "memory.total_bytes", "DEVICE_MEMORY_UNKNOWN", "device memory must be positive"
        )
    profile = profile or MemoryProfile()
    policy_budget = int(total_bytes * config.gpu_memory_utilization)
    fixed_reserve = config.gpu_reserved_bytes + config.activation_reserve_bytes
    available = policy_budget - profile.non_kv_bytes - fixed_reserve
    if available < 0:
        raise ConfigError(
            "memory",
            "NON_KV_MEMORY_EXCEEDS_BUDGET",
            "weights, workspaces and reserves exceed the configured GPU memory budget",
            context={"available_bytes": available},
        )
    kv_bytes = config.kv_cache_memory_bytes or available
    if kv_bytes > available:
        raise ConfigError(
            "memory.kv_cache_memory_bytes",
            "KV_MEMORY_EXCEEDS_BUDGET",
            "explicit KV cache allocation exceeds memory remaining after non-KV allocations",
            context={"requested_bytes": kv_bytes, "available_bytes": available},
        )
    return DeviceMemoryPlan(
        total_bytes=total_bytes,
        policy_budget_bytes=policy_budget,
        non_kv_bytes=profile.non_kv_bytes,
        fixed_reserve_bytes=fixed_reserve,
        kv_cache_bytes=kv_bytes,
        measured=profile.measured,
    )
