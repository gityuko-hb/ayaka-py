"""Physical cluster topology, rank placement and launch/failure policy."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.distributed.device import DeviceRef
from ayaka.types import DeviceKind

__all__ = [
    "CollectiveBackend",
    "DistributedRuntimeConfig",
    "ExecutorBackend",
    "FaultToleranceConfig",
    "RankPlacement",
    "ResolvedDistributedPlan",
]


class CollectiveBackend(enum.StrEnum):
    NONE = "none"
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


class ExecutorBackend(enum.StrEnum):
    LOCAL = "local"
    MULTIPROCESSING = "multiprocessing"
    RAY = "ray"
    EXTERNAL = "external_launcher"


@dataclass(frozen=True, slots=True)
class RankPlacement(ConfigMixin):
    global_rank: int
    node_rank: int
    local_rank: int
    device_index: int

    def __post_init__(self) -> None:
        if min(self.global_rank, self.node_rank, self.local_rank, self.device_index) < 0:
            raise ConfigError(
                "distributed.placements",
                "RANK_PLACEMENT_NEGATIVE",
                "rank and device indices must be non-negative",
            )


@dataclass(frozen=True, slots=True)
class FaultToleranceConfig(ConfigMixin):
    max_worker_restarts: int = 0
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 30.0
    fail_fast_on_rank_loss: bool = True

    def __post_init__(self) -> None:
        if self.max_worker_restarts < 0:
            raise ConfigError(
                "distributed.fault_tolerance.max_worker_restarts",
                "RESTART_COUNT_NEGATIVE",
                "max worker restarts must be non-negative",
            )
        if self.heartbeat_interval_s <= 0 or self.heartbeat_timeout_s <= 0:
            raise ConfigError(
                "distributed.fault_tolerance",
                "HEARTBEAT_INTERVAL_INVALID",
                "heartbeat intervals must be positive",
            )
        if self.heartbeat_timeout_s <= self.heartbeat_interval_s:
            raise ConfigError(
                "distributed.fault_tolerance.heartbeat_timeout_s",
                "HEARTBEAT_TIMEOUT_TOO_SMALL",
                "heartbeat timeout must exceed heartbeat interval",
            )


@dataclass(frozen=True, slots=True)
class DistributedRuntimeConfig(ConfigMixin):
    num_nodes: int = 1
    node_rank: int = 0
    local_world_size: int | None = None
    collective_backend: CollectiveBackend = CollectiveBackend.NONE
    executor_backend: ExecutorBackend = ExecutorBackend.LOCAL
    init_method: str | None = None
    master_addr: str | None = None
    master_port: int | None = None
    init_timeout_s: float = 300.0
    placements: tuple[RankPlacement, ...] = ()
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)

    def __post_init__(self) -> None:
        if self.num_nodes < 1 or not 0 <= self.node_rank < self.num_nodes:
            raise ConfigError(
                "distributed.node_rank",
                "NODE_RANK_INVALID",
                "node_rank must be in [0, num_nodes)",
            )
        if self.local_world_size is not None and self.local_world_size < 1:
            raise ConfigError(
                "distributed.local_world_size",
                "LOCAL_WORLD_SIZE_INVALID",
                "local world size must be positive",
            )
        if self.init_timeout_s <= 0:
            raise ConfigError(
                "distributed.init_timeout_s",
                "DISTRIBUTED_TIMEOUT_INVALID",
                "distributed init timeout must be positive",
            )
        if self.master_port is not None and not 1 <= self.master_port <= 65535:
            raise ConfigError(
                "distributed.master_port", "MASTER_PORT_INVALID", "master port is outside 1..65535"
            )
        if self.placements:
            global_ranks = [item.global_rank for item in self.placements]
            if len(global_ranks) != len(set(global_ranks)):
                raise ConfigError(
                    "distributed.placements",
                    "GLOBAL_RANK_DUPLICATE",
                    "global ranks must be unique",
                )
            local_pairs = [(item.node_rank, item.local_rank) for item in self.placements]
            if len(local_pairs) != len(set(local_pairs)):
                raise ConfigError(
                    "distributed.placements",
                    "LOCAL_RANK_DUPLICATE",
                    "local rank must be unique within a node",
                )

    def validate_world_size(self, world_size: int) -> None:
        if world_size < 1:
            raise ConfigError(
                "parallel.world_size", "WORLD_SIZE_INVALID", "world size must be positive"
            )
        if world_size > 1 and self.collective_backend is CollectiveBackend.NONE:
            raise ConfigError(
                "distributed.collective_backend",
                "COLLECTIVE_BACKEND_REQUIRED",
                "multi-rank execution requires a collective backend",
            )
        if self.placements:
            ranks = sorted(item.global_rank for item in self.placements)
            if ranks != list(range(world_size)):
                raise ConfigError(
                    "distributed.placements",
                    "GLOBAL_RANK_COVERAGE_INVALID",
                    "placements must cover every global rank exactly once",
                )
            if any(item.node_rank >= self.num_nodes for item in self.placements):
                raise ConfigError(
                    "distributed.placements",
                    "PLACEMENT_NODE_OUT_OF_RANGE",
                    "placement references a node outside num_nodes",
                )
            return
        self.derive_local_world_size(world_size)

    def derive_local_world_size(self, world_size: int) -> int:
        """Resolve the rank count for one node from the global mesh.

        This is the sole derivation used by validation, placement synthesis and
        engine resolution.  Keeping it here prevents the old split-brain bug
        where validation assumed ``world_size`` but device mapping assumed 1.
        """

        if world_size < 1:
            raise ConfigError(
                "parallel.world_size", "WORLD_SIZE_INVALID", "world size must be positive"
            )
        if self.local_world_size is not None:
            local = self.local_world_size
        else:
            local, remainder = divmod(world_size, self.num_nodes)
            if remainder:
                raise ConfigError(
                    "distributed.local_world_size",
                    "LOCAL_WORLD_NOT_DERIVABLE",
                    "global world size must divide num_nodes when local_world_size is omitted",
                )
        if local * self.num_nodes != world_size:
            raise ConfigError(
                "distributed.local_world_size",
                "LOCAL_GLOBAL_WORLD_MISMATCH",
                "local_world_size * num_nodes must equal global world size "
                "without explicit placements",
            )
        return local

    def local_rank_count(self, world_size: int | None = None) -> int:
        if self.placements:
            return sum(item.node_rank == self.node_rank for item in self.placements)
        if self.local_world_size is not None:
            return self.local_world_size
        if world_size is None:
            raise ConfigError(
                "distributed.local_world_size",
                "GLOBAL_WORLD_REQUIRED_FOR_DERIVATION",
                "world_size is required when local_world_size and placements are omitted",
            )
        return self.derive_local_world_size(world_size)

    def local_device_indices(self, world_size: int | None = None) -> tuple[int, ...]:
        if not self.placements:
            return tuple(range(self.local_rank_count(world_size)))
        return tuple(
            item.device_index
            for item in sorted(
                (item for item in self.placements if item.node_rank == self.node_rank),
                key=lambda item: item.local_rank,
            )
        )

    def resolved_placements(self, world_size: int) -> tuple[RankPlacement, ...]:
        self.validate_world_size(world_size)
        if self.placements:
            return tuple(sorted(self.placements, key=lambda item: item.global_rank))
        local = self.derive_local_world_size(world_size)
        return tuple(
            RankPlacement(
                global_rank=node * local + local_rank,
                node_rank=node,
                local_rank=local_rank,
                device_index=local_rank,
            )
            for node in range(self.num_nodes)
            for local_rank in range(local)
        )

    def resolve(self, world_size: int) -> ResolvedDistributedPlan:
        placements = self.resolved_placements(world_size)
        local = tuple(item for item in placements if item.node_rank == self.node_rank)
        init_method = self.init_method or ("local://" if world_size == 1 else "env://")
        return ResolvedDistributedPlan(
            world_size=world_size,
            num_nodes=self.num_nodes,
            node_rank=self.node_rank,
            local_world_size=len(local),
            collective_backend=self.collective_backend,
            executor_backend=self.executor_backend,
            init_method=init_method,
            master_addr=self.master_addr,
            master_port=self.master_port,
            init_timeout_s=self.init_timeout_s,
            placements=placements,
            local_placements=local,
            fault_tolerance=self.fault_tolerance,
        )


@dataclass(frozen=True, slots=True)
class ResolvedDistributedPlan(ConfigMixin):
    """Concrete launch and rank placement ABI consumed by workers."""

    world_size: int
    num_nodes: int
    node_rank: int
    local_world_size: int
    collective_backend: CollectiveBackend
    executor_backend: ExecutorBackend
    init_method: str
    master_addr: str | None
    master_port: int | None
    init_timeout_s: float
    placements: tuple[RankPlacement, ...]
    local_placements: tuple[RankPlacement, ...]
    fault_tolerance: FaultToleranceConfig

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ConfigError(
                "distributed.world_size",
                "WORLD_SIZE_INVALID",
                "resolved world size must be positive",
            )
        if tuple(item.global_rank for item in self.placements) != tuple(range(self.world_size)):
            raise ConfigError(
                "distributed.placements",
                "RESOLVED_RANK_COVERAGE_INVALID",
                "resolved placements must cover every global rank exactly once in order",
            )
        if self.local_world_size != len(self.local_placements):
            raise ConfigError(
                "distributed.local_placements",
                "RESOLVED_LOCAL_WORLD_MISMATCH",
                "local placement count must equal resolved local world size",
            )
        if any(item.node_rank != self.node_rank for item in self.local_placements):
            raise ConfigError(
                "distributed.local_placements",
                "RESOLVED_LOCAL_NODE_MISMATCH",
                "local placements must belong to the resolved node",
            )

    def placement_for(self, global_rank: int) -> RankPlacement:
        """Placement record for one global rank."""
        if not 0 <= global_rank < self.world_size:
            raise ConfigError(
                "distributed.placements",
                "GLOBAL_RANK_OUT_OF_RANGE",
                f"global rank {global_rank} is outside [0, {self.world_size})",
            )
        return self.placements[global_rank]

    def device_refs(self, device_kind: DeviceKind = DeviceKind.CUDA) -> tuple[DeviceRef, ...]:
        """One device reference per global rank, in rank order."""
        return tuple(
            DeviceRef(kind=device_kind, index=placement.device_index)
            for placement in self.placements
        )
