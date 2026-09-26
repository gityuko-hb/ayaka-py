from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.configs.model import ArchitectureConfig
from ayaka.utils.import_utils import CapabilityError, has_module

__all__ = [
    "AllToAllBackend",
    "AxisRanks",
    "CollectivePolicy",
    "ParallelConfig",
    "ResolvedParallelPlan",
]


class AllToAllBackend(enum.StrEnum):
    """Backend implementations for All-to-All collective communication primitives.

    Used by distributed runtimes to route tokens during Mixture-of-Experts (MoE)
    expert dispatch/combine phases and DeepSpeed Ulysses sequence-parallel transposes.
    """

    AUTO = "auto"
    """Automatic select the optimal backend-base on hardware topology and library availability."""

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
        """Whether the backend native supports fine-grained communication-computation pipelining."""
        return self in (AllToAllBackend.DEEPEP, AllToAllBackend.NVSHMEM)

    @property
    def required_module(self) -> str | None:
        """Optional runtime module backing this backend; ``None`` when built into torch."""
        if self is AllToAllBackend.NVSHMEM:
            return "nvshmem"
        if self is AllToAllBackend.DEEPEP:
            return "deep_ep"
        return None

    def is_available(self) -> bool:
        """Whether the backend's optional runtime module is importable here."""
        module = self.required_module
        return module is None or has_module(module)

    def require_available(self) -> None:
        """Refuse the backend before execution when its runtime module is missing."""
        module = self.required_module
        if module is not None and not has_module(module):
            raise CapabilityError(
                f"parallel.all_to_all.{self.value}",
                detail=f"runtime module {module!r} is not importable",
                remedy=f"install {module} or select AllToAllBackend.NCCL",
            )

    @classmethod
    def preferred_moe_backend(
        cls, *, available: Callable[[AllToAllBackend], bool] | None = None
    ) -> AllToAllBackend:
        """Select the best available backend for MoE dispatch/combine.

        Prefers specialized kernels (DEEPEP, then NVSHMEM) over the NCCL
        collective baseline. ``available`` overrides the module probe so tests
        and capability gates can inject their own result.
        """
        probe = available if available is not None else (lambda backend: backend.is_available())
        for backend in (cls.DEEPEP, cls.NVSHMEM):
            if probe(backend):
                return backend
        return cls.NCCL


class CollectivePolicy(enum.StrEnum):
    """User-facing selection policy for the tensor-parallel collective backend.

    ``torch`` is the compatible default and the operational rollback. ``auto``
    is allowed to fall to Torch when the group cannot agree on a custom backend
    *before* any data launch. ``custom_required`` refuses the step instead of
    silently changing the decision. The policy itself is DC3 metadata; the
    physical process-group backend lives in ``CollectiveBackend``.
    """

    TORCH = "torch"
    """Only the Torch/NCCL reference backend; no custom communicator is built."""

    AUTO = "auto"
    """Use an agreed custom backend when the whole group proved the capability,
    otherwise delegate every unsupported call to Torch before launch."""

    CUSTOM_REQUIRED = "custom_required"
    """Refuse any case the agreed custom backend cannot serve; never fall back."""


@dataclass(frozen=True, slots=True)
class AxisRanks:
    """One world rank decomposed into its per-axis coordinates."""

    tp_rank: int
    pp_rank: int
    dp_rank: int
    ep_rank: int
    cp_rank: int

    def __post_init__(self) -> None:
        for name in ("tp_rank", "pp_rank", "dp_rank", "ep_rank", "cp_rank"):
            if getattr(self, name) < 0:
                raise ConfigError(
                    f"parallel.{name}", "AXIS_RANK_NEGATIVE", f"{name} must be non-negative"
                )


def _validate_layout(
    layout: ParallelConfig | ResolvedParallelPlan, arch: ArchitectureConfig
) -> None:
    """Check a parallel layout against a concrete model geometry.

    Shared by the user-facing config and the resolved plan so a hand-built plan
    cannot reach allocation without the same divisibility and coverage checks.
    """
    if arch.num_attention_heads % layout.tp_size:
        raise ConfigError(
            "parallel.tp_size",
            "ATTENTION_HEADS_NOT_TP_DIVISIBLE",
            "tp_size must divide attention heads",
        )
    for group in arch.cache_groups:
        group.local_kv_heads(layout.tp_size)
    if arch.intermediate_size % layout.tp_size:
        raise ConfigError(
            "parallel.tp_size",
            "INTERMEDIATE_NOT_TP_DIVISIBLE",
            "tp_size must divide the dense intermediate size",
        )
    if layout.pp_size > arch.num_layers:
        raise ConfigError(
            "parallel.pp_size",
            "PP_EXCEEDS_LAYERS",
            "pipeline stages cannot exceed model layers",
        )
    if layout.pipeline_layer_ranges:
        expected_start = 0
        for stage, (start, end) in enumerate(layout.pipeline_layer_ranges):
            if start != expected_start or end <= start:
                raise ConfigError(
                    f"parallel.pipeline_layer_ranges.{stage}",
                    "PIPELINE_RANGES_NOT_CONTIGUOUS",
                    "manual PP ranges must be non-empty and contiguous",
                )
            expected_start = end
        if expected_start != arch.num_layers:
            raise ConfigError(
                "parallel.pipeline_layer_ranges",
                "PIPELINE_RANGES_INCOMPLETE",
                "manual PP ranges must cover every model layer exactly once",
            )
    if layout.ep_size > 1:
        if not arch.is_moe:
            raise ConfigError(
                "parallel.ep_size", "EP_ON_DENSE_MODEL", "expert parallelism requires MoE"
            )
        assert arch.num_experts is not None
        if arch.num_experts % layout.ep_size:
            raise ConfigError(
                "parallel.ep_size",
                "EXPERTS_NOT_EP_DIVISIBLE",
                "ep_size must divide num_experts",
            )
        if layout.ep_size > layout.world_size:
            raise ConfigError(
                "parallel.ep_size",
                "EP_EXCEEDS_WORLD",
                "EP shards experts over existing ranks and cannot exceed world size",
            )
    if layout.expert_redundancy and not arch.is_moe:
        raise ConfigError(
            "parallel.expert_redundancy",
            "EXPERT_REDUNDANCY_ON_DENSE_MODEL",
            "expert redundancy requires MoE",
        )


@dataclass(frozen=True, slots=True)
class ParallelConfig(ConfigMixin):
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    cp_size: int = 1
    enable_sequence_parallel: bool = False
    enable_dp_attention: bool = False
    all_to_all_backend: AllToAllBackend = AllToAllBackend.AUTO
    collective_policy: CollectivePolicy = CollectivePolicy.TORCH
    pipeline_microbatches: int = 1
    pipeline_layer_ranges: tuple[tuple[int, int], ...] = ()
    expert_redundancy: int = 0
    enable_expert_load_balancing: bool = False

    def __post_init__(self) -> None:
        for name in ("tp_size", "pp_size", "dp_size", "ep_size", "cp_size"):
            if getattr(self, name) < 1:
                raise ConfigError(
                    f"parallel.{name}", "PARALLEL_SIZE_INVALID", f"{name} must be >= 1"
                )
        if not isinstance(self.collective_policy, CollectivePolicy):
            raise ConfigError(
                "parallel.collective_policy",
                "COLLECTIVE_POLICY_INVALID",
                "collective policy must be one of CollectivePolicy",
            )
        if self.enable_sequence_parallel and self.tp_size == 1:
            raise ConfigError(
                "parallel.enable_sequence_parallel",
                "SEQUENCE_PARALLEL_WITHOUT_TP",
                "sequence parallelism requires tp_size > 1",
            )
        if self.enable_dp_attention and self.dp_size == 1:
            raise ConfigError(
                "parallel.enable_dp_attention",
                "DP_ATTENTION_WITHOUT_DP",
                "DP attention requires dp_size > 1",
            )
        if self.pipeline_microbatches < 1:
            raise ConfigError(
                "parallel.pipeline_microbatches",
                "PIPELINE_MICROBATCH_INVALID",
                "pipeline microbatches must be >= 1",
            )
        if self.pp_size > 1 and self.pipeline_microbatches < self.pp_size:
            raise ConfigError(
                "parallel.pipeline_microbatches",
                "PIPELINE_CANNOT_FILL",
                "pipeline_microbatches must be at least pp_size",
            )
        if self.pipeline_layer_ranges and len(self.pipeline_layer_ranges) != self.pp_size:
            raise ConfigError(
                "parallel.pipeline_layer_ranges",
                "PIPELINE_RANGE_COUNT_MISMATCH",
                "manual pipeline ranges must contain one range per PP stage",
            )
        for stage, span in enumerate(self.pipeline_layer_ranges):
            if (
                type(span) is not tuple
                or len(span) != 2
                or any(type(bound) is not int for bound in span)
            ):
                raise ConfigError(
                    f"parallel.pipeline_layer_ranges.{stage}",
                    "PIPELINE_RANGE_MALFORMED",
                    "each pipeline range must be a (start, end) integer pair",
                )
        if self.expert_redundancy < 0:
            raise ConfigError(
                "parallel.expert_redundancy",
                "EXPERT_REDUNDANCY_NEGATIVE",
                "expert redundancy must be non-negative",
            )
        if self.ep_size > 1 and (self.dp_factor * self.tp_size) % self.ep_size:
            raise ConfigError(
                "parallel.ep_size",
                "EP_NOT_RANK_DIVISIBLE",
                "ep_size must divide the flattened dp x tp rank plane",
            )

    @property
    def dp_factor(self) -> int:
        """Physical DP coordinate count; DP attention shares ranks instead of adding them."""
        return 1 if self.enable_dp_attention else self.dp_size

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.dp_factor * self.cp_size

    @property
    def logical_replica_count(self) -> int:
        return self.dp_size

    @property
    def is_single_process(self) -> bool:
        return self.world_size == 1 and self.ep_size == 1

    def validate_against(self, arch: ArchitectureConfig) -> None:
        _validate_layout(self, arch)

    def layer_range_for(self, arch: ArchitectureConfig, pp_rank: int) -> tuple[int, int]:
        if not 0 <= pp_rank < self.pp_size:
            raise ConfigError(
                "parallel.pp_rank", "PP_RANK_OUT_OF_RANGE", "PP rank is outside the mesh"
            )
        if self.pipeline_layer_ranges:
            return self.pipeline_layer_ranges[pp_rank]
        base, extra = divmod(arch.num_layers, self.pp_size)
        start = pp_rank * base + min(pp_rank, extra)
        return start, start + base + (1 if pp_rank < extra else 0)

    def resolve(self, arch: ArchitectureConfig) -> ResolvedParallelPlan:
        self.validate_against(arch)
        all_to_all: AllToAllBackend | None = None
        if arch.is_moe and self.ep_size > 1:
            all_to_all = (
                AllToAllBackend.preferred_moe_backend()
                if self.all_to_all_backend is AllToAllBackend.AUTO
                else self.all_to_all_backend
            )
            all_to_all.require_available()
        return ResolvedParallelPlan(
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            cp_size=self.cp_size,
            world_size=self.world_size,
            logical_replica_count=self.logical_replica_count,
            sequence_parallel=self.enable_sequence_parallel,
            dp_attention=self.enable_dp_attention,
            all_to_all_backend=all_to_all,
            pipeline_microbatches=self.pipeline_microbatches,
            pipeline_layer_ranges=tuple(
                self.layer_range_for(arch, rank) for rank in range(self.pp_size)
            ),
            expert_redundancy=self.expert_redundancy,
            expert_load_balancing=self.enable_expert_load_balancing,
            collective_policy=self.collective_policy,
        )


@dataclass(frozen=True, slots=True)
class ResolvedParallelPlan(ConfigMixin):
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    cp_size: int
    world_size: int
    logical_replica_count: int
    sequence_parallel: bool
    dp_attention: bool
    all_to_all_backend: AllToAllBackend | None
    pipeline_microbatches: int
    pipeline_layer_ranges: tuple[tuple[int, int], ...]
    expert_redundancy: int
    expert_load_balancing: bool
    collective_policy: CollectivePolicy = CollectivePolicy.TORCH

    def __post_init__(self) -> None:
        if not isinstance(self.collective_policy, CollectivePolicy):
            raise ConfigError(
                "parallel.collective_policy",
                "RESOLVED_COLLECTIVE_POLICY_INVALID",
                "resolved collective policy must be one of CollectivePolicy",
            )
        if len(self.pipeline_layer_ranges) != self.pp_size:
            raise ConfigError(
                "parallel.pipeline_layer_ranges",
                "RESOLVED_PIPELINE_RANGE_COUNT_MISMATCH",
                "resolved plan must contain one layer range per PP stage",
            )
        if self.ep_size > 1 and self.all_to_all_backend is None:
            raise ConfigError(
                "parallel.all_to_all_backend",
                "RESOLVED_ALL_TO_ALL_MISSING",
                "expert parallel execution requires a concrete all-to-all backend",
            )
        if self.world_size != self.tp_size * self.pp_size * self.dp_factor * self.cp_size:
            raise ConfigError(
                "parallel.world_size",
                "RESOLVED_WORLD_SIZE_MISMATCH",
                "resolved world size must equal tp_size * pp_size * dp_factor * cp_size",
            )
        if self.ep_size > 1 and (self.dp_factor * self.tp_size) % self.ep_size:
            raise ConfigError(
                "parallel.ep_size",
                "EP_NOT_RANK_DIVISIBLE",
                "ep_size must divide the flattened dp x tp rank plane",
            )

    @property
    def dp_factor(self) -> int:
        """Physical DP coordinate count; DP attention shares ranks instead of adding them."""
        return 1 if self.dp_attention else self.dp_size

    def validate_against(self, arch: ArchitectureConfig) -> None:
        """Recheck this frozen plan against a model geometry before allocation."""
        _validate_layout(self, arch)

    def _decompose(self, world_rank: int) -> AxisRanks:
        tp_rank = world_rank % self.tp_size
        rest = world_rank // self.tp_size
        dp_rank = rest % self.dp_factor
        rest //= self.dp_factor
        pp_rank = rest % self.pp_size
        cp_rank = rest // self.pp_size
        ep_rank = (dp_rank * self.tp_size + tp_rank) % self.ep_size
        return AxisRanks(tp_rank, pp_rank, dp_rank, ep_rank, cp_rank)

    def axis_ranks(self, world_rank: int) -> AxisRanks:
        """Decompose a world rank; tp is the fastest axis and cp the slowest."""
        if not 0 <= world_rank < self.world_size:
            raise ConfigError(
                "parallel.world_rank",
                "WORLD_RANK_OUT_OF_RANGE",
                f"world rank {world_rank} is outside [0, {self.world_size})",
            )
        return self._decompose(world_rank)

    def _ranks_sharing(self, origin: AxisRanks, axes: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(
            rank
            for rank in range(self.world_size)
            if all(getattr(self._decompose(rank), axis) == getattr(origin, axis) for axis in axes)
        )

    def tp_group_ranks(self, world_rank: int) -> tuple[int, ...]:
        """Global ranks of the tensor-parallel collective owning ``world_rank``."""
        return self._ranks_sharing(self.axis_ranks(world_rank), ("pp_rank", "dp_rank", "cp_rank"))

    def ep_group_ranks(self, world_rank: int) -> tuple[int, ...]:
        """Global ranks of the expert-parallel group owning ``world_rank``.

        EP spans the flattened ``dp x tp`` plane inside each ``(pp, cp)`` slice:
        contiguous stripes of ``ep_size`` ranks share one expert placement.
        """
        origin = self.axis_ranks(world_rank)
        origin_stripe = (origin.dp_rank * self.tp_size + origin.tp_rank) // self.ep_size
        ranks: list[int] = []
        for rank in range(self.world_size):
            coords = self._decompose(rank)
            stripe = (coords.dp_rank * self.tp_size + coords.tp_rank) // self.ep_size
            if (
                coords.pp_rank == origin.pp_rank
                and coords.cp_rank == origin.cp_rank
                and stripe == origin_stripe
            ):
                ranks.append(rank)
        return tuple(ranks)

    def layer_range(self, pp_rank: int) -> tuple[int, int]:
        """Half-open layer span executed by one pipeline stage."""
        if not 0 <= pp_rank < self.pp_size:
            raise ConfigError(
                "parallel.pp_rank", "PP_RANK_OUT_OF_RANGE", "PP rank is outside the mesh"
            )
        return self.pipeline_layer_ranges[pp_rank]
