"""KV, prefix, tiering and hybrid state-cache policy."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.types import DType, KVLayoutKind

if TYPE_CHECKING:
    from ayaka.configs.model import ArchitectureConfig

__all__ = [
    "CacheConfig",
    "CacheGroupGeometry",
    "CacheGroupPlan",
    "CacheLayerKind",
    "CachePlan",
    "EvictionPolicy",
    "HybridCacheConfig",
    "HybridCacheMode",
    "KVScaleGranularity",
    "KVScalePolicy",
    "PrefixCacheConfig",
    "PrefixHashAlgorithm",
    "PrefixReuseMode",
    "PhysicalCacheGroupPlan",
    "PhysicalCachePlan",
    "TieredCacheConfig",
    "plan_cache",
]


class EvictionPolicy(enum.StrEnum):
    LRU = "lru"
    LFU = "lfu"
    SLRU = "slru"
    PREFIX_AWARE = "prefix_aware"
    COST_AWARE = "cost_aware"
    PRIORITY = "priority"


class PrefixHashAlgorithm(enum.StrEnum):
    SHA256 = "sha256"
    CBOR_SHA256 = "cbor_sha256"
    XXHASH = "xxhash"


class PrefixReuseMode(enum.StrEnum):
    FULL_ATTENTION_ONLY = "full_attention_only"
    PER_GROUP = "per_group"


@dataclass(frozen=True, slots=True)
class PrefixCacheConfig(ConfigMixin):
    enabled: bool = True
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    hash_algorithm: PrefixHashAlgorithm = PrefixHashAlgorithm.SHA256
    reuse_mode: PrefixReuseMode = PrefixReuseMode.FULL_ATTENTION_ONLY
    allow_sliding_window_reuse: bool = False
    allow_mamba_state_reuse: bool = False
    max_cached_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_cached_tokens is not None and self.max_cached_tokens < 1:
            raise ConfigError(
                "cache.prefix.max_cached_tokens",
                "PREFIX_TOKEN_LIMIT_INVALID",
                "max_cached_tokens must be positive",
            )
        if self.allow_mamba_state_reuse and self.reuse_mode is not PrefixReuseMode.PER_GROUP:
            raise ConfigError(
                "cache.prefix.allow_mamba_state_reuse",
                "MAMBA_PREFIX_MODE_CONFLICT",
                "Mamba state reuse requires per-group prefix matching",
            )
        if self.allow_sliding_window_reuse and self.reuse_mode is not PrefixReuseMode.PER_GROUP:
            raise ConfigError(
                "cache.prefix.allow_sliding_window_reuse",
                "SWA_PREFIX_MODE_CONFLICT",
                "sliding-window reuse requires per-group prefix matching",
            )


class KVScaleGranularity(enum.StrEnum):
    PER_TENSOR = "per_tensor"
    PER_HEAD = "per_head"
    PER_CHANNEL = "per_channel"


@dataclass(frozen=True, slots=True)
class KVScalePolicy(ConfigMixin):
    """Scale granularity for a quantized KV pool.

    Only per-tensor scales are implemented: FP8 storage carries one fp32 host
    scalar per (layer, plane) — see
    :class:`~ayaka.kvcache.storage.quantization.KVQuantization`. Per-head and
    per-channel granularities are declared but rejected, because accepting the
    config while no kernel consumes the extra scales would silently ignore them.
    """

    granularity: KVScaleGranularity = KVScaleGranularity.PER_TENSOR
    calculate_scales: bool = False
    scale_path: str | None = None

    def __post_init__(self) -> None:
        if self.granularity is not KVScaleGranularity.PER_TENSOR:
            raise ConfigError(
                "cache.kv_scales.granularity",
                "KV_SCALE_GRANULARITY_UNSUPPORTED",
                "only per-tensor KV scales are implemented",
            )
        if self.calculate_scales and self.scale_path:
            raise ConfigError(
                "cache.kv_scales",
                "KV_SCALE_SOURCE_CONFLICT",
                "calculate_scales and scale_path are mutually exclusive",
            )
        if self.calculate_scales:
            raise ConfigError(
                "cache.kv_scales.calculate_scales",
                "KV_SCALE_CALIBRATION_UNSUPPORTED",
                "runtime scale calibration is not implemented; supply calibrated scales",
            )


@dataclass(frozen=True, slots=True)
class TieredCacheConfig(ConfigMixin):
    host_bytes: int = 0
    nvme_bytes: int = 0
    nvme_path: str | None = None
    swap_bytes: int = 0
    async_transfers: bool = True
    max_inflight_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("host_bytes", "nvme_bytes", "swap_bytes"):
            if getattr(self, name) < 0:
                raise ConfigError(
                    f"cache.tiering.{name}", "CACHE_TIER_BYTES_NEGATIVE", f"{name} must be >= 0"
                )
        if self.nvme_bytes and not self.nvme_path:
            raise ConfigError(
                "cache.tiering.nvme_path",
                "NVME_PATH_REQUIRED",
                "NVMe capacity requires a path",
            )
        if self.nvme_bytes and not self.host_bytes:
            raise ConfigError(
                "cache.tiering.host_bytes",
                "NVME_HOST_STAGE_REQUIRED",
                "NVMe tier requires host staging capacity",
            )
        if self.max_inflight_bytes is not None and self.max_inflight_bytes < 1:
            raise ConfigError(
                "cache.tiering.max_inflight_bytes",
                "TRANSFER_LIMIT_INVALID",
                "max_inflight_bytes must be positive",
            )


class HybridCacheMode(enum.StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"
    GROUPED = "grouped"
    UNIFIED = "unified"


@dataclass(frozen=True, slots=True)
class HybridCacheConfig(ConfigMixin):
    mode: HybridCacheMode = HybridCacheMode.AUTO
    page_major_layout: bool = False
    swa_full_tokens_ratio: float = 0.8
    mamba_page_size: int | None = None
    require_equal_group_capacity: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.swa_full_tokens_ratio <= 1.0:
            raise ConfigError(
                "cache.hybrid.swa_full_tokens_ratio",
                "SWA_RATIO_INVALID",
                "swa_full_tokens_ratio must be in (0, 1]",
            )
        if self.mamba_page_size is not None:
            if self.mamba_page_size < 1 or self.mamba_page_size & (self.mamba_page_size - 1):
                raise ConfigError(
                    "cache.hybrid.mamba_page_size",
                    "MAMBA_PAGE_SIZE_INVALID",
                    "Mamba page size must be a positive power of two",
                )
        if self.page_major_layout and self.mode is HybridCacheMode.DISABLED:
            raise ConfigError(
                "cache.hybrid.page_major_layout",
                "PAGE_MAJOR_WITHOUT_HYBRID",
                "page-major layout requires a hybrid cache manager",
            )


@dataclass(frozen=True, slots=True)
class CacheConfig(ConfigMixin):
    """Physical cache representation. Memory budgets live in MemoryConfig."""

    block_size: int = 16
    kv_dtype: DType = DType.BF16
    layout: KVLayoutKind = KVLayoutKind.NHD
    num_gpu_blocks_override: int | None = None
    prefix: PrefixCacheConfig = field(default_factory=PrefixCacheConfig)
    kv_scales: KVScalePolicy = field(default_factory=KVScalePolicy)
    tiering: TieredCacheConfig = field(default_factory=TieredCacheConfig)
    hybrid: HybridCacheConfig = field(default_factory=HybridCacheConfig)

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.block_size & (self.block_size - 1):
            raise ConfigError(
                "cache.block_size", "CACHE_BLOCK_SIZE_INVALID", "block_size must be a power of two"
            )
        if self.block_size > 256:
            raise ConfigError(
                "cache.block_size", "CACHE_BLOCK_SIZE_TOO_LARGE", "block_size must not exceed 256"
            )
        if self.num_gpu_blocks_override is not None and self.num_gpu_blocks_override < 1:
            raise ConfigError(
                "cache.num_gpu_blocks_override",
                "CACHE_BLOCK_OVERRIDE_INVALID",
                "num_gpu_blocks_override must be positive or None",
            )
        allowed = (DType.FP16, DType.BF16, DType.FP8_E4M3, DType.FP8_E5M2)
        if self.kv_dtype not in allowed:
            raise ConfigError(
                "cache.kv_dtype",
                "KV_DTYPE_UNSUPPORTED",
                "KV dtype must be fp16, bf16, fp8_e4m3 or fp8_e5m2 "
                "(integer and sub-byte storage has no KV representation)",
            )

    @property
    def enable_prefix_caching(self) -> bool:
        return self.prefix.enabled

    @property
    def eviction_policy(self) -> EvictionPolicy:
        return self.prefix.eviction_policy


@dataclass(frozen=True, slots=True)
class CacheGroupPlan(ConfigMixin):
    """Resolved per-group plan: semantic geometry plus page-level payload bytes.

    ``bytes_per_token`` is payload-only and exists for scheduler heuristics and
    reporting. Allocation sizing uses the exact storage figure
    (:meth:`~ayaka.kvcache.storage.geometry.BaseKVStorageSpec.aligned_total_bytes`),
    which also carries alignment padding.
    """

    geometry: CacheGroupGeometry
    page_size: int
    dtype: DType
    bytes_per_token: int
    bytes_per_page: int
    state_bytes_per_sequence: int
    prefix_reusable: bool
    layout: KVLayoutKind | None
    kv_scales: KVScalePolicy | None
    page_major_layout: bool

    @property
    def group_id(self) -> str:
        return self.geometry.group_id

    @property
    def kind(self) -> CacheLayerKind:
        return self.geometry.kind

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return self.geometry.layer_indices

    @property
    def window_size(self) -> int | None:
        return self.geometry.window_size

    def retained_tokens(self, sequence_length: int) -> int:
        if self.kind is CacheLayerKind.MAMBA:
            return 0
        if self.kind is CacheLayerKind.SLIDING_WINDOW and self.window_size is not None:
            return min(sequence_length, self.window_size + self.page_size - 1)
        return sequence_length

    def bytes_for_sequence(self, sequence_length: int) -> int:
        if sequence_length < 0:
            raise ConfigError(
                "cache.sequence_length", "SEQUENCE_LENGTH_NEGATIVE", "sequence length must be >= 0"
            )
        retained = self.retained_tokens(sequence_length)
        pages = (retained + self.page_size - 1) // self.page_size
        return pages * self.bytes_per_page + self.state_bytes_per_sequence


@dataclass(frozen=True, slots=True)
class CachePlan(ConfigMixin):
    groups: tuple[CacheGroupPlan, ...]
    prefix: PrefixCacheConfig
    tiering: TieredCacheConfig
    hybrid_mode: HybridCacheMode
    swa_full_tokens_ratio: float
    require_equal_group_capacity: bool
    num_gpu_blocks_override: int | None

    @property
    def prefix_reuse_mode(self) -> PrefixReuseMode:
        return self.prefix.reuse_mode

    @property
    def is_hybrid(self) -> bool:
        return len(self.groups) > 1 or any(
            group.kind in (CacheLayerKind.SLIDING_WINDOW, CacheLayerKind.MAMBA)
            for group in self.groups
        )

    @property
    def token_bytes_per_rank(self) -> int:
        return sum(group.bytes_per_token for group in self.groups)

    @property
    def state_bytes_per_sequence(self) -> int:
        return sum(group.state_bytes_per_sequence for group in self.groups)

    def bytes_for_sequence(self, sequence_length: int) -> int:
        return sum(group.bytes_for_sequence(sequence_length) for group in self.groups)

    def prefix_matches_by_group(self, matches: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
        """Normalize per-group prefix hits to page boundaries.

        A missing or non-reusable group reports zero. The runtime must keep this
        vector; collapsing it to one scalar loses hybrid-cache correctness.
        """

        normalized: list[tuple[str, int]] = []
        for group in self.groups:
            value = int(matches.get(group.group_id, 0)) if group.prefix_reusable else 0
            if value < 0:
                raise ConfigError(
                    f"cache.prefix_matches.{group.group_id}",
                    "PREFIX_MATCH_NEGATIVE",
                    "prefix match length must be non-negative",
                )
            normalized.append((group.group_id, value - value % group.page_size))
        return tuple(normalized)

    def common_model_prefix(self, matches: Mapping[str, int]) -> int:
        """Tokens the whole model may skip, i.e. the intersection of all groups."""

        values = dict(self.prefix_matches_by_group(matches))
        if not self.groups:
            return 0
        return min(values[group.group_id] for group in self.groups)


class CacheLayerKind(enum.StrEnum):
    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW = "sliding_window"
    MLA = "mla"
    MAMBA = "mamba"


@dataclass(frozen=True, slots=True)
class CacheGroupGeometry(ConfigMixin):
    """Storage semantics shared by a set of model layers."""

    group_id: str
    kind: CacheLayerKind
    layer_indices: tuple[int, ...]
    num_kv_heads: int = 0
    head_dim: int = 0
    window_size: int | None = None
    state_elements_per_layer: int = 0
    conv_state_elements_per_layer: int = 0
    state_tp_sharded: bool = False

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ConfigError("model.cache_groups", "CACHE_GROUP_ID_EMPTY", "group id is required")
        if not self.layer_indices:
            raise ConfigError(
                f"model.cache_groups.{self.group_id}",
                "CACHE_GROUP_EMPTY",
                "cache group must own at least one layer",
            )
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ConfigError(
                f"model.cache_groups.{self.group_id}.layer_indices",
                "CACHE_GROUP_LAYERS_INVALID",
                "layer indices must be sorted, unique and non-negative",
            )
        if self.layer_indices[0] < 0:
            raise ConfigError(
                f"model.cache_groups.{self.group_id}.layer_indices",
                "CACHE_GROUP_LAYER_NEGATIVE",
                "layer indices must be non-negative",
            )
        if self.kind in (
            CacheLayerKind.FULL_ATTENTION,
            CacheLayerKind.SLIDING_WINDOW,
            CacheLayerKind.MLA,
        ):
            if self.num_kv_heads < 1 or self.head_dim < 1:
                raise ConfigError(
                    f"model.cache_groups.{self.group_id}",
                    "ATTENTION_GEOMETRY_INVALID",
                    "attention cache groups require positive KV heads and head dimension",
                )
        if self.kind is CacheLayerKind.SLIDING_WINDOW:
            if self.window_size is None or self.window_size < 1:
                raise ConfigError(
                    f"model.cache_groups.{self.group_id}.window_size",
                    "SWA_WINDOW_INVALID",
                    "sliding-window groups require a positive window",
                )
        elif self.window_size is not None:
            raise ConfigError(
                f"model.cache_groups.{self.group_id}.window_size",
                "WINDOW_ON_NON_SWA_GROUP",
                "window_size only applies to sliding-window groups",
            )
        if self.kind is CacheLayerKind.MAMBA:
            if self.state_elements_per_layer < 1 or self.conv_state_elements_per_layer < 1:
                raise ConfigError(
                    f"model.cache_groups.{self.group_id}",
                    "MAMBA_STATE_GEOMETRY_MISSING",
                    "Mamba groups require recurrent and convolution state sizes",
                )
        elif self.state_elements_per_layer or self.conv_state_elements_per_layer:
            raise ConfigError(
                f"model.cache_groups.{self.group_id}",
                "STATE_ON_ATTENTION_GROUP",
                "recurrent state geometry only applies to Mamba groups",
            )

    def local_kv_heads(self, tp_size: int) -> int:
        if self.kind is CacheLayerKind.MAMBA:
            return 0
        if self.num_kv_heads >= tp_size:
            if self.num_kv_heads % tp_size:
                raise ConfigError(
                    f"model.cache_groups.{self.group_id}.num_kv_heads",
                    "KV_HEADS_NOT_TP_DIVISIBLE",
                    "KV heads must divide TP or be replicated by it",
                )
            return self.num_kv_heads // tp_size
        if tp_size % self.num_kv_heads:
            raise ConfigError(
                f"model.cache_groups.{self.group_id}.num_kv_heads",
                "KV_HEAD_REPLICATION_INVALID",
                "TP size must be a multiple of KV heads when KV is replicated",
            )
        return 1

    def token_elements_per_rank(self, tp_size: int) -> int:
        if self.kind is CacheLayerKind.MAMBA:
            return 0
        streams = 1 if self.kind is CacheLayerKind.MLA else 2
        return streams * self.local_kv_heads(tp_size) * self.head_dim * len(self.layer_indices)

    def sequence_state_elements_per_rank(self, tp_size: int) -> int:
        if self.kind is not CacheLayerKind.MAMBA:
            return 0
        elements = (self.state_elements_per_layer + self.conv_state_elements_per_layer) * len(
            self.layer_indices
        )
        if not self.state_tp_sharded:
            return elements
        if elements % tp_size:
            raise ConfigError(
                f"model.cache_groups.{self.group_id}",
                "MAMBA_STATE_NOT_TP_DIVISIBLE",
                "Mamba state geometry must divide tensor parallel size",
            )
        return elements // tp_size

    def token_bytes(self, dtype: DType, tp_size: int) -> int:
        """Payload bytes one token costs on this rank.

        Semantic only: alignment padding added by the storage allocator is not
        included, so capacity planning must size pools from
        :meth:`~ayaka.kvcache.storage.geometry.BaseKVStorageSpec.aligned_total_bytes`
        rather than from this figure.
        """
        return dtype.nbytes(self.token_elements_per_rank(tp_size))

    def sequence_state_bytes(self, dtype: DType, tp_size: int) -> int:
        """Payload bytes of sequence-owned state on this rank."""
        return dtype.nbytes(self.sequence_state_elements_per_rank(tp_size))


@dataclass(frozen=True, slots=True)
class PhysicalCacheGroupPlan(ConfigMixin):
    """Concrete allocation contract for one GPU-local cache/state pool."""

    group_id: str
    kind: CacheLayerKind
    layout: KVLayoutKind | None
    dtype: DType
    kv_scales: KVScalePolicy | None
    page_size: int
    bytes_per_page: int
    page_major_layout: bool
    capacity_bytes: int
    num_pages: int
    token_capacity: int
    state_slot_capacity: int

    def __post_init__(self) -> None:
        if (
            min(
                self.capacity_bytes,
                self.num_pages,
                self.token_capacity,
                self.state_slot_capacity,
            )
            < 0
        ):
            raise ConfigError(
                f"cache.physical.groups.{self.group_id}",
                "PHYSICAL_CACHE_CAPACITY_NEGATIVE",
                "physical cache capacities must be non-negative",
            )
        if self.kind is CacheLayerKind.MAMBA:
            if self.num_pages or self.token_capacity:
                raise ConfigError(
                    f"cache.physical.groups.{self.group_id}",
                    "MAMBA_TOKEN_POOL_INVALID",
                    "Mamba state pools cannot expose token pages",
                )
        elif self.state_slot_capacity:
            raise ConfigError(
                f"cache.physical.groups.{self.group_id}",
                "ATTENTION_STATE_POOL_INVALID",
                "attention pools cannot expose recurrent state slots",
            )


@dataclass(frozen=True, slots=True)
class PhysicalCachePlan(ConfigMixin):
    """Per-device physical storage partition, ready for allocator creation."""

    device_kv_budget_bytes: int
    allocated_bytes: int
    unallocated_bytes: int
    max_sequences: int
    max_model_len: int
    groups: tuple[PhysicalCacheGroupPlan, ...]

    def __post_init__(self) -> None:
        if self.allocated_bytes + self.unallocated_bytes != self.device_kv_budget_bytes:
            raise ConfigError(
                "cache.physical",
                "PHYSICAL_CACHE_BUDGET_MISMATCH",
                "allocated and unallocated bytes must exactly cover the device KV budget",
            )
        if sum(group.capacity_bytes for group in self.groups) != self.allocated_bytes:
            raise ConfigError(
                "cache.physical.groups",
                "PHYSICAL_CACHE_GROUP_SUM_MISMATCH",
                "group capacities must sum to allocated bytes",
            )

    def group(self, group_id: str) -> PhysicalCacheGroupPlan:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise ConfigError(
            f"cache.physical.groups.{group_id}",
            "PHYSICAL_CACHE_GROUP_UNKNOWN",
            f"cache group {group_id!r} is not present in the physical plan",
        )


def plan_cache(
    config: CacheConfig, architecture: ArchitectureConfig, tp_size: int = 1
) -> CachePlan:
    """Resolve the semantic cache plan for one model configuration.

    Refuses layouts and cache families no runtime module implements, so the
    failure is a config error here instead of an exception inside storage
    construction:

    * recurrent (Mamba) groups have no storage implementation;
    * MLA requires the ``NLD`` plane layout, and ``NLD`` requires MLA;
    * prefix reuse is only layout-compatible for a single full-attention MHA
      group, matching :func:`~ayaka.kvcache.layout.prefix_cache_capability`.
    """
    if tp_size < 1:
        raise ConfigError("parallel.tp_size", "TP_SIZE_INVALID", "tp_size must be >= 1")
    groups = architecture.cache_groups
    is_hybrid = len(groups) > 1 or any(
        group.kind in (CacheLayerKind.SLIDING_WINDOW, CacheLayerKind.MAMBA) for group in groups
    )
    if is_hybrid and config.hybrid.mode is HybridCacheMode.DISABLED:
        raise ConfigError(
            "cache.hybrid.mode",
            "HYBRID_CACHE_DISABLED",
            "model requires grouped cache/state ownership but hybrid cache is disabled",
        )
    resolved_hybrid_mode = config.hybrid.mode
    if resolved_hybrid_mode is HybridCacheMode.AUTO:
        resolved_hybrid_mode = HybridCacheMode.GROUPED if is_hybrid else HybridCacheMode.DISABLED
    for geometry in groups:
        if geometry.kind is CacheLayerKind.MAMBA:
            raise ConfigError(
                f"model.cache_groups.{geometry.group_id}",
                "RECURRENT_CACHE_UNSUPPORTED",
                "recurrent/state-space cache has no storage or backend implementation",
            )
        if geometry.kind is CacheLayerKind.MLA and config.layout is not KVLayoutKind.NLD:
            raise ConfigError(
                "cache.layout",
                "MLA_REQUIRES_NLD_LAYOUT",
                "MLA groups store a latent/rope plane pair and require the NLD layout",
            )
        if geometry.kind is not CacheLayerKind.MLA and config.layout is KVLayoutKind.NLD:
            raise ConfigError(
                "cache.layout",
                "NLD_REQUIRES_MLA",
                "NLD is an MLA-only layout; MHA/GQA groups use NHD or HND",
            )
    # Layout-level statement only: the grouped manager implements no prefix
    # lookup, so a report must not promise reuse for hybrid or MLA pools.
    prefix_reusable = bool(
        config.prefix.enabled
        and len(groups) == 1
        and groups[0].kind is CacheLayerKind.FULL_ATTENTION
    )
    plans: list[CacheGroupPlan] = []
    for geometry in groups:
        bytes_per_token = geometry.token_bytes(config.kv_dtype, tp_size)
        plans.append(
            CacheGroupPlan(
                geometry=geometry,
                page_size=config.block_size,
                dtype=config.kv_dtype,
                bytes_per_token=bytes_per_token,
                bytes_per_page=bytes_per_token * config.block_size,
                state_bytes_per_sequence=geometry.sequence_state_bytes(config.kv_dtype, tp_size),
                prefix_reusable=prefix_reusable,
                layout=config.layout,
                kv_scales=config.kv_scales,
                page_major_layout=config.hybrid.page_major_layout,
            )
        )
    return CachePlan(
        groups=tuple(plans),
        prefix=config.prefix,
        tiering=config.tiering,
        hybrid_mode=resolved_hybrid_mode,
        swa_full_tokens_ratio=config.hybrid.swa_full_tokens_ratio,
        require_equal_group_capacity=config.hybrid.require_equal_group_capacity,
        num_gpu_blocks_override=config.num_gpu_blocks_override,
    )
