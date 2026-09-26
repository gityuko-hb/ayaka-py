"""Cross-layer composition: resolve config records into runtime objects.

This is the single module that translates the semantic configuration layer
(:mod:`ayaka.configs.model`, :mod:`ayaka.configs.cache`,
:mod:`ayaka.configs.memory`) into the contracts the runtime modules actually
consume -- :class:`~ayaka.attention.spec.AttentionGroupSpec`,
:class:`~ayaka.kvcache.storage.geometry.BaseKVStorageSpec`,
:class:`~ayaka.kvcache.groups.KVCacheGroup` -- and then materializes a
:class:`~ayaka.kvcache.grouped_manager.KVCacheGroupManager`.

Two rules keep the byte accounting honest:

* The config figures (:attr:`~ayaka.configs.cache.CacheGroupPlan.bytes_per_token`)
  are payload-only and are used for scheduling heuristics and reports.
* Capacity comes only from
  :meth:`~ayaka.kvcache.storage.geometry.BaseKVStorageSpec.aligned_total_bytes`,
  so alignment padding cannot push an allocation past its device budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ayaka.attention.spec import AttentionGroupSpec, AttentionSpec, MLAExtras
from ayaka.configs.base import ConfigError
from ayaka.configs.cache import (
    CacheConfig,
    CacheGroupPlan,
    CacheLayerKind,
    CachePlan,
    PhysicalCacheGroupPlan,
    PhysicalCachePlan,
    plan_cache,
)
from ayaka.configs.distributed import (
    CollectiveBackend,
    DistributedRuntimeConfig,
    RankPlacement,
    ResolvedDistributedPlan,
)
from ayaka.configs.memory import (
    DeviceMemoryPlan,
    MemoryConfig,
    MemoryProfile,
    plan_device_memory,
)
from ayaka.configs.model import ArchitectureConfig
from ayaka.configs.parallel import CollectivePolicy, ParallelConfig, ResolvedParallelPlan
from ayaka.distributed import env as distributed_env
from ayaka.distributed.device import CommunicationBackend, DeviceGroup, DeviceRef
from ayaka.kvcache.groups import KVCacheGroup
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.materialize import KVStorageLease, materialize_kv_storage
from ayaka.kvcache.retention.policy import FullRetention, SlidingWindowRetention
from ayaka.kvcache.storage.dtypes import to_storage_dtype
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec, MHAStorageSpec, MLAStorageSpec
from ayaka.kvcache.storage.layout import DEFAULT_KV_ALIGNMENT_BYTES
from ayaka.memory.ledger import MemoryLedger
from ayaka.plan import ComputePlan, ExecutionPlan, ParallelPlan
from ayaka.types import AttentionType, DeviceKind, DType, KVCacheDtype, MaskKind
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from ayaka.distributed.collective_backend import CollectiveSetup
    from ayaka.distributed.parallel import ParallelContext
    from ayaka.kvcache.grouped_manager import KVCacheGroupManager
    from ayaka.kvcache.storage.ports import KVStorage

__all__ = [
    "ParallelRuntime",
    "ResidentKVPlan",
    "bind_resident_kv",
    "build_attention_groups",
    "build_cache_groups",
    "build_execution_plan",
    "build_parallel_plan",
    "build_parallel_runtime",
    "build_resident_kv",
    "build_storage_specs",
    "materialize_resident_kv",
    "plan_physical_cache",
    "plan_resident_kv",
    "resident_ledger",
]

#: Recurrent groups are classified by the config layer and then refused by
#: ``plan_cache``. Linear-attention models need a storage/backend contract that
#: does not exist yet, so this table deliberately has no MAMBA entry.
_ATTENTION_TYPE_BY_KIND = {
    CacheLayerKind.FULL_ATTENTION: AttentionType.FULL,
    CacheLayerKind.SLIDING_WINDOW: AttentionType.SWA,
    CacheLayerKind.MLA: AttentionType.MLA,
}


@dataclass(frozen=True, slots=True)
class ResidentKVPlan:
    """A fully resolved resident KV configuration for one rank.

    Holds every object needed to allocate the pools and to describe them to the
    scheduler: the semantic plan, the exact physical partition, the attention
    group contracts, the storage geometries, and the grouped ownership records.
    """

    cache: CachePlan
    physical: PhysicalCachePlan
    memory: DeviceMemoryPlan
    attention_groups: tuple[AttentionGroupSpec, ...]
    storage_specs: tuple[tuple[str, BaseKVStorageSpec], ...]
    cache_groups: tuple[KVCacheGroup, ...]
    max_num_seqs: int
    max_model_len: int
    parallel: ResolvedParallelPlan | None = None

    def storage_spec(self, group_id: str) -> BaseKVStorageSpec:
        for candidate, spec in self.storage_specs:
            if candidate == group_id:
                return spec
        raise ConfigError(
            f"cache.physical.groups.{group_id}",
            "PHYSICAL_CACHE_GROUP_UNKNOWN",
            f"cache group {group_id!r} has no storage spec",
        )


def _spec_for(storage_specs: Mapping[str, BaseKVStorageSpec], group_id: str) -> BaseKVStorageSpec:
    try:
        return storage_specs[group_id]
    except KeyError:
        raise ConfigError(
            f"cache.physical.groups.{group_id}",
            "PHYSICAL_CACHE_GROUP_UNKNOWN",
            f"cache group {group_id!r} has no storage geometry",
        ) from None


def _require_address_bounds(spec: BaseKVStorageSpec, group_id: str) -> BaseKVStorageSpec:
    """Fail a plan that would overflow page/slot metadata before any byte math."""
    try:
        spec.validate_address_bounds()
    except ValueError as exc:
        raise ConfigError(
            f"cache.physical.groups.{group_id}",
            "PHYSICAL_CACHE_METADATA_OVERFLOW",
            str(exc),
        ) from exc
    return spec


def build_storage_specs(
    architecture: ArchitectureConfig,
    cache_config: CacheConfig,
    cache_plan: CachePlan,
    *,
    tp_size: int = 1,
    capacities: Mapping[str, int] | None = None,
) -> dict[str, BaseKVStorageSpec]:
    """Build the per-group storage geometry, keyed by cache-group id.

    ``capacities`` supplies the final page count per group; when omitted each
    group gets the minimum a manager can accept (one padding page plus one
    usable page). The same function is therefore usable as the cost model for
    capacity planning and as the final allocation spec.
    """

    dtype_name = to_storage_dtype(cache_config.kv_dtype)
    resolved = capacities or {}
    specs: dict[str, BaseKVStorageSpec] = {}
    for group in cache_plan.groups:
        pages = resolved.get(group.group_id, 2)
        if pages < 2:
            raise ConfigError(
                f"cache.physical.groups.{group.group_id}",
                "PHYSICAL_CACHE_PAGE_COUNT_INVALID",
                "a grouped cache needs at least one padding page and one usable page",
            )
        if group.kind is CacheLayerKind.MLA:
            latent_dim = architecture.kv_lora_rank
            rope_dim = architecture.qk_rope_head_dim
            if latent_dim is None or rope_dim is None:
                raise ConfigError(
                    f"model.cache_groups.{group.group_id}",
                    "MLA_LATENT_GEOMETRY_MISSING",
                    "an MLA group requires kv_lora_rank and qk_rope_head_dim",
                )
            specs[group.group_id] = _require_address_bounds(
                MLAStorageSpec(
                    latent_dim=latent_dim,
                    rope_dim=rope_dim,
                    page_size=group.page_size,
                    capacity_pages=pages,
                    num_layers=len(group.layer_indices),
                    dtype=dtype_name,
                ),
                group.group_id,
            )
            continue
        specs[group.group_id] = _require_address_bounds(
            MHAStorageSpec(
                num_layers=len(group.layer_indices),
                num_kv_heads_local=group.geometry.local_kv_heads(tp_size),
                head_dim=group.geometry.head_dim,
                page_size=group.page_size,
                capacity_pages=pages,
                dtype=dtype_name,
                layout=cache_config.layout,
            ),
            group.group_id,
        )
    return specs


def build_attention_groups(
    architecture: ArchitectureConfig,
    cache_config: CacheConfig,
    cache_plan: CachePlan,
    *,
    tp_size: int = 1,
) -> tuple[AttentionGroupSpec, ...]:
    """Translate resolved cache groups into attention backend contracts.

    Query heads are TP-local; KV heads may be replicated when TP exceeds the KV
    head count, which is exactly what
    :meth:`~ayaka.configs.cache.CacheGroupGeometry.local_kv_heads` resolves.
    ``AttentionGroupSpec.group_id`` is an ordinal matching cache-plan order;
    the stable string identity stays on the cache group.
    """

    if tp_size < 1:
        raise ConfigError("parallel.tp_size", "TP_SIZE_INVALID", "tp_size must be >= 1")
    total_qo_heads = architecture.num_attention_heads
    if total_qo_heads < 1 or total_qo_heads % tp_size:
        raise ConfigError(
            "parallel.tp_size",
            "TP_HEADS_NOT_DIVISIBLE",
            f"tp_size={tp_size} must divide num_attention_heads={total_qo_heads}",
        )
    local_qo_heads = total_qo_heads // tp_size
    kv_cache_dtype = KVCacheDtype.from_dtype(cache_config.kv_dtype)
    groups: list[AttentionGroupSpec] = []
    for ordinal, group in enumerate(cache_plan.groups):
        geometry = group.geometry
        attn_type = _ATTENTION_TYPE_BY_KIND.get(group.kind)
        if attn_type is None:
            raise ConfigError(
                f"model.cache_groups.{group.group_id}",
                "RECURRENT_CACHE_UNSUPPORTED",
                "recurrent/state-space cache has no attention backend",
            )
        mla: MLAExtras | None = None
        if attn_type is AttentionType.MLA:
            if architecture.kv_lora_rank is None or architecture.qk_rope_head_dim is None:
                raise ConfigError(
                    f"model.cache_groups.{group.group_id}",
                    "MLA_LATENT_GEOMETRY_MISSING",
                    "an MLA group requires kv_lora_rank and qk_rope_head_dim",
                )
            head_dim_vo = architecture.kv_lora_rank
            mla = MLAExtras(
                kv_lora_rank=architecture.kv_lora_rank,
                qk_rope_head_dim=architecture.qk_rope_head_dim,
            )
            mask = MaskKind.CAUSAL
            window = None
        else:
            head_dim_vo = geometry.head_dim
            mask = MaskKind.SLIDING if attn_type is AttentionType.SWA else MaskKind.CAUSAL
            window = geometry.window_size if attn_type is AttentionType.SWA else None
        spec = AttentionSpec(
            num_qo_heads=local_qo_heads,
            num_kv_heads=geometry.local_kv_heads(tp_size),
            head_dim_qk=geometry.head_dim,
            head_dim_vo=head_dim_vo,
            sm_scale=geometry.head_dim**-0.5,
            mask=mask,
            sliding_window=window,
        )
        groups.append(
            AttentionGroupSpec(
                group_id=ordinal,
                layer_ids=group.layer_indices,
                attn_type=attn_type,
                spec=spec,
                page_size=group.page_size,
                kv_layout=cache_config.layout,
                kv_cache_dtype=kv_cache_dtype,
                mla=mla,
            )
        )
    return tuple(groups)


def _retention_for(group: CacheGroupPlan) -> FullRetention | SlidingWindowRetention:
    if group.kind in (CacheLayerKind.FULL_ATTENTION, CacheLayerKind.MLA):
        return FullRetention()
    if group.kind is CacheLayerKind.SLIDING_WINDOW:
        window = group.window_size
        if window is None:
            raise ConfigError(
                f"model.cache_groups.{group.group_id}.window_size",
                "SWA_WINDOW_INVALID",
                "sliding-window groups require a positive window",
            )
        return SlidingWindowRetention(window_size=window)
    raise ConfigError(
        f"model.cache_groups.{group.group_id}",
        "RECURRENT_CACHE_UNSUPPORTED",
        "recurrent/state-space cache has no retention contract",
    )


def build_cache_groups(
    cache_plan: CachePlan,
    storage_specs: Mapping[str, BaseKVStorageSpec],
) -> tuple[KVCacheGroup, ...]:
    """Build the runtime ownership records with plan-stable group ids."""

    groups: list[KVCacheGroup] = []
    for group in cache_plan.groups:
        spec = _spec_for(storage_specs, group.group_id)
        if spec.num_layers != len(group.layer_indices):
            raise ConfigError(
                f"cache.physical.groups.{group.group_id}",
                "STORAGE_LAYER_COUNT_MISMATCH",
                "storage geometry layer count must match the cache group",
            )
        groups.append(
            KVCacheGroup(
                name=group.group_id,
                layer_ids=group.layer_indices,
                storage_spec=spec,
                retention=_retention_for(group),
            )
        )
    return tuple(groups)


def _group_cost(
    spec: BaseKVStorageSpec,
    pages: int,
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES,
) -> int:
    """Exact reservation bytes for ``pages`` pages, padding included."""
    return spec.with_capacity_pages(pages).aligned_total_bytes(alignment_bytes)


def plan_physical_cache(
    cache: CachePlan,
    storage_specs: Mapping[str, BaseKVStorageSpec],
    device_kv_budget_bytes: int,
    *,
    max_sequences: int,
    max_model_len: int,
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES,
) -> PhysicalCachePlan:
    """Partition one device's KV budget into exact per-group page pools.

    Each group is first sized from its semantic byte cost (SWA expressed as a
    ratio of full-attention token capacity, as the config specifies) and then
    corrected until the storage geometries' ``aligned_total_bytes`` fit the
    budget. The remainder is reported as ``unallocated_bytes``; nothing is
    silently rounded away. The alignment used for the charge is recorded in the
    returned plan so the materializer cannot diverge from the cost model.
    """

    if device_kv_budget_bytes < 1:
        raise ConfigError(
            "cache.physical.device_kv_budget_bytes",
            "PHYSICAL_CACHE_BUDGET_INVALID",
            "device KV cache budget must be positive",
        )
    if max_sequences < 1 or max_model_len < 1:
        raise ConfigError(
            "cache.physical",
            "PHYSICAL_CACHE_LIMIT_INVALID",
            "max_sequences and max_model_len must be positive",
        )
    if (
        isinstance(alignment_bytes, bool)
        or not isinstance(alignment_bytes, int)
        or alignment_bytes <= 0
    ):
        raise ConfigError(
            "cache.physical.alignment_bytes",
            "PHYSICAL_CACHE_ALIGNMENT_INVALID",
            "cache alignment must be a positive integer number of bytes",
        )

    groups = cache.groups
    for group in groups:
        if group.kind is CacheLayerKind.MAMBA:
            raise ConfigError(
                f"cache.physical.groups.{group.group_id}",
                "RECURRENT_CACHE_UNSUPPORTED",
                "recurrent state is sequence-owned and has no storage implementation",
            )
    specs = {group.group_id: _spec_for(storage_specs, group.group_id) for group in groups}
    minimum_pages = {
        group.group_id: max(2, div_ceil(group.retained_tokens(max_model_len), group.page_size))
        for group in groups
    }

    if cache.num_gpu_blocks_override is not None:
        page_counts = {group.group_id: cache.num_gpu_blocks_override for group in groups}
    else:
        ratios = {
            group.group_id: (
                1.0
                if cache.require_equal_group_capacity
                or group.kind is not CacheLayerKind.SLIDING_WINDOW
                else cache.swa_full_tokens_ratio
            )
            for group in groups
        }

        def counts_for(common_tokens: int) -> dict[str, int]:
            return {
                group.group_id: max(
                    minimum_pages[group.group_id],
                    int(common_tokens * ratios[group.group_id] / group.page_size),
                )
                for group in groups
            }

        weighted_bytes_per_token = sum(
            group.bytes_per_token * ratios[group.group_id] for group in groups
        )
        if weighted_bytes_per_token < 1:
            raise ConfigError(
                "cache.physical",
                "PHYSICAL_CACHE_GROUP_BYTES_INVALID",
                "every token-backed cache group must cost at least one byte per token",
            )
        common_tokens = int(device_kv_budget_bytes / weighted_bytes_per_token)
        page_counts = counts_for(common_tokens)
        while (
            sum(
                _group_cost(specs[group.group_id], page_counts[group.group_id], alignment_bytes)
                for group in groups
            )
            > device_kv_budget_bytes
            and common_tokens > 0
        ):
            total = sum(
                _group_cost(specs[group.group_id], page_counts[group.group_id], alignment_bytes)
                for group in groups
            )
            common_tokens = min(
                common_tokens - 1,
                int(common_tokens * device_kv_budget_bytes / total),
            )
            page_counts = counts_for(common_tokens)

    allocated = sum(
        _group_cost(specs[group.group_id], page_counts[group.group_id], alignment_bytes)
        for group in groups
    )
    if allocated > device_kv_budget_bytes:
        raise ConfigError(
            "cache.physical",
            (
                "PHYSICAL_CACHE_OVERRIDE_EXCEEDS_BUDGET"
                if cache.num_gpu_blocks_override is not None
                else "MINIMUM_CAPACITY_EXCEEDS_BUDGET"
            ),
            "the smallest pool that can hold one max-length sequence exceeds the device budget",
            context={"required_bytes": allocated, "available_bytes": device_kv_budget_bytes},
        )

    physical: list[PhysicalCacheGroupPlan] = []
    for group in groups:
        pages = page_counts[group.group_id]
        physical.append(
            PhysicalCacheGroupPlan(
                group_id=group.group_id,
                kind=group.kind,
                layout=group.layout,
                dtype=group.dtype,
                kv_scales=group.kv_scales,
                page_size=group.page_size,
                bytes_per_page=group.bytes_per_page,
                page_major_layout=group.page_major_layout,
                capacity_bytes=_group_cost(specs[group.group_id], pages, alignment_bytes),
                num_pages=pages,
                token_capacity=pages * group.page_size,
                state_slot_capacity=0,
            )
        )
    return PhysicalCachePlan(
        device_kv_budget_bytes=device_kv_budget_bytes,
        allocated_bytes=allocated,
        unallocated_bytes=device_kv_budget_bytes - allocated,
        max_sequences=max_sequences,
        max_model_len=max_model_len,
        groups=tuple(physical),
        alignment_bytes=alignment_bytes,
    )


def _resolve_parallel(
    architecture: ArchitectureConfig,
    parallel: ParallelConfig | ResolvedParallelPlan | None,
    tp_size: int,
) -> ResolvedParallelPlan | None:
    if parallel is None:
        return None
    resolved = parallel.resolve(architecture) if isinstance(parallel, ParallelConfig) else parallel
    if tp_size != 1 and tp_size != resolved.tp_size:
        raise ConfigError(
            "parallel.tp_size",
            "PARALLEL_TP_CONFLICT",
            "pass either tp_size or a parallel plan with the same tp_size",
            context={"tp_size": tp_size, "plan_tp_size": resolved.tp_size},
        )
    resolved.validate_against(architecture)
    return resolved


def plan_resident_kv(
    architecture: ArchitectureConfig,
    cache_config: CacheConfig,
    *,
    device_total_bytes: int,
    max_num_seqs: int,
    memory_config: MemoryConfig | None = None,
    profile: MemoryProfile | None = None,
    parallel: ParallelConfig | ResolvedParallelPlan | None = None,
    tp_size: int = 1,
    max_model_len: int | None = None,
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES,
) -> ResidentKVPlan:
    """Resolve the full memory -> scheduler -> attention -> storage chain.

    Args:
        architecture: Model geometry (layer kinds, heads, MLA latent sizes).
        cache_config: Physical cache representation and policy.
        device_total_bytes: Total device memory used to derive the KV budget.
        max_num_seqs: Scheduler sequence cap; bounds recurrent/slot pools.
        memory_config: Utility/reserve policy; defaults to ``MemoryConfig()``.
        profile: Measured non-KV allocations; defaults to an optimistic profile.
        parallel: Parallel layout; a raw config is validated and resolved here.
        tp_size: Tensor-parallel size; heads are sharded or replicated by it.
            Only used when ``parallel`` is omitted, or as a consistency check.
        max_model_len: Sequence-length bound; defaults to the model context.
        alignment_bytes: Per-allocation alignment the physical plan is charged
            with; the same value must reach the materializer.
    """

    resolved_parallel = _resolve_parallel(architecture, parallel, tp_size)
    effective_tp = tp_size if resolved_parallel is None else resolved_parallel.tp_size
    cache_plan = plan_cache(cache_config, architecture, effective_tp)
    memory_plan = plan_device_memory(device_total_bytes, memory_config or MemoryConfig(), profile)
    resolved_max_model_len = (
        architecture.max_context_len if max_model_len is None else max_model_len
    )
    if resolved_max_model_len < 1:
        raise ConfigError(
            "cache.physical.max_model_len",
            "PHYSICAL_CACHE_LIMIT_INVALID",
            "max_model_len must be positive",
        )
    if max_model_len is not None and max_model_len > architecture.max_context_len:
        # The RoPE-effective context is the checkpoint ceiling; an override may
        # reserve less, never promise more than the model can position.
        raise ConfigError(
            "cache.physical.max_model_len",
            "CONTEXT_EXCEEDS_MODEL",
            f"max_model_len={max_model_len} exceeds the model context "
            f"{architecture.max_context_len}",
        )
    placeholder_specs = build_storage_specs(
        architecture, cache_config, cache_plan, tp_size=effective_tp
    )
    physical = plan_physical_cache(
        cache_plan,
        placeholder_specs,
        memory_plan.kv_cache_bytes,
        max_sequences=max_num_seqs,
        max_model_len=resolved_max_model_len,
        alignment_bytes=alignment_bytes,
    )
    specs = build_storage_specs(
        architecture,
        cache_config,
        cache_plan,
        tp_size=effective_tp,
        capacities={group.group_id: group.num_pages for group in physical.groups},
    )
    return ResidentKVPlan(
        cache=cache_plan,
        physical=physical,
        memory=memory_plan,
        attention_groups=build_attention_groups(
            architecture, cache_config, cache_plan, tp_size=effective_tp
        ),
        storage_specs=tuple(specs.items()),
        cache_groups=build_cache_groups(cache_plan, specs),
        max_num_seqs=max_num_seqs,
        max_model_len=resolved_max_model_len,
        parallel=resolved_parallel,
    )


def build_resident_kv(
    plan: ResidentKVPlan,
    *,
    device: str = "cuda",
    storages: Mapping[str, KVStorage] | None = None,
) -> KVCacheGroupManager:
    """Allocate the pools and build the grouped manager for one rank.

    Imports are local so that resolving a plan stays importable without
    PyTorch; only materialization needs the device stack.
    """

    from ayaka.kvcache.build import build_kv_storage
    from ayaka.kvcache.grouped_manager import KVCacheGroupManager
    from ayaka.kvcache.storage.validation import validate_kv_storage_support
    from ayaka.utils.torch_utils import compute_capability as torch_compute_capability

    normalized = str(device).strip().lower()
    device_type = normalized.partition(":")[0]
    device_index = 0
    if ":" in normalized:
        try:
            device_index = int(normalized.partition(":")[2])
        except ValueError as exc:
            raise ValueError(f"invalid device {device!r}") from exc
    capability = torch_compute_capability(device_index) if device_type == "cuda" else None

    supplied = dict(storages or {})
    for group_id, spec in plan.storage_specs:
        # The resident path bypasses the ledger, so capability validation has to
        # happen here as well: a plan must never allocate an unsupported dtype,
        # layout or geometry just because no charge is reserved.
        validate_kv_storage_support(
            spec,
            device_type=device_type,
            compute_capability=capability,
        ).require_compatible()
        if group_id not in supplied:
            supplied[group_id] = build_kv_storage(spec, device=device)
    return KVCacheGroupManager(
        cache_groups=plan.cache_groups,
        max_sequences=plan.max_num_seqs,
        max_sequence_tokens=plan.max_model_len,
        storages=supplied,
    )


def resident_ledger(
    memory: DeviceMemoryPlan,
    *,
    device: str,
    device_total_bytes: int,
    device_index: int = 0,
) -> MemoryLedger:
    """Build the ledger every resident KV lease is charged against.

    The account capacity is the full policy budget, not the KV portion: every
    backing owner (weights, KV, activation, workspace, graph, staging) is a
    claim inside one budget, and using the KV figure here would hide the
    non-KV allocations behind a second, virtual account. A CPU diagnostic run
    materializes into host-pageable memory, so that tier carries the same
    capacity; only it is charged on that lane (the DEVICE tier stays empty).
    """
    if device_total_bytes < 1:
        raise ConfigError(
            "memory.total_bytes", "DEVICE_MEMORY_UNKNOWN", "device memory must be positive"
        )
    if memory.policy_budget_bytes < 1:
        raise ConfigError(
            "memory.policy_budget_bytes", "DEVICE_MEMORY_UNKNOWN", "policy budget must be positive"
        )
    if device.strip().lower() == "cpu":
        return MemoryLedger.for_device(
            device_budget_bytes=memory.policy_budget_bytes,
            device_total_bytes=device_total_bytes,
            host_pageable_bytes=memory.policy_budget_bytes,
            host_total_bytes=device_total_bytes,
            device_index=device_index,
        )
    return MemoryLedger.for_device(
        device_budget_bytes=memory.policy_budget_bytes,
        device_total_bytes=device_total_bytes,
        device_index=device_index,
    )


def materialize_resident_kv(
    plan: ResidentKVPlan,
    *,
    ledger: MemoryLedger,
    device: str = "cuda",
    zero_initialize: bool = False,
) -> tuple[KVStorageLease, ...]:
    """Materialize every planned group slab, charged to ``ledger``.

    Labels are the cache-group ids, so a lease and its manager entry cannot
    drift apart. A mid-way failure closes what was already materialized, which
    keeps the ledger free of orphaned claims.
    """
    leases: list[KVStorageLease] = []
    try:
        for group_id, spec in plan.storage_specs:
            leases.append(
                materialize_kv_storage(
                    spec,
                    ledger=ledger,
                    label=group_id,
                    device=device,
                    zero_initialize=zero_initialize,
                    alignment_bytes=plan.physical.alignment_bytes,
                )
            )
    except BaseException:
        for lease in reversed(leases):
            with suppress(Exception):
                lease.close()
        raise
    return tuple(leases)


def bind_resident_kv(
    plan: ResidentKVPlan,
    leases: tuple[KVStorageLease, ...],
) -> tuple[KVCacheGroupManager, LogicalKVManager]:
    """Bind materialized leases to the grouped manager and logical facade.

    The manager keeps the same storage objects the leases own, so the ledger
    claim, the allocator and the logical manager can never disagree about
    which slab serves a group.
    """
    by_label = {lease.label: lease for lease in leases}
    expected = {group_id for group_id, _ in plan.storage_specs}
    if set(by_label) != expected:
        raise ConfigError(
            "cache.physical.groups",
            "PHYSICAL_CACHE_LEASE_MISMATCH",
            "one materialized lease is required per cache group",
            context={
                "expected_groups": sorted(expected),
                "materialized_groups": sorted(by_label),
            },
        )
    storages = {group_id: by_label[group_id].storage for group_id in expected}
    from ayaka.kvcache.grouped_manager import KVCacheGroupManager

    manager = KVCacheGroupManager(
        cache_groups=plan.cache_groups,
        max_sequences=plan.max_num_seqs,
        max_sequence_tokens=plan.max_model_len,
        storages=storages,
    )
    return manager, LogicalKVManager(manager, by_label)


def build_execution_plan(
    architecture: ArchitectureConfig,
    cache_config: CacheConfig,
    plan: ResidentKVPlan,
    *,
    plan_id: str,
    model_id: str,
    model_revision: str,
    weights_revision: str,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool = True,
    compute_dtype: DType = DType.BF16,
    parallel: ParallelPlan | None = None,
) -> ExecutionPlan:
    """Bind resolved attention groups and the rank-aware parallel plan to one identity.

    The attention group tuple is the same one the scheduler, the backend
    selector and the time estimator read, so pool descriptors cannot drift from
    the layers they describe.
    """
    resolved_parallel = parallel
    if resolved_parallel is None:
        if plan.parallel is not None and plan.parallel.world_size > 1:
            raise ConfigError(
                "parallel",
                "PARALLEL_RANK_REQUIRED",
                "multi-rank execution requires an explicit rank-aware ParallelPlan",
            )
        resolved_parallel = ParallelPlan(
            tp_size=plan.parallel.tp_size if plan.parallel is not None else 1
        )
    # A pipeline stage owns exactly its declared layer span; single-rank plans
    # and full-span stages keep the full model range. The config-level
    # ResolvedParallelPlan owns the ranges; the runtime plan carries the rank.
    layer_range = (0, architecture.num_layers)
    if resolved_parallel.pp_size > 1:
        if plan.parallel is None:
            raise ConfigError(
                "parallel",
                "PARALLEL_PLAN_REQUIRED",
                "pipeline stages require a resolved parallel plan with layer ranges",
            )
        layer_range = plan.parallel.layer_range(resolved_parallel.pp_rank)
    return ExecutionPlan(
        plan_id=plan_id,
        model_id=model_id,
        model_revision=model_revision,
        weights_revision=weights_revision,
        compute=ComputePlan(
            dtype=compute_dtype,
            kv_dtype=cache_config.kv_dtype,
            layer_range=layer_range,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
        ),
        parallel=resolved_parallel,
        attention_groups=plan.attention_groups,
    )


def build_parallel_plan(parallel: ResolvedParallelPlan, world_rank: int) -> ParallelPlan:
    """Map the config-level plan onto the runtime rank-aware execution plan.

    ``dp_size`` is the physical DP coordinate count
    (:attr:`~ayaka.configs.parallel.ResolvedParallelPlan.dp_factor`); logical DP
    replicas stay on ``logical_replica_count``. The plan carries no layer range:
    callers take that from :meth:`ResolvedParallelPlan.layer_range`.
    """

    ranks = parallel.axis_ranks(world_rank)
    return ParallelPlan(
        tp_size=parallel.tp_size,
        tp_rank=ranks.tp_rank,
        pp_size=parallel.pp_size,
        pp_rank=ranks.pp_rank,
        dp_size=parallel.dp_factor,
        dp_rank=ranks.dp_rank,
        ep_size=parallel.ep_size,
        ep_rank=ranks.ep_rank,
        cp_size=parallel.cp_size,
        cp_rank=ranks.cp_rank,
        sp_enabled=parallel.sequence_parallel,
    )


@dataclass(frozen=True, slots=True)
class ParallelRuntime:
    """Rank-local parallel view for one process.

    Holds the runtime execution plan, this rank's pipeline layer span, the
    physical placement, and the borrowed communication objects layers consume.
    """

    plan: ParallelPlan
    layer_range: tuple[int, int]
    distributed: ResolvedDistributedPlan
    placement: RankPlacement
    tp_group: DeviceGroup | None
    ep_group: DeviceGroup | None
    communication: CommunicationBackend | None
    context: ParallelContext
    collective_policy: CollectivePolicy = CollectivePolicy.TORCH

    def layer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments accepted by the layer ``**runtime`` seam."""
        return {
            "tp_group": self.tp_group,
            "ep_group": self.ep_group,
            "communication": self.communication,
        }


def _device_group(
    name: str,
    ranks: tuple[int, ...],
    devices: tuple[DeviceRef, ...],
    world_rank: int,
) -> DeviceGroup:
    return DeviceGroup(
        name=name,
        devices=tuple(devices[rank] for rank in ranks),
        ranks=ranks,
        local_rank=ranks.index(world_rank),
    )


def _resolve_distributed(
    parallel: ResolvedParallelPlan,
    distributed: DistributedRuntimeConfig | ResolvedDistributedPlan | None,
) -> ResolvedDistributedPlan:
    if isinstance(distributed, DistributedRuntimeConfig):
        resolved = distributed.resolve(parallel.world_size)
    elif distributed is None:
        if parallel.world_size > 1:
            raise ConfigError(
                "distributed",
                "DISTRIBUTED_CONFIG_REQUIRED",
                "multi-rank execution requires DistributedRuntimeConfig",
            )
        resolved = DistributedRuntimeConfig().resolve(parallel.world_size)
    else:
        resolved = distributed
    if resolved.world_size != parallel.world_size:
        raise ConfigError(
            "distributed.world_size",
            "WORLD_SIZE_MISMATCH",
            "parallel and distributed plans must describe the same world size",
            context={
                "parallel_world_size": parallel.world_size,
                "distributed_world_size": resolved.world_size,
            },
        )
    return resolved


def build_parallel_runtime(
    parallel: ResolvedParallelPlan,
    distributed: DistributedRuntimeConfig | ResolvedDistributedPlan | None = None,
    *,
    world_rank: int | None = None,
    devices: Sequence[DeviceRef] | None = None,
    communication: CommunicationBackend | None = None,
    collective_setup: CollectiveSetup | None = None,
    device_kind: DeviceKind = DeviceKind.CUDA,
) -> ParallelRuntime:
    """Build the rank-local parallel runtime from the two resolved plans.

    The distributed plan supplies physical placement and the collective
    backend; the parallel plan supplies the axis decomposition. Multi-rank
    execution requires an explicit distributed config and a non-MPI collective
    backend. A missing communication backend is deferred to collective time,
    so construction itself needs no device stack.

    ``collective_setup`` is the runtime owner's opt-in to group agreement:
    when supplied for a non-torch policy the source rank's profile is exchanged
    once and the returned dispatcher is installed as ``communication``. Under
    ``custom_required`` a multi-rank build without a setup is refused before
    launch instead of silently running Torch.
    """

    resolved_distributed = _resolve_distributed(parallel, distributed)
    if parallel.world_size > 1 and resolved_distributed.collective_backend is CollectiveBackend.MPI:
        raise CapabilityError(
            "distributed.collective_backend",
            detail="MPI process groups have no runtime implementation",
            remedy="select CollectiveBackend.NCCL or CollectiveBackend.GLOO",
        )
    rank = distributed_env.rank() if world_rank is None else world_rank
    placement = resolved_distributed.placement_for(rank)
    resolved_devices = (
        tuple(devices) if devices is not None else resolved_distributed.device_refs(device_kind)
    )
    if len(resolved_devices) != parallel.world_size:
        raise ConfigError(
            "parallel.devices",
            "DEVICE_COUNT_MISMATCH",
            "one device reference is required per world rank",
        )
    ranks = parallel.axis_ranks(rank)
    tp_group = (
        _device_group("tp", parallel.tp_group_ranks(rank), resolved_devices, rank)
        if parallel.tp_size > 1
        else None
    )
    ep_group = (
        _device_group("ep", parallel.ep_group_ranks(rank), resolved_devices, rank)
        if parallel.ep_size > 1
        else None
    )

    policy = parallel.collective_policy
    if collective_setup is not None:
        if policy is CollectivePolicy.TORCH:
            raise ConfigError(
                "parallel.collective_policy",
                "COLLECTIVE_SETUP_ON_TORCH_POLICY",
                "a collective setup cannot be supplied while the policy is torch",
            )
        if tp_group is not None:
            from ayaka.distributed.collective_backend import setup_collective_backend

            communication = setup_collective_backend(
                policy=policy,
                group=tp_group,
                control=collective_setup.control,
                fallback=communication,
                capability=collective_setup.capability,
                custom_factory=collective_setup.custom_factory,
                timeout_s=collective_setup.timeout_s,
            )
    elif policy is CollectivePolicy.CUSTOM_REQUIRED and tp_group is not None:
        raise ConfigError(
            "parallel.collective_policy",
            "COLLECTIVE_SETUP_REQUIRED",
            "custom_required needs a probed collective setup before launch",
        )

    from ayaka.distributed.parallel import LocalParallelContext, RuntimeParallelContext

    context = (
        RuntimeParallelContext(tp_group, communication)
        if tp_group is not None
        else LocalParallelContext()
    )
    return ParallelRuntime(
        plan=build_parallel_plan(parallel, rank),
        layer_range=parallel.layer_range(ranks.pp_rank),
        distributed=resolved_distributed,
        placement=placement,
        tp_group=tp_group,
        ep_group=ep_group,
        communication=communication,
        context=context,
        collective_policy=policy,
    )
