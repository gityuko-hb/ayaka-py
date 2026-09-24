"""Host-tier capability policy for the config-driven resident KV runtime.

This is a structural runtime gate, not a pretrained-model certification record.
It inspects the resolved group plan before any ledger or storage is allocated.
The declared :data:`HOST_TIER_MATRIX` is the single source of truth: an
assessment matches the resolved plan against those rows and reports one
structured issue per axis no row permits. The un-tiered resident path retains
its existing capability checks and behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

from ayaka.configs.cache import CacheConfig, TieredCacheConfig
from ayaka.configs.model import ArchitectureConfig
from ayaka.kvcache.retention.policy import FullRetention, SlidingWindowRetention
from ayaka.kvcache.storage.layout import KVStorageKind
from ayaka.memory.tiering import TieringConfig
from ayaka.types import DType, KVLayoutKind

if TYPE_CHECKING:
    from ayaka.configs.assembly import ResidentKVPlan


class TierCapabilityIssueCode(StrEnum):
    """Reasons a requested host-tier capability is not available."""

    UNSUPPORTED_TIER = "unsupported_tier"
    HOST_CAPACITY_REQUIRED = "host_capacity_required"
    HOST_CAPACITY_TOO_SMALL = "host_capacity_too_small"
    TRANSFER_CAPACITY_TOO_SMALL = "transfer_capacity_too_small"
    ASYNC_TRANSFER_POLICY_UNSUPPORTED = "async_transfer_policy_unsupported"
    MODEL_FAMILY_UNSUPPORTED = "model_family_unsupported"
    STORAGE_UNSUPPORTED = "storage_unsupported"
    RETENTION_UNSUPPORTED = "retention_unsupported"
    DTYPE_UNSUPPORTED = "dtype_unsupported"
    PREFIX_UNSUPPORTED = "prefix_unsupported"
    GRAPH_UNSUPPORTED = "graph_unsupported"
    DISTRIBUTED_UNSUPPORTED = "distributed_unsupported"
    COMBINATION_UNSUPPORTED = "combination_unsupported"
    """Every axis is individually allowed but no declared row combines them."""


@dataclass(frozen=True, slots=True)
class TierCapabilityIssue:
    """One structured reason a host-tier request cannot be served."""

    code: TierCapabilityIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class HostTierMatrixRow:
    """One structurally supported host-tier combination.

    ``model_class`` refers to attention-only model geometry. The concrete
    ``ArchitectureConfig.architecture`` string is retained in each assessment
    but cannot establish checkpoint certification by itself. Every group in a
    matched plan must use the row's ``storage_kind`` and ``storage_layout``.
    """

    model_class: str
    group_layout: str
    retention_layout: str
    storage_kind: KVStorageKind
    storage_layout: KVLayoutKind
    kv_dtypes: frozenset[DType]
    prefix: bool
    graph: bool
    tiering: str
    topology: str


HOST_TIER_MATRIX: tuple[HostTierMatrixRow, ...] = (
    HostTierMatrixRow(
        model_class="attention_only",
        group_layout="homogeneous",
        retention_layout="full",
        storage_kind=KVStorageKind.MHA,
        storage_layout=KVLayoutKind.NHD,
        kv_dtypes=frozenset({DType.BF16}),
        prefix=True,
        graph=False,
        tiering="host",
        topology="single_process",
    ),
    HostTierMatrixRow(
        model_class="attention_only",
        group_layout="grouped",
        retention_layout="full+swa",
        storage_kind=KVStorageKind.MHA,
        storage_layout=KVLayoutKind.NHD,
        kv_dtypes=frozenset({DType.BF16}),
        prefix=True,
        graph=False,
        tiering="host",
        topology="single_process",
    ),
)


@dataclass(frozen=True, slots=True)
class HostTierCapability:
    """The declared capability axes for one resolved resident plan."""

    model_family: str
    model_class: str
    group_layout: str
    storage_contracts: tuple[tuple[str, str, str], ...]
    retention: tuple[tuple[str, str], ...]
    retention_layout: str
    kv_dtype: DType
    prefix: bool
    graph: bool
    tiering: str
    topology: str
    issues: tuple[TierCapabilityIssue, ...]

    @property
    def supported(self) -> bool:
        return not self.issues

    def require_supported(self) -> None:
        """Reject an unsupported host-tier request before resource allocation."""
        if self.issues:
            raise HostTierCapabilityError(self.issues, capability=self)


class HostTierCapabilityError(ValueError):
    """A requested host-tier combination is outside the declared matrix."""

    def __init__(
        self,
        issues: Sequence[TierCapabilityIssue],
        *,
        capability: HostTierCapability | None = None,
    ) -> None:
        self.capability = capability
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.code.value}: {issue.message}" for issue in self.issues)
        super().__init__(f"unsupported host-tier KV capability: {details}")


@dataclass(frozen=True, slots=True)
class GroupedHostTierPlan:
    """Per-group host mirror capacities resolved inside one byte budget."""

    configs: Mapping[str, TieringConfig]
    charged_bytes: int


@dataclass(frozen=True, slots=True)
class _TierRequest:
    """The matrix axes a resolved plan currently declares."""

    model_class: str
    group_layout: str
    retention_layout: str
    storage_kind: KVStorageKind | None
    """``None`` when groups disagree on storage kind."""
    storage_layout: KVLayoutKind | None
    """``None`` when groups disagree on storage layout."""
    kv_dtype: DType
    prefix: bool
    graph: bool
    tiering: str
    topology: str


def _retention_layout(plan: ResidentKVPlan) -> str:
    groups = plan.cache_groups
    if len(groups) == 1 and isinstance(groups[0].retention, FullRetention):
        return "full"
    if (
        len(groups) > 1
        and any(isinstance(group.retention, FullRetention) for group in groups)
        and any(isinstance(group.retention, SlidingWindowRetention) for group in groups)
        and all(
            isinstance(group.retention, (FullRetention, SlidingWindowRetention)) for group in groups
        )
    ):
        return "full+swa"
    return "other"


def _requested_tiering(cache_config: CacheConfig) -> str:
    tier = cache_config.tiering
    requested: list[str] = []
    if tier.host_bytes:
        requested.append("host")
    if tier.nvme_bytes or tier.nvme_path:
        requested.append("nvme")
    if tier.swap_bytes:
        requested.append("swap")
    if not requested and tier.max_inflight_bytes is not None:
        requested.append("transfer_only")
    return "+".join(requested) if requested else "disabled"


def _minimum_host_bytes(plan: ResidentKVPlan) -> int:
    """Bytes required to hold one page in every cache group."""
    return sum(spec.bytes_per_page for _, spec in plan.storage_specs)


def _largest_page_bytes(plan: ResidentKVPlan) -> int:
    """Bytes of the largest single page across every cache group."""
    return max(spec.bytes_per_page for _, spec in plan.storage_specs)


def _capacity_issues(plan: ResidentKVPlan, tier: TieredCacheConfig) -> list[TierCapabilityIssue]:
    """Host and in-flight transfer capacity a bounded tier must cover."""
    issues: list[TierCapabilityIssue] = []
    if tier.host_bytes < 1:
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.HOST_CAPACITY_REQUIRED,
                "host_bytes must be positive",
            )
        )
    else:
        minimum_host_bytes = _minimum_host_bytes(plan)
        if tier.host_bytes < minimum_host_bytes:
            issues.append(
                TierCapabilityIssue(
                    TierCapabilityIssueCode.HOST_CAPACITY_TOO_SMALL,
                    f"host_bytes needs {minimum_host_bytes} bytes for one page per group",
                )
            )
    if tier.max_inflight_bytes is not None and plan.storage_specs:
        minimum_transfer_bytes = _largest_page_bytes(plan)
        if tier.max_inflight_bytes < minimum_transfer_bytes:
            issues.append(
                TierCapabilityIssue(
                    TierCapabilityIssueCode.TRANSFER_CAPACITY_TOO_SMALL,
                    "max_inflight_bytes must cover one page of the largest group "
                    f"({minimum_transfer_bytes} bytes)",
                )
            )
    return issues


def _request_axes(
    architecture: ArchitectureConfig,
    cache_config: CacheConfig,
    plan: ResidentKVPlan,
    *,
    prefix_enabled: bool,
    graph_enabled: bool,
) -> _TierRequest:
    groups = plan.cache_groups
    kinds = {group.storage_spec.kind for group in groups}
    layouts = {group.storage_spec.layout for group in groups}
    tiering = "host" if cache_config.tiering.host_bytes else "none"
    return _TierRequest(
        model_class="attention_only" if not architecture.is_moe else "moe",
        group_layout="homogeneous" if len(groups) == 1 else "grouped",
        retention_layout=_retention_layout(plan),
        storage_kind=next(iter(kinds)) if len(kinds) == 1 else None,
        storage_layout=next(iter(layouts)) if len(layouts) == 1 else None,
        kv_dtype=cache_config.kv_dtype,
        prefix=prefix_enabled,
        graph=graph_enabled,
        tiering=tiering,
        topology=(
            "single_process"
            if plan.parallel is None
            or (plan.parallel.world_size == 1 and plan.parallel.ep_size == 1)
            else "distributed"
        ),
    )


def _row_matches(row: HostTierMatrixRow, request: _TierRequest) -> bool:
    return (
        row.model_class == request.model_class
        and row.group_layout == request.group_layout
        and row.retention_layout == request.retention_layout
        and row.storage_kind is request.storage_kind
        and row.storage_layout is request.storage_layout
        and request.kv_dtype in row.kv_dtypes
        and row.prefix == request.prefix
        and row.graph == request.graph
        and row.tiering == request.tiering
        and row.topology == request.topology
    )


def _unmatched_axis_issues(
    request: _TierRequest, rows: Sequence[HostTierMatrixRow]
) -> list[TierCapabilityIssue]:
    """One issue per axis no declared row permits for this request.

    The matrix is authoritative: a mismatch is attributed to the axis whose
    value appears in no row, so a new row is the only way to widen support.
    """
    issues: list[TierCapabilityIssue] = []
    if not any(row.model_class == request.model_class for row in rows):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.MODEL_FAMILY_UNSUPPORTED,
                "tiered resident KV supports attention-only model geometry",
            )
        )
    if not any(
        (row.group_layout, row.retention_layout) == (request.group_layout, request.retention_layout)
        for row in rows
    ):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.RETENTION_UNSUPPORTED,
                "tiering requires one full group or grouped full+SWA retention",
            )
        )
    if not any(
        row.storage_kind is request.storage_kind and row.storage_layout is request.storage_layout
        for row in rows
    ):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.STORAGE_UNSUPPORTED,
                "tiered resident KV supports MHA/GQA NHD storage only",
            )
        )
    if not any(request.kv_dtype in row.kv_dtypes for row in rows):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.DTYPE_UNSUPPORTED,
                "tiered resident KV is gated to BF16",
            )
        )
    if not any(row.prefix == request.prefix for row in rows):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.PREFIX_UNSUPPORTED,
                "config-driven host tiering requires a canonical prefix provider",
            )
        )
    if not any(row.graph == request.graph for row in rows):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.GRAPH_UNSUPPORTED,
                "tiered resident graph capture/replay needs a separate gate",
            )
        )
    if request.tiering == "host" and not any(row.tiering == request.tiering for row in rows):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.UNSUPPORTED_TIER,
                "the declared matrix supports host tiering only",
            )
        )
    if not any(row.topology == request.topology for row in rows):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.DISTRIBUTED_UNSUPPORTED,
                "distributed tiering needs rank-local readiness and collective ordering proof",
            )
        )
    return issues


def assess_host_tier_capability(
    architecture: ArchitectureConfig,
    cache_config: CacheConfig,
    plan: ResidentKVPlan,
    *,
    prefix_enabled: bool,
    graph_enabled: bool,
) -> HostTierCapability:
    """Assess config-driven resident tiering using resolved per-group contracts.

    ``prefix_enabled`` means a prefix provider is actually attached; the
    configuration default alone does not turn on prefix reuse. ``graph_enabled``
    means capture/replay is requested by the runner. Unsupported tier requests
    are reported as structured issues before ledger or storage creation, and
    the declared :data:`HOST_TIER_MATRIX` decides which combinations pass.
    """
    groups = plan.cache_groups
    requested_tiering = _requested_tiering(cache_config)
    request = _request_axes(
        architecture,
        cache_config,
        plan,
        prefix_enabled=prefix_enabled,
        graph_enabled=graph_enabled,
    )
    retention = tuple((group.name, group.retention.policy_id) for group in groups)
    storage_contracts = tuple(
        (group.name, group.storage_spec.kind.value, group.storage_spec.layout.value)
        for group in groups
    )

    def result(issues: tuple[TierCapabilityIssue, ...]) -> HostTierCapability:
        return HostTierCapability(
            model_family=architecture.architecture,
            model_class=request.model_class,
            group_layout=request.group_layout,
            storage_contracts=storage_contracts,
            retention=retention,
            retention_layout=request.retention_layout,
            kv_dtype=request.kv_dtype,
            prefix=request.prefix,
            graph=request.graph,
            tiering=requested_tiering,
            topology=request.topology,
            issues=issues,
        )

    # Preserve the existing resident fast path and its independently validated
    # graph/prefix/distributed combinations when host tiering is disabled.
    if requested_tiering == "disabled":
        return result(())

    issues: list[TierCapabilityIssue] = []
    tier = cache_config.tiering
    if tier.nvme_bytes or tier.nvme_path or tier.swap_bytes:
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.UNSUPPORTED_TIER,
                "host tiering only; NVMe and swap are separate gates",
            )
        )
    issues.extend(_capacity_issues(plan, tier))
    if not tier.async_transfers:
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.ASYNC_TRANSFER_POLICY_UNSUPPORTED,
                "async_transfers=false is not wired in the resident tier runtime",
            )
        )
    if (
        prefix_enabled
        and any(isinstance(group.retention, SlidingWindowRetention) for group in groups)
        and not cache_config.prefix.allow_sliding_window_reuse
    ):
        issues.append(
            TierCapabilityIssue(
                TierCapabilityIssueCode.PREFIX_UNSUPPORTED,
                "sliding-window prefix reuse requires allow_sliding_window_reuse=true",
            )
        )
    if request.tiering == "host" and not any(
        _row_matches(row, request) for row in HOST_TIER_MATRIX
    ):
        axis_issues = _unmatched_axis_issues(request, HOST_TIER_MATRIX)
        issues.extend(
            axis_issues
            or [
                TierCapabilityIssue(
                    TierCapabilityIssueCode.COMBINATION_UNSUPPORTED,
                    "no declared matrix row combines these capability axes",
                )
            ]
        )
    return result(tuple(issues))


def plan_group_host_tiers(plan: ResidentKVPlan, tiering: TieredCacheConfig) -> GroupedHostTierPlan:
    """Size independent group mirrors inside one frozen host byte budget.

    Every group gets one page first. Additional whole pages are divided in
    proportion to each group's device slab size; any remainder goes to the
    smallest page geometry without exceeding the byte budget. A budget the
    declared matrix cannot serve raises the same structured issues the
    assessment reports instead of a plain capacity error.
    """
    issues = _capacity_issues(plan, tiering)
    if issues:
        raise HostTierCapabilityError(issues)
    specs = dict(plan.storage_specs)
    minimum_bytes = _minimum_host_bytes(plan)
    remaining = tiering.host_bytes - minimum_bytes
    weights = {
        name: spec.bytes_per_page * (spec.capacity_pages - 1) for name, spec in specs.items()
    }
    total_weight = sum(weights.values())
    pages = (
        {
            name: 1 + (remaining * weights[name] // total_weight) // spec.bytes_per_page
            for name, spec in specs.items()
        }
        if total_weight
        else {name: 1 for name in specs}
    )
    charged = sum(pages[name] * spec.bytes_per_page for name, spec in specs.items())
    remaining = tiering.host_bytes - charged
    for name, spec in sorted(specs.items(), key=lambda item: (item[1].bytes_per_page, item[0])):
        extra, remaining = divmod(remaining, spec.bytes_per_page)
        pages[name] += extra
    charged = sum(pages[name] * spec.bytes_per_page for name, spec in specs.items())
    configs = {name: TieringConfig(host_capacity_pages=pages[name]) for name in specs}
    return GroupedHostTierPlan(MappingProxyType(configs), charged)
