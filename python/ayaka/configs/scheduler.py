"""Immutable scheduler contracts for single-flight execution.

Resolution checks declared execution/ledger capacities without allocating. It
does not certify a runnable engine or replace transactional memory admission.
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.memory.ledger import LedgerSnapshot, TierAccountSnapshot
from ayaka.plan import (
    ComputePlan,
    ExecutionPlan,
    GraphMode,
    MemoryPlan,
    ParallelPlan,
    WorkspaceRequest,
)
from ayaka.request.schema import Request
from ayaka.types import MemoryTier
from ayaka.utils.math_utils import align_down, align_up
from ayaka.utils.validation import require_frozen, require_int

__all__ = [
    "PreemptionMode",
    "ResolvedSchedulerPlan",
    "SchedulerCapabilities",
    "SchedulerConfig",
    "SchedulingPolicy",
    "scheduler_config_from_dict",
]


class SchedulingPolicy(enum.StrEnum):
    """Waiting order, independent of phase allocation and victim selection.

    LPM requires a prefix-capable runtime. Remaining-length scheduling remains
    unsupported until output-length estimates have an execution contract.
    """

    FCFS = "fcfs"
    PRIORITY = "priority"
    LONGEST_PREFIX_MATCH = "longest_prefix_match"
    SHORTEST_REMAINING = "shortest_remaining"


class PreemptionMode(enum.StrEnum):
    """RECOMPUTE requires a runtime that releases and rebinds KV; SWAP is unsupported."""

    NONE = "none"
    RECOMPUTE = "recompute"
    SWAP = "swap"


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerCapabilities(ConfigMixin):
    """Runner facts absent from ExecutionPlan, declared before allocation.

    max_num_seqs is metadata capacity. token_padding_multiple describes packed
    batch padding, not per-sequence padding. Chunking needs an explicit runner
    declaration. pipeline_stages declares a runner that executes declared
    layer-range stages and their boundary hand-offs (in-process PP only).
    This descriptor cannot enable multi-step, speculative or SWAP.
    """

    max_num_seqs: int
    token_padding_multiple: int = 1
    chunked_prefill: bool = False
    graph_mode: GraphMode = GraphMode.EAGER
    recompute_preemption: bool = False
    prefix_cache: bool = False
    pipeline_stages: bool = False

    def __post_init__(self) -> None:
        require_int(self.max_num_seqs, "capabilities.max_num_seqs", minimum=1)
        require_int(self.token_padding_multiple, "capabilities.token_padding_multiple", minimum=1)
        if type(self.chunked_prefill) is not bool:
            raise TypeError("capabilities.chunked_prefill must be bool")
        for name in ("recompute_preemption", "prefix_cache", "pipeline_stages"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"capabilities.{name} must be bool")
        if not isinstance(self.graph_mode, GraphMode):
            raise TypeError("capabilities.graph_mode must be GraphMode")

    def physical_token_slots(self, query_tokens: int) -> int:
        """Packed slots after padding; no speculative expansion is supported."""
        require_int(query_tokens, "query_tokens")
        return align_up(query_tokens, self.token_padding_multiple)


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerConfig(ConfigMixin):
    """Canonical settings; parse external/legacy keys with the adapter.

    max_num_requests counts all admitted, nonterminal requests once, including
    waiting/running/partial prompts. max_queued_requests is an optional additional
    waiting-only cap. Terminal requests may still hold ticket resources.
    max_num_seqs independently bounds distinct sequences in a single batch.

    Scheduled tokens are real queries; batched tokens include runner padding.
    The chunk cap bounds one request's prompt queries in one iteration. A partial
    prefill retains its slot across iterations until completion or lifecycle
    release; enforcing this state belongs to the concrete scheduler.

    tier_limits bounds ledger charges (committed + pending), not resident bytes
    or reserved virtual addresses. Missing tiers use ledger capacity.
    transfer_bytes bounds aggregate outstanding transfer bytes until retirement.
    iteration_ns is a soft batch target; retry_ns is resource backoff that capacity
    events may interrupt. Neither is a hard latency gate or request deadline.
    """

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_num_scheduled_tokens: int | None = None
    max_num_requests: int | None = 256
    max_queued_requests: int | None = None
    enable_chunked_prefill: bool = True
    max_prefill_chunk_tokens: int | None = None
    max_num_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FCFS
    preemption_mode: PreemptionMode = PreemptionMode.NONE
    priority_preemption: bool = False
    max_decode_steps_per_schedule: int = 1
    max_inflight: int = 1
    max_bypass: int = 8
    iteration_ns: int = 1_000_000_000
    retry_ns: int = 1_000_000
    transfer_bytes: int = 64 << 20
    tier_limits: tuple[tuple[MemoryTier, int], ...] = ()
    sampling_support_max_tokens: int = 512

    def __post_init__(self) -> None:
        for name in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_num_partial_prefills",
            "max_decode_steps_per_schedule",
            "max_inflight",
            "max_bypass",
            "iteration_ns",
            "retry_ns",
        ):
            require_int(getattr(self, name), f"scheduler.{name}", minimum=1)
        for name in (
            "max_num_scheduled_tokens",
            "max_num_requests",
            "max_queued_requests",
            "max_prefill_chunk_tokens",
        ):
            value = getattr(self, name)
            if value is not None:
                require_int(value, f"scheduler.{name}", minimum=1)
        require_int(self.long_prefill_token_threshold, "scheduler.long_prefill_token_threshold")
        require_int(self.transfer_bytes, "scheduler.transfer_bytes")
        require_int(
            self.sampling_support_max_tokens,
            "scheduler.sampling_support_max_tokens",
            minimum=1,
        )
        if type(self.enable_chunked_prefill) is not bool:
            raise TypeError("scheduler.enable_chunked_prefill must be bool")
        if type(self.priority_preemption) is not bool:
            raise TypeError("scheduler.priority_preemption must be bool")
        if not isinstance(self.scheduling_policy, SchedulingPolicy):
            raise TypeError("scheduler.scheduling_policy must be SchedulingPolicy")
        if not isinstance(self.preemption_mode, PreemptionMode):
            raise TypeError("scheduler.preemption_mode must be PreemptionMode")
        if self.effective_scheduled_token_budget > self.max_num_batched_tokens:
            raise ConfigError(
                "scheduler.max_num_scheduled_tokens",
                "SCHEDULED_TOKEN_BUDGET_TOO_LARGE",
                "issue budget cannot exceed execution capacity",
            )
        if self.max_num_partial_prefills > self.max_num_seqs:
            raise ConfigError(
                "scheduler.max_num_partial_prefills",
                "PARTIAL_PREFILL_LIMIT_TOO_LARGE",
                "partial-prefill limit exceeds the sequence cap",
            )
        if not self.enable_chunked_prefill and self.max_prefill_chunk_tokens is not None:
            raise ConfigError(
                "scheduler.max_prefill_chunk_tokens", "CHUNKING_DISABLED", "chunking is disabled"
            )
        if (
            self.max_prefill_chunk_tokens is not None
            and self.max_prefill_chunk_tokens > self.effective_scheduled_token_budget
        ):
            raise ConfigError(
                "scheduler.max_prefill_chunk_tokens",
                "CHUNK_BUDGET_TOO_LARGE",
                "chunk cap cannot exceed the issue budget",
            )
        for name, supported in (
            ("max_decode_steps_per_schedule", 1),
            ("max_num_partial_prefills", 1),
            ("long_prefill_token_threshold", 0),
            ("priority_preemption", False),
        ):
            if getattr(self, name) != supported:
                raise ConfigError(
                    f"scheduler.{name}", "UNSUPPORTED_FEATURE", f"baseline requires {supported!r}"
                )
        if self.preemption_mode is PreemptionMode.SWAP:
            raise ConfigError(
                "scheduler.preemption_mode", "UNSUPPORTED_FEATURE", "SWAP is not integrated"
            )
        if self.scheduling_policy not in (
            SchedulingPolicy.FCFS,
            SchedulingPolicy.PRIORITY,
            SchedulingPolicy.LONGEST_PREFIX_MATCH,
        ):
            raise ConfigError(
                "scheduler.scheduling_policy",
                "UNSUPPORTED_POLICY",
                "supported policies are FCFS, PRIORITY, and capability-gated LPM",
            )
        if type(self.tier_limits) is not tuple:
            raise TypeError("scheduler.tier_limits must be a tuple of tuple pairs")
        tiers: dict[MemoryTier, int] = {}
        for index, pair in enumerate(self.tier_limits):
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError(f"scheduler.tier_limits[{index}] must be a two-element tuple")
            tier, limit = pair
            if not isinstance(tier, MemoryTier):
                raise TypeError(f"scheduler.tier_limits[{index}] must name a MemoryTier")
            if not tier.allocatable:
                raise ConfigError(
                    f"scheduler.tier_limits[{index}]",
                    "INVALID_MEMORY_TIER",
                    "expected an allocatable MemoryTier",
                )
            if tier in tiers:
                raise ConfigError(
                    f"scheduler.tier_limits[{index}]",
                    "DUPLICATE_MEMORY_TIER",
                    "duplicate memory tier limit",
                )
            require_int(limit, f"scheduler.tier_limits[{index}].bytes")
            tiers[tier] = limit
        object.__setattr__(self, "tier_limits", tuple(sorted(tiers.items())))

    @property
    def effective_scheduled_token_budget(self) -> int:
        """Unpadded issue budget; resolve() also accounts for runner padding."""
        return (
            self.max_num_batched_tokens
            if self.max_num_scheduled_tokens is None
            else self.max_num_scheduled_tokens
        )

    def resolve(
        self,
        max_model_len: int,
        *,
        execution: ExecutionPlan,
        capabilities: SchedulerCapabilities,
        memory: LedgerSnapshot,
        workspace: MemoryPlan,
    ) -> ResolvedSchedulerPlan:
        """Bind execution/memory descriptors without allocation.

        max_model_len already incorporates model context resolution. Capacity is
        copied from the ledger snapshot, never from transient free bytes. Requiring
        execution/runner/memory inputs is an intentional break: context alone cannot
        certify compatible limits. workspace is the runner's static minimum plan;
        actual per-step reservations must still be checked during preparation.
        """
        require_int(max_model_len, "model.max_model_len", minimum=1)
        if not isinstance(capabilities, SchedulerCapabilities):
            raise TypeError("capabilities must be SchedulerCapabilities")
        require_frozen(capabilities, "capabilities")
        if not isinstance(memory, LedgerSnapshot):
            raise TypeError("memory must be LedgerSnapshot")
        if not isinstance(memory.tiers, Mapping):
            raise TypeError("memory.tiers must be a mapping of tier accounts")
        pairs = []
        for tier, account in memory.tiers.items():
            if not isinstance(account, TierAccountSnapshot) or tier is not account.tier:
                raise ConfigError("memory.tiers", "MEMORY_TIER_MISMATCH", "invalid tier account")
            pairs.append((tier, account.capacity_bytes))
        values = {item.name: getattr(self, item.name) for item in fields(SchedulerConfig)}
        issue = self.max_num_scheduled_tokens
        if issue is None:
            multiple = capabilities.token_padding_multiple
            issue = align_down(self.max_num_batched_tokens, multiple)
            if issue < 1:
                raise ConfigError(
                    "scheduler.max_num_scheduled_tokens",
                    "PADDED_CAPACITY_EXCEEDED",
                    "execution capacity is smaller than one padding unit",
                )
        values["max_num_scheduled_tokens"] = issue
        if self.enable_chunked_prefill and self.max_prefill_chunk_tokens is None:
            values["max_prefill_chunk_tokens"] = min(issue, max_model_len)
        return ResolvedSchedulerPlan(
            **values,
            max_model_len=max_model_len,
            execution=execution,
            capabilities=capabilities,
            memory_device_index=memory.device_index,
            memory_capacity_bytes=tuple(pairs),
            workspace=workspace,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedSchedulerPlan(SchedulerConfig):
    """Validated settings bound to immutable execution, runner and tier capacity.

    Direct construction repeats config and cross-module checks. Its fingerprint
    covers these host settings and execution identity, not a compiled graph key or
    acquired KV. No mutable ledger map is retained. Engine construction must pass
    this max_inflight explicitly to the executor; runtime wiring belongs to S1.
    """

    max_num_scheduled_tokens: int = field()  # type: ignore
    max_prefill_chunk_tokens: int | None = field()  # type: ignore
    max_model_len: int
    execution: ExecutionPlan
    capabilities: SchedulerCapabilities
    memory_device_index: int
    memory_capacity_bytes: tuple[tuple[MemoryTier, int], ...]
    workspace: MemoryPlan

    def __post_init__(self) -> None:
        SchedulerConfig.__post_init__(self)
        require_int(self.max_model_len, "model.max_model_len", minimum=1)
        require_int(self.memory_device_index, "memory.device_index")
        require_int(self.max_num_scheduled_tokens, "scheduler.max_num_scheduled_tokens", minimum=1)
        if self.enable_chunked_prefill and self.max_prefill_chunk_tokens is None:
            raise ConfigError(
                "scheduler.max_prefill_chunk_tokens", "UNRESOLVED_VALUE", "missing chunk cap"
            )
        for value, expected, name in (
            (self.execution, ExecutionPlan, "execution"),
            (self.capabilities, SchedulerCapabilities, "capabilities"),
            (self.workspace, MemoryPlan, "workspace"),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
            require_frozen(value, name)
        if type(self.memory_capacity_bytes) is not tuple:
            raise TypeError("memory.capacity_bytes must be a tuple of tuple pairs")
        capacities: dict[MemoryTier, int] = {}
        for index, pair in enumerate(self.memory_capacity_bytes):
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError(f"memory.capacity_bytes[{index}] must be a two-element tuple")
            tier, limit = pair
            if not isinstance(tier, MemoryTier):
                raise TypeError(f"memory.capacity_bytes[{index}] must name a MemoryTier")
            if not tier.allocatable:
                raise ConfigError(
                    f"memory.capacity_bytes[{index}]",
                    "INVALID_MEMORY_TIER",
                    "expected an allocatable MemoryTier",
                )
            if tier in capacities:
                raise ConfigError(
                    f"memory.capacity_bytes[{index}]",
                    "DUPLICATE_MEMORY_TIER",
                    "duplicate memory tier limit",
                )
            require_int(limit, f"memory.capacity_bytes[{index}].bytes")
            capacities[tier] = limit
        object.__setattr__(self, "memory_capacity_bytes", tuple(sorted(capacities.items())))

        compute = self.execution.compute
        if not isinstance(compute, ComputePlan):
            raise TypeError("execution.compute must be ComputePlan")
        require_frozen(compute, "execution.compute")
        parallel = self.execution.parallel
        if not isinstance(parallel, ParallelPlan):
            raise TypeError("execution.parallel must be ParallelPlan")
        require_frozen(parallel, "execution.parallel")
        for axis in ("tp", "pp", "dp", "ep", "cp"):
            require_int(
                getattr(parallel, f"{axis}_size"),
                f"execution.parallel.{axis}_size",
                minimum=1,
            )
            require_int(getattr(parallel, f"{axis}_rank"), f"execution.parallel.{axis}_rank")
        if type(parallel.sp_enabled) is not bool:
            raise TypeError("execution.parallel.sp_enabled must be bool")
        require_int(
            compute.max_num_batched_tokens,
            "execution.compute.max_num_batched_tokens",
            minimum=1,
        )
        if type(compute.enable_chunked_prefill) is not bool:
            raise TypeError("execution.compute.enable_chunked_prefill must be bool")
        if self.max_num_batched_tokens > compute.max_num_batched_tokens:
            raise ConfigError(
                "scheduler.max_num_batched_tokens",
                "EXECUTION_CAPACITY_EXCEEDED",
                "exceeds ComputePlan capacity",
            )
        if self.max_num_seqs > self.capabilities.max_num_seqs:
            raise ConfigError(
                "scheduler.max_num_seqs",
                "METADATA_CAPACITY_EXCEEDED",
                "exceeds runner sequence capacity",
            )
        if (
            self.capabilities.physical_token_slots(self.max_num_scheduled_tokens)
            > self.max_num_batched_tokens
        ):
            raise ConfigError(
                "scheduler.max_num_scheduled_tokens",
                "PADDED_CAPACITY_EXCEEDED",
                "padded issue budget exceeds execution capacity",
            )
        if (
            self.preemption_mode is PreemptionMode.RECOMPUTE
            and not self.capabilities.recompute_preemption
        ):
            raise ConfigError(
                "scheduler.preemption_mode",
                "PREEMPTION_UNSUPPORTED",
                "runtime must support recompute and resume",
            )
        if (
            self.scheduling_policy is SchedulingPolicy.LONGEST_PREFIX_MATCH
            and not self.capabilities.prefix_cache
        ):
            raise ConfigError(
                "scheduler.scheduling_policy",
                "PREFIX_CACHE_UNSUPPORTED",
                "runtime must supply prefix ownership and ranking",
            )
        if self.enable_chunked_prefill and not (
            compute.enable_chunked_prefill and self.capabilities.chunked_prefill
        ):
            raise ConfigError(
                "scheduler.enable_chunked_prefill",
                "CHUNKING_UNSUPPORTED",
                "compute plan and runner must both support chunking",
            )
        if not self.enable_chunked_prefill and self.max_model_len > self.max_num_scheduled_tokens:
            raise ConfigError(
                "scheduler.max_num_scheduled_tokens",
                "UNCHUNKED_MODEL_TOO_LARGE",
                "unchunked baseline requires model context to fit the issue budget",
            )
        if self.execution.parallel.sp_enabled:
            raise ConfigError(
                "execution.parallel.sp_enabled",
                "SP_UNSUPPORTED",
                "sequence parallelism is not integrated",
            )
        staged = (
            parallel.tp_size == 1
            and parallel.dp_size == 1
            and parallel.cp_size == 1
            and parallel.ep_size == 1
            and parallel.pp_size > 1
            and self.capabilities.pipeline_stages
        )
        if not parallel.is_single_process and not staged:
            raise ConfigError(
                "execution.parallel",
                "MULTI_RANK_UNAVAILABLE",
                "only a single-rank plan or a declared in-process pipeline is wired",
            )
        if self.capabilities.graph_mode not in (GraphMode.EAGER, GraphMode.REPLAY):
            raise ConfigError(
                "capabilities.graph_mode",
                "GRAPH_MODE_UNSUPPORTED",
                "eager and replay are the wired graph modes; capture is internal to the runner",
            )
        if compute.num_micro_batches > self.max_inflight:
            raise ConfigError(
                "scheduler.max_inflight",
                "INFLIGHT_BELOW_MICROBATCHES",
                "max_inflight must cover every concurrently submitted micro-batch",
            )

        if not capacities:
            raise ConfigError(
                "memory.capacity_bytes",
                "EMPTY_MEMORY_CAPACITY",
                "at least one ledger tier is required",
            )
        for tier, limit in self.tier_limits:
            if tier not in capacities or limit > capacities[tier]:
                raise ConfigError(
                    "scheduler.tier_limits",
                    "TIER_CAPACITY_EXCEEDED",
                    "tier limit exceeds the configured ledger capacity",
                )
            capacities[tier] = limit
        required: dict[MemoryTier, int] = {}
        for item in self.workspace.workspaces:
            if not isinstance(item, WorkspaceRequest):
                raise TypeError("workspace.request must be WorkspaceRequest")
            require_frozen(item, "workspace.request")
            if not isinstance(item.tier, MemoryTier):
                raise TypeError("workspace.tier must be MemoryTier")
            if not item.tier.allocatable:
                raise ConfigError(
                    "workspace.tier", "INVALID_MEMORY_TIER", "workspace tier must be allocatable"
                )
            require_int(item.nbytes, "workspace.nbytes")
            required[item.tier] = required.get(item.tier, 0) + item.nbytes
        require_int(self.workspace.activation_bytes, "workspace.activation_bytes")
        if self.workspace.activation_bytes:
            required[MemoryTier.DEVICE] = (
                required.get(MemoryTier.DEVICE, 0) + self.workspace.activation_bytes
            )
        for tier, nbytes in required.items():
            if tier not in capacities or nbytes > capacities[tier]:
                raise ConfigError(
                    "workspace",
                    "WORKSPACE_CAPACITY_EXCEEDED",
                    "workspace plus activations exceed the tier charge ceiling",
                )

    def validate_request(self, request: Request) -> None:
        """Check permanent length feasibility before admission without mutation.

        Reject prompt plus requested output beyond context; never clamp output.
        Queue occupancy, available KV pages, prefix acquisition and readiness still
        require the concrete scheduler and the existing memory/request owners.
        """
        if not isinstance(request, Request):
            raise TypeError("request must be Request")
        require_int(request.stop.max_tokens, "request.stop.max_tokens", minimum=1)
        if request.max_total_len > self.max_model_len:
            raise ConfigError(
                "request.max_total_len",
                "REQUEST_TOO_LONG",
                "prompt plus output limit exceeds model context",
            )
        if not self.enable_chunked_prefill and request.prompt_len > self.max_num_scheduled_tokens:
            raise ConfigError(
                "request.prompt_len", "PROMPT_EXCEEDS_BUDGET", "unchunked prompt cannot fit"
            )


def scheduler_config_from_dict(values: Mapping[str, Any]) -> SchedulerConfig:
    """Parse JSON-style canonical/legacy settings without retaining input aliases.

    Unknown keys and conflicting aliases raise ConfigError. Equal old/new values
    are accepted with DeprecationWarning. Legacy issue budgets never enlarge
    execution capacity. Strings are parsed only here; dataclasses require
    canonical enum values. to_dict()/JSON payloads round-trip through here.
    """
    aliases = {
        "max_tokens": "max_num_scheduled_tokens",
        "max_batch_requests": "max_num_seqs",
        "max_requests": "max_num_requests",
        "prefill_chunk": "max_prefill_chunk_tokens",
    }
    if not isinstance(values, Mapping) or any(type(key) is not str for key in values):
        raise ConfigError("scheduler", "INVALID_MAPPING", "expected a mapping with string keys")
    data = dict(values)
    known = {item.name for item in fields(SchedulerConfig)}
    unknown = data.keys() - known - aliases.keys()
    if unknown:
        raise ConfigError("scheduler", "UNKNOWN_KEY", f"unknown keys: {', '.join(sorted(unknown))}")
    used_aliases = []
    for old, new in aliases.items():
        if old not in data:
            continue
        value = data.pop(old)
        if new in data and (type(data[new]) is not type(value) or data[new] != value):
            raise ConfigError(f"scheduler.{old}", "ALIAS_CONFLICT", f"conflicts with {new}")
        data[new] = value
        used_aliases.append(f"{old} -> {new}")
    for name, enum_type in (
        ("scheduling_policy", SchedulingPolicy),
        ("preemption_mode", PreemptionMode),
    ):
        if name in data and type(data[name]) is str:
            try:
                data[name] = enum_type(data[name])
            except ValueError as exc:
                raise ConfigError(f"scheduler.{name}", "INVALID_ENUM", str(exc)) from exc
    if "tier_limits" in data:
        tiers = data["tier_limits"]
        if type(tiers) not in (list, tuple):
            raise ConfigError(
                "scheduler.tier_limits", "INVALID_TIER_LIMITS", "expected an array of pairs"
            )
        parsed = []
        for index, pair in enumerate(tiers):
            path = f"scheduler.tier_limits[{index}]"
            if type(pair) not in (tuple, list) or len(pair) != 2:
                raise ConfigError(path, "INVALID_TIER_PAIR", "expected a two-element array")
            tier, limit = pair
            try:
                if type(tier) is str:
                    tier = MemoryTier[tier.upper()]
                elif type(tier) is int:
                    tier = MemoryTier(tier)
            except (KeyError, ValueError) as exc:
                raise ConfigError(path, "INVALID_MEMORY_TIER", "unknown tier") from exc
            if not isinstance(tier, MemoryTier):
                raise ConfigError(path, "INVALID_MEMORY_TIER", "unknown tier")
            parsed.append((tier, limit))
        data["tier_limits"] = tuple(parsed)
    config = SchedulerConfig(**data)
    if used_aliases:
        warnings.warn(
            "Deprecated scheduler keys: " + ", ".join(used_aliases),
            DeprecationWarning,
            stacklevel=2,
        )
    return config
