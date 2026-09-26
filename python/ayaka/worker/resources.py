"""Worker-owned resident storage, accounting and immutable capacity.

Construction admits each backing allocation once. The same owner is used for
startup and cache rebuild; scheduler and HTTP objects are never stored here.
Borrowed models are accounted for but are not destroyed by this owner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from ayaka.configs.assembly import (
    ResidentKVPlan,
    bind_resident_kv,
    materialize_resident_kv,
    plan_resident_kv,
    resident_ledger,
)
from ayaka.configs.cache import CacheConfig
from ayaka.configs.memory import MemoryConfig, MemoryProfile, plan_device_memory
from ayaka.configs.model import ArchitectureConfig
from ayaka.configs.parallel import ParallelConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.tier_capability import (
    assess_host_tier_capability,
    plan_group_host_tiers,
)
from ayaka.kvcache.grouped_manager import KVCacheGroupManager
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.materialize import KVStorageLease, materialize_kv_storage
from ayaka.kvcache.resize import (
    LEDGER_OVERHEAD_BYTES,
    CacheRebuildRejected,
    ResizeRejectionReason,
)
from ayaka.kvcache.storage.geometry import MHAStorageSpec
from ayaka.kvcache.storage.layout import DEFAULT_KV_ALIGNMENT_BYTES
from ayaka.memory.caching import CachingAllocator
from ayaka.memory.capacity import (
    CapacityFreeze,
    CapacitySnapshot,
    MemoryLane,
    build_capacity_snapshot,
    claim_tier,
    mint_generation,
    reconcile_actual_usage,
)
from ayaka.memory.host.host_policy import HostMemoryPolicy
from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.source import TorchDeviceSource, TorchHostByteSource
from ayaka.memory.tiering import HostKVStorage, TieringConfig, build_transfer_engine
from ayaka.memory.workspace import WorkspaceManager
from ayaka.obs import runtime_event
from ayaka.prefix.transfer import TransferCredits
from ayaka.runner.buffers import RunnerBuffers, RunnerBufferSpec
from ayaka.types import DType, MemoryOwner, MemoryTier
from ayaka.utils.torch_memory import device_memory
from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class WorkerResourcePlan:
    """All inputs to a resident allocation, independent of request state."""

    device: torch.device
    storage_spec: MHAStorageSpec
    buffer_spec: RunnerBufferSpec
    memory_config: MemoryConfig
    device_total_bytes: int | None
    model_id: str
    model_revision: str
    weights_revision: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_inflight: int
    weights_bytes: int
    activation_bytes: int
    workspace_ceiling_bytes: int
    graph_bytes: int
    staging_bytes: int
    # Logical handles for waiting lifecycle; no KV pages until reservation.
    max_num_requests: int | None = None
    tiering: TieringConfig | None = None
    max_inflight_bytes: int | None = None
    #: Resolved host pin ceiling for the KV mirror. ``None`` keeps the
    #: allocation-time behavior (pin on CUDA, pageable fallback allowed).
    host_policy: HostMemoryPolicy | None = None
    #: Pinned mirror bytes a shorter-lived rebuild still holds (a SWAP holds
    #: both mirrors until the old tail closes). Counted against the pin ceiling.
    already_pinned_host_bytes: int = 0
    #: Alignment the KV slab is charged with; must match the planner's cost
    #: model for this geometry.
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES

    @property
    def lane(self) -> MemoryLane:
        return MemoryLane.CUDA if self.device.type == "cuda" else MemoryLane.CPU

    def budget(self) -> tuple[int, int, int]:
        """Return policy, KV and total budgets before materializing anything."""
        require_int(self.alignment_bytes, "alignment_bytes", minimum=1)
        slab = self.storage_spec.aligned_total_bytes(self.alignment_bytes) + LEDGER_OVERHEAD_BYTES
        if self.lane is MemoryLane.CUDA:
            total = self.device_total_bytes
            if total is None:
                total = device_memory(self.device).driver_total
            profile = MemoryProfile(
                weights_bytes=self.weights_bytes,
                activation_bytes=self.activation_bytes,
                kernel_workspace_bytes=self.workspace_ceiling_bytes,
                graph_pool_bytes=self.graph_bytes,
                runner_buffer_bytes=self.buffer_spec.device_bytes,
                measured=True,
            )
            memory = plan_device_memory(total, self.memory_config, profile)
            if slab > memory.kv_cache_bytes:
                raise CacheRebuildRejected(
                    "KV slab exceeds the budget remaining after non-KV allocations",
                    reason=ResizeRejectionReason.BUDGET,
                    requested_pages=self.storage_spec.capacity_pages,
                    need_bytes=slab,
                    available_bytes=memory.kv_cache_bytes,
                )
            return memory.policy_budget_bytes, memory.kv_cache_bytes, total
        owned = (
            slab
            + self.weights_bytes
            + self.activation_bytes
            + self.workspace_ceiling_bytes
            + self.graph_bytes
            + self.staging_bytes
            + self.buffer_spec.device_bytes
        )
        budget = owned + max(4 << 20, owned // 16)
        return budget, slab, budget

    def build(self) -> WorkerResources:
        """Allocate, reconcile and freeze; roll back every claim on failure."""
        max_sequences = self.max_num_requests
        if max_sequences is None:
            max_sequences = self.max_num_seqs
        require_int(max_sequences, "max_num_requests", minimum=1)
        budget, kv_budget, total = self.budget()
        index = self.device.index or 0
        # The host mirror is charged once, at its frozen capacity, before the
        # ledger exists: a pinned mirror must never exceed the HOST_PINNED
        # account, and a degraded pageable mirror still has a place to charge.
        tier = self.tiering
        mirror_bytes = (
            self.storage_spec.bytes_per_page * tier.host_capacity_pages if tier is not None else 0
        )
        # The host hard cap gates the mirror before a single byte is allocated.
        # A refusal with fallback allowed is a decision, not an error: the
        # mirror is built pageable and charged there.
        mirror_decision = None
        if tier is not None and self.host_policy is not None:
            mirror_decision = self.host_policy.decide_mirror(
                mirror_bytes,
                requested_tier=(
                    MemoryTier.HOST_PINNED
                    if self.lane is MemoryLane.CUDA
                    else MemoryTier.HOST_PAGEABLE
                ),
                already_pinned_bytes=self.staging_bytes + self.already_pinned_host_bytes,
            )
            if not mirror_decision:
                raise CacheRebuildRejected(
                    f"host KV mirror refused by host memory policy: {mirror_decision.reason}",
                    reason=ResizeRejectionReason.BUDGET,
                    requested_pages=tier.host_capacity_pages,
                    need_bytes=mirror_bytes,
                    available_bytes=mirror_decision.headroom_bytes,
                )
        # Both host accounts cover staging plus mirror: staging itself may
        # degrade to pageable, so sizing only one account turns an allowed
        # fallback into a ledger overrun.
        ledger = MemoryLedger.for_device(
            device_budget_bytes=budget,
            device_total_bytes=total,
            host_pinned_bytes=self.staging_bytes + mirror_bytes,
            device_index=index,
            host_pageable_bytes=(
                budget if self.lane is MemoryLane.CPU else self.staging_bytes + mirror_bytes
            ),
            host_total_bytes=total if self.lane is MemoryLane.CPU else 0,
        )
        with ExitStack() as rollback:
            # Registered first: accounting is released only after backing owners.
            for owner in (MemoryOwner.WEIGHT, MemoryOwner.COMPILE, MemoryOwner.WORKSPACE):
                rollback.callback(ledger.release_owner, owner)
            for owner, label, nbytes in (
                (MemoryOwner.WEIGHT, "serving.weights", self.weights_bytes),
                (MemoryOwner.COMPILE, "serving.graph", self.graph_bytes),
                (MemoryOwner.WORKSPACE, "serving.staging", self.staging_bytes),
            ):
                if nbytes:
                    ledger.admit(
                        Reservation.backed(
                            owner,
                            label,
                            nbytes,
                            tier=claim_tier(
                                owner,
                                lane=self.lane,
                                pinned_host=owner is MemoryOwner.WORKSPACE,
                            ),
                            device_index=index,
                        )
                    )
            storage = materialize_kv_storage(
                self.storage_spec,
                ledger=ledger,
                label="serving.kv",
                device=str(self.device),
                zero_initialize=True,
                alignment_bytes=self.alignment_bytes,
            )
            rollback.callback(storage.close)
            tier_release: Callable[[], None] | None = None
            host_mirror: HostKVStorage | None = None
            credits: TransferCredits | None = None
            engine: object | None = None
            if tier is not None:
                allow_pageable = (
                    self.host_policy.allow_pageable_fallback
                    if self.host_policy is not None
                    else True
                )
                host_mirror = HostKVStorage(
                    storage.storage,
                    capacity_pages=tier.host_capacity_pages,
                    pin_memory=(
                        mirror_decision.tier is MemoryTier.HOST_PINNED
                        if mirror_decision is not None
                        else self.lane is MemoryLane.CUDA
                    ),
                    allow_pageable_fallback=allow_pageable,
                )
                if mirror_decision is not None:
                    requested_tier = mirror_decision.tier
                    assert requested_tier is not None
                    if mirror_decision.fallback or host_mirror.actual_tier is not requested_tier:
                        runtime_event(
                            "host_mirror",
                            owner_id="tier",
                            status="pageable_fallback",
                            detail=(
                                f"requested={requested_tier.name} "
                                f"actual={host_mirror.actual_tier.name}: {mirror_decision.reason}"
                            ),
                        )
                # The mirror is page-locked host staging for KV content: the
                # R01 owner table charges it to WORKSPACE on the host tier
                # exactly like runner-buffer staging, never as a second KV
                # residency on the device tier.
                mirror_reservation = Reservation.backed(
                    MemoryOwner.WORKSPACE,
                    "serving.kv.tier",
                    host_mirror.total_bytes,
                    tier=claim_tier(
                        MemoryOwner.WORKSPACE,
                        lane=self.lane,
                        pinned_host=host_mirror.pinned,
                        pageable_host=not host_mirror.pinned,
                    ),
                    device_index=index,
                )
                ledger.admit(mirror_reservation)
                rollback.callback(ledger.release, mirror_reservation.label)
                # One shared in-flight staging budget for every host copy the
                # tier makes; the mirror capacity is the default ceiling.
                credits = TransferCredits(
                    self.max_inflight_bytes
                    if self.max_inflight_bytes is not None
                    else host_mirror.total_bytes
                )
                engine = build_transfer_engine(host_mirror)

                def _tier_teardown() -> None:
                    # The mirror tensors and its ledger charge die together,
                    # after the manager proved a clean, settled tier.
                    manager.release_tier()
                    ledger.release(mirror_reservation.label)

                tier_release = _tier_teardown

            manager = RuntimeMemoryManager(
                total_pages=self.storage_spec.capacity_pages,
                page_size=self.storage_spec.page_size,
                max_sequences=max_sequences,
                max_sequence_tokens=self.max_model_len,
                storage=storage.storage,
                tiering=tier,
                host_storage=host_mirror,
                transfer_engine=engine,
                transfer_budget=credits,
            )
            kv = LogicalKVManager(manager, {"default": storage})
            rollback.callback(kv.close)
            source = (
                TorchDeviceSource(device_index=index)
                if self.lane is MemoryLane.CUDA
                else TorchHostByteSource(pinned=False)
            )
            allocator = CachingAllocator(
                source,
                ledger=ledger,
                tier=claim_tier(MemoryOwner.WORKSPACE, lane=self.lane),
                device_index=index,
                ledger_label="serving.workspace",
            )
            rollback.callback(allocator.close)
            workspace = WorkspaceManager(
                allocator,
                workspace_ceiling_bytes=self.workspace_ceiling_bytes,
            )
            rollback.callback(workspace.close)
            workspace.initialize(self.activation_bytes)
            buffers = RunnerBuffers(
                self.buffer_spec,
                device=self.device,
                pin_staging=self.lane is MemoryLane.CUDA,
                ledger=ledger,
                device_index=index,
                label="serving.runner_buffers",
                reserve_staging=False,
            )
            rollback.callback(buffers.close)
            generation = mint_generation(
                model_id=self.model_id,
                model_revision=self.model_revision,
                weights_revision=self.weights_revision,
                kv_storage=kv.fingerprint,
                backend=type(kv.backend).__name__,
                workspace=workspace.workspace_generation,
                buffers=buffers.generation,
            )
            capacity = build_capacity_snapshot(
                generation=generation,
                lane=self.lane,
                dtype=str(self.storage_spec.dtype).removeprefix("torch."),
                kv_dtype=str(self.storage_spec.dtype).removeprefix("torch."),
                group_pages={"default": self.storage_spec.capacity_pages},
                group_page_sizes={"default": self.storage_spec.page_size},
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_inflight=self.max_inflight,
                activation_bytes=allocator.bytes_by_owner().get(MemoryOwner.ACTIVATION, 0),
                workspace_ceiling_bytes=self.workspace_ceiling_bytes,
                graph_bytes=self.graph_bytes,
                staging_bytes=self.staging_bytes + mirror_bytes,
                budget_bytes=budget,
                kv_budget_bytes=kv_budget,
                weights_bytes=self.weights_bytes,
                ledger=ledger,
                staging_tier=(
                    MemoryTier.HOST_PAGEABLE
                    if host_mirror is not None and not host_mirror.pinned
                    else MemoryTier.HOST_PINNED
                ),
                runner_buffer_bytes=self.buffer_spec.device_bytes,
            )
            kv.bind_capacity(capacity)
            resources = WorkerResources(
                ledger,
                kv,
                (storage,),
                capacity,
                workspace,
                allocator,
                buffers,
                tier_release=tier_release,
            )
            rollback.pop_all()
            return resources


class WorkerResources:
    """One owner for the ledger, frozen generation and physical resident tail.

    Ticket leases remain owned by the executor. Closing refuses live logical
    ownership, and never releases a slab after a failed manager close.
    """

    def __init__(
        self,
        ledger: MemoryLedger,
        kv: LogicalKVManager,
        storages: tuple[KVStorageLease, ...],
        capacity: CapacitySnapshot,
        workspace: WorkspaceManager | None = None,
        allocator: CachingAllocator | None = None,
        buffers: RunnerBuffers | None = None,
        *,
        tier_release: Callable[[], None] | None = None,
    ) -> None:
        reconcile_actual_usage(ledger, capacity)
        self.ledger = ledger
        self.kv = kv
        self.storages = storages
        self.freeze = CapacityFreeze(capacity)
        self.workspace = workspace
        self.allocator = allocator
        self.buffers = buffers
        self.tier_release = tier_release
        self.closed = False

    @property
    def capacity(self) -> CapacitySnapshot:
        return self.freeze.current

    def close(self) -> None:
        """Release a quiescent tail; retain failed owners for inspection/retry."""
        if self.closed:
            return
        # Consumers must prove quiescence before the resident slab is released.
        # RunnerBuffers.close() refuses live views, so checking it first keeps
        # the KV owner intact for diagnosis and a later retry.
        if self.buffers is not None:
            self.buffers.close()
        if self.workspace is not None:
            self.workspace.close()
        if self.allocator is not None:
            self.allocator.close()
        if not self.kv.closed:
            self.kv.reclaim_deferred()
            self.kv.close()
        if self.tier_release is not None:
            self.tier_release()
        for storage in reversed(self.storages):
            storage.close()
        for owner in (MemoryOwner.WEIGHT, MemoryOwner.COMPILE, MemoryOwner.WORKSPACE):
            self.ledger.release_owner(owner)
        if any(self.ledger.committed(tier) for tier in MemoryTier):
            raise RuntimeError("worker resource shutdown left ledger claims")
        self.closed = True


def build_configured_resources(
    architecture: ArchitectureConfig,
    cache_config: CacheConfig,
    *,
    scheduler: SchedulerConfig,
    device_total_bytes: int,
    model_id: str,
    model_revision: str,
    weights_revision: str,
    memory_config: MemoryConfig | None,
    profile: MemoryProfile | None,
    parallel: ParallelConfig,
    max_model_len: int | None,
    compute_dtype: DType,
    device: str,
    device_index: int,
    zero_initialize: bool,
    prefix_enabled: bool = False,
    graph_enabled: bool = False,
) -> tuple[ResidentKVPlan, WorkerResources]:
    """Build the config-driven engine's resident owners before scheduler wiring."""
    plan = plan_resident_kv(
        architecture,
        cache_config,
        device_total_bytes=device_total_bytes,
        max_num_seqs=scheduler.max_num_seqs,
        memory_config=memory_config,
        profile=profile,
        parallel=parallel,
        max_model_len=max_model_len,
    )
    # Tiering changes the supported capability and host accounting. Resolve
    # both before materializing any device slab.
    grouped_tiers: Mapping[str, TieringConfig] = {}
    mirror_bytes = 0
    tier_policy = cache_config.tiering
    tier_requested = bool(
        tier_policy.host_bytes
        or tier_policy.nvme_bytes
        or tier_policy.swap_bytes
        or tier_policy.nvme_path is not None
        or tier_policy.max_inflight_bytes is not None
    )
    if tier_requested:
        assess_host_tier_capability(
            architecture,
            cache_config,
            plan,
            prefix_enabled=prefix_enabled,
            graph_enabled=graph_enabled,
        ).require_supported()
    if tier_policy.host_bytes:
        host_tier_plan = plan_group_host_tiers(plan, tier_policy)
        grouped_tiers = host_tier_plan.configs
        mirror_bytes = host_tier_plan.charged_bytes
    lane = MemoryLane.CPU if torch.device(device).type == "cpu" else MemoryLane.CUDA
    host_policy = None
    mirror_decision = None
    if grouped_tiers:
        host_policy = HostMemoryPolicy.from_limits(
            tier_policy.host_bytes,
            (memory_config or MemoryConfig()).host,
        )
        mirror_decision = host_policy.decide_mirror(
            mirror_bytes,
            requested_tier=(
                MemoryTier.HOST_PINNED if lane is MemoryLane.CUDA else MemoryTier.HOST_PAGEABLE
            ),
            already_pinned_bytes=0,
        )
        if not mirror_decision:
            raise CacheRebuildRejected(
                f"grouped host KV mirror refused by host memory policy: {mirror_decision.reason}",
                reason=ResizeRejectionReason.BUDGET,
                need_bytes=mirror_bytes,
                available_bytes=mirror_decision.headroom_bytes,
            )
    if grouped_tiers:
        # CUDA mirrors can degrade to pageable memory. Reserve both possible
        # host accounts at the frozen ceiling; charge only the actual kind.
        if (
            lane is MemoryLane.CPU
            and plan.physical.allocated_bytes + mirror_bytes > plan.memory.policy_budget_bytes
        ):
            raise ValueError("cache.tiering.host_bytes exceeds the CPU resident memory budget")
        ledger = MemoryLedger.for_device(
            device_budget_bytes=plan.memory.policy_budget_bytes,
            device_total_bytes=device_total_bytes,
            host_pinned_bytes=mirror_bytes if lane is MemoryLane.CUDA else 0,
            host_pageable_bytes=(
                plan.memory.policy_budget_bytes if lane is MemoryLane.CPU else mirror_bytes
            ),
            host_total_bytes=device_total_bytes if lane is MemoryLane.CPU else 0,
            device_index=device_index,
        )
    else:
        ledger = resident_ledger(
            plan.memory,
            device=device,
            device_total_bytes=device_total_bytes,
            device_index=device_index,
        )
    with ExitStack() as rollback:
        leases = materialize_resident_kv(
            plan,
            ledger=ledger,
            device=device,
            zero_initialize=zero_initialize,
        )
        for lease in leases:
            rollback.callback(lease.close)
        tier_release: Callable[[], None] | None = None
        # ExitStack runs callbacks in reverse order: close KV (and its
        # canonical host entries) before releasing mirror claims, then slabs.
        rollback.callback(lambda: tier_release() if tier_release is not None else None)
        backend, kv = bind_resident_kv(plan, leases)
        rollback.callback(kv.close)
        mirror_pinned = (
            mirror_decision.tier is MemoryTier.HOST_PINNED
            if mirror_decision is not None
            else lane is MemoryLane.CUDA
        )
        if grouped_tiers:
            if not isinstance(backend, KVCacheGroupManager):
                raise TypeError("grouped host tier requires KVCacheGroupManager")
            by_group = {lease.label: lease for lease in leases}
            charged_labels: list[str] = []
            allow_pageable = (
                host_policy.allow_pageable_fallback if host_policy is not None else True
            )
            try:
                mirrors = {
                    name: HostKVStorage(
                        by_group[name].storage,
                        capacity_pages=config.host_capacity_pages,
                        pin_memory=mirror_pinned,
                        allow_pageable_fallback=allow_pageable,
                    )
                    for name, config in grouped_tiers.items()
                }
                if sum(mirror.total_bytes for mirror in mirrors.values()) != mirror_bytes:
                    raise RuntimeError("grouped host mirror bytes disagree with the resolved plan")
                pin_kinds = {mirror.pinned for mirror in mirrors.values()}
                if len(pin_kinds) != 1:
                    raise RuntimeError("grouped host mirrors must use one host memory tier")
                mirror_pinned = next(iter(pin_kinds))
                if (
                    host_policy is not None
                    and mirror_decision is not None
                    and mirror_decision.fallback
                ):
                    runtime_event(
                        "host_mirror",
                        owner_id="grouped_tier",
                        status="policy_fallback",
                        detail=(
                            f"requested={MemoryTier.HOST_PINNED.name} "
                            f"actual={MemoryTier.HOST_PAGEABLE.name}: {mirror_decision.reason}"
                        ),
                    )
                credits = TransferCredits(cache_config.tiering.max_inflight_bytes or mirror_bytes)
                for name, mirror in mirrors.items():
                    label = f"serving.kv.tier.{name}"
                    ledger.admit(
                        Reservation.backed(
                            MemoryOwner.WORKSPACE,
                            label,
                            mirror.total_bytes,
                            tier=claim_tier(
                                MemoryOwner.WORKSPACE,
                                lane=lane,
                                pinned_host=mirror.pinned,
                                pageable_host=not mirror.pinned,
                            ),
                            device_index=device_index,
                        )
                    )
                    charged_labels.append(label)
                    backend.attach_tier(
                        name,
                        config=grouped_tiers[name],
                        host_storage=mirror,
                        transfer_engine=build_transfer_engine(
                            mirror,
                            prefer_async=cache_config.tiering.async_transfers,
                        ),
                        transfer_budget=credits,
                    )
            except BaseException:
                backend.shutdown_tier()
                backend.release_tier()
                for label in reversed(charged_labels):
                    ledger.release(label)
                raise

            def _release_grouped_tier() -> None:
                backend.shutdown_tier()
                backend.release_tier()
                for label in reversed(charged_labels):
                    ledger.release(label)

            tier_release = _release_grouped_tier
        generation = mint_generation(
            model_id=model_id,
            model_revision=model_revision,
            weights_revision=weights_revision,
            kv_storage=kv.fingerprint,
            backend=type(kv.backend).__name__,
        )
        capacity = build_capacity_snapshot(
            generation=generation,
            lane=lane,
            dtype=compute_dtype.label,
            kv_dtype=cache_config.kv_dtype.label,
            group_pages={group.group_id: group.num_pages for group in plan.physical.groups},
            group_page_sizes={group_id: spec.page_size for group_id, spec in plan.storage_specs},
            max_model_len=plan.max_model_len,
            max_num_seqs=scheduler.max_num_seqs,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
            max_inflight=scheduler.max_inflight,
            activation_bytes=profile.activation_bytes if profile else 0,
            workspace_ceiling_bytes=profile.kernel_workspace_bytes if profile else 0,
            graph_bytes=profile.graph_pool_bytes if profile else 0,
            staging_bytes=mirror_bytes,
            budget_bytes=plan.memory.policy_budget_bytes,
            kv_budget_bytes=plan.memory.kv_cache_bytes,
            weights_bytes=profile.weights_bytes if profile else 0,
            ledger=ledger,
            staging_tier=(MemoryTier.HOST_PINNED if mirror_pinned else MemoryTier.HOST_PAGEABLE),
        )
        kv.bind_capacity(capacity)
        resources = WorkerResources(ledger, kv, leases, capacity, tier_release=tier_release)
        rollback.pop_all()
        return plan, resources
