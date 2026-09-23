"""Worker-owned resident storage, accounting and immutable capacity.

Construction admits each backing allocation once. The same owner is used for
startup and cache rebuild; scheduler and HTTP objects are never stored here.
Borrowed models are accounted for but are not destroyed by this owner.
"""

from __future__ import annotations

from collections.abc import Callable
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
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.materialize import KVStorageLease, materialize_kv_storage
from ayaka.kvcache.resize import (
    LEDGER_OVERHEAD_BYTES,
    CacheRebuildRejected,
    ResizeRejectionReason,
)
from ayaka.kvcache.storage.geometry import MHAStorageSpec
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
from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.source import TorchDeviceSource, TorchHostByteSource
from ayaka.memory.tiering import HostKVStorage, TieringConfig, build_transfer_engine
from ayaka.memory.workspace import WorkspaceManager
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

    @property
    def lane(self) -> MemoryLane:
        return MemoryLane.CUDA if self.device.type == "cuda" else MemoryLane.CPU

    def budget(self) -> tuple[int, int, int]:
        """Return policy, KV and total budgets before materializing anything."""
        slab = self.storage_spec.aligned_total_bytes(256) + LEDGER_OVERHEAD_BYTES
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
        ledger = MemoryLedger.for_device(
            device_budget_bytes=budget,
            device_total_bytes=total,
            host_pinned_bytes=self.staging_bytes + mirror_bytes,
            device_index=index,
            host_pageable_bytes=budget if self.lane is MemoryLane.CPU else mirror_bytes,
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
            )
            rollback.callback(storage.close)
            tier_release: Callable[[], None] | None = None
            host_mirror: HostKVStorage | None = None
            credits: TransferCredits | None = None
            engine: object | None = None
            if tier is not None:
                host_mirror = HostKVStorage(
                    storage.storage,
                    capacity_pages=tier.host_capacity_pages,
                    pin_memory=self.lane is MemoryLane.CUDA,
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
                        pinned_host=self.lane is MemoryLane.CUDA,
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
                page_size=self.storage_spec.page_size,
                group_pages={"default": self.storage_spec.capacity_pages},
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
                staging_pinned=self.lane is MemoryLane.CUDA,
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
        for storage in reversed(self.storages):
            storage.close()
        if self.tier_release is not None:
            self.tier_release()
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
        _, kv = bind_resident_kv(plan, leases)
        rollback.callback(kv.close)
        generation = mint_generation(
            model_id=model_id,
            model_revision=model_revision,
            weights_revision=weights_revision,
            kv_storage=kv.fingerprint,
            backend=type(kv.backend).__name__,
        )
        capacity = build_capacity_snapshot(
            generation=generation,
            lane=MemoryLane.CPU if torch.device(device).type == "cpu" else MemoryLane.CUDA,
            dtype=compute_dtype.label,
            kv_dtype=cache_config.kv_dtype.label,
            page_size=plan.storage_specs[0][1].page_size,
            group_pages={group.group_id: group.num_pages for group in plan.physical.groups},
            max_model_len=plan.max_model_len,
            max_num_seqs=scheduler.max_num_seqs,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
            max_inflight=scheduler.max_inflight,
            activation_bytes=profile.activation_bytes if profile else 0,
            workspace_ceiling_bytes=profile.kernel_workspace_bytes if profile else 0,
            graph_bytes=profile.graph_pool_bytes if profile else 0,
            staging_bytes=0,
            budget_bytes=plan.memory.policy_budget_bytes,
            kv_budget_bytes=plan.memory.kv_cache_bytes,
            weights_bytes=profile.weights_bytes if profile else 0,
            ledger=ledger,
        )
        kv.bind_capacity(capacity)
        resources = WorkerResources(ledger, kv, leases, capacity)
        rollback.pop_all()
        return plan, resources
