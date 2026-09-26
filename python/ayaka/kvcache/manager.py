"""Logical KV facade binding existing managers to physical storage owners.

The homogeneous implementation remains importable from ayaka.memory.manager.
Schedulers use logical sequence handles; allocation lifetime stays with leases.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ayaka.exceptions import InvalidHandleError, InvalidStateTransitionError
from ayaka.handles import KVPageHandle, KVReservationHandle, SequenceHandle, StepMemoryLeaseHandle
from ayaka.kvcache.grouped_manager import KVCacheGroupManager
from ayaka.kvcache.materialize import KVStorageLease, KVStoragePin
from ayaka.kvcache.retention.range import retained_page_range
from ayaka.memory.capacity import CapacitySnapshot, ResourceGeneration
from ayaka.memory.ledger import MemoryLedger
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.pressure import (
    MemoryPressureResult,
    SequencePreemptionResult,
    SequenceTruncationResult,
)
from ayaka.memory.sequence import GroupedSequenceSnapshot, SequenceMemorySnapshot
from ayaka.memory.state import ReleaseStatus, ReservationFailure
from ayaka.memory.views import (
    ExecutionMemoryView,
    GroupedExecutionMemoryView,
    GroupedLeakReport,
    GroupedMemorySnapshot,
    LeakReport,
    MemorySnapshot,
)
from ayaka.prefix.global_index import PrefixIndexPublisher
from ayaka.prefix.service import PrefixService
from ayaka.sched.plan import BatchStepPlan

MemoryView = ExecutionMemoryView | GroupedExecutionMemoryView
SequenceSnapshotView = SequenceMemorySnapshot | GroupedSequenceSnapshot
MemorySnapshotView = MemorySnapshot | GroupedMemorySnapshot
LeakReportView = LeakReport | GroupedLeakReport


class KVCapacityError(RuntimeError):
    """A structured reservation failure, preserved across boundary."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        reason: ReservationFailure | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.reason = reason


class LogicalKVManager:
    """Bind a logical manager to charged slabs for its entire live lifetime.

    Engine-thread serialized. Existing sequence, prefix, and retention APIs stay
    on backend. Close only after all sequences, cache entries, and leases are
    gone; callers then close the storage leases explicitly.
    """

    def __init__(
        self,
        backend: RuntimeMemoryManager | KVCacheGroupManager,
        storages: Mapping[str, KVStorageLease],
        *,
        index_publisher: PrefixIndexPublisher | None = None,
    ) -> None:
        self.backend = backend
        self.storages = MappingProxyType(dict(storages))
        self._pins: list[KVStoragePin] = []
        self.closed = False
        self.capacity: CapacitySnapshot | None = None
        self._prefix_service: PrefixService | None = None
        self._index_publisher = index_publisher
        self._validate_storage_bindings()
        ledgers = {id(lease.ledger) for lease in self.storages.values()}
        if len(ledgers) != 1:
            raise ValueError("all KV groups must share one ledger")
        self.ledger: MemoryLedger = next(iter(self.storages.values())).ledger
        try:
            for lease in self.storages.values():
                self._pins.append(lease.pin())
        except BaseException:
            for pin in reversed(self._pins):
                pin.close()
            raise

    @property
    def prefix_service(self) -> PrefixService:
        """Return the single canonical resident prefix policy for this KV owner."""
        if self._prefix_service is None:
            self._prefix_service = PrefixService(self, index_publisher=self._index_publisher)
        return self._prefix_service

    @property
    def group_names(self) -> tuple[str, ...]:
        if isinstance(self.backend, KVCacheGroupManager):
            return tuple(group.name for group in self.backend.cache_groups)
        return ("default",)

    @property
    def fingerprint(self) -> tuple[str, ...]:
        """Immutable identity of the bound physical slabs.

        One entry per group naming the group, its ledger claim, the full storage
        compatibility key (family, local head/latent geometry, page size, dtype,
        layout, quantization), the layer count and the materialized capacity. A
        capacity snapshot must carry exactly this fingerprint before it can be
        frozen onto this manager: a geometry change that keeps page size and
        page count identical (dtype, layout, local heads, quantization) still
        changes the identity.
        """
        entries: list[str] = []
        for name, lease in sorted(self.storages.items()):
            spec = lease.storage.spec
            key = "|".join(str(part) for part in spec.compatibility_key)
            entries.append(f"{name}:{lease.label}:{key}:{spec.num_layers}:{spec.capacity_pages}")
        return tuple(entries)

    @property
    def generation(self) -> ResourceGeneration | None:
        """Frozen resource generation, or None while capacity is unbound."""
        return None if self.capacity is None else self.capacity.generation

    @property
    def padding_page(self) -> int:
        """Reserved padding page dummy decode lanes write to (homogeneous KV).

        Grouped backends keep one padding page per group; a single graph slot
        has no meaning there, so callers get a loud error instead of a guess.
        """
        self._require_open()
        if not isinstance(self.backend, RuntimeMemoryManager):
            raise RuntimeError("grouped KV has no single padding page")
        return self.backend.padding_physical_page

    @property
    def padding_slot(self) -> int:
        """Flat slot address of the reserved padding page's first position."""
        self._require_open()
        if not isinstance(self.backend, RuntimeMemoryManager):
            raise RuntimeError("grouped KV has no single padding slot")
        return self.backend.padding_slot

    def bind_capacity(self, snapshot: CapacitySnapshot) -> None:
        """Freeze one capacity snapshot for this slab set, exactly once."""
        self._require_open()
        if self.capacity is not None:
            raise RuntimeError("capacity is already bound to this logical KV manager")
        if not isinstance(snapshot, CapacitySnapshot):
            raise TypeError("snapshot must be a CapacitySnapshot")
        if snapshot.generation.kv_storage != self.fingerprint:
            raise ValueError("capacity snapshot does not describe this manager's KV storage")
        self.capacity = snapshot

    # ------------------------------------------------------------------
    # public sequence/step boundary
    #
    # The runtime adapters (sequence allocator, request preparer, execution
    # resources, step runtime) own policy and call these verbs. They must not
    # reach into ``backend`` or its allocator: physical page state, transactions
    # and deferred reclaim stay behind this facade.
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("logical KV manager is closed")

    def create_sequence(self, request_id: str) -> SequenceHandle:
        """Allocate a generation-safe sequence identity for admission."""
        self._require_open()
        return self.backend.create_sequence(request_id)

    def release_sequence(
        self, sequence: SequenceHandle, *, safe_epoch: int | None = None
    ) -> ReleaseStatus:
        """Drop request ownership now or behind the sequence's active step."""
        self._require_open()
        return self.backend.release_sequence(sequence, safe_epoch=safe_epoch)

    def get_sequence(self, sequence: SequenceHandle) -> SequenceSnapshotView:
        """Return the immutable logical state used for staleness validation."""
        self._require_open()
        return self.backend.get_sequence(sequence)

    def preempt_sequence(
        self, sequence: SequenceHandle, *, safe_epoch: int | None = None
    ) -> SequencePreemptionResult:
        """Release request-owned KV while preserving sequence identity."""
        self._require_open()
        return self.backend.preempt_sequence(sequence, safe_epoch=safe_epoch)

    def truncate_sequence(
        self,
        sequence: SequenceHandle,
        num_tokens: int,
        *,
        safe_epoch: int | None = None,
    ) -> SequenceTruncationResult:
        """Release a sequence's committed suffix above ``num_tokens``.

        Suffix-only on the homogeneous and full-retention grouped paths: pages
        still shared with a fork sibling, prefix cache, or pin stay allocated
        under their remaining owners while this sequence's request reference
        is dropped. The sequence version bumps, so frozen plans and execution
        views become stale and are rejected by :meth:`validate_view`.
        """
        self._require_open()
        return self.backend.truncate_sequence(sequence, num_tokens, safe_epoch=safe_epoch)

    @property
    def current_epoch(self) -> int:
        """Epoch of the newest completed step, common across groups."""
        return self.backend.current_epoch

    def advance_epoch(self, completed_epoch: int) -> int:
        """Publish a completed-step watermark; never regresses."""
        self._require_open()
        return self.backend.advance_epoch(completed_epoch)

    def mark_step_in_flight(self, lease: StepMemoryLeaseHandle) -> None:
        """Acquire transient page ownership before the executor enqueues work."""
        self._require_open()
        self.backend.mark_step_in_flight(lease)

    def commit_step(
        self,
        lease: StepMemoryLeaseHandle,
        *,
        written_tokens: Mapping[KVReservationHandle, int] | None = None,
    ) -> None:
        """Publish committed KV only under executor-authored completion proof."""
        self._require_open()
        self.backend.commit_step(lease, written_tokens=written_tokens)

    def retire_step(self, lease: StepMemoryLeaseHandle) -> None:
        """Drop last-use execution ownership after logical settlement."""
        self._require_open()
        self.backend.retire_step(lease)

    def abort_prepared_step(self, lease: StepMemoryLeaseHandle) -> None:
        """Roll back a not-yet-launched step; the caller must still own it."""
        self._require_open()
        self.backend.abort_prepared_step(lease)

    def fail_in_flight_step(self, lease: StepMemoryLeaseHandle, *, safe_epoch: int) -> None:
        """Abandon partially written KV into deferred reclaim until ``safe_epoch``."""
        self._require_open()
        self.backend.fail_in_flight_step(lease, safe_epoch=safe_epoch)

    def reclaim_deferred(self) -> MemoryPressureResult:
        """Reclaim pages whose deferred epoch has completed."""
        self._require_open()
        return self.backend.reclaim_deferred()

    def shutdown_transfers(self) -> None:
        """Drain every tier transfer and release quarantined destinations.

        Shutdown-only: the copy stream is synchronized, so a failed transfer's
        partially written destination can finally be reclaimed. Tiering-off
        backends are a no-op. Must run before slab teardown; after it, the
        tier leak report must be clean.
        """
        self._require_open()
        backend = self.backend
        if isinstance(backend, (RuntimeMemoryManager, KVCacheGroupManager)):
            backend.shutdown_tier()

    def evict_prefixes_for_pressure(self, required_pages: int) -> MemoryPressureResult:
        """Release cache-only pages under reservation pressure."""
        self._require_open()
        return self.backend.evict_prefixes_for_pressure(required_pages)

    def privatize_prefix_tail(self, sequence: SequenceHandle) -> bool:
        """Relinquish cache sharing on a sole-consumer partial tail.

        Grouped boundaries pin whole per-group page sets, so unsharing is not
        implemented there; the backend reports ``False`` and pressure handling
        reports an explicit capacity error instead of guessing.
        """
        self._require_open()
        return self.backend.privatize_prefix_tail(sequence)

    def clear_prefix_cache(self, *, safe_epoch: int | None = None) -> int:
        """Teardown-only: drop every resident prefix entry; policy stays in PrefixService."""
        self._require_open()
        return self.backend.clear_prefix_cache(safe_epoch=safe_epoch)

    def physical_page(self, group_name: str, page: KVPageHandle) -> int:
        """Resolve a generation-safe page handle to its kernel slot address."""
        self._require_open()
        if isinstance(self.backend, KVCacheGroupManager):
            return self.backend.physical_page(group_name, page)
        if group_name != "default":
            raise ValueError(f"homogeneous KV has no group {group_name!r}")
        return self.backend.allocator.physical_id(page).value

    def snapshot(self) -> MemorySnapshotView:
        """Capacity accounting for diagnostics and status reporting."""
        self._require_open()
        return self.backend.snapshot()

    def leak_report(self) -> LeakReportView:
        """Deterministic ownership report; ``clean`` gates storage teardown."""
        return self.backend.leak_report()

    def _validate_storage_bindings(self) -> None:
        if self.closed:
            raise RuntimeError("logical KV manager is closed")
        if set(self.storages) != set(self.group_names):
            raise ValueError("one physical storage lease is required per KV group")
        if len({id(lease.storage) for lease in self.storages.values()}) != len(self.storages):
            raise ValueError("independent group page namespaces cannot share a slab")
        if isinstance(self.backend, RuntimeMemoryManager):
            # Homogeneous tiering is certified from R12B: readiness states,
            # transfer ownership, ledger accounting and quarantine are all
            # enforced on the backend before any sequence binds pages.
            expected = {"default": self.backend.storage}
        else:
            expected = self.backend.storages
        for name, lease in self.storages.items():
            if lease.storage is not expected.get(name):
                raise ValueError(f"physical storage binding changed for {name}")
            if lease.ledger.get(lease.label) is None:
                raise ValueError(f"KV slab {name} has no ledger claim")

    def validate_step(self, step: BatchStepPlan) -> None:
        self._validate_storage_bindings()
        for value in step.inputs:
            state = self.backend.get_sequence(value.sequence)
            if (
                state.request_id != value.request_id
                or state.version != value.state_version
                or state.committed_tokens != value.computed_tokens
                or state.release_requested
                or state.busy
            ):
                raise InvalidStateTransitionError(
                    f"request snapshot disagrees with logical KV: request={value.request_id!r} "
                    f"snapshot(version={value.state_version}, computed={value.computed_tokens}) "
                    f"state(request={state.request_id!r}, version={state.version}, "
                    f"committed={state.committed_tokens}, release={state.release_requested}, "
                    f"busy={state.busy})"
                )
        if isinstance(self.backend, KVCacheGroupManager):
            for scheduled in step.slices:
                for group in self.backend.cache_groups:
                    retained = retained_page_range(
                        group.retention,
                        layer_id=group.layer_ids[0],
                        sequence_length=scheduled.query_end,
                        page_size=group.storage_spec.page_size,
                    )
                    if len(retained) > group.storage_spec.capacity_pages - 1:
                        raise KVCapacityError(
                            f"{scheduled.request_id}: retained KV exceeds "
                            f"group {group.name} capacity",
                            request_id=scheduled.request_id,
                            reason=ReservationFailure.REQUEST_TOO_LARGE,
                        )
        if step.kv_requirements:
            if {r.group_id for r in step.kv_requirements} != set(range(len(self.group_names))):
                raise ValueError("KV requirements must cover every group by stable ordinal")
            for requirement in step.kv_requirements:
                if requirement.append_tokens != step.num_tokens:
                    raise ValueError("KV append requirement disagrees with packed token count")
                if requirement.restore_bytes or requirement.growth_bytes:
                    raise ValueError("resident append/COW does not support restore or slab growth")

    def reserve(self, step: BatchStepPlan) -> MemoryView:
        """Atomically reserve all sequences/groups; prepare never commits progress."""
        self.validate_step(step)
        transaction = self.backend.begin_transaction(step.step_id)
        lease = None
        try:
            for value, scheduled in zip(step.inputs, step.slices, strict=True):
                result = self.backend.try_reserve(
                    transaction, value.sequence, scheduled.query_count
                )
                if not result.ok:
                    raise KVCapacityError(
                        f"{value.request_id}: {result.reason}",
                        request_id=value.request_id,
                        reason=result.reason,
                    )
            lease = self.backend.prepare_step(transaction)
            return self.backend.build_execution_view(lease)
        except BaseException:
            if lease is not None:
                self.backend.abort_prepared_step(lease)
            else:
                try:
                    self.backend.rollback_transaction(transaction)
                except InvalidHandleError:
                    # prepare_step already rolled back its rejected transaction.
                    pass
            raise

    def validate_view(self, view: MemoryView) -> None:
        self._validate_storage_bindings()
        if isinstance(self.backend, RuntimeMemoryManager):
            if not isinstance(view, ExecutionMemoryView):
                raise TypeError("homogeneous manager requires homogeneous metadata")
            self.backend.validate_execution_view(view)
        else:
            if not isinstance(view, GroupedExecutionMemoryView):
                raise TypeError("grouped manager requires grouped metadata")
            self.backend.validate_execution_view(view)

    def close(self) -> None:
        """Close the shared prefix service, then release slab pins when clean.

        The prefix service belongs to this manager: closing it drops cache
        ownership (after transfers retire) so the leak check can pass. A
        pending transfer refuses the close; retire it first. Callers then
        close the storage leases explicitly.

        With tiering, the drain runs first: a failed transfer keeps its
        destination owners until the copy stream is proven quiet, and the
        leak check can only pass once the settlement released them.
        """
        if self.closed:
            return
        if isinstance(self.backend, RuntimeMemoryManager):
            self.backend.shutdown_tier()
            service = self._prefix_service
            if service is not None and not service.closed and not service.close():
                raise RuntimeError("prefix service still has active transfers")
        else:
            # Drain grouped copies before releasing canonical device pins and
            # host slots. A failed copy keeps its destination quarantined
            # until the transfer engine proves quiescence.
            self.backend.shutdown_tier()
            self.backend.clear_prefix_cache()
            self.backend.reclaim_deferred()
        if not self.backend.leak_report().clean:
            raise RuntimeError("logical KV still owns sequences, cache pages, or execution leases")
        for pin in reversed(self._pins):
            pin.close()
        self._pins.clear()
        self.closed = True
