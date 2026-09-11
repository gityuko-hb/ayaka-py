"""Logical KV facade binding existing managers to physical storage owners.

The homogeneous implementation remains importable from ayaka.memory.manager.
Schedulers use logical sequence handles; allocation lifetime stays with leases.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ayaka.exceptions import InvalidHandleError, InvalidStateTransitionError
from ayaka.kvcache.grouped_manager import KVCacheGroupManager
from ayaka.kvcache.materialize import KVStorageLease, KVStoragePin
from ayaka.memory.ledger import MemoryLedger
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.views import ExecutionMemoryView, GroupedExecutionMemoryView
from ayaka.sched.plan import BatchStepPlan

MemoryView = ExecutionMemoryView | GroupedExecutionMemoryView


class KVCapacityError(RuntimeError):
    """A sequence or group could not satisfy an atomic append reservation."""


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
    ) -> None:
        self.backend = backend
        self.storages = MappingProxyType(dict(storages))
        self._pins: list[KVStoragePin] = []
        self.closed = False
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
    def group_names(self) -> tuple[str, ...]:
        if isinstance(self.backend, KVCacheGroupManager):
            return tuple(group.name for group in self.backend.cache_groups)
        return ("default",)

    def _validate_storage_bindings(self) -> None:
        if self.closed:
            raise RuntimeError("logical KV manager is closed")
        if set(self.storages) != set(self.group_names):
            raise ValueError("one physical storage lease is required per KV group")
        if len({id(lease.storage) for lease in self.storages.values()}) != len(self.storages):
            raise ValueError("independent group page namespaces cannot share a slab")
        if isinstance(self.backend, RuntimeMemoryManager):
            if self.backend.tiering_enabled:
                raise ValueError("P2 requires resident KV; tiering readiness is unsupported")
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
                raise InvalidStateTransitionError("request snapshot disagrees with logical KV")
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
                    raise KVCapacityError(f"{value.request_id}: {result.reason}")
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
        """Release slab pins only when the logical manager has no remaining owners."""
        if self.closed:
            return
        if not self.backend.leak_report().clean:
            raise RuntimeError("logical KV still owns sequences, cache pages, or execution leases")
        for pin in reversed(self._pins):
            pin.close()
        self._pins.clear()
        self.closed = True
