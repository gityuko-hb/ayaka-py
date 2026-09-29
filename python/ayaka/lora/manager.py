"""Authoritative bounded residency and request lifetime management."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from time import perf_counter
from typing import Protocol

from ayaka.lora.cache import DeviceAdapterPool
from ayaka.lora.config import LoRAConfig
from ayaka.lora.variant import AdapterIdentity
from ayaka.lora.weights import AdapterWeights
from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.types import MemoryOwner, MemoryTier


class AdapterCapacityError(MemoryError):
    """No unpinned, unleased cache victim can satisfy the requested residency."""


class ResidencyState(StrEnum):
    HOST_READY = "host_ready"
    DEVICE_LOADING = "device_loading"
    DEVICE_READY = "device_ready"
    ACTIVE = "active"
    LEASED = "leased"
    EVICTABLE = "evictable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AdapterBinding:
    identity: AdapterIdentity
    slot: int
    generation: int
    content_digest: str


@dataclass(frozen=True, slots=True)
class AdapterStatus:
    identity: AdapterIdentity
    state: ResidencyState
    host_resident: bool
    device_resident: bool
    pinned: bool
    leases: int
    binding: AdapterBinding | None
    active_in_batch: bool = False


class BatchCompletion(Protocol):
    def query(self) -> bool: ...


class AdapterLease:
    """Exactly-once request ownership, released only after executor retirement."""

    def __init__(self, owner: LoRAManager, binding: AdapterBinding) -> None:
        self.owner, self.binding, self.closed = owner, binding, False

    def validate(self) -> None:
        with self.owner.lock:
            if self.closed:
                raise ValueError("adapter lease is retired")
            self.owner.validate(self.binding)

    def close(self) -> None:
        with self.owner.lock:
            if not self.closed:
                self.validate()
                self.owner._leases[self.binding.slot] -= 1
                self.closed = True


class LoRAManager:
    """One lock, one slot table, independent bounded host and device LRUs.

    Device-resident adapters retain their host copy for transactional rollback.
    Host LRU may discard unpinned/unleased host-only registrations; callers must
    reload those sources explicitly. Digest tombstones prohibit revision reuse
    with different content for this owner's lifetime, even after unload.
    """

    def __init__(self, config: LoRAConfig, pool: DeviceAdapterPool) -> None:
        self.config, self.pool = config, pool
        self.lock = RLock()
        self._host: OrderedDict[AdapterIdentity, AdapterWeights] = OrderedDict()
        self._bindings: OrderedDict[AdapterIdentity, AdapterBinding] = OrderedDict()
        self._digests: dict[AdapterIdentity, str] = {}
        self._pins: set[AdapterIdentity] = set()
        self._failed: set[AdapterIdentity] = set()
        self._generations = [0] * (config.max_adapters + 1)
        self._leases = [0] * (config.max_adapters + 1)
        self._readers = 0
        self._executing: Counter[AdapterIdentity] = Counter()
        self._batch_events: list[tuple[tuple[AdapterIdentity, ...], BatchCompletion]] = []
        self.closed = False
        self._ledger: MemoryLedger | None = None
        self._backend_fallback: str | None = None
        self._counters = dict(
            host_cache_hits=0,
            host_cache_misses=0,
            device_cache_hits=0,
            device_cache_misses=0,
            evictions=0,
            load_seconds=0.0,
            h2d_seconds=0.0,
            fallback_count=0,
        )

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("LoRA owner is closed")

    @contextmanager
    def execution_guard(self, adapters: tuple[AdapterIdentity, ...] = ()) -> Iterator[None]:
        """Serialize enqueue versus slot mutation, without synchronizing requests."""
        with self.lock:
            self._require_open()
            self._retire_batch_events()
            self._executing.update(adapters)
            try:
                yield
            finally:
                self._executing.subtract(adapters)
                for identity in adapters:
                    if not self._executing[identity]:
                        del self._executing[identity]
                self.pool.record_read()
                if adapters and (event := self.pool.batch_completion()) is not None:
                    self._batch_events.append((adapters, event))

    def _retire_batch_events(self) -> None:
        self._batch_events = [
            (ids, event) for ids, event in self._batch_events if not event.query()
        ]

    def _active_in_batch(self, identity: AdapterIdentity) -> bool:
        return self._executing[identity] > 0 or any(
            identity in ids for ids, _ in self._batch_events
        )

    def record_batch(self, slots: list[int]) -> None:
        with self.lock:
            self._counters["distinct_adapters_per_batch"] = len(set(slots) - {0})

    def note_backend_fallback(self, reason: str) -> None:
        """Account an ``auto`` backend fallback once at bootstrap, with its reason."""
        with self.lock:
            self._counters["fallback_count"] += 1
            self._backend_fallback = reason

    def bind_host_ledger(self, ledger: MemoryLedger) -> None:
        """Attach the composition root's ledger; device is already charged by it."""
        with self.lock:
            if ledger is self._ledger:
                return
            cap = self.config.host_memory_bytes
            if ledger.get("lora.host") is None:
                ledger.admit(
                    Reservation(
                        MemoryOwner.WEIGHT,
                        "lora.host",
                        cap,
                        sum(w.nbytes for w in self._host.values()),
                        tier=MemoryTier.HOST_PAGEABLE,
                        charged_bytes=cap,
                    )
                )
            if self._ledger is not None:
                self._ledger.release("lora.host")
            self._ledger = ledger
            self._account_host()

    def _account_host(self) -> None:
        if self._ledger is not None and self._ledger.get("lora.host") is not None:
            cap = self.config.host_memory_bytes
            self._ledger.update(
                "lora.host",
                reserved_bytes=cap,
                charged_bytes=cap,
                backed_bytes=sum(w.nbytes for w in self._host.values()),
            )

    def ensure_host_resident(
        self, weights: AdapterWeights, *, replacing: AdapterIdentity | None = None
    ) -> None:
        with self.lock:
            self._require_open()
            identity = weights.identity
            known = self._digests.get(identity)
            if known is not None and known != weights.content_digest:
                raise ValueError(
                    "adapter revision collision: immutable identity has different content"
                )
            if identity in self._host:
                self._host.move_to_end(identity)
                self._counters["host_cache_hits"] += 1
                return
            self._counters["host_cache_misses"] += 1
            # Decide every victim before mutation. Invalid/exhausted load leaves cache intact.
            size, count = sum(w.nbytes for w in self._host.values()), len(self._host)
            victims = []
            for key, old in self._host.items():
                if (
                    count < self.config.max_host_adapters
                    and size + weights.nbytes <= self.config.host_memory_bytes
                ):
                    break
                if key not in self._pins and (key not in self._bindings or key == replacing):
                    victims.append(key)
                    size -= old.nbytes
                    count -= 1
            if (
                count >= self.config.max_host_adapters
                or size + weights.nbytes > self.config.host_memory_bytes
            ):
                raise AdapterCapacityError(
                    "host adapter cache exhausted by resident/pinned adapters"
                )
            for victim in victims:
                del self._host[victim]
                self._counters["evictions"] += 1
            self._host[identity] = weights
            self._digests[identity] = weights.content_digest
            self._failed.discard(identity)
            self._account_host()

    def _device_slot(self) -> tuple[int, AdapterIdentity | None]:
        used = {b.slot for b in self._bindings.values()} | self.pool.quarantined
        for slot in range(1, self.config.max_adapters + 1):
            if slot not in used:
                return slot, None
        for key, binding in self._bindings.items():
            if key not in self._pins and not self._leases[binding.slot]:
                return binding.slot, key
        raise AdapterCapacityError(
            "adapter capacity exhausted: all device slots leased/pinned/quarantined"
        )

    def ensure_device_resident(
        self, identity: AdapterIdentity, *, rollback_weights: AdapterWeights | None = None
    ) -> AdapterBinding:
        with self.lock:
            self._require_open()
            if identity in self._bindings:
                self._counters["device_cache_hits"] += 1
                self._bindings.move_to_end(identity)
                self._host.move_to_end(identity)
                return self._bindings[identity]
            if identity not in self._host:
                raise ValueError("adapter revision is not host resident; load it first")
            self._counters["device_cache_misses"] += 1
            slot, victim = self._device_slot()
            started = perf_counter()
            try:
                self.pool.copy(slot, self._host[identity])
            except BaseException:
                self._failed.add(identity)
                try:
                    if victim is not None:
                        previous = self._host.get(victim, rollback_weights)
                        assert previous is not None
                        self.pool.copy(slot, previous)
                except BaseException:
                    assert victim is not None
                    self.pool.quarantined.add(slot)
                    self._bindings.pop(victim, None)
                    self._failed.add(victim)
                if victim is None:
                    self.pool.quarantined.add(slot)
                raise
            self._counters["h2d_seconds"] += perf_counter() - started
            if victim is not None:
                del self._bindings[victim]
                self._counters["evictions"] += 1
            self._generations[slot] += 1
            binding = AdapterBinding(
                identity, slot, self._generations[slot], self._host[identity].content_digest
            )
            self._bindings[identity] = binding
            self._host.move_to_end(identity)
            self._failed.discard(identity)
            return binding

    def load(self, weights: AdapterWeights, *, device: bool = True) -> AdapterBinding | None:
        with self.lock:
            started = perf_counter()
            victim = None
            # Capacity refusal must not discard a valid host entry unnecessarily.
            if device and weights.identity not in self._bindings:
                _, victim = self._device_slot()
            old_host = self._host.copy()
            self.ensure_host_resident(weights, replacing=victim)
            try:
                result = (
                    self.ensure_device_resident(
                        weights.identity, rollback_weights=old_host.get(victim) if victim else None
                    )
                    if device
                    else None
                )
            except BaseException:
                self._host = old_host
                self._account_host()
                raise
            self._counters["load_seconds"] += perf_counter() - started
            return result

    def validate(self, binding: AdapterBinding) -> None:
        with self.lock:
            if self.closed or self._bindings.get(binding.identity) != binding:
                raise ValueError("stale adapter slot/revision")

    def acquire(self, identity: AdapterIdentity) -> AdapterLease:
        with self.lock:
            binding = self.ensure_device_resident(identity)
            self._leases[binding.slot] += 1
            return AdapterLease(self, binding)

    def pin(self, identity: AdapterIdentity) -> None:
        with self.lock:
            self.ensure_device_resident(identity)
            self._pins.add(identity)

    def unpin(self, identity: AdapterIdentity) -> None:
        with self.lock:
            self._require_open()
            if identity not in self._host:
                raise KeyError(identity)
            self._pins.discard(identity)

    def evict(self, identity: AdapterIdentity) -> None:
        """Drop device residency only; retain a reloadable canonical host copy."""
        with self.lock:
            self._require_open()
            binding = self._bindings.get(identity)
            if identity in self._pins or (binding is not None and self._leases[binding.slot]):
                raise RuntimeError("adapter is leased or pinned; drain/unpin before eviction")
            if binding is not None:
                del self._bindings[identity]
                self._counters["evictions"] += 1

    def unload(self, identity: AdapterIdentity) -> None:
        with self.lock:
            if identity not in self._host:
                raise KeyError(identity)
            self.evict(identity)
            del self._host[identity]
            self._failed.discard(identity)
            self._account_host()

    def prefix_identity(self, identity: AdapterIdentity) -> str:
        with self.lock:
            self._require_open()
            if identity not in self._host:
                raise ValueError("adapter revision is not loaded")
            return self._host[identity].content_digest

    def retain_request(self) -> None:
        with self.lock:
            self._require_open()
            self._readers += 1

    def release_request(self) -> None:
        with self.lock:
            if self._readers <= 0:
                raise RuntimeError("unbalanced LoRA request lifetime")
            self._readers -= 1

    def get_adapter_status(self, identity: AdapterIdentity) -> AdapterStatus:
        with self.lock:
            self._retire_batch_events()
            if identity not in self._host and identity not in self._failed:
                raise KeyError(identity)
            binding = self._bindings.get(identity)
            leases = 0 if binding is None else self._leases[binding.slot]
            state = ResidencyState.HOST_READY
            if identity in self._failed:
                state = ResidencyState.FAILED
            elif self._active_in_batch(identity):
                state = ResidencyState.ACTIVE
            elif leases:
                state = ResidencyState.LEASED
            elif binding is not None:
                state = (
                    ResidencyState.DEVICE_READY
                    if identity in self._pins
                    else ResidencyState.EVICTABLE
                )
            return AdapterStatus(
                identity,
                state,
                identity in self._host,
                binding is not None,
                identity in self._pins,
                leases,
                binding,
                self._active_in_batch(identity),
            )

    def list_adapters(self) -> tuple[AdapterStatus, ...]:
        with self.lock:
            return tuple(self.get_adapter_status(key) for key in self._host)

    def stats(self) -> dict[str, int | float | str | None]:
        with self.lock:
            return {
                **{f"lora_{key}": value for key, value in self._counters.items()},
                "lora_host_cache_bytes": sum(w.nbytes for w in self._host.values()),
                "lora_device_cache_bytes": self.pool.bytes,
                "lora_active_leases": sum(self._leases),
                "lora_active_slots": len(self._bindings),
                "lora_host_adapters": len(self._host),
                "lora_backend": "torch_reference",
                "lora_backend_fallback_reason": self._backend_fallback,
                "lora_kernel_seconds": None,  # measured only by explicit event profiling
                "lora_distinct_adapters_per_batch": self._counters.get(
                    "distinct_adapters_per_batch", 0
                ),
            }

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            if self._readers or any(self._leases):
                raise RuntimeError("cannot close leased adapters")
            self.pool.wait_reads()
            self.closed = True
            self._host.clear()
            self._bindings.clear()
            self._pins.clear()
            self.pool.weights.clear()
            self._batch_events.clear()
            self._executing.clear()
            if self._ledger is not None:
                self._ledger.release("lora.host")
                self._ledger = None
