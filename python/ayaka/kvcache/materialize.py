from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from threading import Lock

from ayaka.kvcache.build import build_kv_storage
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec
from ayaka.kvcache.storage.layout import DEFAULT_KV_ALIGNMENT_BYTES
from ayaka.kvcache.storage.ports import KVStorage
from ayaka.kvcache.storage.validation import validate_kv_storage_support
from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.types import MemoryOwner, MemoryTier
from ayaka.utils.torch_utils import compute_capability as torch_compute_capability

StorageFactory = Callable[[BaseKVStorageSpec, str, bool], KVStorage]


class KVStorageLease:
    """Own one materialized KV storage and its committed ledger claim.

    ``close`` is explicit and idempotent. The runtime must drain every stream
    that can touch the slab before calling it; Python finalizers cannot enforce
    CUDA completion ordering and are intentionally not used.
    """

    __slots__ = ("_closed", "_charged_bytes", "_ledger", "_lock", "_storage", "_pins", "label")

    def __init__(
        self,
        storage: KVStorage,
        ledger: MemoryLedger,
        label: str,
        *,
        charged_bytes: int,
    ) -> None:
        self._storage: KVStorage | None = storage
        self._ledger = ledger
        self.label = label
        self._charged_bytes = charged_bytes
        self._closed = False
        self._lock = Lock()
        self._pins = 0

    @property
    def storage(self) -> KVStorage:
        with self._lock:
            if self._storage is None:
                raise RuntimeError(f"KV storage lease {self.label!r} is closed")
            return self._storage

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def materialized_bytes(self) -> int:
        """Bytes the storage tensors actually hold, measured from the tensors."""
        return self.storage.materialized_bytes

    @property
    def charged_bytes(self) -> int:
        """Bytes charged to the ledger: payload plus alignment padding.

        Always greater than or equal to :attr:`materialized_bytes`. The gap is
        the padding the allocator holds but no tensor reports, and it is the
        reason the ledger is charged this number rather than the measured one.
        """
        return self._charged_bytes

    @property
    def ledger(self) -> MemoryLedger:
        """The single accounting authority for this physical slab."""
        return self._ledger

    @property
    def pin_count(self) -> int:
        with self._lock:
            return self._pins

    def pin(self) -> KVStoragePin:
        """Keep the slab alive while a logical manager owns any page or lease."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot pin closed KV storage")
            self._pins += 1
            return KVStoragePin(self)

    def close(self) -> None:
        """Close only after every owner released its pin; idempotent."""
        with self._lock:
            if self._closed:
                return
            if self._pins:
                raise RuntimeError("KV storage is pinned by a live logical manager")
            storage = self._storage
            assert storage is not None
            if self._ledger.get(self.label) is None:
                raise RuntimeError(f"ledger claim {self.label!r} disappeared before storage close")
            storage.close()
            self._ledger.release(self.label)
            self._storage = None
            self._closed = True

    def __enter__(self) -> KVStorageLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class KVStoragePin:
    """Explicit, idempotent lifetime hold. No finalizer guesses device completion."""

    def __init__(self, lease: KVStorageLease) -> None:
        self._lease = lease
        self._closed = False

    def close(self) -> None:
        with self._lease._lock:
            if self._closed:
                return
            self._lease._pins -= 1
            self._closed = True


def materialize_kv_storage(
    spec: BaseKVStorageSpec,
    *,
    ledger: MemoryLedger,
    label: str,
    device: str = "cuda",
    device_index: int | None = None,
    zero_initialize: bool = False,
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES,
    storage_factory: StorageFactory | None = None,
    validate_support: bool = True,
) -> KVStorageLease:
    """Reserve, allocate, materialize and publish one KV slab atomically.

    Allocation failure leaves no capacity charge or label claim. The committed
    claim is released only by the returned lease, so a production caller cannot
    materialize an unaccounted slab by forgetting a later bootstrap hook.

    Args:
        spec: Planned geometry. Must be the spec the capacity planner produced.
        ledger: Owner of the device capacity account.
        label: Unique ledger claim label.
        device: ``"cuda"`` or ``"cuda:N"``; CPU is supported for host diagnostics.
        device_index: Defaults to the ledger's device.
        zero_initialize: Fill the slab instead of leaving it uninitialized.
        alignment_bytes: Per-allocation alignment used for the charge.
        storage_factory: Injection point for tests and diagnostics.
        validate_support: Check dtype / device / layout / capability before
            allocating. Turn off only when the caller has already validated.

    Raises:
        KVStorageCompatibilityError: when the spec cannot run on this device.
        ValueError: on a device or accounting disagreement.
    """
    if not isinstance(spec, BaseKVStorageSpec):
        raise TypeError(f"spec must be a BaseKVStorageSpec, got {type(spec).__name__}")
    if not label.strip():
        raise ValueError("KV storage needs a non-empty ledger label")
    if alignment_bytes <= 0:
        raise ValueError("alignment_bytes must be positive")
    resolved_index = ledger.device_index if device_index is None else device_index
    if resolved_index != ledger.device_index:
        raise ValueError(
            f"KV storage device {resolved_index} disagrees with ledger device {ledger.device_index}"
        )
    resolved_device = _resolve_device(
        device,
        device_index=resolved_index,
        custom_factory=storage_factory is not None,
    )

    if validate_support:
        # Before the reservation, so a rejected configuration leaves the ledger
        # untouched rather than relying on the rollback path.
        device_type = resolved_device.partition(":")[0]
        validate_kv_storage_support(
            spec,
            device_type=device_type,
            compute_capability=_compute_capability(device_type, resolved_index),
        ).require_compatible()

    charged_bytes = spec.aligned_total_bytes(alignment_bytes)
    reservation = Reservation(
        owner=MemoryOwner.KV,
        label=label,
        reserved_bytes=charged_bytes,
        backed_bytes=charged_bytes,
        charged_bytes=charged_bytes,
        tier=(MemoryTier.HOST_PAGEABLE if resolved_device == "cpu" else MemoryTier.DEVICE),
        device_index=resolved_index,
    )
    ticket = ledger.reserve(reservation)
    storage: KVStorage | None = None
    try:
        if storage_factory is not None:
            storage = storage_factory(spec, resolved_device, zero_initialize)
            if not isinstance(storage, KVStorage):
                raise TypeError(
                    "storage factory returned an object that does not implement KVStorage"
                )
        else:
            storage = build_kv_storage(
                spec,
                device=resolved_device,
                zero_initialize=zero_initialize,
            )
        if storage.spec != spec:
            raise ValueError("materialized KV storage does not match the planned specification")

        measured = storage.materialized_bytes
        if measured > charged_bytes:
            # The reservation is derived from the same spec, so this can only
            # mean the plan and the allocation disagree about geometry. Failing
            # here keeps the ledger's per-owner total truthful; letting it pass
            # would leave the excess permanently uncharged.
            raise ValueError(
                f"KV storage materialized {measured} bytes but only {charged_bytes} were "
                "reserved; the capacity plan and the storage layout disagree"
            )
        # Charge the reservation, not the measurement. The difference is the
        # alignment padding, which is held whether or not a tensor reports it,
        # and the caching allocator rounds every block up on top of that. A
        # ledger may over-account; it may never under-account.
        ledger.materialize(ticket, actual_bytes=charged_bytes)
        ledger.commit(ticket)
    except BaseException:
        if storage is not None:
            with suppress(Exception):
                storage.close()
        with suppress(KeyError, ValueError):
            ledger.rollback(ticket)
        raise
    return KVStorageLease(storage, ledger, label, charged_bytes=charged_bytes)


def _compute_capability(device_type: str, device_index: int) -> tuple[int, int] | None:
    """Query the CUDA compute capability, or return None when unknowable.

    Returning None rather than raising keeps a CPU-only host and a torch-free
    planner on the same code path: validation treats an unknown capability as a
    reason to refuse FP8, which is the safe direction.
    """
    if device_type != "cuda":
        return None
    return torch_compute_capability(device_index)


def _resolve_device(device: str, *, device_index: int, custom_factory: bool) -> str:
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return f"cuda:{device_index}"
    if normalized.startswith("cuda:"):
        try:
            explicit_index = int(normalized.partition(":")[2])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device {device!r}") from exc
        if explicit_index != device_index:
            raise ValueError(
                f"KV storage device {explicit_index} disagrees with ledger device {device_index}"
            )
        return normalized
    if not custom_factory:
        raise ValueError("device KV storage must materialize on CUDA")
    if not normalized:
        raise ValueError("device must be non-empty")
    return normalized
