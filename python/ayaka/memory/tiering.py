from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from ayaka.exceptions import (
    InvalidHandleError,
    InvalidStateTransitionError,
    InvariantViolationError,
    StorageUnavailableError,
)
from ayaka.handles import KVPageHandle, PrefixHandle
from ayaka.memory.allocator import PageAllocator
from ayaka.memory.state import PageAllocationState
from ayaka.prefix.identity import PrefixBlockIdentity, PrefixCacheContext
from ayaka.prefix.interface import CachedBlockInfo
from ayaka.utils.torch_utils import require_torch


class TierState(Enum):
    """Authoritative placement of one cached block.

    Exactly one of these holds for a tracked block at any instant; the two
    transient states are the only ones that touch both tiers, and each owns
    its destination resource until the transfer resolves.
    """

    DEVICE = auto()
    """Bytes live in a GPU page owned by the prefix cache."""
    EVICTING = auto()
    """A device-to-host copy is in flight; the host slot is already reserved."""
    HOST = auto()
    """Bytes live only in a host slot; the GPU page has been released."""
    PROMOTING = auto()
    """A host-to-device copy is in flight into a RESERVED GPU page."""

class TransferDirection(Enum):
    """Direction of one page-sized tier transfer."""

    DEVICE_TO_HOST = auto()
    HOST_TO_DEVICE = auto()


class TransferState(Enum):
    """Lifecycle of one submitted transfer."""

    PENDING = auto()
    """Submitted; the copy may still be executing."""
    COMPLETED = auto()
    """Terminal: the bytes are readable at the destination."""
    FAILED = auto()
    """Terminal: the copy raised; the caller must abort its state machine."""


@dataclass(frozen=True, slots=True)
class TransferTicket:
    """Opaque identity of one submitted transfer.

    ``issue_epoch`` is the completion watermark at submission time; the tier
    never releases a page for reuse before it.
    """

    ticket_id: int
    direction: TransferDirection
    device_page: int
    """Physical page index on the device side of the copy."""
    host_slot: int
    issue_epoch: int

@dataclass(frozen=True, slots=True)
class TransferOutcome:
    """Result of one resolved transfer."""

    ticket: TransferTicket
    state: TransferState
    error: str | None = None
    """Exception text when ``state`` is ``FAILED``; None otherwise."""

    def __post_init__(self) -> None:
        if self.state is TransferState.PENDING:
            raise ValueError("a transfer outcome cannot still be pending")
        if (self.error is None) == (self.state is TransferState.FAILED):
            raise ValueError("a failed outcome must carry an error and a success must not")

@dataclass(frozen=True, slots=True)
class TransferMetrics:
    """Cumulative transfer accounting; diagnostics only."""

    submitted_total: int
    completed_total: int
    failed_total: int
    bytes_to_host_total: int
    bytes_to_device_total: int
    pending: int


@runtime_checkable
class TransferEngine(Protocol):
    """Moves whole pages between the device KV storage and a host mirror.

    Implementations are free to be synchronous; the tier only requires that
    :meth:`poll` never reports a transfer complete before its bytes are
    actually readable at the destination.
    """

    @property
    def bytes_per_page(self) -> int: ...

    def submit(
        self,
        direction: TransferDirection,
        *,
        device_page: int,
        host_slot: int,
        issue_epoch: int,
    ) -> TransferTicket: ...

    def poll(self) -> tuple[TransferOutcome, ...]:
        """Return every transfer that resolved since the last call."""

        ...

    def drain(self) -> tuple[TransferOutcome, ...]:
        """Block until no transfer is pending, then return the outcomes."""

        ...

    @property
    def metrics(self) -> TransferMetrics: ...

class HostKVStorage:
    """Page-for-page host mirror of a device KV storage (A11-01).

    Built from the device storage's own ``buffers()`` families, so it works
    for both MHA (key/value) and MLA (latent/rope) geometries without knowing
    which it mirrors. Host tensors are page-major with the same trailing shape
    and dtype as their device counterparts, so a spill is one contiguous
    ``copy_`` per family per layer.

    Pinned memory is used whenever CUDA is available, because only pinned
    staging buffers give asynchronous, overlap-capable copies; on a CPU-only
    host the mirror degrades to ordinary CPU tensors and stays functional.
    """

    def __init__(
        self,
        device_storage: Any,
        *,
        capacity_pages: int,
        pin_memory: bool | None = None,
    ) -> None:
        if capacity_pages <= 0:
            raise ValueError("host capacity_pages must be positive")
        torch = require_torch()
        first_family, second_family = device_storage.buffers()
        if not first_family or len(first_family) != len(second_family):
            raise ValueError("device storage must expose two equally sized buffer families")

        self._torch = torch
        self._device_storage = device_storage
        self._capacity_pages = int(capacity_pages)
        self._page_size = int(device_storage.page_size)
        # One mirror is one tier. Falling back per buffer would create a mixed
        # allocation that can be neither accounted nor copied correctly, so a
        # failed pinned attempt is discarded in full before pageable retry.
        requested_pinned = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        try:
            first, second = self._allocate_all(
                first_family,
                second_family,
                capacity_pages=capacity_pages,
                pin_memory=requested_pinned,
            )
            pinned = requested_pinned
        except RuntimeError as exc:  # pragma: no cover - real failure is host-dependent
            if not requested_pinned:
                raise StorageUnavailableError("host KV mirror allocation failed") from exc
            try:
                first, second = self._allocate_all(
                    first_family,
                    second_family,
                    capacity_pages=capacity_pages,
                    pin_memory=False,
                )
            except RuntimeError as pageable_exc:  # pragma: no cover - host-dependent
                raise StorageUnavailableError("host KV mirror allocation failed") from pageable_exc
            pinned = False
        self._pinned = pinned
        self._first_buffers = first
        self._second_buffers = second
        self._bytes_per_page = sum(
            buffer[0].numel() * buffer[0].element_size()
            for buffer in (*self._first_buffers, *self._second_buffers)
        )

    def _allocate_all(
        self,
        first_family: Sequence[Any],
        second_family: Sequence[Any],
        *,
        capacity_pages: int,
        pin_memory: bool,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Allocate both families atomically with respect to memory kind."""
        first: list[Any] = []
        second: list[Any] = []
        try:
            for buffer in first_family:
                first.append(self._mirror(buffer, capacity_pages, pin_memory=pin_memory))
            for buffer in second_family:
                second.append(self._mirror(buffer, capacity_pages, pin_memory=pin_memory))
        except RuntimeError:
            # Drop every successful allocation from this attempt before retry.
            first.clear()
            second.clear()
            raise
        return tuple(first), tuple(second)

    def _mirror(self, buffer: Any, capacity_pages: int, *, pin_memory: bool) -> Any:
        """Allocate one host twin; fallback belongs to the family transaction."""
        shape = (capacity_pages, *tuple(buffer.shape[1:]))
        return self._torch.empty(
            shape,
            dtype=buffer.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

    @property
    def capacity_pages(self) -> int:
        return self._capacity_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def pinned(self) -> bool:
        """Whether the mirror is page-locked and therefore DMA-capable."""
        return self._pinned

    @property
    def bytes_per_page(self) -> int:
        """Total bytes one page occupies across every layer and family."""
        return int(self._bytes_per_page)

    @property
    def total_bytes(self) -> int:
        return int(self._bytes_per_page) * self._capacity_pages

    def buffers(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Return the ``(first_family, second_family)`` host tensor families."""
        return self._first_buffers, self._second_buffers

    def copy_device_to_host(self, *, device_page: int, host_slot: int) -> None:
        """Copy one whole page out of every device buffer into the mirror."""
        self._validate(device_page=device_page, host_slot=host_slot)
        device_first, device_second = self._device_storage.buffers()
        for host, device in self._pairs(device_first, device_second):
            host[host_slot].copy_(device[device_page], non_blocking=self._pinned)

    def copy_host_to_device(self, *, host_slot: int, device_page: int) -> None:
        """Copy one whole page from the mirror back into every device buffer."""
        self._validate(device_page=device_page, host_slot=host_slot)
        device_first, device_second = self._device_storage.buffers()
        for host, device in self._pairs(device_first, device_second):
            device[device_page].copy_(host[host_slot], non_blocking=self._pinned)

    def synchronize(self) -> None:
        """Block until every copy issued from this mirror has actually landed.

        ``copy_device_to_host``/``copy_host_to_device`` issue ``non_blocking``
        copies whenever the mirror is pinned, and those are asynchronous with
        respect to the host. Any caller whose contract is "the bytes are
        readable when I return" -- notably
        :class:`SynchronousTransferEngine` -- must wait here first, or the
        tier will recycle a page while its DMA is still in flight.
        """
        if self._pinned:
            self._torch.cuda.synchronize()

    def _pairs(
        self,
        device_first: Sequence[Any],
        device_second: Sequence[Any],
    ) -> Iterable[tuple[Any, Any]]:
        """Zip host and device buffers across both families and every layer."""
        yield from zip(self._first_buffers, device_first, strict=True)
        yield from zip(self._second_buffers, device_second, strict=True)

    def _validate(self, *, device_page: int, host_slot: int) -> None:
        """Bounds-check both ends of a copy before touching any tensor."""
        if not 0 <= host_slot < self._capacity_pages:
            raise IndexError("host slot is outside the host mirror")
        if not 0 <= device_page < self._device_storage.capacity_pages:
            raise IndexError("device page is outside the device storage")


class SynchronousTransferEngine:
    """Reference engine: every copy completes before :meth:`submit` returns.

    This is the default whenever CUDA is unavailable, and the engine tests use
    to exercise the tier state machines deterministically. It is also the
    correctness oracle the asynchronous engine is compared against.
    """

    def __init__(self, copier: Any, *, bytes_per_page: int | None = None) -> None:
        self._copier = copier
        self._bytes_per_page = int(
            getattr(copier, "bytes_per_page", 0) if bytes_per_page is None else bytes_per_page
        )
        self._ids = count()
        self._resolved: list[TransferOutcome] = []
        self._submitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._bytes_to_host_total = 0
        self._bytes_to_device_total = 0
        self._lock = RLock()

    @property
    def bytes_per_page(self) -> int:
        return self._bytes_per_page

    def submit(
        self,
        direction: TransferDirection,
        *,
        device_page: int,
        host_slot: int,
        issue_epoch: int,
    ) -> TransferTicket:
        """Run one page copy immediately and queue its outcome for polling."""
        with self._lock:
            ticket = TransferTicket(
                ticket_id=next(self._ids),
                direction=direction,
                device_page=device_page,
                host_slot=host_slot,
                issue_epoch=issue_epoch,
            )
            self._submitted_total += 1
            try:
                self._run(ticket)
            except Exception as exc:
                self._failed_total += 1
                self._resolved.append(
                    TransferOutcome(ticket=ticket, state=TransferState.FAILED, error=str(exc))
                )
                return ticket
            self._completed_total += 1
            if direction is TransferDirection.DEVICE_TO_HOST:
                self._bytes_to_host_total += self._bytes_per_page
            else:
                self._bytes_to_device_total += self._bytes_per_page
            self._resolved.append(TransferOutcome(ticket=ticket, state=TransferState.COMPLETED))
            return ticket

    def _run(self, ticket: TransferTicket) -> None:
        """Dispatch one ticket to the copier, then wait for it to land.

        This engine's entire contract is that a copy is finished once
        ``submit`` returns, and the tier relies on it: a spill releases its
        source page as soon as the outcome is COMPLETED. A pinned host mirror
        issues asynchronous copies, so an un-drained copier would hand back a
        page the DMA is still reading. Copiers without a ``synchronize`` hook
        (test doubles, pageable mirrors) are already synchronous.
        """
        if ticket.direction is TransferDirection.DEVICE_TO_HOST:
            self._copier.copy_device_to_host(
                device_page=ticket.device_page,
                host_slot=ticket.host_slot,
            )
        else:
            self._copier.copy_host_to_device(
                host_slot=ticket.host_slot,
                device_page=ticket.device_page,
            )
        synchronize = getattr(self._copier, "synchronize", None)
        if synchronize is not None:
            synchronize()

    def poll(self) -> tuple[TransferOutcome, ...]:
        """Drain the outcomes accumulated since the last poll."""
        with self._lock:
            outcomes = tuple(self._resolved)
            self._resolved.clear()
            return outcomes

    def drain(self) -> tuple[TransferOutcome, ...]:
        """Nothing is ever pending here, so this is exactly :meth:`poll`."""
        return self.poll()

    @property
    def metrics(self) -> TransferMetrics:
        with self._lock:
            return TransferMetrics(
                submitted_total=self._submitted_total,
                completed_total=self._completed_total,
                failed_total=self._failed_total,
                bytes_to_host_total=self._bytes_to_host_total,
                bytes_to_device_total=self._bytes_to_device_total,
                # Nothing is ever in flight here: submit() does not return
                # until the copy has landed. Unpolled outcomes are finished
                # work, not pending work.
                pending=0,
            )


class CudaTransferEngine:
    """Asynchronous engine on a dedicated copy stream (A11-04).

    Every copy is issued on a side stream and stamped with a CUDA event, so
    :meth:`poll` can report completion without synchronizing the compute
    stream. The event is the ordering proof the tier needs: a promoted page is
    only published after its event has actually fired, so no consumer can read
    bytes the DMA has not written yet.
    """

    def __init__(self, copier: Any, *, bytes_per_page: int | None = None) -> None:
        torch = require_torch()
        if not torch.cuda.is_available():  # pragma: no cover - guarded by callers
            raise StorageUnavailableError("CudaTransferEngine requires an available CUDA device")
        self._torch = torch
        self._copier = copier
        self._bytes_per_page = int(
            getattr(copier, "bytes_per_page", 0) if bytes_per_page is None else bytes_per_page
        )
        self._stream = torch.cuda.Stream()
        self._ids = count()
        self._pending: dict[int, tuple[TransferTicket, Any]] = {}
        self._failed: list[TransferOutcome] = []
        self._submitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._bytes_to_host_total = 0
        self._bytes_to_device_total = 0
        self._lock = RLock()

    @property
    def bytes_per_page(self) -> int:
        return self._bytes_per_page

    @property
    def stream(self) -> Any:
        """The dedicated copy stream; exposed for diagnostics only."""
        return self._stream

    def submit(
        self,
        direction: TransferDirection,
        *,
        device_page: int,
        host_slot: int,
        issue_epoch: int,
    ) -> TransferTicket:
        """Issue one page copy on the side stream and stamp it with an event."""
        with self._lock:
            ticket = TransferTicket(
                ticket_id=next(self._ids),
                direction=direction,
                device_page=device_page,
                host_slot=host_slot,
                issue_epoch=issue_epoch,
            )
            self._submitted_total += 1
            try:
                # Wait for work already queued on the compute stream so the
                # copy never races a kernel that is still writing this page.
                self._stream.wait_stream(self._torch.cuda.current_stream())
                with self._torch.cuda.stream(self._stream):
                    if direction is TransferDirection.DEVICE_TO_HOST:
                        self._copier.copy_device_to_host(
                            device_page=device_page,
                            host_slot=host_slot,
                        )
                    else:
                        self._copier.copy_host_to_device(
                            host_slot=host_slot,
                            device_page=device_page,
                        )
                    event = self._torch.cuda.Event()
                    event.record(self._stream)
            except Exception as exc:
                self._failed_total += 1
                self._failed.append(
                    TransferOutcome(ticket=ticket, state=TransferState.FAILED, error=str(exc))
                )
                return ticket
            self._pending[ticket.ticket_id] = (ticket, event)
            return ticket

    def poll(self) -> tuple[TransferOutcome, ...]:
        """Report transfers whose event has fired, without any synchronization."""
        with self._lock:
            outcomes = list(self._failed)
            self._failed.clear()
            for ticket_id, (ticket, event) in list(self._pending.items()):
                if not event.query():
                    continue
                del self._pending[ticket_id]
                outcomes.append(self._complete(ticket))
            return tuple(outcomes)

    def drain(self) -> tuple[TransferOutcome, ...]:
        """Synchronize the copy stream, then report every resolved transfer."""
        with self._lock:
            if self._pending:
                self._stream.synchronize()
            return self.poll()

    def _complete(self, ticket: TransferTicket) -> TransferOutcome:
        """Record accounting for one finished transfer."""
        self._completed_total += 1
        if ticket.direction is TransferDirection.DEVICE_TO_HOST:
            self._bytes_to_host_total += self._bytes_per_page
        else:
            self._bytes_to_device_total += self._bytes_per_page
        return TransferOutcome(ticket=ticket, state=TransferState.COMPLETED)

    @property
    def metrics(self) -> TransferMetrics:
        with self._lock:
            return TransferMetrics(
                submitted_total=self._submitted_total,
                completed_total=self._completed_total,
                failed_total=self._failed_total,
                bytes_to_host_total=self._bytes_to_host_total,
                bytes_to_device_total=self._bytes_to_device_total,
                # Only genuinely in-flight copies; a failed transfer has
                # already resolved and is awaiting collection, not execution.
                pending=len(self._pending),
            )


def build_transfer_engine(
    copier: Any,
    *,
    prefer_async: bool = True,
) -> TransferEngine:
    """Pick the strongest engine the host actually supports.

    Falls back to the synchronous reference engine whenever CUDA is missing,
    so tiering behaves identically (just serialized) on a CPU-only host.
    """

    if prefer_async:
        try:
            torch = require_torch()
        except StorageUnavailableError:
            return SynchronousTransferEngine(copier)
        if torch.cuda.is_available():
            return CudaTransferEngine(copier)
    return SynchronousTransferEngine(copier)


class HostSlotPool:
    """Deterministic index-ordered pool of host mirror slots.

    Mirrors the page allocator's discipline (index-ordered deque, explicit
    double-free detection) so host capacity accounting is as auditable as
    device capacity accounting.
    """

    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages <= 0:
            raise ValueError("host slot capacity must be positive")
        self._capacity = int(capacity_pages)
        self._free: deque[int] = deque(range(self._capacity))
        self._allocated: set[int] = set()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def free_slots(self) -> int:
        return len(self._free)

    @property
    def used_slots(self) -> int:
        return len(self._allocated)

    def acquire(self) -> int | None:
        """Take the lowest free slot, or None when the mirror is full."""
        if not self._free:
            return None
        slot = self._free.popleft()
        self._allocated.add(slot)
        return slot

    def release(self, slot: int) -> None:
        """Return a slot to the pool, rejecting double frees."""
        if slot not in self._allocated:
            raise InvariantViolationError("host slot was not allocated")
        self._allocated.discard(slot)
        self._free.append(slot)

    def assert_invariants(self) -> None:
        """Check the pool partitions its capacity exactly once."""
        free = list(self._free)
        if len(free) != len(set(free)):
            raise InvariantViolationError("duplicate host slot in the free pool")
        if set(free) & self._allocated:
            raise InvariantViolationError("host slot is both free and allocated")
        if len(free) + len(self._allocated) != self._capacity:
            raise InvariantViolationError("host slot accounting does not equal capacity")


@dataclass(frozen=True, slots=True)
class TieringConfig:
    """Opt-in configuration for the host KV tier.

    Tiering is off unless a manager is constructed with one of these, and the
    GPU-only path stays byte-for-byte the same code when it is absent.
    """

    host_capacity_pages: int
    """Host mirror size in pages; must be positive."""
    enable_spill: bool = True
    """Whether cache pressure may move blocks to the host tier."""
    enable_promotion: bool = True
    """Whether a lookup may pull host blocks back onto the device."""
    max_inflight_transfers: int = 8
    """Ceiling on simultaneously pending transfers in either direction."""
    max_promotion_blocks: int = 8
    """Ceiling on blocks one lookup may request promotion for."""
    promotion_min_free_pages: int = 1
    """Free pages a promotion must leave behind.

    Promotion competes with reservations for the same free pool. Without this
    floor a tier under pressure spills a block, promotes it straight back on
    the next lookup, and turns every step into a pressure step; the floor is
    what stops eviction and promotion from fighting each other.
    """

    def __post_init__(self) -> None:
        if self.host_capacity_pages <= 0:
            raise ValueError("host_capacity_pages must be positive")
        if self.max_inflight_transfers <= 0:
            raise ValueError("max_inflight_transfers must be positive")
        if self.max_promotion_blocks <= 0:
            raise ValueError("max_promotion_blocks must be positive")
        if self.promotion_min_free_pages < 0:
            raise ValueError("promotion_min_free_pages must be non-negative")


@dataclass(slots=True)
class _TierRecord:
    """The single authoritative placement record for one block identity.

    Invariant 17 is structural here: there is one record per identity and one
    ``state`` field on it. The paired resource fields are what each state is
    allowed to own, and :meth:`TierManager.assert_invariants` enforces the
    pairing.
    """

    identity: PrefixBlockIdentity
    parent_identity: PrefixBlockIdentity | None
    block_token_ids: tuple[int, ...]
    context: PrefixCacheContext
    state: TierState
    device_page: KVPageHandle | None = None
    host_slot: int | None = None
    ticket: TransferTicket | None = None
    issue_epoch: int = 0
    tier_request_ref: bool = False
    """Whether the tier holds its own request reference on ``device_page``."""
    copy_landed: bool = False
    """A promotion copy finished but the block is not published yet."""
    position_at_submit: tuple[int, bool, int] | None = None
    """``(last_access, terminal, children)`` of the source block at submit."""
    release_anchor: bool = False
    """Whether dropping this block's cache ref is what prunes its chain."""
    promotions: int = 0
    """How many times this block has been pulled back onto the device."""
    spills: int = 0
    """How many times this block has been pushed out to the host."""


@dataclass(frozen=True, slots=True)
class PromotedBlock:
    """One block whose host-to-device copy finished and is ready to publish."""

    identity: PrefixBlockIdentity
    token_ids: tuple[int, ...]
    """Full token prefix through this block, ready for a cache insert."""
    context: PrefixCacheContext
    page: KVPageHandle
    """RESERVED page holding the promoted bytes; the caller commits it."""
    ancestor_pages: tuple[KVPageHandle, ...]
    """Device pages of the already-cached ancestors, in chain order."""


@dataclass(frozen=True, slots=True)
class EvictedBlock:
    """One block whose device-to-host copy finished; its GPU page can go."""

    identity: PrefixBlockIdentity
    prefix_handle: PrefixHandle | None
    """Cache entry still holding the page, or None if it already vanished."""
    safe_epoch: int
    """Never earlier than the epoch the copy was issued in."""


@dataclass(frozen=True, slots=True)
class HostTierSnapshot:
    """Placement accounting for the whole KV host tier, in pages and blocks.

    Exactly one of these describes a :class:`TierManager`: how many blocks sit
    in each :class:`TierState`, how much of the host mirror is occupied, and
    the cumulative spill/promotion counters.

    Not to be confused with
    :class:`~ayaka.memory.ledger.TierAccountSnapshot`, which is per-
    :class:`~ayaka.types.MemoryTier` *byte* accounting against a capacity.
    """

    host_capacity_pages: int
    host_used_slots: int
    device_blocks: int
    host_blocks: int
    evicting_blocks: int
    promoting_blocks: int
    spills_total: int
    promotions_total: int
    spill_aborts_total: int
    promotion_aborts_total: int
    dropped_host_blocks_total: int
    """Host entries discarded because the mirror was full or a chain vanished."""
    transfers: TransferMetrics

    @property
    def tracked_blocks(self) -> int:
        """Blocks with an authoritative placement in any tier state."""
        return self.device_blocks + self.host_blocks + self.evicting_blocks + self.promoting_blocks


class TierManager:
    """Single authority over where each cached block's bytes live (A11-02).

    It owns no scheduler-visible surface: it never sees requests, never
    touches sequence page tables, and reaches the allocator only through the
    same public reference APIs the prefix cache already uses. Callers drive it
    with :meth:`reconcile`, :meth:`begin_eviction`, and :meth:`begin_promotion`,
    then collect resolved work from :meth:`poll`.

    While a spill is in flight the tier holds its own *request* reference on
    the source page. That is the same tentative-ownership mechanism
    ``try_reserve`` uses for prefix pages, and it means a concurrent cache
    eviction can never pull the bytes out from under a running copy.
    """

    def __init__(
        self,
        *,
        allocator: PageAllocator,
        engine: TransferEngine,
        config: TieringConfig,
    ) -> None:
        self._allocator = allocator
        self._engine = engine
        self._config = config
        self._slots = HostSlotPool(config.host_capacity_pages)
        self._records: dict[PrefixBlockIdentity, _TierRecord] = {}
        self._by_ticket: dict[int, PrefixBlockIdentity] = {}
        self._spills_total = 0
        self._promotions_total = 0
        self._spill_aborts_total = 0
        self._promotion_aborts_total = 0
        self._dropped_host_blocks_total = 0
        self._lock = RLock()

    @property
    def config(self) -> TieringConfig:
        return self._config

    @property
    def engine(self) -> TransferEngine:
        return self._engine

    @property
    def host_free_slots(self) -> int:
        """Host mirror slots still available for a spill."""
        with self._lock:
            return self._slots.free_slots

    @property
    def has_host_blocks(self) -> bool:
        """Whether anything is currently held only on the host.

        Callers use this to skip promotion bookkeeping entirely while nothing
        has ever been spilled, which is the steady state of a tier that is
        enabled but not under pressure.
        """

        with self._lock:
            return self._slots.used_slots > 0

    def reconcile(self, cached: Sequence[CachedBlockInfo]) -> None:
        """Mirror the prefix cache's current device blocks into the index.

        The cache stays the sole authority over what is cached; this only
        keeps placement records in step with it. Three cases matter:

        * a cached identity with no record starts life as ``DEVICE``;
        * an identity that reappeared on the device while the tier believed it
          was host resident makes the device copy authoritative again, so the
          host slot is released (and an in-flight promotion is aborted);
        * a ``DEVICE`` record whose identity is gone was dropped without a
          spill, so the record goes with it.
        """

        with self._lock:
            seen: set[PrefixBlockIdentity] = set()
            for info in cached:
                seen.add(info.identity)
                record = self._records.get(info.identity)
                if record is None:
                    self._records[info.identity] = _TierRecord(
                        identity=info.identity,
                        parent_identity=info.parent_identity,
                        block_token_ids=info.block_token_ids,
                        context=info.identity.context,
                        state=TierState.DEVICE,
                        device_page=info.page,
                    )
                    continue
                record.parent_identity = info.parent_identity
                if record.state is TierState.DEVICE:
                    # A re-inserted block may sit on a different physical page.
                    record.device_page = info.page
                elif record.state is TierState.HOST:
                    # Recomputed on the device: drop the now-redundant copy.
                    self._release_host_slot(record)
                    record.state = TierState.DEVICE
                    record.device_page = info.page
                elif record.state is TierState.PROMOTING:
                    # The promotion raced a recompute and lost; roll it back.
                    self._abort_promotion(record)
                    self._release_host_slot(record)
                    record.state = TierState.DEVICE
                    record.device_page = info.page
                    record.copy_landed = False
            for identity in [
                identity
                for identity, record in self._records.items()
                if record.state is TierState.DEVICE and identity not in seen
            ]:
                self._records.pop(identity)
            self._collect_orphans()

    def _collect_orphans(self) -> None:
        """Release host entries whose ancestor chain no longer exists.

        A host block is only useful if every ancestor can be walked back to
        the chain root, because promotion re-inserts the whole chain. A block
        whose parent was dropped is unreachable forever, so its mirror slot is
        returned instead of being pinned by a dead entry.
        """

        while True:
            orphans = [
                identity
                for identity, record in self._records.items()
                if record.state is TierState.HOST and not self._chain_intact(record)
            ]
            if not orphans:
                return
            for identity in orphans:
                record = self._records.pop(identity)
                self._release_host_slot(record)
                self._dropped_host_blocks_total += 1

    def _chain_intact(self, record: _TierRecord) -> bool:
        """Whether every ancestor of a block still has a placement record."""
        parent = record.parent_identity
        guard = len(self._records) + 1
        while parent is not None:
            ancestor = self._records.get(parent)
            if ancestor is None:
                return False
            parent = ancestor.parent_identity
            guard -= 1
            if guard < 0:  # pragma: no cover - cycle guard
                return False
        return True

    def placement(self, identity: PrefixBlockIdentity) -> TierState | None:
        """Return the authoritative placement of a block, or None if untracked."""
        with self._lock:
            record = self._records.get(identity)
            return None if record is None else record.state

    def host_resident(self, identity: PrefixBlockIdentity) -> bool:
        """Whether a block currently lives only in the host tier."""
        return self.placement(identity) is TierState.HOST

    def begin_eviction(
        self,
        info: CachedBlockInfo,
        *,
        release_anchor: bool = True,
    ) -> TransferTicket | None:
        """Start moving one cached block to the host tier (A11-06).

        Refuses any source page a launched step still owns, so a copy can
        never overlap a window in which a step could touch the page. Returns
        None when the block cannot be spilled right now; the caller then falls
        back to dropping it outright, which is always safe.

        ``release_anchor`` marks the block whose cache reference the caller
        will drop. Dropping a leaf prunes its unshared ancestors too, so those
        ancestors are spilled as non-anchor members of the same cascade: their
        bytes must reach the host before the anchor's release removes them.
        """

        with self._lock:
            if not self._config.enable_spill:
                return None
            if self._inflight_transfers() >= self._config.max_inflight_transfers:
                return None
            record = self._records.get(info.identity)
            if record is None or record.state is not TierState.DEVICE:
                return None
            if not self._spillable(info.page):
                return None
            slot = self._slots.acquire()
            if slot is None:
                return None

            issue_epoch = self._allocator.current_epoch
            # Tier-owned request ref: keeps the source bytes alive for the whole
            # copy even if the cache drops the block meanwhile.
            self._allocator.acquire_request_ref(info.page)
            record.state = TierState.EVICTING
            record.device_page = info.page
            record.host_slot = slot
            record.issue_epoch = issue_epoch
            record.tier_request_ref = True
            record.position_at_submit = (info.last_access, info.terminal, info.children)
            record.release_anchor = release_anchor
            try:
                ticket = self._engine.submit(
                    TransferDirection.DEVICE_TO_HOST,
                    device_page=self._allocator.physical_id(info.page).value,
                    host_slot=slot,
                    issue_epoch=issue_epoch,
                )
            except Exception:
                self._abort_eviction(record, still_cached=True)
                raise
            record.ticket = ticket
            self._by_ticket[ticket.ticket_id] = record.identity
            return ticket

    def begin_promotion(self, identity: PrefixBlockIdentity) -> TransferTicket | None:
        """Start pulling one host block back onto the device (A11-05).

        Allocates a fresh RESERVED page as the destination. No launched step
        can reach a page in that state, which is exactly why the copy is safe
        to run concurrently with execution. Returns None when the block is not
        host resident or the device has no page to spare -- an ordinary
        capacity outcome, not an error.
        """

        with self._lock:
            if not self._config.enable_promotion:
                return None
            if self._inflight_transfers() >= self._config.max_inflight_transfers:
                return None
            record = self._records.get(identity)
            if record is None or record.state is not TierState.HOST:
                return None
            if record.host_slot is None:
                raise InvariantViolationError("a host-resident block has no host slot")
            if self._allocator.available_pages() <= self._config.promotion_min_free_pages:
                return None

            allocated = self._allocator.allocate(1)
            if not allocated:
                return None
            page = allocated[0]

            issue_epoch = self._allocator.current_epoch
            record.state = TierState.PROMOTING
            record.device_page = page
            record.issue_epoch = issue_epoch
            try:
                ticket = self._engine.submit(
                    TransferDirection.HOST_TO_DEVICE,
                    device_page=self._allocator.physical_id(page).value,
                    host_slot=record.host_slot,
                    issue_epoch=issue_epoch,
                )
            except Exception:
                self._abort_promotion(record)
                raise
            record.ticket = ticket
            self._by_ticket[ticket.ticket_id] = record.identity
            return ticket

    def poll(
        self,
        *,
        cached: Sequence[CachedBlockInfo],
        drain: bool = False,
    ) -> tuple[tuple[EvictedBlock, ...], tuple[PromotedBlock, ...]]:
        """Resolve finished transfers into work the caller must publish.

        Returns ``(evicted, promoted)``: the caller drops the cache reference
        named by each evicted block and inserts each promoted page back into
        the cache. Both lists are empty when nothing resolved, which is the
        common case on a quiet step.
        """

        outcomes = self._engine.drain() if drain else self._engine.poll()
        by_identity = {info.identity: info for info in cached}
        with self._lock:
            evicted: list[EvictedBlock] = []
            for outcome in outcomes:
                identity = self._by_ticket.pop(outcome.ticket.ticket_id, None)
                if identity is None:
                    continue
                record = self._records.get(identity)
                if record is None or record.ticket != outcome.ticket:
                    continue
                record.ticket = None
                if outcome.state is TransferState.FAILED:
                    self._abort(record, by_identity)
                    continue
                if record.state is TierState.EVICTING:
                    settled = self._settle_eviction(record, by_identity)
                    if settled is not None:
                        evicted.append(settled)
                elif record.state is TierState.PROMOTING:
                    record.copy_landed = True

            # Promotions publish in chain order: a block can only re-enter the
            # cache once its parent is back on the device, so a landed copy
            # whose parent is still promoting simply waits for the next pass.
            promoted: list[PromotedBlock] = []
            for record in list(self._records.values()):
                if record.state is not TierState.PROMOTING or not record.copy_landed:
                    continue
                ready = self._settle_promotion(record, by_identity)
                if ready is not None:
                    promoted.append(ready)
            return tuple(evicted), tuple(promoted)

    def publish_promotion(
        self,
        identity: PrefixBlockIdentity,
        *,
        device_page: KVPageHandle,
    ) -> None:
        """Acknowledge that a promoted block is cached again on the device.

        Releases the host slot, completing ``PROMOTING -> DEVICE``: the block
        has exactly one authoritative placement again. ``device_page`` is the
        page the cache actually kept, which is not always the page that was
        copied into: an insert reuses an existing canonical block when one
        reappeared while the copy ran.
        """

        with self._lock:
            record = self._records.get(identity)
            if record is None or record.state is not TierState.PROMOTING:
                raise InvalidStateTransitionError("block is not awaiting promotion publication")
            self._release_host_slot(record)
            record.state = TierState.DEVICE
            record.device_page = device_page
            record.copy_landed = False
            record.promotions += 1
            self._promotions_total += 1

    def abort_promotion(self, identity: PrefixBlockIdentity) -> None:
        """Roll a promotion back to ``HOST`` when publication is impossible.

        Used when a cache insert fails after the copy landed: the reserved
        page is rolled back and the host copy remains authoritative.
        """

        with self._lock:
            record = self._records.get(identity)
            if record is None or record.state is not TierState.PROMOTING:
                raise InvalidStateTransitionError("block is not awaiting promotion publication")
            self._abort_promotion(record)

    def forget(self, identity: PrefixBlockIdentity) -> None:
        """Drop a host block permanently, releasing its mirror slot."""
        with self._lock:
            record = self._records.get(identity)
            if record is None:
                return
            if record.state is not TierState.HOST:
                raise InvalidStateTransitionError("only a host-resident block can be forgotten")
            self._release_host_slot(record)
            self._records.pop(identity, None)
            self._dropped_host_blocks_total += 1

    def request_ref_pages(self) -> tuple[KVPageHandle, ...]:
        """Pages the tier itself holds a request reference on, for auditing.

        Cross-component invariant checks must count these alongside sequence
        page tables, since they are ordinary request references.
        """

        with self._lock:
            return tuple(
                record.device_page
                for record in self._records.values()
                if record.tier_request_ref and record.device_page is not None
            )

    def reserved_pages(self) -> tuple[KVPageHandle, ...]:
        """Pages the tier holds as RESERVED promotion destinations.

        Reservation accounting in the manager must count these, since they are
        ordinary allocator reservations that no transaction owns.
        """

        with self._lock:
            return tuple(
                record.device_page
                for record in self._records.values()
                if record.state is TierState.PROMOTING and record.device_page is not None
            )

    def host_chain_after(
        self,
        identities: Sequence[PrefixBlockIdentity],
        *,
        start: int,
    ) -> tuple[PrefixBlockIdentity, ...]:
        """Return the contiguous run of host blocks continuing a device match.

        Promotion is front-anchored because the prefix cache is a chain: a
        block can only be re-inserted once every ancestor is device resident.
        The run stops at the first identity that is not purely host resident.
        """

        with self._lock:
            run: list[PrefixBlockIdentity] = []
            for identity in identities[start : start + self._config.max_promotion_blocks]:
                record = self._records.get(identity)
                if record is None or record.state is not TierState.HOST:
                    break
                run.append(identity)
            return tuple(run)

    def snapshot(self) -> HostTierSnapshot:
        """Return placement and capacity accounting for diagnostics."""
        with self._lock:
            states = [record.state for record in self._records.values()]
            return HostTierSnapshot(
                host_capacity_pages=self._slots.capacity,
                host_used_slots=self._slots.used_slots,
                device_blocks=states.count(TierState.DEVICE),
                host_blocks=states.count(TierState.HOST),
                evicting_blocks=states.count(TierState.EVICTING),
                promoting_blocks=states.count(TierState.PROMOTING),
                spills_total=self._spills_total,
                promotions_total=self._promotions_total,
                spill_aborts_total=self._spill_aborts_total,
                promotion_aborts_total=self._promotion_aborts_total,
                dropped_host_blocks_total=self._dropped_host_blocks_total,
                transfers=self._engine.metrics,
            )

    def assert_invariants(self) -> None:
        """Debug-only proof of invariants 17 and 18 for the tier index.

        Checks that every block has exactly one placement, that each state
        owns exactly the resources it is allowed to own, that host slots are
        never shared, and that no tracked device page is FREE.
        """

        with self._lock:
            self._slots.assert_invariants()
            slots_in_use: dict[int, PrefixBlockIdentity] = {}
            expected_resources = {
                TierState.DEVICE: (True, False),
                TierState.HOST: (False, True),
                TierState.EVICTING: (True, True),
                TierState.PROMOTING: (True, True),
            }
            for identity, record in self._records.items():
                if record.identity != identity:
                    raise InvariantViolationError("tier record is filed under a foreign identity")
                owned = (record.device_page is not None, record.host_slot is not None)
                if owned != expected_resources[record.state]:
                    raise InvariantViolationError(
                        f"tier state {record.state.name} owns the wrong tier resources"
                    )
                if record.tier_request_ref and record.state is not TierState.EVICTING:
                    raise InvariantViolationError("tier request ref outlived its eviction")
                if record.host_slot is not None:
                    if record.host_slot in slots_in_use:
                        raise InvariantViolationError("two blocks claim the same host slot")
                    slots_in_use[record.host_slot] = identity
                if record.device_page is not None:
                    meta = self._allocator.get_meta(record.device_page)
                    if meta.allocation_state is PageAllocationState.FREE:
                        raise InvariantViolationError("a tracked tier page is free")
                    if (
                        record.state is TierState.PROMOTING
                        and meta.allocation_state is not PageAllocationState.RESERVED
                    ):
                        raise InvariantViolationError("a promoting page is not reserved")
            if len(slots_in_use) != self._slots.used_slots:
                raise InvariantViolationError("host slot pool disagrees with tier records")

    def _settle_eviction(
        self,
        record: _TierRecord,
        by_identity: dict[PrefixBlockIdentity, CachedBlockInfo],
    ) -> EvictedBlock | None:
        """Finish ``EVICTING``: publish the host copy, or abort back to device.

        The copy is abandoned when the block's position in the cache changed
        while it ran -- it was touched, gained a child, or became a retained
        endpoint. Spilling then would evict exactly the wrong entry, or would
        drop a reference that no longer prunes what the caller planned for.
        """

        info = by_identity.get(record.identity)
        if info is not None and (info.last_access, info.terminal, info.children) != (
            record.position_at_submit
        ):
            self._abort_eviction(record, still_cached=True)
            return None

        safe_epoch = max(self._allocator.current_epoch, record.issue_epoch)
        self._release_tier_request_ref(record, safe_epoch=safe_epoch)
        record.state = TierState.HOST
        record.device_page = None
        record.spills += 1
        self._spills_total += 1
        return EvictedBlock(
            identity=record.identity,
            prefix_handle=info.handle if info is not None and record.release_anchor else None,
            safe_epoch=safe_epoch,
        )

    def _settle_promotion(
        self,
        record: _TierRecord,
        by_identity: dict[PrefixBlockIdentity, CachedBlockInfo],
    ) -> PromotedBlock | None:
        """Finish ``PROMOTING``: hand the page over, or abort back to host."""
        chain = self._resolve_chain(record, by_identity)
        if chain is None:
            if self._chain_pending(record):
                # An ancestor is still promoting; retry once it publishes.
                return None
            # An ancestor left the cache mid-copy, so the block can no longer
            # be re-inserted; the host copy stays authoritative.
            self._abort_promotion(record)
            return None
        tokens, ancestor_pages = chain
        page = record.device_page
        if page is None:
            raise InvariantViolationError("a promoting block has no device page")
        return PromotedBlock(
            identity=record.identity,
            token_ids=tokens,
            context=record.context,
            page=page,
            ancestor_pages=ancestor_pages,
        )

    def _resolve_chain(
        self,
        record: _TierRecord,
        by_identity: dict[PrefixBlockIdentity, CachedBlockInfo],
    ) -> tuple[tuple[int, ...], tuple[KVPageHandle, ...]] | None:
        """Rebuild a block's full token prefix and ancestor pages, or None.

        Every ancestor must currently be cached and device resident, because a
        prefix insert re-walks the whole chain from the root and needs one page
        per complete block.
        """

        blocks: list[tuple[int, ...]] = [record.block_token_ids]
        pages: list[KVPageHandle] = []
        parent = record.parent_identity
        guard = len(self._records) + 1
        while parent is not None:
            ancestor = self._records.get(parent)
            info = by_identity.get(parent)
            if ancestor is None or ancestor.state is not TierState.DEVICE or info is None:
                return None
            blocks.append(ancestor.block_token_ids)
            pages.append(info.page)
            parent = ancestor.parent_identity
            guard -= 1
            if guard < 0:  # pragma: no cover - cycle guard
                raise InvariantViolationError("tier parent chain contains a cycle")
        tokens: list[int] = []
        for block in reversed(blocks):
            tokens.extend(block)
        return tuple(tokens), tuple(reversed(pages))

    def _chain_pending(self, record: _TierRecord) -> bool:
        """Whether an unresolved chain is merely waiting on a promoting parent."""
        parent = record.parent_identity
        guard = len(self._records) + 1
        while parent is not None:
            ancestor = self._records.get(parent)
            if ancestor is None:
                return False
            if ancestor.state is TierState.PROMOTING:
                return True
            if ancestor.state is not TierState.DEVICE:
                return False
            parent = ancestor.parent_identity
            guard -= 1
            if guard < 0:  # pragma: no cover - cycle guard
                return False
        return False

    def _abort(
        self,
        record: _TierRecord,
        by_identity: dict[PrefixBlockIdentity, CachedBlockInfo],
    ) -> None:
        """Roll one transient record back to its origin tier."""
        if record.state is TierState.EVICTING:
            self._abort_eviction(record, still_cached=record.identity in by_identity)
        elif record.state is TierState.PROMOTING:
            self._abort_promotion(record)

    def _abort_eviction(self, record: _TierRecord, *, still_cached: bool) -> None:
        """``EVICTING -> DEVICE``: give the host slot back, keep the GPU page.

        When the cache dropped the block while the copy ran there is no device
        placement left to keep authority over, so the record is discarded and
        the tier's request reference releases the page.
        """

        self._release_host_slot(record)
        safe_epoch = max(self._allocator.current_epoch, record.issue_epoch)
        self._release_tier_request_ref(record, safe_epoch=safe_epoch)
        self._forget_ticket(record)
        self._spill_aborts_total += 1
        if still_cached:
            record.state = TierState.DEVICE
            return
        record.device_page = None
        self._records.pop(record.identity, None)

    def _abort_promotion(self, record: _TierRecord) -> None:
        """``PROMOTING -> HOST``: roll the reserved page back, keep the copy.

        The destination page is only rolled back while it is still RESERVED.
        A publication that failed midway may already have committed it, in
        which case its ownership release has queued it for deferred free and
        the allocator is the one that finishes the job.
        """

        page = record.device_page
        record.device_page = None
        record.state = TierState.HOST
        self._forget_ticket(record)
        record.copy_landed = False
        self._promotion_aborts_total += 1
        if page is None:
            return
        try:
            meta = self._allocator.get_meta(page)
        except InvalidHandleError:
            return
        if meta.allocation_state is PageAllocationState.RESERVED:
            self._allocator.rollback_reserved([page])

    def _forget_ticket(self, record: _TierRecord) -> None:
        """Drop a record's ticket and its reverse index entry together.

        An abort resolves a transfer the engine may never report, so clearing
        only ``record.ticket`` would strand the ``_by_ticket`` entry forever.
        """
        if record.ticket is not None:
            self._by_ticket.pop(record.ticket.ticket_id, None)
        record.ticket = None

    def _release_host_slot(self, record: _TierRecord) -> None:
        """Return a record's mirror slot to the pool if it holds one."""
        if record.host_slot is not None:
            self._slots.release(record.host_slot)
            record.host_slot = None

    def _release_tier_request_ref(self, record: _TierRecord, *, safe_epoch: int) -> None:
        """Drop the tier's own request reference on an eviction source page."""
        if not record.tier_request_ref:
            return
        record.tier_request_ref = False
        if record.device_page is not None:
            self._allocator.release_request_ref(record.device_page, safe_epoch=safe_epoch)

    def _spillable(self, page: KVPageHandle) -> bool:
        """Whether a page may be read as an eviction source right now.

        Cache blocks are immutable, but a page a launched step still owns is
        refused anyway: that is the conservative epoch rule, and it keeps the
        copy strictly outside any window a step could touch.
        """

        meta = self._allocator.get_meta(page)
        return (
            meta.allocation_state is PageAllocationState.LIVE
            and meta.inflight_refs == 0
            and meta.pending_free_epoch is None
            and meta.cache_refs > 0
        )

    def _inflight_transfers(self) -> int:
        """Count transfers this tier is currently waiting on."""
        return sum(record.ticket is not None for record in self._records.values())
