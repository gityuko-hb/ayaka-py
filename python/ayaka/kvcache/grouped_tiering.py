"""Host placement for canonical, all-group KV prefix checkpoints.

The grouped prefix cache remains the token-identity owner.  A host placement
belongs to that cache entry, and every host slot is qualified by its group and
logical block.  No page becomes device-readable until *all* groups have landed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayaka.exceptions import InvariantViolationError
from ayaka.handles import KVPageHandle
from ayaka.memory.allocator import PageAllocator
from ayaka.memory.state import PageAllocationState
from ayaka.memory.tiering import (
    HostKVStorage,
    HostSlotPool,
    HostTierSnapshot,
    ReadinessState,
    TieringConfig,
    TierState,
    TransferBudget,
    TransferDirection,
    TransferEngine,
    TransferState,
    TransferTicket,
)

if TYPE_CHECKING:
    from ayaka.kvcache.grouped_manager import GroupedPrefixSnapshot, KVCacheGroupManager
    from ayaka.kvcache.grouped_prefix import GroupedPrefixEntry


@dataclass(frozen=True, slots=True)
class GroupHostPage:
    """One page in a host checkpoint; slot numbers are local to ``group_name``."""

    group_name: str
    logical_block: int
    valid_tokens: int
    host_slot: int
    source_page: KVPageHandle


@dataclass(slots=True)
class _GroupBinding:
    allocator: PageAllocator
    storage: HostKVStorage
    engine: TransferEngine
    config: TieringConfig
    slots: HostSlotPool


@dataclass(slots=True)
class _Move:
    host_page: GroupHostPage
    device_page: KVPageHandle
    ticket: TransferTicket | None = None
    credit_owner: object | None = None
    completed: bool = False


@dataclass(slots=True)
class _Placement:
    entry: GroupedPrefixEntry
    state: TierState
    host_pages: tuple[GroupHostPage, ...]
    source_snapshot: GroupedPrefixSnapshot | None = None
    moves: list[_Move] = field(default_factory=list)
    quarantined: bool = False
    submitted: int = 0


class GroupedTierManager:
    """Bounded transfers for whole canonical grouped checkpoints.

    Device pages remain pinned while a spill is pending.  Promotion pages are
    RESERVED until every transfer succeeds; failures retain all destinations
    until a drained engine proves quiescence.  A whole checkpoint is the
    publication unit because each group's retention geometry may differ.
    """

    def __init__(self, manager: KVCacheGroupManager) -> None:
        self._manager = manager
        self._groups: dict[str, _GroupBinding] = {}
        self._placements: dict[int, _Placement] = {}
        self._by_ticket: dict[tuple[str, int], tuple[int, _Move]] = {}
        self._budget: TransferBudget | None = None
        self._spills = 0
        self._promotions = 0
        self._spill_aborts = 0
        self._promotion_aborts = 0
        self._dropped = 0

    @property
    def enabled(self) -> bool:
        return len(self._groups) == len(self._manager.cache_groups)

    def attach_group(
        self,
        group_name: str,
        *,
        config: TieringConfig,
        host_storage: HostKVStorage,
        transfer_engine: TransferEngine,
        transfer_budget: TransferBudget | None,
    ) -> None:
        if group_name in self._groups:
            raise ValueError(f"tier for group {group_name!r} is already attached")
        runtime = self._manager._runtime_by_name.get(group_name)
        if runtime is None:
            raise ValueError(f"unknown KV group {group_name!r}")
        spec = runtime.descriptor.storage_spec
        if runtime.storage is None:
            raise ValueError(f"group {group_name!r} has no bound device storage")
        if host_storage.capacity_pages != config.host_capacity_pages:
            raise ValueError("host mirror capacity disagrees with tier configuration")
        if host_storage.page_size != spec.page_size:
            raise ValueError("host mirror page size disagrees with the group")
        if host_storage.bytes_per_page != spec.bytes_per_page:
            raise ValueError("host mirror page bytes disagree with the group")
        if transfer_engine.bytes_per_page != spec.bytes_per_page:
            raise ValueError("transfer engine page bytes disagree with the group")
        if self._groups and transfer_budget is not self._budget:
            raise ValueError("all grouped tiers must share one transfer budget")
        if not self._groups:
            self._budget = transfer_budget
        self._groups[group_name] = _GroupBinding(
            runtime.allocator,
            host_storage,
            transfer_engine,
            config,
            HostSlotPool(config.host_capacity_pages),
        )

    def readiness(self, entry_id: int) -> ReadinessState:
        placement = self._placements.get(entry_id)
        if placement is None:
            return ReadinessState.DEVICE_VALID
        if placement.quarantined:
            return ReadinessState.FAILED_CANCELLED
        if placement.state is TierState.EVICTING:
            return ReadinessState.DEVICE_VALID
        if placement.state is TierState.PROMOTING:
            return ReadinessState.TRANSFER_PENDING
        return ReadinessState.HOST_VALID

    def spill(self, entry: GroupedPrefixEntry) -> bool:
        """Begin an all-group spill of a cache-exclusive, idle checkpoint."""
        if not self.enabled or entry.snapshot is None or entry.entry_id in self._placements:
            return False
        snapshot = entry.snapshot
        host_pages: list[GroupHostPage] = []
        for name, entries in snapshot.groups:
            binding = self._groups[name]
            if not binding.config.enable_spill or binding.slots.free_slots < len(entries):
                return False
            for page_entry in entries:
                meta = binding.allocator.get_meta(page_entry.page)
                if (
                    meta.allocation_state is not PageAllocationState.LIVE
                    or meta.request_refs
                    or meta.cache_refs
                    or meta.inflight_refs
                    or meta.pin_refs != 1
                    or meta.valid_tokens < page_entry.valid_tokens
                ):
                    return False
        try:
            for name, entries in snapshot.groups:
                binding = self._groups[name]
                for page_entry in entries:
                    slot = binding.slots.acquire()
                    if slot is None:
                        raise InvariantViolationError("host slot reservation changed during spill")
                    host_pages.append(
                        GroupHostPage(
                            name,
                            page_entry.logical_block,
                            page_entry.valid_tokens,
                            slot,
                            page_entry.page,
                        )
                    )
        except BaseException:
            self._release_slots(host_pages)
            raise
        placement = _Placement(entry, TierState.EVICTING, tuple(host_pages), snapshot)
        placement.moves = [_Move(value, value.source_page) for value in host_pages]
        self._placements[entry.entry_id] = placement
        self._issue_available(placement)
        if placement.submitted == 0 and not placement.quarantined:
            self._release_slots(host_pages)
            del self._placements[entry.entry_id]
            return False
        return True

    def promote(self, entry: GroupedPrefixEntry) -> bool:
        """Reserve all device destinations before copying any host page."""
        placement = self._placements.get(entry.entry_id)
        if not self.enabled or placement is None or placement.state is not TierState.HOST:
            return False
        allocated: dict[str, tuple[KVPageHandle, ...]] = {}
        try:
            for name, binding in self._groups.items():
                pages = tuple(value for value in placement.host_pages if value.group_name == name)
                if not binding.config.enable_promotion:
                    return False
                if (
                    binding.allocator.available_pages() - len(pages)
                    < binding.config.promotion_min_free_pages
                ):
                    return False
                reserved = binding.allocator.allocate(len(pages))
                if reserved is None:
                    return False
                allocated[name] = reserved
        finally:
            if len(allocated) != len(self._groups):
                for name, pages in allocated.items():
                    self._groups[name].allocator.rollback_reserved(pages)
        if len(allocated) != len(self._groups):
            return False
        by_group = {name: iter(pages) for name, pages in allocated.items()}
        placement.state = TierState.PROMOTING
        placement.moves = [
            _Move(value, next(by_group[value.group_name])) for value in placement.host_pages
        ]
        placement.submitted = 0
        placement.quarantined = False
        self._issue_available(placement)
        if placement.submitted == 0 and not placement.quarantined:
            self._abort_promotion(placement)
            return False
        return True

    def poll(self, *, drain: bool = False) -> int:
        """Settle transfer outcomes and publish only complete checkpoints."""
        landed = 0
        while True:
            progress = False
            for name, binding in self._groups.items():
                outcomes = binding.engine.drain() if drain else binding.engine.poll()
                for outcome in outcomes:
                    key = (name, outcome.ticket.ticket_id)
                    tracked = self._by_ticket.pop(key, None)
                    if tracked is None:
                        continue
                    entry_id, move = tracked
                    placement = self._placements.get(entry_id)
                    if placement is None or move.ticket != outcome.ticket:
                        raise InvariantViolationError("grouped transfer ticket changed owner")
                    self._release_credit(move)
                    move.ticket = None
                    progress = True
                    if outcome.state is TransferState.COMPLETED:
                        move.completed = True
                    else:
                        placement.quarantined = True
            for placement in tuple(self._placements.values()):
                if not placement.quarantined:
                    progress |= self._issue_available(placement)
                    if placement.moves and all(move.completed for move in placement.moves):
                        if placement.state is TierState.EVICTING:
                            self._finish_spill(placement)
                        else:
                            self._finish_promotion(placement)
                        landed += 1
                        progress = True
                elif drain and not any(move.ticket is not None for move in placement.moves):
                    self._settle_quarantine(placement)
                    progress = True
            if not drain or not progress:
                break
        return landed

    def cancel(self, entry_id: int) -> bool:
        """Quarantine a pending migration until a drained copy stream is quiet."""
        placement = self._placements.get(entry_id)
        if placement is None or placement.state not in (TierState.EVICTING, TierState.PROMOTING):
            return False
        placement.quarantined = True
        return True

    def can_forget(self, entry_id: int) -> bool:
        placement = self._placements.get(entry_id)
        return placement is None or placement.state is TierState.HOST

    def forget(self, entry_id: int) -> bool:
        placement = self._placements.get(entry_id)
        if placement is None:
            return True
        if placement.state is not TierState.HOST:
            return False
        self._release_slots(placement.host_pages)
        del self._placements[entry_id]
        self._dropped += 1
        return True

    def shutdown(self) -> None:
        """Drain copies, then prove every failed destination is quiescent."""
        self.poll(drain=True)
        if any(
            place.state in (TierState.EVICTING, TierState.PROMOTING)
            for place in self._placements.values()
        ):
            raise InvariantViolationError("grouped tier has unsettled transfers after drain")

    def release(self) -> None:
        if self._placements or self._by_ticket:
            raise InvariantViolationError("grouped tier still owns prefix or transfer resources")
        for binding in self._groups.values():
            binding.slots.assert_invariants()
            if binding.slots.used_slots:
                raise InvariantViolationError("grouped tier still owns host slots")
        self._groups.clear()
        self._budget = None

    def snapshots(self) -> dict[str, HostTierSnapshot]:
        result: dict[str, HostTierSnapshot] = {}
        for name, binding in self._groups.items():
            states = [
                place.state
                for place in self._placements.values()
                for page in place.host_pages
                if page.group_name == name
            ]
            result[name] = HostTierSnapshot(
                host_capacity_pages=binding.slots.capacity,
                host_used_slots=binding.slots.used_slots,
                device_blocks=sum(
                    len(entries)
                    for entry in (
                        self._manager._prefix_cache._entries.values()
                        if self._manager._prefix_cache is not None
                        else ()
                    )
                    if entry.entry_id not in self._placements and entry.snapshot is not None
                    for group_name, entries in entry.snapshot.groups
                    if group_name == name
                ),
                host_blocks=states.count(TierState.HOST),
                evicting_blocks=states.count(TierState.EVICTING),
                promoting_blocks=states.count(TierState.PROMOTING),
                spills_total=self._spills,
                promotions_total=self._promotions,
                spill_aborts_total=self._spill_aborts,
                promotion_aborts_total=self._promotion_aborts,
                dropped_host_blocks_total=self._dropped,
                quarantined_blocks=sum(
                    sum(page.group_name == name for page in place.host_pages)
                    for place in self._placements.values()
                    if place.quarantined
                ),
                transfers=binding.engine.metrics,
            )
        return result

    def reserved_pages(self, group_name: str) -> tuple[KVPageHandle, ...]:
        return tuple(
            move.device_page
            for place in self._placements.values()
            if place.state is TierState.PROMOTING
            for move in place.moves
            if move.host_page.group_name == group_name
        )

    def assert_invariants(self) -> None:
        for name, binding in self._groups.items():
            binding.slots.assert_invariants()
            slots = [
                page.host_slot
                for place in self._placements.values()
                for page in place.host_pages
                if page.group_name == name
            ]
            if len(slots) != len(set(slots)) or len(slots) != binding.slots.used_slots:
                raise InvariantViolationError("grouped host slots disagree with placement records")
        for entry_id, place in self._placements.items():
            if place.entry.entry_id != entry_id:
                raise InvariantViolationError("grouped tier entry identity drifted")
            if place.state is TierState.HOST and place.entry.snapshot is not None:
                raise InvariantViolationError("host checkpoint still binds device pages")
            if place.state is TierState.EVICTING and place.entry.snapshot is None:
                raise InvariantViolationError("spill lost its device source pins")
            if place.state is TierState.PROMOTING and place.entry.snapshot is not None:
                raise InvariantViolationError("promotion became device-visible before publication")

    def _issue_available(self, place: _Placement) -> bool:
        if place.state not in (TierState.EVICTING, TierState.PROMOTING) or place.quarantined:
            return False
        progressed = False
        direction = (
            TransferDirection.DEVICE_TO_HOST
            if place.state is TierState.EVICTING
            else TransferDirection.HOST_TO_DEVICE
        )
        for move in place.moves:
            if move.completed or move.ticket is not None:
                continue
            binding = self._groups[move.host_page.group_name]
            active = sum(
                other.ticket is not None
                for placement in self._placements.values()
                for other in placement.moves
                if other.host_page.group_name == move.host_page.group_name
            )
            if active >= binding.config.max_inflight_transfers:
                continue
            # A unique owner keeps the shared byte semaphore correct across
            # repeated migrations and groups whose ticket IDs overlap.
            owner = object()
            if self._budget is not None:
                try:
                    self._budget.acquire(owner, binding.engine.bytes_per_page)
                except Exception:
                    continue
            try:
                ticket = binding.engine.submit(
                    direction,
                    device_page=binding.allocator.physical_id(move.device_page).value,
                    host_slot=move.host_page.host_slot,
                    issue_epoch=binding.allocator.current_epoch,
                )
            except Exception:
                if self._budget is not None:
                    self._budget.release(owner)
                place.quarantined = True
                break
            move.ticket = ticket
            move.credit_owner = owner if self._budget is not None else None
            self._by_ticket[(move.host_page.group_name, ticket.ticket_id)] = (
                place.entry.entry_id,
                move,
            )
            place.submitted += 1
            progressed = True
        return progressed

    def _release_credit(self, move: _Move) -> None:
        if move.credit_owner is not None and self._budget is not None:
            self._budget.release(move.credit_owner)
            move.credit_owner = None

    def _finish_spill(self, place: _Placement) -> None:
        snapshot = place.source_snapshot
        if snapshot is None or place.entry.snapshot is not snapshot:
            raise InvariantViolationError("grouped spill lost its canonical source")
        place.entry.snapshot = None
        self._manager.unpin_prefix(snapshot)
        place.source_snapshot = None
        place.moves.clear()
        place.state = TierState.HOST
        self._spills += 1

    def _finish_promotion(self, place: _Placement) -> None:
        from ayaka.kvcache.grouped_manager import GroupedPrefixSnapshot
        from ayaka.memory.sequence import GroupPageTableEntry

        groups = []
        for name in self._manager._runtime_by_name:
            moves = [move for move in place.moves if move.host_page.group_name == name]
            binding = self._groups[name]
            pages = tuple(move.device_page for move in moves)
            for move in moves:
                binding.allocator.set_valid_tokens(move.device_page, move.host_page.valid_tokens)
            binding.allocator.commit_reserved(pages)
            for page in pages:
                binding.allocator.pin_page(page)
                binding.allocator.release_request_ref(
                    page, safe_epoch=binding.allocator.current_epoch
                )
            groups.append(
                (
                    name,
                    tuple(
                        GroupPageTableEntry(
                            move.host_page.logical_block,
                            move.device_page,
                            move.host_page.valid_tokens,
                        )
                        for move in moves
                    ),
                )
            )
        self._manager._next_snapshot += 1
        snapshot = GroupedPrefixSnapshot(
            id(self._manager),
            self._manager._next_snapshot,
            place.entry.logical_position,
            tuple(groups),
        )
        self._manager._prefix_snapshots[snapshot.snapshot_id] = snapshot
        place.entry.snapshot = snapshot
        self._release_slots(place.host_pages)
        del self._placements[place.entry.entry_id]
        self._promotions += 1

    def _settle_quarantine(self, place: _Placement) -> None:
        if place.state is TierState.EVICTING:
            self._release_slots(place.host_pages)
            del self._placements[place.entry.entry_id]
            self._spill_aborts += 1
        else:
            self._abort_promotion(place)

    def _abort_promotion(self, place: _Placement) -> None:
        for name, binding in self._groups.items():
            pages = tuple(
                move.device_page for move in place.moves if move.host_page.group_name == name
            )
            if pages:
                binding.allocator.abandon_reserved(
                    pages, safe_epoch=binding.allocator.current_epoch
                )
                binding.allocator.reclaim_completed()
        place.moves.clear()
        place.state = TierState.HOST
        place.quarantined = False
        self._promotion_aborts += 1

    def _release_slots(self, pages: list[GroupHostPage] | tuple[GroupHostPage, ...]) -> None:
        for page in pages:
            self._groups[page.group_name].slots.release(page.host_slot)
