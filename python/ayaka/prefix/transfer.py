"""Explicit spill/restore tickets with all-layer completion and last-use ownership.

These maintenance tickets use the same backend transaction/lease lifecycle as
model steps. They never change a request's computed counter. A restored prefix
becomes eligible for request attachment only after the entire copy succeeds.
"""

from __future__ import annotations

from contextlib import nullcontext
from itertools import count
from typing import Any

from ayaka.executor.ticket import TicketState
from ayaka.kvcache.manager import KVCapacityError
from ayaka.memory.buffer import BufferAllocator
from ayaka.plan import WorkspaceRequest
from ayaka.prefix.resume import ValidResume
from ayaka.types import MemoryOwner, MemoryTier
from ayaka.utils.torch_utils import require_torch
from ayaka.utils.validation import require_int

_TRANSFER_IDS = count(1)


class TransferCredits:
    """Shared byte semaphore; reservations last until whole-ticket retirement."""

    def __init__(self, limit: int):
        require_int(limit, "transfer limit")
        self.limit = limit
        self._claims: dict[object, int] = {}

    @property
    def held(self):
        return sum(self._claims.values())

    def acquire(self, owner, nbytes):
        require_int(nbytes, "transfer bytes")
        if owner in self._claims:
            raise ValueError("transfer owner already reserved credits")
        if self.held + nbytes > self.limit:
            raise KVCapacityError("transfer credits exhausted")
        self._claims[owner] = nbytes

    def release(self, owner):
        if owner not in self._claims:
            raise ValueError("unknown transfer owner")
        del self._claims[owner]


class HostPrefix:
    """Immutable completed host checkpoint with one shared-ledger buffer claim.

    Owned by the caller after a successful spill. Close explicitly after all
    restore tickets retire. No host snapshot is readable/restorable while its
    producer is pending, failed or cancelled. Contents are private; mutating a
    borrowed tensor invalidates the snapshot by version check before restore.
    """

    def __init__(self, cache, value, buffer, tensor):
        self.cache_id = cache.cache_id
        self.context = value.context
        self.token_ids = value.token_ids
        self._buffer = buffer
        self._tensor: Any = tensor
        self._version = None
        self._borrowers = 0
        self.ready = self.closed = False

    def validate(self, cache):
        if (
            self.closed
            or not self.ready
            or cache.cache_id != self.cache_id
            or self._tensor._version != self._version
        ):
            raise ValueError("host prefix is pending, mutated, closed or from another cache")

    def close(self):
        if self.closed:
            return
        if self._borrowers:
            raise RuntimeError("host prefix still belongs to an active transfer")
        self._tensor = None
        self._buffer.close()
        self.closed = True


class PrefixTransfer:
    """One explicitly adopted spill or restore, polled without synchronization.

    ``launch`` transfers ownership before the first copy. ``poll`` publishes
    only after the final event; on failure it drains the same stream. If a drain
    fence cannot be recorded, state stays QUARANTINED and all buffers/pages and
    credits remain owned. Retrying poll may re-establish a drain fence. Caller
    must continue polling even after cancel; no destructor frees live resources.
    """

    def __init__(
        self, cache, credits, *, host: HostPrefix | None = None, value: ValidResume | None = None
    ):
        if (host is None) == (value is None):
            raise ValueError("provide exactly one host checkpoint or resident entry")
        cache.validate_runner(cache.runner)
        self.cache, self.credits = cache, credits
        self.backend = cache.backend
        self.storage = cache.kv.storages["default"].storage
        self.id = next(_TRANSFER_IDS)
        self.restoring = host is not None
        self.state = None
        self._settlement_failed = False
        self.retired = False
        self.succeeded = False
        self.cancelled = False
        self.error = ""
        self.stream = self.event = None
        self.sequence = self.lease = None
        self._submitted = False
        self._pins = ()
        self.result = None
        self._claimed = False
        spec = self.storage.spec
        if host is not None:
            n = len(host.token_ids)
        else:
            assert value is not None
            n = len(value.token_ids)
        self.nbytes = (
            n
            * spec.num_layers
            * 2
            * spec.num_kv_heads_local
            * spec.head_dim
            * self.storage.torch_dtype.itemsize
        )
        self.host = host
        try:
            credits.acquire(self, self.nbytes)
            self._claimed = True
            if self.restoring:
                assert host is not None
                host.validate(cache)
                host._borrowers += 1
                self._host_borrowed = True
                self.sequence = self.backend.create_sequence(
                    f"prefix-restore/{cache.cache_id}/{self.id}"
                )
                tx = self.backend.begin_transaction(self.id)
                try:
                    result = self.backend.try_reserve(tx, self.sequence, n)
                    if not result.ok:
                        raise KVCapacityError(f"restore reservation failed: {result.reason}")
                    self.lease = self.backend.prepare_step(tx)
                except BaseException:
                    # prepare_step rolls back a rejected transaction itself.
                    from ayaka.exceptions import InvalidHandleError

                    try:
                        self.backend.rollback_transaction(tx)
                    except InvalidHandleError:
                        pass
                    raise
                self.view = self.backend.build_execution_view(self.lease)
            else:
                assert value is not None
                if value.cache_id != cache.cache_id or cache._entries.get(value.entry_id) != value:
                    raise ValueError("spill requires a current complete cache entry")
                pins = []
                try:
                    for entry in value.pages:
                        self.backend.allocator.pin_page(entry.page)
                        pins.append(entry)
                finally:
                    self._pins = tuple(pins)
                tier = (
                    MemoryTier.HOST_PINNED
                    if self.storage.device.type == "cuda"
                    else MemoryTier.HOST_PAGEABLE
                )
                buffer = BufferAllocator(cache.kv.ledger).allocate(
                    WorkspaceRequest("prefix-host", self.nbytes, MemoryOwner.KV, tier),
                    label=f"prefix/{cache.cache_id}/{self.id}/host",
                )
                assert buffer is not None
                try:
                    tensor = buffer.tensor.view(self.storage.torch_dtype).view(
                        spec.num_layers, 2, n, spec.num_kv_heads_local, spec.head_dim
                    )
                    self.host = HostPrefix(cache, value, buffer, tensor)
                except BaseException:
                    buffer.close()
                    raise
                self.host._borrowers += 1
                self._host_borrowed = True
        except BaseException:
            self._cleanup(succeeded=False)
            raise
        # Registration is the adoption boundary: cache.close cannot free us.
        cache._transfers.add(self)
        self.state = TicketState.ADOPTED

    @classmethod
    def spill(cls, cache, value, credits):
        """Reserve host storage and pin source; no data movement until launch."""
        return cls(cache, credits, value=value)

    @classmethod
    def restore(cls, cache, host, credits):
        """Reserve every destination page and borrow a completed host snapshot."""
        return cls(cache, credits, host=host)

    def launch(self):
        if self.state is not TicketState.ADOPTED or self.retired:
            raise RuntimeError("transfer can be launched exactly once after adoption")
        assert self.host is not None
        torch = require_torch(capability="prefix transfer")
        try:
            if self.restoring:
                self.host.validate(self.cache)
                self.backend.validate_execution_view(self.view)
                self.backend.mark_step_in_flight(self.lease)
                self._submitted = True
            if self.storage.device.type == "cuda":
                self.stream = torch.cuda.Stream(device=self.storage.device)
            self.state = TicketState.SUBMITTED
            with (
                torch.cuda.stream(self.stream) if self.stream is not None else nullcontext(),
                torch.no_grad(),
            ):
                self._enqueue()
                self._record_event()
        except BaseException as exc:
            self.error = str(exc)
            self._drain()
            if not isinstance(exc, Exception):
                raise
        return self

    def _enqueue(self):
        """All layers share one stream; override only for controlled fault injection."""
        assert self.host is not None
        n = len(self.host.token_ids)
        if self.restoring:
            pages = self.view.sequences[0].block_table
        else:
            pages = tuple(self.backend.allocator.physical_id(e.page).value for e in self._pins)
        for layer in range(self.storage.spec.num_layers):
            for index, kind in enumerate(("key", "value")):
                plane = self.storage.plane(layer, kind)
                for block, page in enumerate(pages):
                    start = block * self.backend.page_size
                    count = min(self.backend.page_size, n - start)
                    device = plane[page, :count]
                    host = self.host._tensor[layer, index, start : start + count]
                    destination, source = (device, host) if self.restoring else (host, device)
                    destination.copy_(source, non_blocking=True)

    def _record_event(self):
        if self.stream is not None:
            torch = require_torch(capability="prefix completion fence")
            event = torch.cuda.Event()
            event.record(self.stream)
            self.event = event

    def _drain(self):
        self.state = TicketState.DRAINING
        try:
            self._record_event()
        except Exception as exc:
            self.event = None
            self.error = self.error or str(exc)
            self.state = TicketState.QUARANTINED

    def cancel(self):
        """Suppress publication while preserving ownership until last use."""
        if self.retired:
            return
        self.cancelled = True
        if self.state is TicketState.ADOPTED:
            self._finish(False)

    def poll(self) -> bool:
        """Return True only when retired; a False result retains all ownership."""
        if self.retired:
            return True
        if self._settlement_failed:
            return False
        if self.state is TicketState.ADOPTED:
            return False
        if self.state is TicketState.QUARANTINED:
            self._drain()
            if self.state is TicketState.QUARANTINED:
                return False
        try:
            if self.event is not None and not self.event.query():
                return False
        except Exception as exc:
            self.error = str(exc)
            self._drain()
            return False
        self._finish(not self.error and not self.cancelled)
        return self.retired

    def _finish(self, succeeded):
        assert self.host is not None
        try:
            if succeeded and self.restoring:
                self.backend.commit_step(self.lease)
                self.backend.retire_step(self.lease)
                self.lease = None
                pins = self.backend.pin_prefix(self.sequence, len(self.host.token_ids))
                try:
                    self.result = self.cache._insert(self.host.context, self.host.token_ids, pins)
                except BaseException:
                    self.backend.unpin_prefix(pins)
                    raise
            elif succeeded:
                self.host._version = self.host._tensor._version
                self.host.ready = True
                self.result = self.host
            self._cleanup(succeeded=succeeded)
        except Exception as exc:
            # A settlement error is not repaired by repeating commit or cleanup.
            self.error = f"transfer settlement failed: {exc}"
            self._settlement_failed = True
            self.state = TicketState.QUARANTINED
            return
        self.succeeded = succeeded
        self.retired = True
        self.state = TicketState.RETIRED
        self.cache._transfers.discard(self)

    def _cleanup(self, *, succeeded):
        if self.lease is not None:
            if self._submitted:
                self.backend.fail_in_flight_step(self.lease, safe_epoch=self.backend.current_epoch)
            else:
                self.backend.abort_prepared_step(self.lease)
            self.lease = None
        if self.sequence is not None:
            self.backend.release_sequence(self.sequence)
            self.sequence = None
        if self._pins:
            self.backend.unpin_prefix(self._pins)
            self._pins = ()
        self.backend.advance_epoch(self.backend.current_epoch)
        if getattr(self, "_host_borrowed", False):
            assert self.host is not None
            self.host._borrowers -= 1
            self._host_borrowed = False
        if not self.restoring and self.host is not None and not succeeded:
            self.host.close()
        if self._claimed:
            self.credits.release(self)
            self._claimed = False
