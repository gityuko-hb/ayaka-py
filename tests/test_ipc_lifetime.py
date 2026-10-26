"""IPC export/import lifetime on CPU (research P0 lifetime row, E16 owners part).

The research asks that a pin, a local hold, a duplicate or missing close ACK and
a lost peer all *refuse* the close and keep the allocation until every owner is
released, so a peer can never read recycled memory. ``IpcRegionRegistry`` takes
an injectable backend; this fake maps "handles" onto CPU byte tensors so the
registry's refcount and ACK protocol is exercised without CUDA. The native
primitives themselves are covered by ``test_ipc_csrc.py`` on GPU hosts.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from ayaka.runtime.ipc_engine import (
    IpcGenerationError,
    IpcLifetimeError,
    IpcRegionDescriptor,
    IpcRegionRegistry,
)
from ayaka.utils.import_utils import CapabilityError

HANDLE_SIZE = 8


class FakeIpcBackend:
    """Maps each exported tensor to a fixed-size handle; ``open`` aliases it."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.allocations: dict[bytes, torch.Tensor] = {}
        self.opened: list[bytes] = []

    def ipc_available(self) -> bool:
        return self.available

    def legacy_ipc_capable(self, tensor: torch.Tensor) -> bool:
        return isinstance(tensor, torch.Tensor) and tensor.is_contiguous()

    def export_allocation(self, tensor: torch.Tensor) -> tuple[bytes, int, int, int]:
        handle = tensor.data_ptr().to_bytes(HANDLE_SIZE, "little")
        self.allocations[handle] = tensor.view(torch.uint8).reshape(-1)
        nbytes = tensor.numel() * tensor.element_size()
        return handle, 0, nbytes, 0

    def open_allocation(self, handle: bytes, allocation_nbytes: int, device: int) -> torch.Tensor:
        self.opened.append(handle)
        return self.allocations[handle][:allocation_nbytes]

    def byte_view(self, tensor: torch.Tensor, nbytes: int) -> torch.Tensor:
        return tensor.view(torch.uint8).reshape(-1)[:nbytes]

    def ipc_handle_size(self) -> int:
        return HANDLE_SIZE


@pytest.fixture
def backend() -> FakeIpcBackend:
    return FakeIpcBackend()


def _registry(backend: FakeIpcBackend, **kwargs: object) -> IpcRegionRegistry:
    return IpcRegionRegistry(backend=backend, **kwargs)  # type: ignore[arg-type]


def _region(nbytes: int = 64) -> torch.Tensor:
    return torch.arange(nbytes, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Exporter: holds, graph pins and peer close ACKs keep the allocation alive
# ---------------------------------------------------------------------------


def test_release_without_owners_drops_the_allocation(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    tensor = _region()
    owner = weakref.ref(tensor)
    descriptor = registry.export(tensor, offset=8, nbytes=16)
    assert (descriptor.offset, descriptor.nbytes, descriptor.end) == (8, 16, 24)
    del tensor
    backend.allocations.clear()
    gc.collect()
    assert owner() is not None  # the registry holds the exported allocation
    registry.release(descriptor)
    gc.collect()
    assert registry.exports == 0 and owner() is None


def test_descriptors_on_one_allocation_share_one_owner(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    tensor = _region()
    first = registry.export(tensor, offset=0, nbytes=8)
    second = registry.export(tensor, offset=32, nbytes=8)
    assert registry.exports == 1 and first.handle == second.handle
    registry.release(first)
    assert registry.exports == 1  # the second descriptor still owns it
    registry.release(second)
    assert registry.exports == 0


def test_local_hold_refuses_release_until_unheld(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    descriptor = registry.export(_region())
    registry.hold(descriptor)
    with pytest.raises(IpcLifetimeError, match="1 local holds.*allocation retained"):
        registry.release(descriptor)
    assert registry.exports == 1
    with pytest.raises(KeyError):
        registry.release(descriptor)  # the deferred close is not a second release
    registry.unhold(descriptor)
    assert registry.exports == 0
    with pytest.raises(KeyError):
        registry.unhold(descriptor)


def test_missing_peer_ack_retains_the_allocation(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    descriptor = registry.export(_region(), close_acks=(1, 2))
    with pytest.raises(IpcLifetimeError, match=r"close-ACK missing from ranks \[1, 2\]"):
        registry.release(descriptor)
    registry.confirm_close(descriptor, 1)  # a valid ACK while rank 2 is owed: no error
    registry.confirm_close(descriptor, 1)  # duplicate: idempotent, not rank 2's ACK
    assert registry.exports == 1
    registry.confirm_close(descriptor, 2)
    assert registry.exports == 0


def test_ack_from_an_undeclared_rank_is_refused_and_not_recorded(
    backend: FakeIpcBackend,
) -> None:
    registry = _registry(backend)
    tensor = _region()
    descriptor = registry.export(tensor, close_acks=(1,))
    with pytest.raises(IpcLifetimeError, match="rank 9, which this export never declared"):
        registry.confirm_close(descriptor, 9)
    later = registry.export(tensor, offset=8, nbytes=8, close_acks=(9,))
    registry.confirm_close(descriptor, 1)
    registry.release(descriptor)
    with pytest.raises(IpcLifetimeError, match=r"ranks \[9\]"):
        registry.release(later)  # the early stray ACK did not pre-satisfy rank 9
    registry.confirm_close(later, 9)
    assert registry.exports == 0


def test_lost_peer_never_frees_the_allocation(backend: FakeIpcBackend) -> None:
    """Without the peer's ACK the export stays owned; only force abandons it."""
    registry = _registry(backend)
    tensor = _region()
    owner = weakref.ref(tensor)
    descriptor = registry.export(tensor, close_acks=(3,))
    del tensor
    backend.allocations.clear()
    with pytest.raises(IpcLifetimeError):
        registry.release(descriptor)
    gc.collect()
    assert registry.exports == 1 and owner() is not None
    registry.close_all()  # exports are dropped by an explicit teardown only
    gc.collect()
    assert registry.exports == 0 and owner() is None


def test_ack_received_before_release_counts(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    descriptor = registry.export(_region(), close_acks=(1,))
    registry.confirm_close(descriptor, 1)  # peer closed first
    assert registry.exports == 1  # the exporter has not released yet
    registry.release(descriptor)
    assert registry.exports == 0


def test_re_export_adds_peers_to_the_shared_allocation(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    tensor = _region()
    first = registry.export(tensor, close_acks=(1,))
    second = registry.export(tensor, offset=16, nbytes=16, close_acks=(2,))
    registry.release(first)
    assert registry.exports == 1  # the second descriptor still owns the allocation
    with pytest.raises(IpcLifetimeError, match=r"ranks \[1, 2\]"):
        registry.release(second)
    registry.confirm_close(second, 2)
    assert registry.exports == 1
    registry.confirm_close(first, 1)  # ACKs are per allocation, not per descriptor
    assert registry.exports == 0


def test_hold_and_ack_must_both_clear(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    descriptor = registry.export(_region(), close_acks=(1,))
    registry.hold(descriptor)
    with pytest.raises(IpcLifetimeError, match="local holds.*close-ACK missing"):
        registry.release(descriptor)
    registry.unhold(descriptor)  # the hold is returned; the ACK is still owed
    assert registry.exports == 1
    registry.confirm_close(descriptor, 1)
    assert registry.exports == 0


def test_graph_close_after_a_refused_release_stays_consistent(
    backend: FakeIpcBackend,
) -> None:
    """``collective_graph`` returns its pin and hold in sequence after a refusal.

    Neither return may raise while the peer ACK is still owed; otherwise the
    graph would keep ``_pinned`` set and later return the hold twice.
    """
    exporter, importer = _registry(backend), _registry(backend)
    own = exporter.export(_region(), close_acks=(1,))
    peer = exporter.export(_region(32))
    importer.open(peer).close()
    exporter.hold(own)
    importer.pin(peer)
    with pytest.raises(IpcLifetimeError):
        exporter.release(own)
    importer.unpin(peer)
    exporter.unhold(own)
    with pytest.raises(IpcLifetimeError, match="no matching export hold"):
        exporter.unhold(own)
    exporter.confirm_close(own, 1)
    assert exporter.exports == 1  # only ``peer`` is still exported
    importer.close(peer)
    exporter.release(peer)
    assert (exporter.exports, importer.opens) == (0, 0)


def test_export_validation_happens_before_the_native_export(backend: FakeIpcBackend) -> None:
    registry = _registry(backend)
    tensor = _region(16)
    with pytest.raises(ValueError, match="outside"):
        registry.export(tensor, offset=16)
    with pytest.raises(ValueError, match="outside"):
        registry.export(tensor, offset=8, nbytes=9)
    with pytest.raises(ValueError, match="close-ACK ranks"):
        registry.export(tensor, close_acks=(-1,))
    with pytest.raises(ValueError, match="contiguous"):
        registry.export(torch.zeros(4, 4, dtype=torch.uint8).t())
    assert registry.exports == 0


def test_unavailable_backend_fails_closed(backend: FakeIpcBackend) -> None:
    backend.available = False
    registry = _registry(backend)
    with pytest.raises(CapabilityError):
        registry.export(_region())
    descriptor = IpcRegionDescriptor(
        handle=b"x" * HANDLE_SIZE, offset=0, nbytes=4, allocation_nbytes=4
    )
    with pytest.raises(CapabilityError):
        registry.open(descriptor)
    assert backend.opened == []


# ---------------------------------------------------------------------------
# Importer: views, GPU flights and graph pins keep the mapping alive
# ---------------------------------------------------------------------------


def test_view_aliases_the_exported_bytes(backend: FakeIpcBackend) -> None:
    exporter, importer = _registry(backend), _registry(backend)
    tensor = _region()
    descriptor = exporter.export(tensor, offset=8, nbytes=16)
    with importer.open(descriptor) as view:
        assert view.tensor.tolist() == list(range(8, 24))
        tensor[8] = 200
        assert int(view.tensor[0]) == 200
    assert view.closed
    with pytest.raises(IpcLifetimeError):
        _ = view.tensor
    view.close()  # idempotent


def test_live_view_refuses_close(backend: FakeIpcBackend) -> None:
    exporter, importer = _registry(backend), _registry(backend)
    descriptor = exporter.export(_region())
    view = importer.open(descriptor)
    with pytest.raises(IpcLifetimeError, match="1 live views; mapping retained"):
        importer.close(descriptor)
    assert importer.opens == 1
    view.close()
    importer.close(descriptor)
    assert importer.opens == 0


def test_flight_and_pin_defer_the_unmap(backend: FakeIpcBackend) -> None:
    exporter, importer = _registry(backend), _registry(backend)
    descriptor = exporter.export(_region())
    importer.open(descriptor).close()
    importer.begin_flight(descriptor)
    importer.pin(descriptor)
    with pytest.raises(IpcLifetimeError, match="GPU flights in progress"):
        importer.close(descriptor)
    importer.end_flight(descriptor)
    with pytest.raises(IpcLifetimeError, match="graph pins"):
        importer.close(descriptor)
    importer.unpin(descriptor)
    importer.close(descriptor)
    assert importer.opens == 0
    with pytest.raises(KeyError):
        importer.end_flight(descriptor)


def test_unbalanced_flight_and_pin_releases_are_refused(backend: FakeIpcBackend) -> None:
    exporter, importer = _registry(backend), _registry(backend)
    descriptor = exporter.export(_region())
    importer.open(descriptor).close()
    with pytest.raises(IpcLifetimeError, match="no matching GPU flight"):
        importer.end_flight(descriptor)
    with pytest.raises(IpcLifetimeError, match="no matching graph pin"):
        importer.unpin(descriptor)
    importer.close(descriptor)


def test_one_mapping_is_shared_per_handle_and_session(backend: FakeIpcBackend) -> None:
    exporter, importer = _registry(backend), _registry(backend)
    tensor = _region()
    first = exporter.export(tensor, offset=0, nbytes=8)
    second = exporter.export(tensor, offset=8, nbytes=8)
    a, b = importer.open(first), importer.open(second)
    assert importer.opens == 1 and len(backend.opened) == 1
    a.close()
    b.close()
    importer.close(first)
    assert importer.opens == 1  # the second descriptor still references the mapping
    importer.close(second)
    assert importer.opens == 0


def test_close_all_refuses_live_consumers_unless_forced(backend: FakeIpcBackend) -> None:
    exporter, importer = _registry(backend), _registry(backend)
    descriptor = exporter.export(_region())
    view = importer.open(descriptor)
    importer.begin_flight(descriptor)
    with pytest.raises(IpcLifetimeError, match="2 live consumers across 1 mapping"):
        importer.close_all()
    assert importer.opens == 1
    importer.close_all(force=True)
    assert importer.opens == 0
    assert view.tensor.numel() == 64  # the caller's own reference stays valid
    view.close()


# ---------------------------------------------------------------------------
# Identity: stale sessions and foreign devices are rejected before mapping
# ---------------------------------------------------------------------------


def test_stale_session_generation_is_rejected_before_mapping(backend: FakeIpcBackend) -> None:
    exporter = _registry(backend, session_generation=(1, 1))
    descriptor = exporter.export(_region())
    assert descriptor.session_generation == (1, 1)
    importer = _registry(backend, session_generation=(1, 2))
    with pytest.raises(IpcGenerationError, match="belongs to session"):
        importer.open(descriptor)
    assert backend.opened == [] and importer.opens == 0
    with _registry(backend, session_generation=(1, 1)).open(descriptor):
        pass


def test_foreign_physical_device_is_rejected_before_mapping(backend: FakeIpcBackend) -> None:
    exporter = _registry(backend)
    exported = exporter.export(_region())
    foreign = IpcRegionDescriptor(
        handle=exported.handle,
        offset=0,
        nbytes=exported.nbytes,
        allocation_nbytes=exported.allocation_nbytes,
        physical_device_uuid="GPU-a",
    )
    importer = _registry(backend, physical_device_uuid="GPU-b")
    with pytest.raises(IpcGenerationError, match="physical device"):
        importer.open(foreign)
    assert backend.opened == []


def test_wrong_handle_size_and_short_mapping_are_rejected(backend: FakeIpcBackend) -> None:
    importer = _registry(backend)
    short_handle = IpcRegionDescriptor(handle=b"abc", offset=0, nbytes=4, allocation_nbytes=4)
    with pytest.raises(ValueError, match="handle size mismatch"):
        importer.open(short_handle)
    exported = _registry(backend).export(_region(16))
    overstated = IpcRegionDescriptor(
        handle=exported.handle, offset=0, nbytes=16, allocation_nbytes=1024
    )
    with pytest.raises(IpcGenerationError, match="covers 16 bytes"):
        importer.open(overstated)
    assert importer.opens == 0


def test_descriptor_round_trips_through_the_wire_form(backend: FakeIpcBackend) -> None:
    descriptor = _registry(backend, session_generation=(3, 4)).export(_region(), offset=4)
    assert IpcRegionDescriptor.from_dict(descriptor.to_dict()) == descriptor
    tampered = descriptor.to_dict() | {"schema_version": 2}
    with pytest.raises(ValueError, match="schema_version"):
        IpcRegionDescriptor.from_dict(tampered)
