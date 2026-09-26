"""Legacy CUDA IPC region descriptor and registry.

The registry owns the bytes-and-lifetime half of the IPC contract:

* exporters hold a strong reference to the exported allocation and keep it
  alive until every descriptor is released, every local hold/graph pin is
  retired and every expected peer has acknowledged the close;
* importers cache one native mapping per ``(handle, device, session)`` and
  only drop it once no descriptor, view, flight or pin references it;
* session generation and physical device identity are validated before a
  pointer is dereferenced (INV-6), and a stale generation is rejected rather
  than silently re-used (INV-8).

Importing this module is torch-free and side-effect-free: capability probing
and the native JIT build live behind the injectable :class:`IpcBackend`, whose
default implementation lazily imports :mod:`ayaka.kernel.ipc`.
"""

from __future__ import annotations

import base64
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.validation import require_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "IPC_SCHEMA_VERSION",
    "IpcBackend",
    "IpcGenerationError",
    "IpcLifetimeError",
    "IpcRegionDescriptor",
    "IpcRegionRegistry",
    "IpcRegionView",
]

logger = logging.getLogger(__name__)

#: Descriptor schema accepted by this implementation. A different value is
#: rejected instead of guessed at.
IPC_SCHEMA_VERSION = 1

#: Python ints are unbounded; these bound what may cross the native boundary.
_MAX_IPC_BYTES = (1 << 63) - 1

_CAPABILITY = "cuda_ipc"
_CAPABILITY_REMEDY = (
    "run on Linux with a CUDA runtime matching torch.version.cuda, or leave "
    "the collective policy on torch"
)


class IpcLifetimeError(RuntimeError):
    """A close/release was refused because a consumer or peer ACK is outstanding."""


class IpcGenerationError(RuntimeError):
    """A descriptor or mapping belongs to a stale/foreign session or device."""


# ---------------------------------------------------------------------------
# Backend protocol and default lazy adapter
# ---------------------------------------------------------------------------


@runtime_checkable
class IpcBackend(Protocol):
    """The six native primitives the registry depends on."""

    def ipc_available(self) -> bool: ...

    def legacy_ipc_capable(self, tensor: Any) -> bool: ...

    def export_allocation(self, tensor: Any) -> tuple[bytes, int, int, int]: ...

    def open_allocation(self, handle: bytes, allocation_nbytes: int, device: int) -> Any: ...

    def byte_view(self, tensor: Any, nbytes: int) -> Any: ...

    def ipc_handle_size(self) -> int: ...


class _KernelIpcBackend:
    """Lazy proxy to :mod:`ayaka.kernel.ipc`; imports stay torch-free."""

    @staticmethod
    def _module() -> Any:
        from ayaka.kernel import ipc as kernel_ipc

        return kernel_ipc

    def ipc_available(self) -> bool:
        return bool(self._module().ipc_available())

    def legacy_ipc_capable(self, tensor: Any) -> bool:
        return bool(self._module().legacy_ipc_capable(tensor))

    def export_allocation(self, tensor: Any) -> tuple[bytes, int, int, int]:
        return self._module().export_allocation(tensor)

    def open_allocation(self, handle: bytes, allocation_nbytes: int, device: int) -> Any:
        return self._module().open_allocation(handle, allocation_nbytes, device)

    def byte_view(self, tensor: Any, nbytes: int) -> Any:
        return self._module().byte_view(tensor, nbytes)

    def ipc_handle_size(self) -> int:
        return int(self._module().ipc_handle_size())


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free hosts
        return None
    return torch


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def _require_plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class IpcRegionDescriptor:
    """One byte range inside an exported allocation (schema v1).

    ``handle`` is opaque; ``offset``/``nbytes`` locate the region inside the
    exporter's allocation of ``allocation_nbytes``. ``device`` and
    ``exporter_rank`` are informational (an ordinal only means something in
    the exporting process); ``physical_device_uuid`` and
    ``session_generation`` are the binding identities the importer checks
    before dereferencing anything.
    """

    handle: bytes
    offset: int
    nbytes: int
    allocation_nbytes: int
    device: int = 0
    exporter_rank: int = 0
    schema_version: int = IPC_SCHEMA_VERSION
    session_generation: tuple[int, int] | None = None
    physical_device_uuid: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.handle, (bytearray, memoryview)):
            object.__setattr__(self, "handle", bytes(self.handle))
        if not isinstance(self.handle, bytes):
            raise TypeError("IPC region handle must be bytes")
        if not self.handle:
            raise ValueError("IPC region handle must not be empty")

        if self.schema_version != IPC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported IPC descriptor schema_version {self.schema_version!r}; "
                f"expected {IPC_SCHEMA_VERSION}"
            )

        offset = _require_plain_int(self.offset, "IPC region offset")
        if offset < 0:
            raise ValueError("IPC region offset must be non-negative")
        nbytes = _require_plain_int(self.nbytes, "IPC region nbytes")
        if nbytes <= 0:
            raise ValueError("IPC region nbytes must be positive")
        allocation = _require_plain_int(self.allocation_nbytes, "IPC region allocation_nbytes")
        if allocation < 0:
            raise ValueError("IPC region allocation_nbytes must be non-negative")
        if max(offset, nbytes, allocation) > _MAX_IPC_BYTES:
            raise ValueError("IPC region byte range overflows the native int64 range")
        if offset + nbytes > allocation:
            raise ValueError(
                "IPC region [offset, offset+nbytes) lies outside its allocation extent"
            )

        _require_plain_int(self.device, "IPC region device")
        if self.device < 0:
            raise ValueError("IPC region device must be non-negative")
        _require_plain_int(self.exporter_rank, "IPC region exporter_rank")
        if self.exporter_rank < 0:
            raise ValueError("IPC region exporter_rank must be non-negative")

        if self.session_generation is not None:
            generation = self.session_generation
            if (
                type(generation) is not tuple
                or len(generation) != 2
                or any(type(part) is not int or part <= 0 for part in generation)
            ):
                raise ValueError(
                    "IPC region session_generation must be a "
                    "(owner_incarnation, worker_generation) pair of positive integers"
                )
        if self.physical_device_uuid:
            require_text(self.physical_device_uuid, "IPC region physical_device_uuid")

    @property
    def end(self) -> int:
        """One-past-the-last exported byte."""
        return self.offset + self.nbytes

    def to_dict(self) -> dict[str, Any]:
        """Wire form for the control channel (JSON-compatible, no pointers)."""
        return {
            "schema_version": self.schema_version,
            "handle": base64.b64encode(self.handle).decode("ascii"),
            "offset": self.offset,
            "nbytes": self.nbytes,
            "allocation_nbytes": self.allocation_nbytes,
            "device": self.device,
            "exporter_rank": self.exporter_rank,
            "session_generation": (
                list(self.session_generation) if self.session_generation is not None else None
            ),
            "physical_device_uuid": self.physical_device_uuid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IpcRegionDescriptor:
        """Rebuild a descriptor from :meth:`to_dict`, validating every field."""
        if not isinstance(payload, Mapping):
            raise TypeError("IPC descriptor payload must be a mapping")
        try:
            encoded = payload["handle"]
        except KeyError as exc:
            raise ValueError("IPC descriptor payload is missing 'handle'") from exc
        if not isinstance(encoded, str):
            raise TypeError("IPC descriptor handle must be a base64 string")
        try:
            handle = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("IPC descriptor handle is not valid base64") from exc
        generation = payload.get("session_generation")
        if generation is not None:
            if not isinstance(generation, (list, tuple)) or len(generation) != 2:
                raise ValueError("IPC descriptor session_generation must be a two-element sequence")
            generation = (generation[0], generation[1])
        return cls(
            handle=handle,
            offset=payload.get("offset", 0),
            nbytes=payload.get("nbytes", 0),
            allocation_nbytes=payload.get("allocation_nbytes", 0),
            device=payload.get("device", 0),
            exporter_rank=payload.get("exporter_rank", 0),
            schema_version=payload.get("schema_version", 0),
            session_generation=generation,
            physical_device_uuid=payload.get("physical_device_uuid", ""),
        )


# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ExportedRegion:
    handle: bytes
    allocation_nbytes: int
    device: int
    owner: Any
    descriptors: list[IpcRegionDescriptor] = field(default_factory=list)
    pending_peers: set[int] = field(default_factory=set)
    acked_peers: set[int] = field(default_factory=set)
    holds: int = 0
    pinned: int = 0
    closing: bool = False

    @property
    def outstanding_peers(self) -> set[int]:
        return self.pending_peers - self.acked_peers


@dataclass(slots=True)
class _OpenedRegion:
    handle: bytes
    allocation_nbytes: int
    device: int
    session_generation: tuple[int, int] | None
    mapping: Any
    descriptors: list[IpcRegionDescriptor] = field(default_factory=list)
    views: int = 0
    flights: int = 0
    pinned: int = 0
    closing: bool = False


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------


class IpcRegionView:
    """A live byte view over an imported IPC region.

    Closing the view is idempotent; the registry drops the native mapping
    only after the last view, descriptor, flight and pin are gone.
    """

    def __init__(
        self,
        registry: IpcRegionRegistry,
        key: tuple[bytes, int, tuple[int, int] | None],
        tensor: Any,
    ) -> None:
        self._registry = registry
        self._key = key
        self._tensor = tensor
        self._closed = False

    @property
    def tensor(self) -> Any:
        """The narrowed ``uint8`` tensor. Valid until :meth:`close`."""
        if self._closed:
            raise IpcLifetimeError("IPC region view is already closed")
        return self._tensor

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tensor, self._tensor = self._tensor, None
        self._registry._release_view(self._key, tensor)

    def __enter__(self) -> IpcRegionView:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class IpcRegionRegistry:
    """Own exports, import mappings and their refcounts for one process.

    Args:
        device: default local CUDA ordinal used when ``open`` is not given an
            explicit device.
        session_generation: when set, every export/open is bound to this
            ``(owner_incarnation, worker_generation)`` and a descriptor from
            another session is rejected.
        physical_device_uuid: when set, an importer refuses a descriptor whose
            exporter UUID differs (a different physical GPU).
        backend: native primitive adapter; defaults to the lazy kernel module
            and is injectable so registry semantics are testable on CPU.
    """

    def __init__(
        self,
        *,
        device: int | None = None,
        session_generation: tuple[int, int] | None = None,
        physical_device_uuid: str | None = None,
        backend: IpcBackend | None = None,
    ) -> None:
        if device is not None and (type(device) is not int or device < 0):
            raise ValueError("registry device must be a non-negative integer")
        if session_generation is not None:
            if (
                type(session_generation) is not tuple
                or len(session_generation) != 2
                or any(type(part) is not int or part <= 0 for part in session_generation)
            ):
                raise ValueError(
                    "session_generation must be an (owner_incarnation, "
                    "worker_generation) pair of positive integers"
                )
        if physical_device_uuid is not None:
            require_text(physical_device_uuid, "physical_device_uuid")
        self._backend: IpcBackend = backend if backend is not None else _KernelIpcBackend()
        self._device = device
        self._session_generation = session_generation
        self._physical_device_uuid = physical_device_uuid
        self._exports: dict[bytes, _ExportedRegion] = {}
        self._imports: dict[tuple[bytes, int, tuple[int, int] | None], _OpenedRegion] = {}
        self._lock = threading.RLock()

    # -- introspection -----------------------------------------------------

    @property
    def exports(self) -> int:
        """Number of live exported allocations (not descriptors)."""
        with self._lock:
            return len(self._exports)

    @property
    def opens(self) -> int:
        """Number of live imported mappings."""
        with self._lock:
            return len(self._imports)

    @property
    def session_generation(self) -> tuple[int, int] | None:
        return self._session_generation

    # -- exporter side -----------------------------------------------------

    def export(
        self,
        tensor: torch.Tensor,
        *,
        offset: int = 0,
        nbytes: int | None = None,
        close_acks: Iterable[int] = (),
    ) -> IpcRegionDescriptor:
        """Export the dense byte region ``[offset, offset+nbytes)`` of ``tensor``.

        The allocation is exported once: several descriptors on the same
        allocation share one handle and one owner reference. ``close_acks``
        names the peer ranks whose close acknowledgement must arrive before
        :meth:`release` is allowed to drop the allocation.
        """
        offset = _require_plain_int(offset, "export offset")
        if offset < 0:
            raise ValueError("export offset must be non-negative")
        torch = _optional_torch()
        if torch is None or not isinstance(tensor, torch.Tensor):
            raise TypeError("IPC export requires a torch.Tensor")
        tensor_bytes = self._dense_byte_length(tensor)
        if offset >= tensor_bytes:
            raise ValueError(f"export offset {offset} is outside the tensor's {tensor_bytes} bytes")
        if nbytes is None:
            nbytes = tensor_bytes - offset
        else:
            nbytes = _require_plain_int(nbytes, "export nbytes")
            if nbytes <= 0:
                raise ValueError("export nbytes must be positive")
            if offset + nbytes > tensor_bytes:
                raise ValueError(
                    f"export region [{offset}, {offset + nbytes}) is outside the "
                    f"tensor's {tensor_bytes} bytes"
                )

        module = self._require_export_capability(tensor)
        handle, allocation_offset, allocation_nbytes, device = module.export_allocation(tensor)
        handle = bytes(handle)
        peers = {_require_plain_int(rank, "close-ACK rank") for rank in close_acks}
        if any(rank < 0 for rank in peers):
            raise ValueError("close-ACK ranks must be non-negative")

        with self._lock:
            entry = self._exports.get(handle)
            if entry is None:
                entry = _ExportedRegion(
                    handle=handle,
                    allocation_nbytes=int(allocation_nbytes),
                    device=int(device),
                    owner=tensor,
                    pending_peers=peers,
                )
                self._exports[handle] = entry
            else:
                if int(allocation_nbytes) != entry.allocation_nbytes:
                    raise IpcGenerationError(
                        "IPC handle re-exported with a different allocation extent"
                    )
                if int(device) != entry.device:
                    raise IpcGenerationError("IPC handle re-exported from a different device")
                entry.pending_peers |= peers
            descriptor = IpcRegionDescriptor(
                handle=handle,
                offset=int(allocation_offset) + offset,
                nbytes=nbytes,
                allocation_nbytes=int(allocation_nbytes),
                device=int(device),
                exporter_rank=0,
                session_generation=self._session_generation,
            )
            entry.descriptors.append(descriptor)
            return descriptor

    def release(self, descriptor: IpcRegionDescriptor) -> None:
        """Drop one export descriptor.

        Refused (and deferred) while local holds/pins remain, while importer
        views/flights are still live, or while a declared peer has not sent
        its close ACK. The allocation is retained in those cases so a peer
        can never observe recycled memory.
        """
        with self._lock:
            entry = self._find_export(descriptor)
            if entry.closing:
                raise IpcLifetimeError("export region is already being released")
            entry.descriptors.remove(descriptor)
            if entry.descriptors:
                return
            entry.closing = True
            self._try_finalize_export(entry)

    def confirm_close(self, descriptor: IpcRegionDescriptor, rank: int) -> None:
        """Record the peer close ACK for ``rank`` and finalize when quiescent."""
        rank = _require_plain_int(rank, "close-ACK rank")
        if rank < 0:
            raise ValueError("close-ACK rank must be non-negative")
        with self._lock:
            entry = self._find_export_entry(descriptor)
            entry.acked_peers.add(rank)
            self._try_finalize_export(entry)

    def hold(self, descriptor: IpcRegionDescriptor) -> None:
        """Take a local exporter lease so release cannot drop the allocation."""
        with self._lock:
            self._find_export(descriptor).holds += 1

    def unhold(self, descriptor: IpcRegionDescriptor) -> None:
        with self._lock:
            entry = self._find_export_entry(descriptor)
            if entry.holds <= 0:
                raise IpcLifetimeError("no matching export hold to release")
            entry.holds -= 1
            self._try_finalize_export(entry)

    def _try_finalize_export(self, entry: _ExportedRegion) -> None:
        if not entry.closing or entry.descriptors:
            return
        blockers: list[str] = []
        if entry.holds:
            blockers.append(f"{entry.holds} local holds")
        if entry.pinned:
            blockers.append(f"{entry.pinned} graph pins")
        outstanding = entry.outstanding_peers
        if outstanding:
            blockers.append(f"close-ACK missing from ranks {sorted(outstanding)}")
        if blockers:
            raise IpcLifetimeError(
                "exporter release refused: " + "; ".join(blockers) + "; allocation retained"
            )
        self._exports.pop(entry.handle, None)
        entry.owner = None

    # -- importer side -----------------------------------------------------

    def open(
        self,
        descriptor: IpcRegionDescriptor,
        *,
        device: int | None = None,
        nbytes: int | None = None,
    ) -> IpcRegionView:
        """Import ``descriptor`` and return a byte view over its region.

        One native mapping is shared per ``(handle, device, session)``;
        validation of session generation, physical device identity and handle
        size happens before the mapping is touched.
        """
        if not isinstance(descriptor, IpcRegionDescriptor):
            raise TypeError("open requires an IpcRegionDescriptor")
        module = self._require_available()
        expected_handle = int(module.ipc_handle_size())
        if len(descriptor.handle) != expected_handle:
            raise ValueError(
                f"IPC handle size mismatch: descriptor has {len(descriptor.handle)} "
                f"bytes, runtime expects {expected_handle}"
            )
        active = self._session_generation
        if active is not None and descriptor.session_generation != active:
            raise IpcGenerationError(
                "IPC descriptor belongs to session "
                f"{descriptor.session_generation!r}, registry is bound to {active!r}"
            )
        if (
            self._physical_device_uuid is not None
            and descriptor.physical_device_uuid
            and descriptor.physical_device_uuid != self._physical_device_uuid
        ):
            raise IpcGenerationError(
                "IPC descriptor was exported from physical device "
                f"{descriptor.physical_device_uuid!r}, registry expects "
                f"{self._physical_device_uuid!r}"
            )
        local_device = self._resolve_device(device, descriptor)
        if nbytes is None:
            view_bytes = descriptor.nbytes
        else:
            view_bytes = _require_plain_int(nbytes, "open nbytes")
            if not 0 < view_bytes <= descriptor.nbytes:
                raise ValueError(
                    f"open nbytes {view_bytes} is outside the descriptor's "
                    f"{descriptor.nbytes} bytes"
                )

        key = (descriptor.handle, local_device, descriptor.session_generation)
        with self._lock:
            entry = self._imports.get(key)
            if entry is None:
                mapping = module.open_allocation(
                    descriptor.handle, descriptor.allocation_nbytes, local_device
                )
                available = int(mapping.numel()) * int(mapping.element_size())
                if available < descriptor.allocation_nbytes:
                    raise IpcGenerationError(
                        f"imported mapping covers {available} bytes but the descriptor "
                        f"declares {descriptor.allocation_nbytes}"
                    )
                entry = _OpenedRegion(
                    handle=descriptor.handle,
                    allocation_nbytes=descriptor.allocation_nbytes,
                    device=local_device,
                    session_generation=descriptor.session_generation,
                    mapping=mapping,
                )
                self._imports[key] = entry
            entry.descriptors.append(descriptor)
            entry.views += 1
            view_tensor = entry.mapping.narrow(0, descriptor.offset, view_bytes)
        return IpcRegionView(self, key, view_tensor)

    def close(self, descriptor: IpcRegionDescriptor) -> None:
        """Drop one imported descriptor; refused while consumers are live."""
        with self._lock:
            key, entry = self._find_import(descriptor)
            if entry.views:
                raise IpcLifetimeError(
                    f"importer close refused: {entry.views} live views; mapping retained"
                )
            if entry.flights:
                raise IpcLifetimeError(
                    f"importer close refused: {entry.flights} GPU flights in progress; "
                    "mapping retained"
                )
            if entry.pinned:
                raise IpcLifetimeError(
                    f"importer close refused: {entry.pinned} graph pins; mapping retained"
                )
            entry.descriptors.remove(descriptor)
            if entry.descriptors:
                return
            entry.closing = True
            self._try_finalize_import(key, entry)

    def begin_flight(self, descriptor: IpcRegionDescriptor) -> None:
        """Mark GPU work reading/writing the imported region as in flight."""
        with self._lock:
            _, entry = self._find_import(descriptor)
            entry.flights += 1

    def end_flight(self, descriptor: IpcRegionDescriptor) -> None:
        with self._lock:
            key, entry = self._find_import(descriptor)
            if entry.flights <= 0:
                raise IpcLifetimeError("no matching GPU flight to end")
            entry.flights -= 1
            self._try_finalize_import(key, entry)

    def pin(self, descriptor: IpcRegionDescriptor) -> None:
        """Pin an imported mapping (graph/workspace identity) against close."""
        with self._lock:
            _, entry = self._find_import(descriptor)
            entry.pinned += 1

    def unpin(self, descriptor: IpcRegionDescriptor) -> None:
        with self._lock:
            key, entry = self._find_import(descriptor)
            if entry.pinned <= 0:
                raise IpcLifetimeError("no matching graph pin to release")
            entry.pinned -= 1
            self._try_finalize_import(key, entry)

    def close_all(self, *, force: bool = False) -> None:
        """Drop all exports, and all imports without live consumers.

        ``force=True`` drops mappings even with live views/flights/pins; the
        storage refcount still keeps a live view's memory valid until the
        view itself is released.
        """
        with self._lock:
            blocked = [
                entry.views + entry.flights + entry.pinned
                for entry in self._imports.values()
                if entry.views + entry.flights + entry.pinned
            ]
            if blocked and not force:
                raise IpcLifetimeError(
                    f"close_all refused: {sum(blocked)} live consumers across "
                    f"{len(blocked)} mapping(s); pass force=True to abandon them"
                )
            for entry in list(self._exports.values()):
                entry.descriptors.clear()
                entry.owner = None
            self._exports.clear()
            for key, entry in list(self._imports.items()):
                self._imports.pop(key, None)
                entry.mapping = None

    # -- internals ---------------------------------------------------------

    def _require_available(self) -> IpcBackend:
        module = self._backend
        if not module.ipc_available():
            raise CapabilityError(
                _CAPABILITY,
                detail="legacy CUDA IPC is not available on this host/tensor",
                remedy=_CAPABILITY_REMEDY,
            )
        return module

    def _require_export_capability(self, tensor: Any) -> IpcBackend:
        module = self._backend
        # The tensor probe is fail-closed and never builds for CPU/VMM inputs,
        # so a rejected export stays cheap while a real CUDA tensor triggers
        # the lazy JIT path.
        if not module.legacy_ipc_capable(tensor):
            raise CapabilityError(
                _CAPABILITY,
                detail=(
                    "tensor does not sit in an exportable dense CUDA allocation "
                    "(VMM-backed, strided, expanded or on CPU)"
                ),
                remedy=_CAPABILITY_REMEDY,
            )
        if not module.ipc_available():
            raise CapabilityError(
                _CAPABILITY,
                detail="legacy CUDA IPC is not available on this host",
                remedy=_CAPABILITY_REMEDY,
            )
        return module

    @staticmethod
    def _dense_byte_length(tensor: Any) -> int:
        if tensor.numel() == 0:
            raise ValueError("IPC export requires a non-empty region")
        if not tensor.is_contiguous():
            raise ValueError("IPC export requires a dense contiguous tensor")
        if tensor.storage_offset() < 0:
            raise ValueError("IPC export tensor has a negative storage offset")
        length = int(tensor.numel()) * int(tensor.element_size())
        if length <= 0 or length > _MAX_IPC_BYTES:
            raise ValueError("IPC export byte length is out of range")
        return length

    def _resolve_device(self, device: int | None, descriptor: IpcRegionDescriptor) -> int:
        resolved = device if device is not None else self._device
        if resolved is None:
            resolved = descriptor.device
        resolved = _require_plain_int(resolved, "import device")
        if resolved < 0:
            raise ValueError("import device must be non-negative")
        return resolved

    def _find_export(self, descriptor: IpcRegionDescriptor) -> _ExportedRegion:
        if not isinstance(descriptor, IpcRegionDescriptor):
            raise TypeError("expected an IpcRegionDescriptor")
        entry = self._exports.get(descriptor.handle)
        if entry is None or descriptor not in entry.descriptors:
            raise KeyError(f"IPC region {descriptor.handle!r} is not exported")
        return entry

    def _find_export_entry(self, descriptor: IpcRegionDescriptor) -> _ExportedRegion:
        """Look up a deferred-close entry by handle (descriptor already retired)."""
        if not isinstance(descriptor, IpcRegionDescriptor):
            raise TypeError("expected an IpcRegionDescriptor")
        entry = self._exports.get(descriptor.handle)
        if entry is None:
            raise KeyError(f"IPC region {descriptor.handle!r} is not exported")
        return entry

    def _find_import(
        self, descriptor: IpcRegionDescriptor
    ) -> tuple[tuple[bytes, int, tuple[int, int] | None], _OpenedRegion]:
        if not isinstance(descriptor, IpcRegionDescriptor):
            raise TypeError("expected an IpcRegionDescriptor")
        for key, entry in self._imports.items():
            if descriptor in entry.descriptors:
                return key, entry
        raise KeyError(f"IPC region {descriptor.handle!r} is not opened")

    def _release_view(
        self,
        key: tuple[bytes, int, tuple[int, int] | None],
        tensor: Any,
    ) -> None:
        del tensor  # dropping the caller's view ref; mapping keeps its own
        with self._lock:
            entry = self._imports.get(key)
            if entry is None:
                return
            entry.views = max(0, entry.views - 1)
            self._try_finalize_import(key, entry)

    def _try_finalize_import(
        self,
        key: tuple[bytes, int, tuple[int, int] | None],
        entry: _OpenedRegion,
    ) -> None:
        if not entry.closing or entry.descriptors or entry.views or entry.flights:
            return
        if entry.pinned:
            return
        self._imports.pop(key, None)
        entry.mapping = None
