from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ayaka.utils.import_utils import CapabilityError


def _optional_torch() -> Any | None:
    """Return torch when importable, else None; multi-GPU code stays testable."""
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised on torch-free hosts
        return None
    return torch


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


@dataclass(frozen=True, slots=True)
class PeerTopology:
    """Measured peer-access map over a set of devices."""

    devices: tuple[str, ...]
    peer_capable: tuple[tuple[bool, ...], ...]
    """``peer_capable[i][j]`` is True when rank i can directly access rank j."""

    def __post_init__(self) -> None:
        size = len(self.devices)
        if len(self.peer_capable) != size or any(len(row) != size for row in self.peer_capable):
            raise ValueError("peer_capable must be a square matrix over devices")

    @property
    def world_size(self) -> int:
        return len(self.devices)

    @property
    def peer_available(self) -> bool:
        """Whether any ordered pair of distinct ranks can peer-access."""
        return any(
            self.peer_capable[i][j]
            for i in range(self.world_size)
            for j in range(self.world_size)
            if i != j
        )

    def can_access(self, source: int, destination: int) -> bool:
        """Whether ``source`` may directly read/write ``destination`` memory."""
        return self.peer_capable[source][destination]

    @classmethod
    def probe(cls, devices: Sequence[str]) -> PeerTopology:
        """Measure peer access for every ordered pair of ``devices``."""
        resolved = tuple(str(device) for device in devices)
        torch = _optional_torch()
        size = len(resolved)
        if torch is None or not torch.cuda.is_available():
            return cls(devices=resolved, peer_capable=tuple((False,) * size for _ in range(size)))

        def index_of(device: str) -> int:
            parsed = torch.device(device)
            return 0 if parsed.index is None else int(parsed.index)

        rows: list[tuple[bool, ...]] = []
        for i in range(size):
            row: list[bool] = []
            for j in range(size):
                if i == j or not (_is_cuda_device(resolved[i]) and _is_cuda_device(resolved[j])):
                    row.append(False)
                    continue
                row.append(
                    bool(
                        torch.cuda.can_device_access_peer(
                            index_of(resolved[i]), index_of(resolved[j])
                        )
                    )
                )
            rows.append(tuple(row))
        return cls(devices=resolved, peer_capable=tuple(rows))


@dataclass(frozen=True, slots=True)
class PhysicalDeviceIdentity:
    """Cross-process identity of one CUDA device.

    ``ordinal`` is a local index that only means something inside the process
    that produced it. ``uuid``/``pci_bus_id`` are the fields safe to compare
    between processes; an identity that has neither (``trustworthy`` is False)
    must never be used to claim two processes see the same physical GPU.
    """

    ordinal: int
    uuid: str = ""
    pci_bus_id: str = ""
    name: str = ""
    sm: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("device ordinal must be a non-negative integer")
        for label, value in (
            ("uuid", self.uuid),
            ("pci_bus_id", self.pci_bus_id),
            ("name", self.name),
        ):
            if not isinstance(value, str):
                raise TypeError(f"device {label} must be a string")
        if self.sm is not None:
            if (
                type(self.sm) is not tuple
                or len(self.sm) != 2
                or any(type(part) is not int or part < 0 for part in self.sm)
            ):
                raise ValueError("device sm must be a (major, minor) pair of non-negative ints")

    @property
    def identity_key(self) -> str:
        """Cross-process key: UUID first, PCI address second, else empty."""
        return self.uuid or self.pci_bus_id

    @property
    def trustworthy(self) -> bool:
        """Whether this identity can identify a physical GPU across processes."""
        return bool(self.identity_key)

    def same_physical_device(self, other: PhysicalDeviceIdentity) -> bool:
        """Whether two identities name the same physical GPU.

        UUID and PCI address are compared independently; a matching local
        ordinal is never sufficient, and an ordinal-only identity answers
        False because nothing can be proven.
        """
        if isinstance(other, Mapping):  # tolerate wire-form dicts in comparisons
            other = PhysicalDeviceIdentity.from_dict(other)
        if self.uuid and other.uuid:
            return self.uuid == other.uuid
        if self.pci_bus_id and other.pci_bus_id:
            return self.pci_bus_id == other.pci_bus_id
        return False

    def to_dict(self) -> dict[str, Any]:
        """Wire form for child results and cache records."""
        return {
            "ordinal": self.ordinal,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "name": self.name,
            "sm": list(self.sm) if self.sm is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhysicalDeviceIdentity:
        if not isinstance(payload, Mapping):
            raise TypeError("device identity payload must be a mapping")
        sm = payload.get("sm")
        if isinstance(sm, (list, tuple)):
            sm = (sm[0], sm[1])
        elif sm is not None:
            raise ValueError("device identity sm must be a two-element sequence or null")
        return cls(
            ordinal=payload.get("ordinal", 0),
            uuid=payload.get("uuid", ""),
            pci_bus_id=payload.get("pci_bus_id", ""),
            name=payload.get("name", ""),
            sm=sm,
        )


def _format_pci_bus_id(properties: Any) -> str:
    """Render ``domain:bus:device.0`` from torch device properties.

    Torch exposes the three integers but no function field; NVIDIA GPUs use
    function 0, so this is the same BDF string NVML and ``nvidia-smi`` print.
    """
    domain = getattr(properties, "pci_domain_id", None)
    bus = getattr(properties, "pci_bus_id", None)
    device = getattr(properties, "pci_device_id", None)
    if not all(type(value) is int and value >= 0 for value in (domain, bus, device)):
        return ""
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def physical_device_identity(index: int = 0) -> PhysicalDeviceIdentity:
    """Read the cross-process identity of local CUDA device ``index``.

    Raises:
        CapabilityError: when torch or CUDA is unavailable.
        ValueError: when ``index`` is not a visible device ordinal.
    """
    if type(index) is not int or index < 0:
        raise ValueError("device index must be a non-negative integer")
    torch = _optional_torch()
    if torch is None or not torch.cuda.is_available():
        raise CapabilityError(
            "cuda_identity",
            detail="CUDA is unavailable, so no physical device identity can be read",
            remedy="run on a Linux/CUDA host with a working driver, or stay on the torch policy",
        )
    if index >= torch.cuda.device_count():
        raise ValueError(f"cuda:{index} is outside the {torch.cuda.device_count()} visible devices")
    properties = torch.cuda.get_device_properties(index)
    raw_uuid = getattr(properties, "uuid", None)
    uuid_text = str(raw_uuid).strip() if raw_uuid is not None else ""
    if uuid_text and not uuid_text.startswith("GPU-"):
        uuid_text = f"GPU-{uuid_text}"
    major, minor = torch.cuda.get_device_capability(index)
    return PhysicalDeviceIdentity(
        ordinal=index,
        uuid=uuid_text,
        pci_bus_id=_format_pci_bus_id(properties),
        name=str(getattr(properties, "name", "")),
        sm=(int(major), int(minor)),
    )


def visible_device_identities() -> tuple[PhysicalDeviceIdentity, ...]:
    """Identity of every CUDA device visible to this process, in ordinal order.

    Reflects ``CUDA_VISIBLE_DEVICES`` exactly as CUDA enumerates it; the tuple
    is empty on a torch-free or CUDA-less host instead of raising.
    """
    torch = _optional_torch()
    if torch is None or not torch.cuda.is_available():
        return ()
    return tuple(physical_device_identity(index) for index in range(torch.cuda.device_count()))
