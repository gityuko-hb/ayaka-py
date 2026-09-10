from __future__ import annotations

from dataclasses import dataclass

from ayaka.configs.hardware import HardwareConfig
from ayaka.distributed.device import DeviceRef, LinkKind

_LINK_COST: dict[LinkKind, int] = {
    LinkKind.SELF: 0,
    LinkKind.NVLINK: 1,
    LinkKind.NVSWITCH: 1,
    LinkKind.PCIE: 2,
    LinkKind.QPI: 3,
    LinkKind.NETWORK: 4,
}


@dataclass(frozen=True, slots=True)
class TopologyLink:
    source: DeviceRef
    destination: DeviceRef
    kind: LinkKind
    peer_access: bool | None
    """True/False when probed; None means the transport did not prove it."""
    same_numa: bool | None
    """None when either endpoint has no NUMA association."""

    @property
    def is_local(self) -> bool:
        return self.kind is LinkKind.SELF

    @property
    def cost(self) -> tuple[int, int, int]:
        """Deterministic ordinal cost, never a fabricated bandwidth number."""
        cross_numa = 1 if self.same_numa is False else 0
        unknown_peer = 1 if self.peer_access is None else 0
        return (_LINK_COST[self.kind], cross_numa, unknown_peer)


@dataclass(frozen=True, slots=True)
class DeviceTopology:
    devices: tuple[DeviceRef, ...]
    links: tuple[tuple[TopologyLink, ...], ...]

    def __post_init__(self) -> None:
        size = len(self.devices)
        if len(set(self.devices)) != size:
            raise ValueError("topology devices must be unique")
        if len(self.links) != size or any(len(row) != size for row in self.links):
            raise ValueError("topology links must be square over devices")
        for src in range(size):
            for dst in range(size):
                link = self.links[src][dst]
                if link.source != self.devices[src] or link.destination != self.devices[dst]:
                    raise ValueError("topology link endpoints disagree with matrix position")
                if src == dst and link.kind is not LinkKind.SELF:
                    raise ValueError("topology diagonal must be SELF")
                if src == dst and link.peer_access is not True:
                    raise ValueError("a topology self-link must have peer access")
                reverse = self.links[dst][src]
                if link.kind is not reverse.kind:
                    raise ValueError("topology link kinds must be symmetric")
                if link.peer_access is not reverse.peer_access:
                    raise ValueError("topology peer-access facts must be symmetric")
                if link.same_numa is not reverse.same_numa:
                    raise ValueError("topology NUMA facts must be symmetric")

    @classmethod
    def from_hardware(cls, hardware: HardwareConfig) -> DeviceTopology:
        devices = hardware.devices
        if len(devices) > 1 and not hardware.links:
            raise ValueError("multi-device topology requires an explicit link matrix")
        matrix: list[list[TopologyLink]] = []
        for src, source in enumerate(devices):
            row: list[TopologyLink] = []
            source_numa = hardware.capabilities[src].numa_node
            for dst, destination in enumerate(devices):
                destination_numa = hardware.capabilities[dst].numa_node
                kind = hardware.link(src, dst)
                same_numa = (
                    True
                    if src == dst
                    else (
                        source_numa == destination_numa
                        if source_numa >= 0 and destination_numa >= 0
                        else None
                    )
                )
                peer_access = (
                    hardware.peer_access[src][dst]
                    if hardware.peer_access
                    else (True if src == dst else None)
                )
                row.append(
                    TopologyLink(
                        source=source,
                        destination=destination,
                        kind=kind,
                        peer_access=peer_access,
                        same_numa=same_numa,
                    )
                )
            matrix.append(row)
        return cls(devices=devices, links=tuple(tuple(row) for row in matrix))

    @property
    def size(self) -> int:
        return len(self.devices)

    def link(self, source: int, destination: int) -> TopologyLink:
        return self.links[source][destination]

    def ranked_peers(self, source: int) -> tuple[DeviceRef, ...]:
        """Other devices ordered by measured path class, then device index."""
        if not 0 <= source < self.size:
            raise IndexError(f"source {source} outside topology size {self.size}")
        peers = [index for index in range(self.size) if index != source]
        peers.sort(key=lambda index: (*self.link(source, index).cost, index))
        return tuple(self.devices[index] for index in peers)

    def pairwise_links(self, ordinals: tuple[int, ...]) -> tuple[TopologyLink, ...]:
        return tuple(
            self.link(ordinals[left], ordinals[right])
            for left in range(len(ordinals))
            for right in range(left + 1, len(ordinals))
        )
