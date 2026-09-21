"""Node-to-node KV transport engine for PD disaggregation.

Extends the tiering ``TransferEngine`` *semantics* (submit → poll → drain,
monotonic tickets, all-or-nothing outcomes, credit flow control) across nodes:

- Same node: the two endpoints share memory — a descriptor is completed by
  reference; the intra-node CUDA peer matrix from
  :class:`~ayaka.distributed.topology.PeerTopology` decides whether a real
  ``PeerPageMover``-style device copy can bypass staging.
- Cross node: :class:`TcpRemoteChannel` frames chunks over sockets behind a
  geometry handshake; :class:`LoopbackRemoteChannel` pairs two engines
  in-process for tests. ``RdmaRemoteChannel`` is the reserved seam.

Chunks arrive one per layer: callers feed
:meth:`ayaka.prefix.transfer.PrefixTransfer.layer_view` bytes so a layer
streams as soon as its fence resolves instead of waiting for the whole
ticket.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count
from threading import RLock
from typing import Protocol, runtime_checkable

from ayaka.distributed.topology import PeerTopology
from ayaka.memory.tiering import TransferState
from ayaka.prefix.transfer import TransferCredits

__all__ = [
    "GeometryMismatchError",
    "LinkKind",
    "LoopbackRemoteChannel",
    "NodeEndpoint",
    "NodeTransferDescriptor",
    "NodeTransferEngine",
    "NodeTransferMetrics",
    "NodeTransferOutcome",
    "NodeTransferTicket",
    "NodeTopology",
    "RemoteChannel",
    "TcpRemoteChannel",
    "build_node_topology",
]

_GEOMETRY_MAGIC = b"ayaka-node-kv-v1"
_HEADER = struct.Struct("<QQ")


class LinkKind:
    """How two endpoints exchange KV bytes."""

    HOST_PEER = "host_peer"
    TCP = "tcp"
    RDMA = "rdma"
    NVLINK = "nvlink"

    _ALL = frozenset({HOST_PEER, TCP, RDMA, NVLINK})

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in cls._ALL:
            raise ValueError(f"unknown link kind {value!r}")
        return value


@dataclass(frozen=True, slots=True)
class NodeEndpoint:
    """Addressable identity of one node in the transfer fabric."""

    node_id: str
    host: str = "127.0.0.1"
    port: int = 0

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if not 0 <= int(self.port) <= 65535:
            raise ValueError("port must be inside 0..65535")


@dataclass(frozen=True, slots=True)
class NodeTopology:
    """Node-scoped view of the transfer fabric; reuses intra-node probing."""

    self_node: str
    endpoints: tuple[NodeEndpoint, ...]
    links: tuple[tuple[tuple[str, str], str], ...]
    """Ordered pair ``(source, destination)`` → link kind override."""
    intra_peer: PeerTopology | None = None
    """CUDA peer matrix inside this node (:meth:`PeerTopology.probe` result)."""

    def __post_init__(self) -> None:
        ids = [endpoint.node_id for endpoint in self.endpoints]
        if self.self_node not in ids:
            raise ValueError("self_node must be one of the endpoints")
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        seen_pairs = {pair for pair, _ in self.links}
        if len(seen_pairs) != len(self.links):
            raise ValueError("link overrides must be unique per pair")
        for (source, destination), kind in self.links:
            if source not in ids or destination not in ids:
                raise ValueError("link override references an unknown node")
            if source == destination:
                raise ValueError("link override must span two distinct nodes")
            LinkKind.validate(kind)

    def endpoint(self, node_id: str) -> NodeEndpoint:
        for endpoint in self.endpoints:
            if endpoint.node_id == node_id:
                return endpoint
        raise KeyError(f"unknown node {node_id!r}")

    def link(self, source: str, destination: str) -> str:
        """The link kind for one ordered pair; intra-node pairs are HOST_PEER."""
        if source == destination:
            return LinkKind.HOST_PEER
        for (pair_source, pair_destination), kind in self.links:
            if (source, destination) == (pair_source, pair_destination):
                return kind
        return LinkKind.TCP

    def peer_capable(self, source_rank: int, destination_rank: int) -> bool:
        """Intra-node CUDA peer capability via the probed matrix."""
        if self.intra_peer is None:
            return False
        return self.intra_peer.can_access(source_rank, destination_rank)


def build_node_topology(
    self_node: str,
    endpoints: Sequence[NodeEndpoint],
    *,
    local_devices: Sequence[str] = (),
    links: Sequence[tuple[tuple[str, str], str]] = (),
) -> NodeTopology:
    """Build a topology; probes intra-node CUDA peer access when devices exist."""
    intra = PeerTopology.probe(tuple(local_devices)) if local_devices else None
    return NodeTopology(
        self_node=self_node,
        endpoints=tuple(endpoints),
        links=tuple(links),
        intra_peer=intra,
    )


@dataclass(frozen=True, slots=True)
class NodeTransferDescriptor:
    """One node-to-node KV move: opaque chunks, one per layer."""

    source_node: str
    destination_node: str
    chunk_sizes: tuple[int, ...]
    token_range: tuple[int, int] = (0, 0)
    layer_count: int = 0

    def __post_init__(self) -> None:
        if any(size < 0 for size in self.chunk_sizes):
            raise ValueError("chunk sizes must be non-negative")
        start, end = self.token_range
        if start < 0 or end < start:
            raise ValueError("token_range must be a non-negative, non-decreasing pair")
        if self.layer_count < 0:
            raise ValueError("layer_count must be non-negative")


@dataclass(frozen=True, slots=True)
class NodeTransferTicket:
    """Opaque identity of one submitted node-to-node transfer."""

    ticket_id: int
    source_node: str
    destination_node: str
    nbytes: int
    chunks: int
    link: str
    issue_epoch: int


@dataclass(frozen=True, slots=True)
class NodeTransferOutcome:
    """Result of one resolved node-to-node transfer."""

    ticket: NodeTransferTicket
    state: TransferState
    error: str | None = None

    def __post_init__(self) -> None:
        if self.state is TransferState.PENDING:
            raise ValueError("a transfer outcome cannot still be pending")
        if (self.error is None) == (self.state is TransferState.FAILED):
            raise ValueError("a failed outcome must carry an error and a success must not")


@dataclass(frozen=True, slots=True)
class NodeTransferMetrics:
    """Cumulative node-transfer accounting; diagnostics only."""

    submitted_total: int
    completed_total: int
    failed_total: int
    bytes_transferred_total: int
    pending: int


class GeometryMismatchError(RuntimeError):
    """The two endpoints disagree on KV page geometry or layout."""


@runtime_checkable
class RemoteChannel(Protocol):
    """Transport seam for one node-to-node session.

    ``handshake`` exchanges the KV geometry blob (magic + page_size + dtype +
    layout) and returns the peer's blob; a mismatch aborts the session before
    any payload moves.
    """

    def send_chunks(self, chunks: Sequence[bytes], *, chunk_size: int) -> None: ...

    def recv_chunks(self, chunk_sizes: Sequence[int]) -> tuple[bytes, ...]: ...

    def handshake(self, geometry: bytes) -> bytes: ...

    def close(self) -> None: ...


def encode_geometry(*, page_size: int, dtype: str, layout: str) -> bytes:
    """One canonical geometry blob for the wire handshake."""
    payload = {
        "magic": _GEOMETRY_MAGIC.decode(),
        "page_size": page_size,
        "dtype": dtype,
        "layout": layout,
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def check_geometry(local: bytes, remote: bytes) -> None:
    if local != remote:
        raise GeometryMismatchError(f"node geometry mismatch: local={local!r} remote={remote!r}")


class TcpRemoteChannel:
    """Length-prefixed chunk framing over one TCP socket.

    ``host``/``port`` describe the *peer*: ``connect`` dials it, while the
    engine on the receiving side accepts a socket handed over by its own
    server loop. Both sides speak the same frame format, so the payload is a
    byte-for-byte match after a full round trip.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        peer: str,
        timeout: float | None = None,
    ) -> None:
        self._sock = sock
        self._peer = peer
        if timeout is not None:
            self._sock.settimeout(timeout)

    @classmethod
    def connect(cls, endpoint: NodeEndpoint, *, timeout: float | None = None) -> TcpRemoteChannel:
        sock = socket.create_connection((endpoint.host, endpoint.port), timeout=timeout)
        return cls(sock, peer=endpoint.node_id, timeout=timeout)

    @property
    def peer(self) -> str:
        return self._peer

    def handshake(self, geometry: bytes) -> bytes:
        _send_frame(self._sock, geometry)
        return _recv_frame(self._sock)

    def send_chunks(self, chunks: Sequence[bytes], *, chunk_size: int) -> None:
        """One length-prefixed frame per chunk; ``chunk_size`` is advisory."""
        del chunk_size  # layer chunks are already sized by the caller
        for chunk in chunks:
            _send_frame(self._sock, chunk)

    def recv_chunks(self, chunk_sizes: Sequence[int]) -> tuple[bytes, ...]:
        received: list[bytes] = []
        for expected in chunk_sizes:
            chunk = bytearray()
            while len(chunk) < expected:
                chunk.extend(_recv_frame(self._sock))
            received.append(bytes(chunk[:expected]))
        return tuple(received)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class LoopbackRemoteChannel:
    """In-process paired channel; send hands chunks to the twin by reference."""

    def __init__(self, name: str, twin: LoopbackRemoteChannel | None = None) -> None:
        self.name = name
        self._twin: LoopbackRemoteChannel | None = twin
        self._inbox: list[bytes] = []
        self._handshake_inbox: bytes | None = None
        self._lock = threading.RLock()
        self._closed = False
        if twin is not None and twin._twin is None:
            twin._twin = self

    @classmethod
    def pair(
        cls, left: str = "left", right: str = "right"
    ) -> tuple[LoopbackRemoteChannel, LoopbackRemoteChannel]:
        first = cls(left)
        second = cls(right)
        first._twin = second
        second._twin = first
        return first, second

    def handshake(self, geometry: bytes) -> bytes:
        twin = self._twin
        if twin is None:
            raise GeometryMismatchError("loopback channel has no twin")
        with twin._lock:
            twin._handshake_inbox = geometry
        return geometry  # the twin echoes the same geometry back

    def send_chunks(self, chunks: Sequence[bytes], *, chunk_size: int) -> None:
        twin = self._twin
        if twin is None:
            raise GeometryMismatchError("loopback channel has no twin")
        with twin._lock:
            for chunk in chunks:
                twin._inbox.append(bytes(chunk))

    def recv_chunks(self, chunk_sizes: Sequence[int]) -> tuple[bytes, ...]:
        received: list[bytes] = []
        with self._lock:
            for expected in chunk_sizes:
                chunk = bytearray()
                while len(chunk) < expected:
                    if not self._inbox:
                        raise RuntimeError("loopback inbox drained before the ticket")
                    chunk.extend(self._inbox.pop(0))
                received.append(bytes(chunk))
        return tuple(received)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._inbox.clear()


class RdmaRemoteChannel:
    """Reserved RDMA seam: not implemented in this milestone."""

    def __init__(self, endpoint: NodeEndpoint) -> None:
        self.endpoint = endpoint

    def handshake(self, geometry: bytes) -> bytes:
        raise NotImplementedError("RDMA transport is a reserved seam; use TCP or the loopback pair")

    def send_chunks(self, chunks: Sequence[bytes], *, chunk_size: int) -> None:
        raise NotImplementedError("RDMA transport is a reserved seam; use TCP or the loopback pair")

    def recv_chunks(self, chunk_sizes: Sequence[int]) -> tuple[bytes, ...]:
        raise NotImplementedError("RDMA transport is a reserved seam; use TCP or the loopback pair")

    def close(self) -> None:
        return None


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(len(payload), 0) + payload)


def _recv_frame(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, _HEADER.size)
    length, _ = _HEADER.unpack(header)
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        data = sock.recv(size - len(chunks))
        if not data:
            raise ConnectionError("peer closed mid-frame")
        chunks.extend(data)
    return bytes(chunks)


@dataclass(slots=True)
class _NodeTransfer:
    """Worker-side bookkeeping for one in-flight node transfer."""

    ticket: NodeTransferTicket
    chunks: tuple[bytes, ...]
    channel: RemoteChannel | None
    thread: threading.Thread | None = None
    error: str | None = None
    cancelled: bool = False


class NodeTransferEngine:
    """Node-to-node transfer engine over the tiering ticket semantics.

    Same-node tickets complete by reference; cross-node tickets stream their
    chunks through the channel on a per-transfer worker thread. ``poll``
    resolves finished tickets, ``drain`` blocks until every worker exits, and
    credits are held for the exact ticket lifetime (acquired at submit,
    released at resolution or failure).
    """

    def __init__(
        self,
        topology: NodeTopology,
        *,
        channel_for: Callable[[str, str], RemoteChannel] | None = None,
        credits: TransferCredits | None = None,
        chunk_size: int = 1 << 20,
        geometry: bytes | None = None,
        connect_timeout: float | None = 10.0,
    ) -> None:
        self._topology = topology
        self._channel_for = channel_for or self._tcp_channel
        self._credits = credits
        self._chunk_size = max(1, int(chunk_size))
        self._connect_timeout = connect_timeout
        self._geometry_blob = geometry or encode_geometry(
            page_size=0, dtype="float16", layout="unset"
        )
        self._ids = count(1)
        self._pending: dict[int, _NodeTransfer] = {}
        self._lock = RLock()
        self._submitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._bytes_total = 0
        self._issue_epoch = 0

    @property
    def topology(self) -> NodeTopology:
        return self._topology

    @property
    def node_id(self) -> str:
        return self._topology.self_node

    def _tcp_channel(self, source: str, destination: str) -> RemoteChannel:
        return TcpRemoteChannel.connect(
            self._topology.endpoint(destination), timeout=self._connect_timeout
        )

    def submit(
        self,
        chunks: Sequence[bytes],
        *,
        source_node: str,
        destination_node: str,
        token_range: tuple[int, int] = (0, 0),
        layer_count: int = 0,
    ) -> NodeTransferTicket:
        """Stage one transfer; bytes move on the next poll/drain cycle."""
        self._topology.endpoint(source_node)
        self._topology.endpoint(destination_node)
        materialized = tuple(bytes(chunk) for chunk in chunks)
        if any(len(chunk) == 0 for chunk in chunks):
            raise ValueError("a node transfer cannot carry an empty chunk")
        nbytes = sum(len(chunk) for chunk in chunks)
        link = self._topology.link(source_node, destination_node)
        with self._lock:
            ticket = NodeTransferTicket(
                ticket_id=next(self._ids),
                source_node=source_node,
                destination_node=destination_node,
                nbytes=nbytes,
                chunks=len(chunks),
                link=link,
                issue_epoch=self._issue_epoch,
            )
            self._issue_epoch += 1
            record = _NodeTransfer(ticket=ticket, chunks=materialized, channel=None)
            if self._credits is not None:
                self._credits.acquire(ticket, nbytes)
            self._pending[ticket.ticket_id] = record
            self._submitted_total += 1
            thread = threading.Thread(
                target=self._run_transfer,
                args=(record,),
                name=f"ayaka-node-transfer-{ticket.ticket_id}",
                daemon=True,
            )
            record.thread = thread
            thread.start()
            return ticket

    def _run_transfer(self, record: _NodeTransfer) -> None:
        try:
            if record.ticket.link == LinkKind.HOST_PEER:
                # Intra-node pair: the bytes are shared by reference; the
                # ticket still keeps its poll contract.
                return
            channel = self._channel_for(record.ticket.source_node, record.ticket.destination_node)
            record.channel = channel
            remote_geometry = channel.handshake(self._geometry_blob)
            check_geometry(self._geometry_blob, remote_geometry)
            for chunk in record.chunks:
                if record.cancelled:
                    record.error = "cancelled by caller"
                    return
                channel.send_chunks((chunk,), chunk_size=self._chunk_size)
        except Exception as exc:  # outcome must carry any failure
            record.error = record.error or str(exc)

    def poll(self) -> tuple[NodeTransferOutcome, ...]:
        with self._lock:
            resolved: list[NodeTransferOutcome] = []
            for ticket_id in list(self._pending):
                record = self._pending[ticket_id]
                if record.thread is not None and record.thread.is_alive():
                    continue
                if record.error is not None:
                    resolved.append(
                        NodeTransferOutcome(
                            ticket=record.ticket,
                            state=TransferState.FAILED,
                            error=record.error,
                        )
                    )
                    self._failed_total += 1
                else:
                    resolved.append(
                        NodeTransferOutcome(ticket=record.ticket, state=TransferState.COMPLETED)
                    )
                    self._completed_total += 1
                    self._bytes_total += record.ticket.nbytes
                self._release(record)
                del self._pending[ticket_id]
            return tuple(resolved)

    def drain(self) -> tuple[NodeTransferOutcome, ...]:
        with self._lock:
            threads = [record.thread for record in self._pending.values() if record.thread]
        for thread in threads:
            thread.join()
        return self.poll()

    def cancel(self, ticket: NodeTransferTicket) -> bool:
        """Suppress publication; ownership is preserved until poll/drain."""
        with self._lock:
            record = self._pending.get(ticket.ticket_id)
            if record is None:
                return False
            record.cancelled = True
            return True

    def _release(self, record: _NodeTransfer) -> None:
        if self._credits is not None:
            self._credits.release(record.ticket)

    @property
    def metrics(self) -> NodeTransferMetrics:
        with self._lock:
            return NodeTransferMetrics(
                submitted_total=self._submitted_total,
                completed_total=self._completed_total,
                failed_total=self._failed_total,
                bytes_transferred_total=self._bytes_total,
                pending=len(self._pending),
            )
