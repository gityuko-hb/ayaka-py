"""Registered CUDA graph for the two-rank signal/epoch SUM.

One graph owns one static input/output buffer and one DC4 workspace. Replay
stages the changing input outside capture, then launches the recorded kernel.
The kernel advances a device counter, so an unchanged graph executable cannot
accept a previous replay's signal. The caller must join each ticket before
reusing the buffer; this is the initial one-flight serialization policy.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ayaka.distributed.collective_backend import (
    CollectivePostLaunchError,
    CollectivePreLaunchError,
)
from ayaka.distributed.custom_all_reduce import _ordered_exchange
from ayaka.kernel.comm_layout import STATUS_TIMEOUT

if TYPE_CHECKING:
    from ayaka.distributed.custom_all_reduce import CustomAllReduce

__all__ = ["CapturedCollective", "CollectiveGraphIdentity", "GraphCollectiveHandle"]


@dataclass(frozen=True, slots=True)
class CollectiveGraphIdentity:
    """The group and local addresses recorded into one graph executable."""

    group_name: str
    ranks: tuple[int, ...]
    session_generation: tuple[int, int]
    backend: str
    algorithm: str
    dtype: torch.dtype
    numel: int
    bucket_bytes: int
    workspace_generation: int
    pointers: tuple[int, ...]


class GraphCollectiveHandle:
    """One replay ticket; join the kernel, then release after consuming output."""

    def __init__(self, owner: CapturedCollective, epoch: int, event: torch.cuda.Event) -> None:
        self._owner = owner
        self.epoch = epoch
        self.event = event
        self.output = owner.output
        self._joined = False
        self._released = False
        self._lock = threading.RLock()

    def is_completed(self) -> bool:
        return bool(self.event.query())

    def wait(self) -> None:
        """Wait within the configured watchdog and verify the device epoch."""
        with self._lock:
            if self._joined:
                return
            self._owner._join(self)
            self._joined = True

    def release(self, *, consumer_stream: torch.cuda.Stream | None = None) -> None:
        """End the output lease after its last consumer is enqueued.

        ``consumer_stream`` is the stream carrying that last consumer; when
        omitted, the current stream is used. The next replay waits on its
        recorded event before overwriting the static buffer.
        """
        with self._lock:
            if self._released:
                return
            if not self._joined:
                raise CollectivePreLaunchError("join graph ticket before releasing its output")
            self._owner._release(self, consumer_stream=consumer_stream)
            self._released = True


class CapturedCollective:
    """A pinned, single-flight CUDA graph for a fixed SUM dtype and shape."""

    def __init__(
        self,
        communicator: CustomAllReduce,
        *,
        dtype: torch.dtype,
        numel: int,
        generation_provider: Callable[[], tuple[int, int]] | None,
    ) -> None:
        self._comm = communicator
        self._lock = threading.RLock()
        self._generation_provider = generation_provider
        self._graph: torch.cuda.CUDAGraph | None = None
        self._active: GraphCollectiveHandle | None = None
        self._invalid_reason: str | None = None
        self._pinned = False
        self._epoch = communicator.epoch
        self._static = torch.empty(numel, dtype=dtype, device=f"cuda:{communicator._device}")
        self._counter = communicator._workspace.narrow(
            0, communicator._layout.pointer_table_offset, 4
        ).view(torch.int32)
        self._counter.fill_(self._epoch)
        self._local_payload = communicator._payload_view(communicator._workspace, dtype)
        self._peer_payload = communicator._payload_view(communicator._peer_workspace, dtype)
        self._ready_event = torch.cuda.Event()
        self._done_event = torch.cuda.Event()
        self._reuse_event = torch.cuda.Event()
        self._has_release = False
        self._identity = CollectiveGraphIdentity(
            group_name=communicator._group.name,
            ranks=tuple(communicator._group.ranks),
            session_generation=communicator._session_generation,
            backend=communicator.name,
            algorithm="signal_epoch_graph_v1",
            dtype=dtype,
            numel=numel,
            bucket_bytes=communicator._config.bucket_bytes,
            workspace_generation=id(communicator._workspace),
            pointers=self._current_pointers(),
        )

    @property
    def identity(self) -> CollectiveGraphIdentity:
        return self._identity

    @property
    def output(self) -> torch.Tensor:
        """Static output; valid until the next replay starts or graph closes."""
        if self._graph is None:
            raise CollectivePreLaunchError("graph has not been captured or was retired")
        return self._static

    @property
    def epoch(self) -> int:
        return self._epoch

    @classmethod
    def capture(
        cls,
        communicator: CustomAllReduce,
        *,
        dtype: torch.dtype,
        numel: int,
        generation_provider: Callable[[], tuple[int, int]] | None = None,
    ) -> CapturedCollective:
        """Preallocate, JIT, agree the bucket on all ranks, then capture."""
        if not torch.cuda.is_available():
            raise CollectivePreLaunchError("CUDA graph capture requires CUDA")
        if type(numel) is not int or numel <= 0 or dtype not in communicator.config.torch_dtypes:
            raise CollectivePreLaunchError("graph bucket requires a supported dtype and size")
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if nbytes < communicator.config.min_bytes or nbytes > communicator.config.bucket_bytes:
            raise CollectivePreLaunchError("graph bucket exceeds the agreed payload bounds")
        if nbytes % communicator.config.alignment:
            raise CollectivePreLaunchError("graph bucket must be 16-byte aligned")
        graph = cls(communicator, dtype=dtype, numel=numel, generation_provider=generation_provider)
        from ayaka.kernel.triton.comm.signal_epoch import launch_signal_epoch_sum_graph

        launch_args: dict[str, Any] = dict(
            x=graph._static,
            local_payload=graph._local_payload,
            peer_payload=graph._peer_payload,
            local_signal=communicator._local_signal,
            peer_signal=communicator._peer_signal,
            status=communicator._status,
            epoch_counter=graph._counter,
            slot_stride=communicator._layout.slot_stride_elements(graph._static.element_size()),
            spin_budget=communicator.config.spin_budget,
            epoch_limit=communicator.config.epoch_limit,
            block=communicator.config.block_elements,
        )
        # The JIT path and all descriptor traffic are outside the captured region.
        try:
            launch_signal_epoch_sum_graph(**launch_args, warmup=True)
        except BaseException as exc:
            raise CollectivePreLaunchError(
                f"graph kernel warmup failed: {type(exc).__name__}"
            ) from exc
        proposal = json.dumps(
            {
                "group": graph.identity.group_name,
                "ranks": graph.identity.ranks,
                "generation": graph.identity.session_generation,
                "backend": graph.identity.backend,
                "algorithm": graph.identity.algorithm,
                "dtype": str(dtype),
                "numel": numel,
                "bucket_bytes": graph.identity.bucket_bytes,
                "epoch": graph._epoch,
            },
            sort_keys=True,
        )
        texts = _ordered_exchange(
            communicator._control,
            communicator._group.local_rank,
            communicator._group.size,
            proposal,
            communicator._timeout_s,
        )
        if any(text != proposal for text in texts):
            raise CollectivePreLaunchError("graph capture sequence or bucket differs across ranks")
        capture_error: BaseException | None = None
        try:
            communicator._registry.hold(communicator._own_descriptor)
            communicator._import_registry.pin(communicator._peer_descriptor)
            graph._pinned = True
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=communicator._stream):
                launch_signal_epoch_sum_graph(**launch_args)
            graph._graph = captured
            # CUDA capture itself should not execute the recorded kernel; read
            # once here to keep the host expectation correct across versions.
            graph._epoch = int(graph._counter.item())
            communicator._epoch = graph._epoch
        except BaseException as exc:
            capture_error = exc
        try:
            agreed = communicator._control.all_grant(
                capture_error is None, timeout_s=communicator._timeout_s
            )
        except BaseException as exc:
            capture_error = exc
            agreed = False
        if not agreed:
            communicator._fail_closed(
                f"graph capture failed: {type(capture_error).__name__}"
                if capture_error is not None
                else "peer graph capture failed"
            )
            # A peer might have captured the imported pointer. Retain pins on
            # failure until group recovery proves quiescence.
            raise CollectivePostLaunchError("distributed graph capture failed") from capture_error
        return graph

    def replay(
        self,
        tensor: torch.Tensor,
        *,
        session_generation: tuple[int, int] | None = None,
    ) -> GraphCollectiveHandle:
        """Stage input and enqueue replay; no probe, exchange, JIT or host sync."""
        with self._lock:
            return self._replay_locked(tensor, session_generation=session_generation)

    def _replay_locked(
        self, tensor: torch.Tensor, *, session_generation: tuple[int, int] | None
    ) -> GraphCollectiveHandle:
        comm = self._comm
        if self._graph is None or self._invalid_reason is not None or comm._closed:
            raise CollectivePreLaunchError("graph was retired or invalidated")
        if comm._quarantined:
            raise CollectivePostLaunchError("graph communicator is quarantined")
        if self._active is not None:
            raise CollectivePreLaunchError("graph workspace still belongs to an active ticket")
        if (
            session_generation is not None
            and session_generation != self.identity.session_generation
        ):
            raise CollectivePreLaunchError("graph session generation changed")
        if (
            self._generation_provider is not None
            and self._generation_provider() != self.identity.session_generation
        ):
            raise CollectivePreLaunchError("graph generation provider changed")
        if self._current_pointers() != self.identity.pointers or comm._peer_view is None:
            raise CollectivePreLaunchError("graph pointer binding changed")
        comm._validate_tensor(tensor)
        if tensor.dtype != self.identity.dtype or tensor.numel() != self.identity.numel:
            raise CollectivePreLaunchError("graph dtype or bucket changed")
        if self._epoch >= comm.config.epoch_limit:
            raise CollectivePreLaunchError("graph epoch limit reached; recapture after recovery")
        current = torch.cuda.current_stream(comm._device)
        self._ready_event.record(current)
        try:
            with torch.cuda.stream(comm._stream):
                comm._stream.wait_event(self._ready_event)
                if self._has_release:
                    comm._stream.wait_event(self._reuse_event)
                self._static.copy_(tensor, non_blocking=True)
                self._graph.replay()
                self._done_event.record(comm._stream)
            current.wait_event(self._done_event)
        except BaseException as exc:
            comm._fail_closed(f"graph replay failed: {type(exc).__name__}")
            raise CollectivePostLaunchError("graph replay failed after enqueue") from exc
        self._epoch += 1
        comm._epoch = self._epoch
        ticket = GraphCollectiveHandle(self, self._epoch, self._done_event)
        self._active = ticket
        return ticket

    def _join(self, ticket: GraphCollectiveHandle) -> None:
        with self._lock:
            self._join_locked(ticket)

    def _join_locked(self, ticket: GraphCollectiveHandle) -> None:
        if ticket is not self._active:
            raise CollectivePreLaunchError("graph ticket is stale or belongs to another graph")
        comm = self._comm
        if not comm._bridge.wait_event(ticket.event, comm.config.watchdog_timeout_s):
            comm._fail_closed("graph replay watchdog deadline exceeded")
            raise CollectivePostLaunchError("graph replay exceeded its watchdog deadline")
        status = comm._bridge.item(comm._status)
        if status == STATUS_TIMEOUT or status != ticket.epoch:
            comm._fail_closed(f"graph replay status {status} differs from epoch {ticket.epoch}")
            raise CollectivePostLaunchError("graph replay status failed; communicator quarantined")

    def _release(
        self, ticket: GraphCollectiveHandle, *, consumer_stream: torch.cuda.Stream | None
    ) -> None:
        with self._lock:
            self._release_locked(ticket, consumer_stream=consumer_stream)

    def _release_locked(
        self, ticket: GraphCollectiveHandle, *, consumer_stream: torch.cuda.Stream | None
    ) -> None:
        if ticket is not self._active:
            raise CollectivePreLaunchError("graph ticket is stale or belongs to another graph")
        stream = consumer_stream or torch.cuda.current_stream(self._comm._device)
        self._reuse_event.record(stream)
        self._has_release = True
        self._active = None

    def invalidate(self, reason: str) -> None:
        """Reject future replay after resize, rebind, membership or recovery."""
        with self._lock:
            self._invalid_reason = reason or "graph invalidated"

    def close(self) -> bool:
        """Retire graph and pins only after the last replay has joined."""
        with self._lock:
            return self._close_locked()

    def _close_locked(self) -> bool:
        if self._active is not None or self._comm._quarantined:
            return False
        if self._graph is None:
            return True
        if self._has_release and not self._comm._bridge.wait_event(
            self._reuse_event, self._comm.config.watchdog_timeout_s
        ):
            self._comm._fail_closed("graph output consumer did not drain before close")
            return False
        self._comm._bridge.synchronize_stream(self._comm._stream)
        self._graph = None
        if self._pinned:
            self._comm._import_registry.unpin(self._comm._peer_descriptor)
            self._comm._registry.unhold(self._comm._own_descriptor)
            self._pinned = False
        # The caller may keep this Python graph object after close. Drop every
        # captured view now so it cannot extend an imported mapping's lifetime.
        del self._peer_payload
        del self._local_payload
        del self._counter
        del self._static
        if self._comm._graph is self:
            self._comm._graph = None
        return True

    def _current_pointers(self) -> tuple[int, ...]:
        comm = self._comm
        return (
            int(self._static.data_ptr()),
            int(comm._workspace.data_ptr()),
            int(comm._peer_workspace.data_ptr()),
            int(comm._local_signal.data_ptr()),
            int(comm._peer_signal.data_ptr()),
            int(comm._status.data_ptr()),
            int(self._counter.data_ptr()),
        )
