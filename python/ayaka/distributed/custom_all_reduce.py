"""Custom two-rank all-reduce communicator.

This module owns the eager, world=2 SUM implementation behind the
:class:`~ayaka.distributed.collective_backend.CustomCollective` protocol. It
does not own a process group, does not commit or publish anything, and never
falls back on its own: every refusal that is safe to retry is raised as
:class:`~ayaka.distributed.collective_backend.CollectivePreLaunchError` before
mutation, and every failure after the data plane was entered fails closed by
quarantining the workspace.

Resource ownership:

* the communicator owns one dedicated legacy-IPC workspace per rank plus the
  imported peer mapping; descriptors and views are bound to the
  ``session_generation`` and released only after the peer close-ACK;
* the runtime owner supplies the control channel and the Torch fallback; the
  helper :func:`build_custom_all_reduce_setup` performs the one-time ordered
  descriptor exchange *before* DC3 agreement so the factory itself is
  channel-free and rollback stays aligned on every rank;
* ``CustomAllReduceHandle`` retains the caller tensor and every workspace view
  until joined; the stream event is the completion proof, the local status word
  is the failure proof.

Importing this module never imports Triton: the kernel launcher is resolved
lazily only when a communicator is actually built.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from ayaka.distributed.collective_backend import (
    CUSTOM_ALIGNMENT_BYTES,
    AgreementChannel,
    CollectiveCapability,
    CollectivePostLaunchError,
    CollectivePreLaunchError,
    CollectiveReason,
    CollectiveSetup,
    CustomCapability,
    probe_collective_capability,
)
from ayaka.distributed.device import AsyncHandle, CommOpType, DeviceGroup
from ayaka.kernel.comm_layout import STATUS_UNSET, WorkspaceLayout
from ayaka.runtime.ipc_engine import IpcRegionDescriptor, IpcRegionRegistry, IpcRegionView
from ayaka.types import DType

if TYPE_CHECKING:
    from ayaka.distributed.collective_graph import CapturedCollective

__all__ = [
    "CUSTOM_ALL_REDUCE_ALGORITHM",
    "CUSTOM_ALL_REDUCE_BACKEND",
    "CUSTOM_ALL_REDUCE_VERSION",
    "DEFAULT_BUCKET_BYTES",
    "DEFAULT_EPOCH_LIMIT",
    "DEFAULT_WATCHDOG_TIMEOUT_S",
    "CustomAllReduce",
    "CustomAllReduceConfig",
    "CustomAllReduceHandle",
    "DeviceBridge",
    "SignalEpochLauncher",
    "build_custom_all_reduce_setup",
    "custom_all_reduce_capability",
    "quarantined_setups",
    "reset_quarantined_setups",
]

logger = logging.getLogger(__name__)

CUSTOM_ALL_REDUCE_BACKEND = "custom_all_reduce"
CUSTOM_ALL_REDUCE_VERSION = "1"
CUSTOM_ALL_REDUCE_ALGORITHM = "signal_epoch"

#: Default payload bucket. The workspace holds two slots plus pads; owners that
#: need a larger layer bucket must raise this explicitly and re-agree the
#: profile, because the agreed capacity is part of the setup fingerprint.
DEFAULT_BUCKET_BYTES = 1 << 20
#: Host watchdog deadline for one collective completion.
DEFAULT_WATCHDOG_TIMEOUT_S = 60.0
#: Monotonic uint32 epoch limit. Reaching it refuses the call (fail-closed, no
#: wrap in the lifetime of a runtime generation); the workspace is retired.
DEFAULT_EPOCH_LIMIT = (1 << 31) - 1
#: Deadline for the descriptor exchange and the close handshake.
DEFAULT_SETUP_TIMEOUT_S = 30.0

_TORCH_DTYPE_BY_LABEL: dict[str, torch.dtype] = {
    DType.FP32.label: torch.float32,
    DType.FP16.label: torch.float16,
    DType.BF16.label: torch.bfloat16,
}

#: Exports retained when setup/close cannot prove peer quiescence. Keeping the
#: registry alive keeps the allocation alive: never recycle memory a peer may
#: still map (INV-4, DC0 14.2).
_QUARANTINED_SETUPS: list[tuple[IpcRegionRegistry, str]] = []


def quarantined_setups() -> tuple[tuple[IpcRegionRegistry, str], ...]:
    """Test hook: exports retained because quiescence was not proven."""
    return tuple(_QUARANTINED_SETUPS)


def reset_quarantined_setups() -> None:
    """Test hook: forget retained exports (never frees their memory)."""
    _QUARANTINED_SETUPS.clear()


@dataclasses.dataclass(frozen=True, slots=True)
class CustomAllReduceConfig:
    """Static capability of one communicator, agreed as the group profile.

    ``bucket_bytes`` is the maximum payload per collective; the workspace
    budget is twice that (ping-pong) plus signal/status/pointer-table pads.
    ``supported_dtypes`` must match the kernels' gates. ``epoch_limit`` is the
    fail-closed wraparound point.
    """

    bucket_bytes: int = DEFAULT_BUCKET_BYTES
    supported_dtypes: tuple[DType, ...] = (DType.FP32, DType.FP16, DType.BF16)
    alignment: int = CUSTOM_ALIGNMENT_BYTES
    min_bytes: int = 16
    spin_budget: int = 1_000_000
    watchdog_timeout_s: float = DEFAULT_WATCHDOG_TIMEOUT_S
    epoch_limit: int = DEFAULT_EPOCH_LIMIT
    block_elements: int = 1024
    enable_graph_capture: bool = False

    def __post_init__(self) -> None:
        if type(self.bucket_bytes) is not int or self.bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be a positive integer")
        if self.bucket_bytes % self.alignment:
            raise ValueError("bucket_bytes must be a multiple of the alignment")
        if type(self.supported_dtypes) is not tuple or not self.supported_dtypes:
            raise ValueError("supported_dtypes must be a non-empty tuple")
        unknown = [
            dtype for dtype in self.supported_dtypes if dtype.label not in _TORCH_DTYPE_BY_LABEL
        ]
        if unknown:
            raise ValueError(f"unsupported custom all-reduce dtypes: {unknown}")
        if len(set(self.supported_dtypes)) != len(self.supported_dtypes):
            raise ValueError("supported_dtypes must be unique")
        if self.alignment != CUSTOM_ALIGNMENT_BYTES:
            raise ValueError("the DC4 protocol requires 16-byte alignment")
        if type(self.min_bytes) is not int or not 0 < self.min_bytes <= self.bucket_bytes:
            raise ValueError("min_bytes must be positive and at most bucket_bytes")
        if type(self.spin_budget) is not int or self.spin_budget < 1:
            raise ValueError("spin_budget must be a positive integer")
        if type(self.watchdog_timeout_s) not in (int, float) or self.watchdog_timeout_s <= 0:
            raise ValueError("watchdog_timeout_s must be positive")
        if type(self.epoch_limit) is not int or self.epoch_limit < 1:
            raise ValueError("epoch_limit must be a positive integer")
        if type(self.block_elements) is not int or self.block_elements < 1:
            raise ValueError("block_elements must be a positive integer")
        if type(self.enable_graph_capture) is not bool:
            raise TypeError("enable_graph_capture must be a boolean")

    @property
    def torch_dtypes(self) -> tuple[torch.dtype, ...]:
        return tuple(_TORCH_DTYPE_BY_LABEL[dtype.label] for dtype in self.supported_dtypes)

    @property
    def layout(self) -> WorkspaceLayout:
        return WorkspaceLayout.for_bucket(self.bucket_bytes)

    @property
    def workspace_bytes(self) -> int:
        return int(self.layout.total_bytes)


@runtime_checkable
class DeviceBridge(Protocol):
    """Device operations the communicator needs, injectable for host tests."""

    def allocate_workspace(self, nbytes: int, device: int) -> Any: ...

    def create_stream(self, device: int) -> Any: ...

    def current_stream(self, device: int) -> Any: ...

    def stream_scope(self, stream: Any) -> contextlib.AbstractContextManager[None]: ...

    def record_event(self, stream: Any) -> Any: ...

    def stream_wait_event(self, stream: Any, event: Any) -> None: ...

    def event_query(self, event: Any) -> bool: ...

    def wait_event(self, event: Any, timeout_s: float) -> bool: ...

    def synchronize_stream(self, stream: Any) -> None: ...

    def item(self, tensor: Any) -> int: ...

    def is_device_tensor(self, tensor: Any, device: int) -> bool: ...


class _TorchDeviceBridge:
    """Default bridge on torch CUDA streams and events."""

    def allocate_workspace(self, nbytes: int, device: int) -> Any:
        return torch.zeros(nbytes, dtype=torch.uint8, device=f"cuda:{device}")

    def create_stream(self, device: int) -> Any:
        return torch.cuda.Stream(device=device)

    def current_stream(self, device: int) -> Any:
        return torch.cuda.current_stream(device=device)

    def stream_scope(self, stream: Any) -> contextlib.AbstractContextManager[None]:
        return torch.cuda.stream(stream)

    def record_event(self, stream: Any) -> Any:
        event = torch.cuda.Event(blocking=False, interprocess=False)
        event.record(stream)
        return event

    def stream_wait_event(self, stream: Any, event: Any) -> None:
        stream.wait_event(event)

    def event_query(self, event: Any) -> bool:
        return bool(event.query())

    def wait_event(self, event: Any, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while not bool(event.query()):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.0005)
        return True

    def synchronize_stream(self, stream: Any) -> None:
        stream.synchronize()

    def item(self, tensor: Any) -> int:
        return int(tensor.item())

    def is_device_tensor(self, tensor: Any, device: int) -> bool:
        return bool(getattr(tensor, "is_cuda", False)) and tensor.device.index == device


@runtime_checkable
class SignalEpochLauncher(Protocol):
    """Kernel surface used by the communicator; the default wraps the DC4 op."""

    def available(self) -> bool: ...

    def launch(
        self,
        *,
        x: Any,
        local_payload: Any,
        peer_payload: Any,
        local_signal: Any,
        peer_signal: Any,
        status: Any,
        epoch: int,
        slot_stride: int,
        spin_budget: int,
    ) -> None: ...

    def warmup(
        self,
        *,
        x: Any,
        local_payload: Any,
        peer_payload: Any,
        local_signal: Any,
        peer_signal: Any,
        status: Any,
        epoch: int,
        slot_stride: int,
        spin_budget: int,
    ) -> bool: ...


class _TritonSignalEpochLauncher:
    """Launch the registered ``ayaka::signal_epoch_sum`` public custom op."""

    @staticmethod
    def _comm() -> Any:
        from ayaka.kernel.triton import comm

        return comm

    def available(self) -> bool:
        try:
            module = self._comm()
        except Exception:
            return False
        return not bool(module.signal_epoch_sum.use_reference())

    def launch(
        self,
        *,
        x: Any,
        local_payload: Any,
        peer_payload: Any,
        local_signal: Any,
        peer_signal: Any,
        status: Any,
        epoch: int,
        slot_stride: int,
        spin_budget: int,
    ) -> None:
        self._comm().signal_epoch_sum(
            x,
            local_payload,
            peer_payload,
            local_signal,
            peer_signal,
            status,
            epoch,
            slot_stride,
            spin_budget,
        )

    def warmup(
        self,
        *,
        x: Any,
        local_payload: Any,
        peer_payload: Any,
        local_signal: Any,
        peer_signal: Any,
        status: Any,
        epoch: int,
        slot_stride: int,
        spin_budget: int,
    ) -> bool:
        return bool(
            self._comm().warmup_signal_epoch_sum(
                x=x,
                local_payload=local_payload,
                peer_payload=peer_payload,
                local_signal=local_signal,
                peer_signal=peer_signal,
                status=status,
                epoch=epoch,
                slot_stride=slot_stride,
                spin_budget=spin_budget,
            )
        )


@dataclasses.dataclass(slots=True)
class _PendingCall:
    """One launched collective whose status has not been reaped yet."""

    epoch: int
    event: Any
    status: Any


class CustomAllReduceHandle:
    """Completion handle for one eager custom collective.

    ``wait()`` polls the stream event under a bounded host watchdog and then
    verifies the local status word. A watchdog expiry or a status mismatch
    quarantines the communicator: mappings are retained, never recycled.
    """

    def __init__(
        self,
        *,
        epoch: int,
        event: Any,
        status: Any,
        buffers: tuple[Any, ...],
        communicator: CustomAllReduce,
    ) -> None:
        self.epoch = epoch
        self.event = event
        self.status = status
        self.buffers = buffers
        self._communicator = communicator
        self._joined = False

    def is_completed(self) -> bool:
        return bool(self._communicator._bridge.event_query(self.event))

    def wait(self) -> None:
        self._communicator._join(self)
        self._joined = True


class CustomAllReduce:
    """World=2 signal/epoch SUM over a dedicated IPC workspace."""

    name = CUSTOM_ALL_REDUCE_BACKEND

    def __init__(
        self,
        *,
        group: DeviceGroup,
        control: AgreementChannel,
        capability: CollectiveCapability,
        session_generation: tuple[int, int],
        device: int,
        config: CustomAllReduceConfig,
        registry: IpcRegionRegistry,
        workspace: Any,
        own_descriptor: IpcRegionDescriptor,
        peer_descriptor: IpcRegionDescriptor,
        peer_view: IpcRegionView,
        import_registry: IpcRegionRegistry,
        bridge: DeviceBridge | None = None,
        launcher: SignalEpochLauncher | None = None,
        timeout_s: float | None = None,
    ) -> None:
        if group.size != 2:
            raise ValueError("the DC4 custom all-reduce is certified for world size 2")
        if capability.custom is None:
            raise ValueError("a custom all-reduce requires a custom capability record")
        self._group = group
        self._control = control
        self._capability = capability
        self._session_generation = session_generation
        self._device = int(device)
        self._config = config
        self._bridge: DeviceBridge = bridge if bridge is not None else _TorchDeviceBridge()
        self._launcher: SignalEpochLauncher = (
            launcher if launcher is not None else _TritonSignalEpochLauncher()
        )
        self._timeout_s = DEFAULT_SETUP_TIMEOUT_S if timeout_s is None else float(timeout_s)
        self._registry = registry
        self._workspace = workspace
        self._own_descriptor = own_descriptor
        self._peer_descriptor = peer_descriptor
        self._import_registry = import_registry
        self._layout = config.layout
        self._stream = self._bridge.create_stream(self._device)
        self._status = self._workspace.narrow(0, self._layout.status_offset, 4).view(torch.int32)
        self._status.fill_(STATUS_UNSET)
        self._local_signal = self._workspace.narrow(0, self._layout.signal_offset, 4).view(
            torch.int32
        )
        self._peer_view: IpcRegionView | None = peer_view
        self._peer_workspace = self._peer_view.tensor
        self._peer_signal = self._peer_workspace.narrow(0, self._layout.signal_offset, 4).view(
            torch.int32
        )
        self._epoch = 0
        self._pending: list[_PendingCall] = []
        self._closed = False
        self._clean_close = False
        self._quarantined = False
        self._failure = ""
        self._exhausted = False
        self._graph: CapturedCollective | None = None
        self._warmup()

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def config(self) -> CustomAllReduceConfig:
        return self._config

    @property
    def workspace_bytes(self) -> int:
        return int(self._layout.total_bytes)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def clean_close(self) -> bool:
        return self._clean_close

    @property
    def failure(self) -> str:
        """Failure class name or reason; never a pointer."""
        return self._failure

    @property
    def pending_calls(self) -> int:
        return len(self._pending)

    # ── collective ───────────────────────────────────────────────────────────

    def all_reduce(
        self,
        tensor: torch.Tensor,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None:
        """Reduce ``tensor`` in place across the two rank-local workspaces."""
        del async_op  # the handle is always returned; callers may join any time
        if not isinstance(group, DeviceGroup):
            raise TypeError("custom all-reduce requires an explicit DeviceGroup")
        if self._closed:
            raise CollectivePostLaunchError("custom all-reduce communicator is closed")
        if self._graph is not None:
            raise CollectivePreLaunchError("eager collective requires retiring the captured graph")
        self._reap_completed()
        if self._quarantined:
            raise CollectivePostLaunchError(
                f"custom all-reduce is quarantined after failure: {self._failure}"
            )
        if (
            group.name != self._group.name
            or tuple(group.ranks) != tuple(self._group.ranks)
            or group.local_rank != self._group.local_rank
        ):
            raise CollectivePreLaunchError("custom all-reduce group differs from the agreed group")
        if op is not CommOpType.SUM:
            raise CollectivePreLaunchError("custom all-reduce only implements SUM")
        self._validate_tensor(tensor)
        if tensor.numel() == 0:
            return None
        epoch = self._epoch + 1
        if epoch > self._config.epoch_limit:
            self._exhausted = True
            raise CollectivePreLaunchError(
                f"custom all-reduce epoch limit {self._config.epoch_limit} reached; "
                "the workspace is retired instead of wrapping"
            )

        element_size = int(tensor.element_size())
        slot_stride = self._layout.slot_stride_elements(element_size)
        local_payload = self._payload_view(self._workspace, tensor.dtype)
        peer_payload = self._payload_view(self._peer_workspace, tensor.dtype)

        current = self._bridge.current_stream(self._device)
        self._bridge.stream_wait_event(self._stream, self._bridge.record_event(current))
        try:
            with self._bridge.stream_scope(self._stream):
                self._launcher.launch(
                    x=tensor,
                    local_payload=local_payload,
                    peer_payload=peer_payload,
                    local_signal=self._local_signal,
                    peer_signal=self._peer_signal,
                    status=self._status,
                    epoch=epoch,
                    slot_stride=slot_stride,
                    spin_budget=self._config.spin_budget,
                )
        except CollectivePreLaunchError:
            raise
        except BaseException as exc:
            self._fail_closed(type(exc).__name__)
            raise CollectivePostLaunchError(
                "custom all-reduce launch failed after the stream was entered: "
                f"{type(exc).__name__}"
            ) from exc
        done = self._bridge.record_event(self._stream)
        self._bridge.stream_wait_event(current, done)
        self._epoch = epoch
        self._pending.append(_PendingCall(epoch=epoch, event=done, status=self._status))
        return CustomAllReduceHandle(
            epoch=epoch,
            event=done,
            status=self._status,
            buffers=(tensor, local_payload, peer_payload, self._workspace),
            communicator=self,
        )

    def quarantine(self) -> None:
        """Mark the communicator failed; resources stay retained."""
        self._quarantined = True

    def capture_graph(
        self,
        *,
        dtype: torch.dtype,
        numel: int,
        generation_provider: Callable[[], tuple[int, int]] | None = None,
    ) -> CapturedCollective:
        """Capture one registered SUM bucket after all ranks agree its identity.

        The returned graph owns the IPC pins. Retire it after consumers drain
        and before closing this communicator or building a different bucket.
        """
        if self._closed or self._quarantined or self._graph is not None:
            raise CollectivePreLaunchError("communicator is unavailable for graph capture")
        if not self._config.enable_graph_capture:
            raise CollectivePreLaunchError("graph capture is disabled for this agreed profile")
        if self._pending:
            self._reap_completed()
            if self._pending:
                raise CollectivePreLaunchError("drain eager collective flights before capture")
        from ayaka.distributed.collective_graph import CapturedCollective

        graph = CapturedCollective.capture(
            self, dtype=dtype, numel=numel, generation_provider=generation_provider
        )
        self._graph = graph
        return graph

    def close(self) -> None:
        """Drain, reap, close imports, ACK the peer and release the export.

        Never raises: an unclean shutdown (peer missing, ACK missing, watchdog
        failure) retains or quarantines resources instead of pretending to be
        clean. ``clean_close`` reports which happened.
        """
        if self._closed:
            return
        self._closed = True
        if self._quarantined:
            # No group-wide quiescence proof exists after a postlaunch or
            # partial-capture failure. Keep both mappings and graph pins live.
            self._clean_close = False
            return
        clean = True
        if self._graph is not None:
            try:
                graph_closed = self._graph.close()
            except BaseException:  # noqa: BLE001 - teardown must fail closed
                graph_closed = False
            if not graph_closed:
                self._fail_closed("graph flight did not drain before close")
                self._clean_close = False
                return
        try:
            self._bridge.synchronize_stream(self._stream)
        except BaseException:  # noqa: BLE001 - teardown must not raise
            clean = False
        if not self._reap_completed(raise_on_failure=False):
            clean = False
        try:
            if self._peer_view is not None:
                self._peer_view.close()
                self._peer_view = None
            self._import_registry.close(self._peer_descriptor)
        except BaseException:  # noqa: BLE001 - teardown must not raise
            clean = False
        try:
            texts = _ordered_exchange(
                self._control,
                self._group.local_rank,
                self._group.size,
                f"close-ack:{self._group.local_rank}",
                self._timeout_s,
            )
            peer_rank = self._group.ranks[1 - self._group.local_rank]
            if all(text.startswith("close-ack:") for text in texts):
                self._registry.confirm_close(self._own_descriptor, peer_rank)
            else:
                clean = False
        except BaseException:  # noqa: BLE001 - teardown must not raise
            clean = False
        try:
            self._registry.release(self._own_descriptor)
        except BaseException:  # noqa: BLE001 - release refused; retention is the safe path
            clean = False
        self._clean_close = clean

    # ── internals ────────────────────────────────────────────────────────────

    def _validate_tensor(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise CollectivePreLaunchError("custom all-reduce requires a torch.Tensor")
        if tensor.numel() == 0:
            # An empty collective is a uniform no-op; callers never reach the
            # device and never consume an epoch.
            return
        if not self._bridge.is_device_tensor(tensor, self._device):
            raise CollectivePreLaunchError(
                "custom all-reduce tensor must live on the communicator's device"
            )
        if tensor.dtype not in self._config.torch_dtypes:
            raise CollectivePreLaunchError(
                f"custom all-reduce does not support dtype {tensor.dtype}"
            )
        if not tensor.is_contiguous() or int(tensor.storage_offset()) < 0:
            raise CollectivePreLaunchError("custom all-reduce requires a contiguous tensor")
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        if nbytes < self._config.min_bytes:
            raise CollectivePreLaunchError(
                f"custom all-reduce payload {nbytes} bytes is below the "
                f"{self._config.min_bytes}-byte floor"
            )
        if nbytes > self._config.bucket_bytes:
            raise CollectivePreLaunchError(
                f"custom all-reduce payload {nbytes} bytes exceeds the agreed "
                f"{self._config.bucket_bytes}-byte bucket"
            )
        if nbytes % self._config.alignment or int(tensor.data_ptr()) % self._config.alignment:
            raise CollectivePreLaunchError(
                "custom all-reduce payload must be 16-byte aligned in pointer and byte length"
            )

    def _payload_view(self, workspace: Any, dtype: torch.dtype) -> torch.Tensor:
        return workspace.narrow(0, 0, self._layout.payload_bytes).view(dtype)

    def _warmup(self) -> None:
        try:
            dummy = torch.empty(
                self._config.min_bytes, dtype=torch.float32, device=f"cuda:{self._device}"
            )
        except BaseException:  # noqa: BLE001 - warmup is best effort
            return
        try:
            self._launcher.warmup(
                x=dummy,
                local_payload=self._payload_view(self._workspace, torch.float32),
                peer_payload=self._payload_view(self._peer_workspace, torch.float32),
                local_signal=self._local_signal,
                peer_signal=self._peer_signal,
                status=self._status,
                epoch=1,
                slot_stride=self._layout.slot_stride_elements(4),
                spin_budget=self._config.spin_budget,
            )
        except BaseException:  # noqa: BLE001 - first launch pays the compile cost
            logger.debug("custom all-reduce warmup skipped", exc_info=True)

    def _reap_completed(self, *, raise_on_failure: bool = True) -> bool:
        """Check finished launches' status words; quarantine on any mismatch."""
        if not self._pending:
            return True
        remaining: list[_PendingCall] = []
        healthy = True
        for call in self._pending:
            if not self._bridge.event_query(call.event):
                remaining.append(call)
                continue
            status = self._bridge.item(call.status)
            if status < call.epoch:
                healthy = False
                self._fail_closed(f"status {status} below epoch {call.epoch}")
                continue
        self._pending = remaining
        if not healthy and raise_on_failure:
            raise CollectivePostLaunchError(
                f"custom all-reduce failed after launch: {self._failure}"
            )
        return healthy

    def _join(self, handle: CustomAllReduceHandle) -> None:
        if self._quarantined:
            raise CollectivePostLaunchError(
                f"custom all-reduce is quarantined after failure: {self._failure}"
            )
        if not self._bridge.wait_event(handle.event, self._config.watchdog_timeout_s):
            self._fail_closed("host watchdog deadline exceeded")
            raise CollectivePostLaunchError("custom all-reduce exceeded its host watchdog deadline")
        self._pending = [call for call in self._pending if call.event is not handle.event]
        status = self._bridge.item(handle.status)
        if status < handle.epoch:
            self._fail_closed(f"status {status} below epoch {handle.epoch}")
            raise CollectivePostLaunchError(
                f"custom all-reduce did not complete epoch {handle.epoch} "
                f"(status {status}); the group is failed"
            )

    def _fail_closed(self, reason: str) -> None:
        if not self._quarantined:
            self._quarantined = True
            self._failure = reason


def custom_all_reduce_capability(
    config: CustomAllReduceConfig | None = None,
) -> CustomCapability:
    """The capability record this build can advertise for the custom backend."""
    resolved = config if config is not None else CustomAllReduceConfig()
    return CustomCapability(
        backend=CUSTOM_ALL_REDUCE_BACKEND,
        version=CUSTOM_ALL_REDUCE_VERSION,
        algorithm=CUSTOM_ALL_REDUCE_ALGORITHM,
        supported_ops=(CommOpType.SUM,),
        supported_dtypes=tuple(resolved.supported_dtypes),
        min_bytes=resolved.min_bytes,
        max_bytes=resolved.bucket_bytes,
        alignment=resolved.alignment,
        workspace_bytes=resolved.workspace_bytes,
        graph_certified=resolved.enable_graph_capture,
    )


# ---------------------------------------------------------------------------
# Setup: probe, ordered descriptor exchange, communicator construction
# ---------------------------------------------------------------------------


def _ordered_exchange(
    control: AgreementChannel,
    rank: int,
    world_size: int,
    own_text: str | None,
    timeout_s: float,
) -> list[str]:
    """Broadcast one text per rank in rank order; every rank receives all."""
    received: list[str] = []
    for source in range(world_size):
        text = own_text if source == rank else None
        received.append(control.broadcast_text(text, source_rank=source, timeout_s=timeout_s))
    return received


def _wire_payload(
    rank: int,
    build: bool,
    descriptor: IpcRegionDescriptor | None,
    uuid: str,
    session_generation: tuple[int, int],
) -> str:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "rank": rank,
        "build": bool(build),
        "session_generation": list(session_generation),
    }
    if build and descriptor is not None:
        payload["descriptor"] = descriptor.to_dict()
        payload["physical_device_uuid"] = uuid
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _downgrade(
    capability: CollectiveCapability, reason: CollectiveReason, detail: str
) -> CollectiveCapability:
    return dataclasses.replace(
        capability, reason=reason, probe_reason=f"{capability.probe_reason}|{detail}"
    )


def _quarantine_export(registry: IpcRegionRegistry | None, reason: str) -> None:
    """Retain an export whose peer quiescence cannot be proven."""
    if registry is not None:
        _QUARANTINED_SETUPS.append((registry, reason))


def _launcher_available(launcher: SignalEpochLauncher) -> bool:
    probe = getattr(launcher, "available", None)
    if not callable(probe):
        return True
    try:
        return bool(probe())
    except BaseException:  # noqa: BLE001 - unavailable means fall back
        return False


def build_custom_all_reduce_setup(
    *,
    group: DeviceGroup,
    control: AgreementChannel,
    session_generation: tuple[int, int],
    config: CustomAllReduceConfig | None = None,
    device_ordinals: Sequence[int] | None = None,
    probe: Callable[..., Any] | None = None,
    registry_factory: Callable[[int, str | None], IpcRegionRegistry] | None = None,
    identity_provider: Callable[[int], str] | None = None,
    bridge: DeviceBridge | None = None,
    launcher: SignalEpochLauncher | None = None,
    timeout_s: float | None = None,
) -> CollectiveSetup:
    """Probe, exchange and build the DC4 communicator, then hand DC3 a setup.

    Called on **every** rank that runs a non-torch policy, even when the local
    capability is not OK: the ordered descriptor exchange with explicit skip
    markers keeps ranks aligned. When any rank cannot build, every rank
    downgrades its capability reason and returns no factory, so DC3 reaches a
    consensual fallback (``auto``) or refusal (``custom_required``).

    Setup failures after a successful export retain (quarantine) that export
    rather than force-freeing it: the peer may already map it, and freeing
    memory a peer could still read is exactly the failure INV-4 forbids.
    """
    resolved_config = config if config is not None else CustomAllReduceConfig()
    resolved_timeout = DEFAULT_SETUP_TIMEOUT_S if timeout_s is None else float(timeout_s)
    if resolved_timeout <= 0:
        raise ValueError("setup timeout must be positive")
    if control.world_size != group.size or control.rank != group.local_rank:
        raise ValueError("the agreement channel must cover the logical group in local-rank order")
    ordinals = (
        tuple(device.index for device in group.devices)
        if device_ordinals is None
        else tuple(int(entry) for entry in device_ordinals)
    )
    if len(ordinals) != group.size:
        raise ValueError("one device ordinal is required per group rank")
    declaration = custom_all_reduce_capability(resolved_config)

    capability = probe_collective_capability(
        group=group,
        session_generation=session_generation,
        custom=declaration,
        probe=probe,
        timeout_s=max(60.0, resolved_timeout),
        need_ordering=True,
        need_remote_atomic=True,
        device_ordinals=ordinals,
    )
    if group.size == 1:
        return CollectiveSetup(
            control=control,
            capability=capability,
            custom_factory=None,
            timeout_s=resolved_timeout,
        )
    if group.size != 2:
        return CollectiveSetup(
            control=control,
            capability=dataclasses.replace(
                capability, reason=CollectiveReason.UNSUPPORTED_PLATFORM
            ),
            custom_factory=None,
            timeout_s=resolved_timeout,
        )

    resolved_factory: Callable[[int, str | None], IpcRegionRegistry]
    if registry_factory is None:
        resolved_factory = lambda device, uuid: IpcRegionRegistry(  # noqa: E731
            device=device,
            session_generation=session_generation,
            physical_device_uuid=uuid,
        )
    else:
        resolved_factory = registry_factory

    active_bridge: DeviceBridge = bridge if bridge is not None else _TorchDeviceBridge()
    active_launcher: SignalEpochLauncher = (
        launcher if launcher is not None else _TritonSignalEpochLauncher()
    )
    rank = group.local_rank
    device = ordinals[rank]
    peer_rank = group.ranks[1 - rank]
    build = capability.reason is CollectiveReason.OK
    if build and not _launcher_available(active_launcher):
        build = False
        capability = _downgrade(
            capability, CollectiveReason.UNSUPPORTED_PLATFORM, "launcher_unavailable"
        )
    registry: IpcRegionRegistry | None = None
    workspace: Any = None
    own_descriptor: IpcRegionDescriptor | None = None
    wire_descriptor: IpcRegionDescriptor | None = None
    own_uuid = ""

    if build:
        try:
            if identity_provider is None:
                from ayaka.distributed.topology import physical_device_identity

                own_uuid = physical_device_identity(device).uuid
            else:
                own_uuid = str(identity_provider(device))
            registry = resolved_factory(device, None)
            workspace = active_bridge.allocate_workspace(
                int(resolved_config.workspace_bytes), device
            )
            own_descriptor = registry.export(workspace, close_acks=(peer_rank,))
            wire_descriptor = dataclasses.replace(
                own_descriptor,
                exporter_rank=rank,
                physical_device_uuid=own_uuid,
                device=device,
            )
        except BaseException:  # noqa: BLE001 - downgrade with reason
            logger.debug("custom all-reduce workspace export failed", exc_info=True)
            _quarantine_export(registry, "export_failed")
            registry = None
            workspace = None
            own_descriptor = None
            wire_descriptor = None
            build = False
            capability = _downgrade(
                capability, CollectiveReason.WORKSPACE_UNAVAILABLE, "export_failed"
            )

    own_text = _wire_payload(rank, build, wire_descriptor, own_uuid, session_generation)
    try:
        received = _ordered_exchange(control, rank, group.size, own_text, resolved_timeout)
    except BaseException:
        _quarantine_export(registry, "exchange_failed")
        raise
    peers_ready = True
    peer_declined = False
    peer_payload: Mapping[str, Any] | None = None
    for index, text in enumerate(received):
        if index == rank:
            continue
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if not isinstance(decoded, Mapping) or decoded.get("rank") != group.ranks[index]:
            peers_ready = False
            break
        if not decoded.get("build"):
            peer_declined = True
            peers_ready = False
            break
        if tuple(decoded.get("session_generation", ())) != session_generation:
            peers_ready = False
            break
        peer_payload = decoded

    if build and peers_ready and peer_payload is not None and workspace is not None:
        import_registry: IpcRegionRegistry | None = None
        peer_view: IpcRegionView | None = None
        peer_descriptor: IpcRegionDescriptor | None = None
        try:
            peer_descriptor = IpcRegionDescriptor.from_dict(peer_payload["descriptor"])
            if peer_descriptor.session_generation != session_generation:
                raise ValueError("peer descriptor belongs to a different session generation")
            peer_uuid = str(peer_payload.get("physical_device_uuid", "")) or None
            import_registry = resolved_factory(device, peer_uuid)
            peer_view = import_registry.open(peer_descriptor)
            communicator = CustomAllReduce(
                group=group,
                control=control,
                capability=capability,
                session_generation=session_generation,
                device=device,
                config=resolved_config,
                registry=registry,  # type: ignore[arg-type]
                workspace=workspace,
                own_descriptor=own_descriptor,  # type: ignore[arg-type]
                peer_descriptor=peer_descriptor,
                peer_view=peer_view,
                import_registry=import_registry,
                bridge=active_bridge,
                launcher=active_launcher,
                timeout_s=resolved_timeout,
            )
        except BaseException:  # noqa: BLE001 - downgrade with reason
            logger.debug("custom all-reduce peer import failed", exc_info=True)
            for action in (
                (peer_view.close if peer_view is not None else None),
                (
                    import_registry.close(peer_descriptor)
                    if import_registry is not None and peer_descriptor is not None
                    else None
                ),
            ):
                if action is None:
                    continue
                try:
                    action()
                except BaseException:  # noqa: BLE001 - rollback must not mask the cause
                    logger.debug("custom all-reduce import rollback failed", exc_info=True)
            _quarantine_export(registry, "import_failed")
            capability = _downgrade(
                capability, CollectiveReason.WORKSPACE_UNAVAILABLE, "import_failed"
            )
        else:
            return CollectiveSetup(
                control=control,
                capability=capability,
                custom_factory=lambda plan: communicator,
                timeout_s=resolved_timeout,
            )

    if peer_declined and registry is not None:
        # The peer explicitly never allocated or imported this setup, so
        # force-releasing the local export cannot dangle a peer mapping.
        try:
            registry.close_all(force=True)
        except BaseException:  # noqa: BLE001 - retention is the safe fallback
            _quarantine_export(registry, "peer_declined_release_failed")
    else:
        _quarantine_export(registry, "peers_not_ready")
    reason = capability.reason
    if capability.reason is CollectiveReason.OK and (not peers_ready or peer_payload is None):
        reason = CollectiveReason.P2P_UNVERIFIED
    return CollectiveSetup(
        control=control,
        capability=dataclasses.replace(capability, reason=reason),
        custom_factory=None,
        timeout_s=resolved_timeout,
    )
