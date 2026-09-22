from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from ayaka.distributed.metadata import DistributedKVMetadata
from ayaka.distributed.topology import _is_cuda_device, _optional_torch
from ayaka.exceptions import InvariantViolationError, RuntimeMemoryError, StorageUnavailableError


class DistributedStepError(RuntimeMemoryError):
    """A step failed on at least one rank and was cleaned up group-wide."""


class DistributedCollectiveTimeout(DistributedStepError):
    """A control-plane collective did not finish before its deadline."""


class TorchDistributedKVProcessGroup:
    """NCCL/Gloo collectives used by the KV control plane.

    The caller initializes ``torch.distributed`` and chooses the group. Every
    operation uses ``async_op=True`` and waits with an explicit deadline, so a
    wedged rank becomes a fail-closed control-plane error rather than an
    unbounded scheduler hang.
    """

    def __init__(
        self,
        *,
        group: Any | None = None,
        device: str | None = None,
        default_timeout_s: float = 30.0,
        max_metadata_bytes: int = 64 * 1024 * 1024,
        owns_process_group: bool = False,
    ) -> None:
        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")
        if not isinstance(max_metadata_bytes, int) or isinstance(max_metadata_bytes, bool):
            raise TypeError("max_metadata_bytes must be an integer")
        if max_metadata_bytes <= 0:
            raise ValueError("max_metadata_bytes must be positive")
        if type(owns_process_group) is not bool:
            raise TypeError("owns_process_group must be a boolean")
        torch = _optional_torch()
        if torch is None or not torch.distributed.is_available():
            raise StorageUnavailableError("torch.distributed is unavailable")
        dist = torch.distributed
        if not dist.is_initialized():
            raise StorageUnavailableError("torch.distributed is not initialized")
        self._torch = torch
        self._dist = dist
        self._group = group
        self._rank = int(dist.get_rank(group))
        self._world_size = int(dist.get_world_size(group))
        self._backend = str(dist.get_backend(group)).lower()
        self._default_timeout_s = float(default_timeout_s)
        self._max_metadata_bytes = max_metadata_bytes
        self._owns_process_group = owns_process_group
        self._destroyed = False
        self._shutdown_complete = False
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if "nccl" in self._backend else "cpu"
        if "nccl" in self._backend and not _is_cuda_device(device):
            raise ValueError("an NCCL control group requires a CUDA control device")
        self._device = str(device)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def default_timeout_s(self) -> float:
        """Deadline used when a collective is called without an explicit one."""
        return self._default_timeout_s

    @property
    def owns_process_group(self) -> bool:
        """Whether this process initialized the default group and may destroy it."""
        return self._owns_process_group

    def _collective(
        self,
        operation: str,
        submit: Callable[[], Any],
        *,
        timeout_s: float | None,
    ) -> None:
        """Submit one async collective and wait with the shared deadline.

        A submission failure (a peer that already tore its transport down, a
        closed communicator) is a collective failure exactly like a wait
        timeout: the caller must fail closed, so both become
        :class:`DistributedCollectiveTimeout` instead of leaking a raw backend
        error into the control plane.
        """
        try:
            work = submit()
        except BaseException as exc:
            raise DistributedCollectiveTimeout(
                f"{operation} could not be submitted on rank {self._rank}: {exc}"
            ) from exc
        self._wait(work, timeout_s=timeout_s, operation=operation)

    def _wait(self, work: Any, *, timeout_s: float | None, operation: str) -> None:
        timeout = self._default_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ValueError("collective timeout must be positive")
        try:
            completed = work.wait(timeout=timedelta(seconds=timeout))
        except BaseException as exc:
            raise DistributedCollectiveTimeout(
                f"{operation} failed or timed out on rank {self._rank}: {exc}"
            ) from exc
        if completed is False:
            raise DistributedCollectiveTimeout(
                f"{operation} timed out after {timeout:.3f}s on rank {self._rank}"
            )

    def all_grant(self, local_grant: bool, *, timeout_s: float | None = None) -> bool:
        tensor = self._torch.tensor(
            [1 if local_grant else 0],
            dtype=self._torch.int32,
            device=self._device,
        )
        self._collective(
            "KV all-grant",
            lambda: self._dist.all_reduce(
                tensor,
                op=self._dist.ReduceOp.MIN,
                group=self._group,
                async_op=True,
            ),
            timeout_s=timeout_s,
        )
        return bool(int(tensor.item()))

    def barrier(self, *, timeout_s: float | None = None) -> None:
        self._collective(
            "KV barrier",
            lambda: self._dist.barrier(group=self._group, async_op=True),
            timeout_s=timeout_s,
        )

    def shutdown(self, local_accepting: bool, *, timeout_s: float | None = None) -> bool:
        """Agreement vote plus a final barrier; True only on a unanimous accept.

        Every rank votes exactly once through the all-grant minimum: a rank
        with an active flight or a failed incarnation votes no. On unanimity
        every rank has passed the vote before any of them proceeds, and the
        barrier orders the teardown so no rank destroys or frees anything while
        a peer still expects a collective. A refusal never enters the barrier —
        the refusing peer may be wedged — so the caller must fail closed. A
        repeat call after a completed handshake is a no-op that returns True:
        the barrier proved every rank finished the first one.
        """
        if self._shutdown_complete:
            return True
        agreed = self.all_grant(bool(local_accepting), timeout_s=timeout_s)
        if agreed:
            self.barrier(timeout_s=timeout_s)
            self._shutdown_complete = True
        return agreed

    def destroy(self, *, force: bool = False) -> None:
        """Destroy the default process group when this process owns it.

        An adopted group (launcher-initialized or explicitly supplied) is left
        untouched: only the initializer may tear the communicator down. Without
        ``force`` this refuses to run before a completed shutdown handshake,
        because destroying a communicator a peer still expects is precisely the
        teardown race the shutdown barrier exists to prevent. Idempotent.
        """
        if not self._owns_process_group or self._destroyed:
            return
        if not self._shutdown_complete and not force:
            raise RuntimeError(
                "refusing to destroy the process group before a clean coordinated shutdown"
            )
        self._destroyed = True
        self._dist.destroy_process_group()

    def _broadcast_tensor(self, tensor: Any, *, source_rank: int) -> Any:
        """Broadcast from a process-group-local rank across PyTorch versions."""
        try:
            return self._dist.broadcast(
                tensor,
                group_src=source_rank,
                group=self._group,
                async_op=True,
            )
        except TypeError:
            global_source = source_rank
            if self._group is not None and hasattr(self._dist, "get_global_rank"):
                global_source = int(self._dist.get_global_rank(self._group, source_rank))
            return self._dist.broadcast(
                tensor,
                src=global_source,
                group=self._group,
                async_op=True,
            )

    def broadcast_text(
        self,
        text: str | None,
        *,
        source_rank: int,
        timeout_s: float | None = None,
    ) -> str:
        """Broadcast one canonical UTF-8 payload with a length prefix.

        The source rank supplies the text; every other rank supplies ``None``.
        The received bytes are returned verbatim so each caller decodes with
        its own checksummed envelope type — this method never interprets the
        payload, only its transport framing.
        """
        if not 0 <= source_rank < self._world_size:
            raise ValueError("source_rank outside process group")
        if self._rank == source_rank:
            if text is None:
                raise ValueError("source rank must provide broadcast text")
            encoded = text.encode("utf-8")
            size = self._torch.tensor([len(encoded)], dtype=self._torch.int64, device=self._device)
        else:
            encoded = b""
            size = self._torch.zeros(1, dtype=self._torch.int64, device=self._device)

        work = self._broadcast_tensor(size, source_rank=source_rank)
        self._wait(work, timeout_s=timeout_s, operation="distributed payload size broadcast")
        payload_size = int(size.item())
        if payload_size <= 0:
            raise InvariantViolationError("distributed broadcast payload is empty")
        if payload_size > self._max_metadata_bytes:
            raise InvariantViolationError(
                "distributed broadcast payload exceeds the configured byte limit"
            )
        if self._rank == source_rank:
            payload = self._torch.tensor(
                list(encoded), dtype=self._torch.uint8, device=self._device
            )
        else:
            payload = self._torch.empty(payload_size, dtype=self._torch.uint8, device=self._device)
        work = self._broadcast_tensor(payload, source_rank=source_rank)
        self._wait(work, timeout_s=timeout_s, operation="distributed payload broadcast")
        return bytes(payload.cpu().tolist()).decode("utf-8")

    def broadcast_metadata(
        self,
        metadata: DistributedKVMetadata | None,
        *,
        source_rank: int,
        timeout_s: float | None = None,
    ) -> DistributedKVMetadata:
        """Broadcast canonical metadata bytes on NCCL or Gloo."""
        text = None
        if self._rank == source_rank:
            if metadata is None:
                raise ValueError("source rank must provide metadata")
            text = metadata.encode()
        received = self.broadcast_text(text, source_rank=source_rank, timeout_s=timeout_s)
        result = DistributedKVMetadata.decode(received)
        if metadata is not None and result.sha256 != metadata.sha256:
            raise InvariantViolationError("source metadata changed during broadcast")
        return result
