from __future__ import annotations

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
    ) -> None:
        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")
        if not isinstance(max_metadata_bytes, int) or isinstance(max_metadata_bytes, bool):
            raise TypeError("max_metadata_bytes must be an integer")
        if max_metadata_bytes <= 0:
            raise ValueError("max_metadata_bytes must be positive")
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
        if device is None:
            device = (
                f"cuda:{torch.cuda.current_device()}"
                if "nccl" in self._backend
                else "cpu"
            )
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
        work = self._dist.all_reduce(
            tensor,
            op=self._dist.ReduceOp.MIN,
            group=self._group,
            async_op=True,
        )
        self._wait(work, timeout_s=timeout_s, operation="KV all-grant")
        return bool(int(tensor.item()))

    def barrier(self, *, timeout_s: float | None = None) -> None:
        work = self._dist.barrier(group=self._group, async_op=True)
        self._wait(work, timeout_s=timeout_s, operation="KV barrier")

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

    def broadcast_metadata(
        self,
        metadata: DistributedKVMetadata | None,
        *,
        source_rank: int,
        timeout_s: float | None = None,
    ) -> DistributedKVMetadata:
        """Broadcast canonical bytes with a length prefix on NCCL or Gloo."""
        if not 0 <= source_rank < self._world_size:
            raise ValueError("source_rank outside process group")
        if self._rank == source_rank:
            if metadata is None:
                raise ValueError("source rank must provide metadata")
            encoded = metadata.encode().encode("utf-8")
            size = self._torch.tensor(
                [len(encoded)], dtype=self._torch.int64, device=self._device
            )
        else:
            encoded = b""
            size = self._torch.zeros(1, dtype=self._torch.int64, device=self._device)

        work = self._broadcast_tensor(size, source_rank=source_rank)
        self._wait(work, timeout_s=timeout_s, operation="KV metadata size broadcast")
        payload_size = int(size.item())
        if payload_size <= 0:
            raise InvariantViolationError("distributed KV metadata payload is empty")
        if payload_size > self._max_metadata_bytes:
            raise InvariantViolationError(
                "distributed KV metadata exceeds the configured byte limit"
            )
        if self._rank == source_rank:
            payload = self._torch.tensor(
                list(encoded), dtype=self._torch.uint8, device=self._device
            )
        else:
            payload = self._torch.empty(
                payload_size, dtype=self._torch.uint8, device=self._device
            )
        work = self._broadcast_tensor(payload, source_rank=source_rank)
        self._wait(work, timeout_s=timeout_s, operation="KV metadata broadcast")
        decoded = bytes(payload.cpu().tolist()).decode("utf-8")
        result = DistributedKVMetadata.decode(decoded)
        if metadata is not None and result.sha256 != metadata.sha256:
            raise InvariantViolationError("source metadata changed during broadcast")
        return result