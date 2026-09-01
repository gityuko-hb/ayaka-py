from __future__ import annotations

import mmap
import os
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from types import TracebackType

from ayaka.exceptions import CheckpointCorruptError
from ayaka.weights.plan import FileReadPlan
from ayaka.weights.spec import WeightSource

def safe_pread(fd: int, size: int, offset: int) -> bytes:
    pread = getattr(os, "pread", None)
    if pread is not None:
        return pread(fd, size, offset)

    # Windows fallback: save current position, seek, read, restore
    current = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, offset, os.SEEK_SET)
        out = bytearray()
        remaining = size
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                raise OSError(f"EOF while reading {size} bytes at offset {offset}")
            out += chunk
            remaining -= len(chunk)
        return bytes(out)
    finally:
        os.lseek(fd, current, os.SEEK_SET)

class ReaderCancelled(CheckpointCorruptError):
    """Raised in place of a read that never ran because the load was cancelled.

    A ``ModelLoadError`` subclass so a caller's single handler covers it, but a
    distinct type so "the checkpoint is broken" and "we gave up" stay
    distinguishable in a log — the second is usually a symptom of the first,
    somewhere else.
    """

@dataclass(frozen=True, slots=True)
class ReadResult:
    """One completed read, tagged with the plan it satisfies."""

    index: int  # submission order
    plan: FileReadPlan
    data: bytes

    def __post_init__(self) -> None:
        if len(self.data) != self.plan.nbytes:
            raise CheckpointCorruptError(
                f"{self.plan.file_uri}: read {len(self.data)} of {self.plan.nbytes} B at "
                f"offset {self.plan.byte_offset} — short read, usually a truncated file"
            )
            
class BoundedCheckpointReader:
    """Executes read plans.  Owns file handles; owns no tensors.

    Not a context manager by accident — the handle cache must be closed
    explicitly, and a load that dies mid-stream must still release descriptors
    on the way out.
    """

    __slots__ = ("_cancel", "_handles", "_lock", "_max_inflight", "_pool", "_use_mmap")

    def __init__(self, *, workers: int = 4, max_inflight: int = 8, use_mmap: bool = False) -> None:
        if workers < 1:
            raise ValueError("reader needs at least one worker")
        if max_inflight < workers:
            raise ValueError(
                f"max_inflight={max_inflight} below workers={workers}: the queue would "
                "throttle below the pool's own width, which is never what was meant"
            )
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ayaka-read")
        self._max_inflight = max_inflight
        self._use_mmap = use_mmap
        self._cancel = threading.Event()
        self._handles: dict[str, int] = {}
        self._lock = threading.Lock()
        
    def __enter__(self) -> BoundedCheckpointReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Cancel first, then join: a pool shut down while workers are still
        # queueing reads takes as long as the whole remaining plan.
        if exc is not None:
            self.cancel()
        self.close()
        
    def cancel(self) -> None:
        """Cooperative.  Reads already in flight finish; queued ones do not start."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            for fd in self._handles.values():
                with suppress(OSError):
                    os.close(fd)
            self._handles.clear()

    def _fd(self, uri: str) -> int:
        with self._lock:
            fd = self._handles.get(uri)
            if fd is None:
                try:
                    fd = os.open(uri, os.O_RDONLY | os.O_BINARY if hasattr(os, "O_BINARY") else 0)
                except OSError as exc:
                    raise CheckpointCorruptError(f"{uri}: cannot open: {exc}") from exc
                self._handles[uri] = fd
            return fd

    def _read_one(self, plan: FileReadPlan) -> bytes:
        if self._cancel.is_set():
            raise ReaderCancelled(f"{plan.file_uri}: load cancelled before this read started")
        fd = self._fd(plan.file_uri)
        if plan.is_contiguous:
            return self._pread(fd, plan.file_uri, plan.byte_offset, plan.nbytes)

        # Strided: `run_count` runs of `run_bytes`, `stride_bytes` apart.  The
        # runs are concatenated in order, which is exactly the layout the
        # destination shard wants — no reassembly step downstream.
        chunks: list[bytes] = []
        offset = plan.byte_offset
        for _ in range(plan.run_count):
            if self._cancel.is_set():
                raise ReaderCancelled(f"{plan.file_uri}: cancelled mid-strided-read")
            chunks.append(self._pread(fd, plan.file_uri, offset, plan.run_bytes))
            offset += plan.stride_bytes or plan.run_bytes
        return b"".join(chunks)

    def _pread(self, fd: int, uri: str, offset: int, nbytes: int) -> bytes:
        if self._use_mmap:
            return self._mmap_read(fd, uri, offset, nbytes)
        out = bytearray()
        remaining = nbytes
        pos = offset
        while remaining:
            # os.pread may return fewer bytes than asked on any platform; the
            # loop is not defensive padding, it is the documented contract, and
            # skipping it produces a short buffer that only shows up as garbage
            # weights on large files.
            try:
                chunk = safe_pread(fd, remaining, pos)
            except OSError as exc:
                raise CheckpointCorruptError(f"{uri}: pread failed at {pos}: {exc}") from exc
            if not chunk:
                raise CheckpointCorruptError(
                    f"{uri}: EOF at offset {pos} with {remaining} B still expected"
                )
            out += chunk
            pos += len(chunk)
            remaining -= len(chunk)
        return bytes(out)

    def _mmap_read(self, fd: int, uri: str, offset: int, nbytes: int) -> bytes:
        # ALLOCATION_GRANULARITY is the mmap page size on all platforms, and mmap() requires
        # the offset to be page-aligned.  The read window is expanded to the nearest page boundary
        # and the returned slice is adjusted to the requested offset.
        page = mmap.ALLOCATIONGRANULARITY
        base = (offset // page) * page
        span = (offset - base) + nbytes
        try:
            with mmap.mmap(fd, span, offset=base, access=mmap.ACCESS_READ) as mapped:
                start = offset - base
                return bytes(mapped[start : start + nbytes])
        except (OSError, ValueError) as exc:
            raise CheckpointCorruptError(f"{uri}: mmap failed at {offset}: {exc}") from exc

    def read_all(self, plans: Sequence[FileReadPlan]) -> Iterator[ReadResult]:
        """Yield results in submission order, never more than ``max_inflight`` live.

        Submission order rather than completion order because the transform and
        H2D stages downstream must see weights in plan order to bind them, and
        because it makes the error deterministic (see the module docstring).
        """
        pending: dict[int, Future[bytes]] = {}
        next_to_yield = 0
        try:
            for index, plan in enumerate(plans):
                while len(pending) >= self._max_inflight:
                    yield self._collect(pending, next_to_yield, plans)
                    next_to_yield += 1
                pending[index] = self._pool.submit(self._read_one, plan)
            while pending:
                yield self._collect(pending, next_to_yield, plans)
                next_to_yield += 1
        except BaseException:
            # Any failure — including the consumer throwing into the generator —
            # stops the remaining reads.  Without this, a caller that abandons
            # the iterator leaves workers reading a checkpoint nobody wants.
            self.cancel()
            for future in pending.values():
                future.cancel()
            raise

    def _collect(
        self, pending: dict[int, Future[bytes]], index: int, plans: Sequence[FileReadPlan]
    ) -> ReadResult:
        future = pending.pop(index)
        data = future.result()
        return ReadResult(index=index, plan=plans[index], data=data)

    def read_one(self, plan: FileReadPlan) -> bytes:
        """Synchronous single read.  For discovery paths and tests."""
        return self._read_one(plan)

    # Satisfied *structurally*.  This module does not import `ayaka.weights` —
    # the two are peers (node 1.04) and the Protocol lives with the consumer
    # that needs it.  `deployment.bootstrap` is the only place that names both.

    def readinto(self, source: WeightSource, *, offset: int, destination: memoryview) -> int:
        """Fill a caller-owned window.  The buffer belongs to the pipeline.

        ``readinto`` rather than ``read`` because a reader that returns bytes
        allocates them, and then two objects own the staging budget while only
        one is counted against it.
        """
        nbytes = len(destination)
        if nbytes == 0:
            return 0
        if self._cancel.is_set():
            raise ReaderCancelled(f"{source.uri}: load cancelled before this read started")
        data = self._pread(self._fd(source.uri), source.uri, offset, nbytes)
        if len(data) != nbytes:  # pragma: no cover - _pread raises first
            raise CheckpointCorruptError(
                f"{source.uri}: filled {len(data)} of {nbytes} B at offset {offset}"
            )
        destination[:] = data
        return nbytes
