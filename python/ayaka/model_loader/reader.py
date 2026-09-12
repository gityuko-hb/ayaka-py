"""Executes read plans against local safetensors files.

Failures are deterministic: :meth:`BoundedCheckpointReader.read_all` yields
results in submission order, so a run that reports a bad read at plan index
``k`` reports the same bad read at index ``k`` on every rerun and every rank —
a load either fails identically everywhere or succeeds identically everywhere.
"""

from __future__ import annotations

import mmap
import os
import threading
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from types import TracebackType

from ayaka.exceptions import CheckpointCorruptError
from ayaka.weights.plan import FileReadPlan
from ayaka.weights.spec import WeightSource

_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0)


def safe_pread(fd: int, size: int, offset: int) -> bytes:
    """Read ``size`` bytes at ``offset`` without disturbing the file position.

    The caller must own the fd exclusively — i.e. one fd per thread.  ``os.pread``
    is atomic with respect to the file offset, but the Windows fallback below is
    a ``lseek``/``read`` pair over a shared position and is only safe when no
    other thread touches the same descriptor concurrently.
    :class:`BoundedCheckpointReader` guarantees that with its per-thread fd
    cache; sharing one fd across worker threads here would silently interleave
    two threads' seek and read.
    """
    pread = getattr(os, "pread", None)
    if pread is not None:
        return pread(fd, size, offset)

    # Windows fallback (no os.pread): save position, seek, read, restore.  Safe
    # only under the one-fd-per-thread discipline documented above.
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

    __slots__ = (
        "_cancel",
        "_local",
        "_lock",
        "_max_inflight",
        "_pool",
        "_thread_maps",
        "_use_mmap",
    )

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
        # One fd per thread.  os.pread does not exist on Windows, so the read
        # path falls back to lseek+read over the *file position* — a shared fd
        # would let two workers interleave seek and read and silently read each
        # other's bytes.  Each worker thread therefore opens its own descriptor;
        # the registry below is what close() uses to release all of them.
        self._local = threading.local()
        self._thread_maps: list[dict[str, int]] = []
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
            for handles in self._thread_maps:
                for fd in handles.values():
                    with suppress(OSError):
                        os.close(fd)
                handles.clear()
            self._thread_maps.clear()

    def _fd(self, uri: str) -> int:
        handles: dict[str, int] | None = getattr(self._local, "handles", None)
        if handles is None:
            handles = {}
            self._local.handles = handles
            with self._lock:
                self._thread_maps.append(handles)
        fd = handles.get(uri)
        if fd is None:
            try:
                fd = os.open(uri, _OPEN_FLAGS)
            except OSError as exc:
                raise CheckpointCorruptError(f"{uri}: cannot open: {exc}") from exc
            handles[uri] = fd
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
            offset += plan.stride_bytes
        return b"".join(chunks)

    def _pread(self, fd: int, uri: str, offset: int, nbytes: int) -> bytes:
        out = bytearray(nbytes)
        self._read_into(fd, uri, offset, memoryview(out))
        return bytes(out)

    def _read_into(self, fd: int, uri: str, offset: int, destination: memoryview) -> int:
        """Fill ``destination`` completely from ``offset``, whatever it takes.

        One dispatcher for both backends so ``readinto`` and the bytes-returning
        ``_pread`` share a single code path — a fix applied to one cannot drift
        from the other.
        """
        if self._use_mmap:
            self._mmap_into(fd, uri, offset, destination)
            return len(destination)
        return self._pread_into(fd, uri, offset, destination)

    def _pread_into(self, fd: int, uri: str, offset: int, destination: memoryview) -> int:
        """pread directly into the caller's buffer, in bounded chunks.

        ``os.pread`` may return fewer bytes than asked on any platform; the
        loop is not defensive padding, it is the documented contract, and
        skipping it produces a short buffer that only shows up as garbage
        weights on large files.  Each chunk is copied into the destination and
        released, so the transient allocation stays at chunk size instead of a
        second full copy of the tensor.
        """
        filled = 0
        remaining = len(destination)
        pos = offset
        while remaining:
            try:
                chunk = safe_pread(fd, remaining, pos)
            except OSError as exc:
                raise CheckpointCorruptError(f"{uri}: pread failed at {pos}: {exc}") from exc
            if not chunk:
                raise CheckpointCorruptError(
                    f"{uri}: EOF at offset {pos} with {remaining} B still expected"
                )
            destination[filled : filled + len(chunk)] = chunk
            filled += len(chunk)
            pos += len(chunk)
            remaining -= len(chunk)
        return filled

    def _mmap_into(self, fd: int, uri: str, offset: int, destination: memoryview) -> None:
        # ALLOCATION_GRANULARITY is the mmap page size on all platforms, and mmap() requires
        # the offset to be page-aligned.  The read window is expanded to the nearest page boundary
        # and the returned slice is adjusted to the requested offset.
        nbytes = len(destination)
        page = mmap.ALLOCATIONGRANULARITY
        base = (offset // page) * page
        span = (offset - base) + nbytes
        try:
            with mmap.mmap(fd, span, offset=base, access=mmap.ACCESS_READ) as mapped:
                start = offset - base
                destination[:] = mapped[start : start + nbytes]
        except (OSError, ValueError) as exc:
            raise CheckpointCorruptError(f"{uri}: mmap failed at {offset}: {exc}") from exc

    def read_all(self, plans: Sequence[FileReadPlan]) -> Generator[ReadResult]:
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
        one is counted against it.  The bytes land in ``destination`` directly —
        mmap slices into the view in one copy, pread fills it in bounded
        chunks — with no full-size intermediate buffer.
        """
        nbytes = len(destination)
        if nbytes == 0:
            return 0
        if self._cancel.is_set():
            raise ReaderCancelled(f"{source.uri}: load cancelled before this read started")
        return self._read_into(self._fd(source.uri), source.uri, offset, destination)
