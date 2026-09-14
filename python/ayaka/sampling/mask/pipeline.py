"""MaskPipeline -- trai tim cua thiet ke."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ayaka.device.backend import DeviceBackend, get_backend
from ayaka.sampling.mask.arena import BitmaskArena, MaskHandle
from ayaka.sampling.mask.producer import MaskProducer


@dataclass(slots=True)
class MaskEntry:
    producer: MaskProducer
    logits_row: int
    mask_row: int
    stream_idx: int = 0
    propose_step: int = 0
    draft: tuple[int, ...] = ()
    commit_tokens: tuple[int, ...] = ()


class MaskPipelineError(RuntimeError):
    pass


class MaskPipeline:
    __slots__ = (
        "_arena",
        "_backend",
        "_caps",
        "_closed",
        "_copy_stream",
        "_lock",
        "_n_rows",
        "_pending",
        "_pool",
    )

    def __init__(
        self,
        arena: BitmaskArena,
        n_workers: int = 4,
        *,
        backend: DeviceBackend | None = None,
    ):
        self._arena = arena
        self._backend = backend if backend is not None else get_backend()
        self._pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ayaka-mask")
        self._copy_stream = self._backend.create_stream(arena.device.index or 0, 0)
        self._pending: list[Future] | None = None
        self._n_rows = 0
        self._caps: np.ndarray | None = None
        self._lock = threading.Lock()
        self._closed = False

    def launch(self, entries: list[MaskEntry], n_streams: int = 0) -> None:
        if self._closed:
            raise MaskPipelineError("pipeline da shutdown")
        if self._pending is not None:
            raise MaskPipelineError("launch() goi hai lan ma chua gather()")
        if not entries:
            self._n_rows = 0
            self._caps = None
            return

        rows, row_indices = self._arena.begin_step()
        n_rows = sum(e.propose_step + 1 for e in entries)
        if n_rows > self._arena.max_rows:
            raise MaskPipelineError(f"can {n_rows} row, arena chi co {self._arena.max_rows}")

        caps = np.full(max(n_streams, 1), 1 << 30, dtype=np.int32)
        for e in entries:
            if not 0 <= e.stream_idx < caps.size:
                raise MaskPipelineError(
                    f"stream_idx {e.stream_idx} ngoai {caps.size} cap (n_streams={n_streams})"
                )

        def one(e: MaskEntry) -> None:
            if e.commit_tokens:
                e.producer.commit(e.commit_tokens)
            span = e.propose_step + 1
            cap = e.producer.emit(e.draft, rows.window(e.mask_row, span))
            caps[e.stream_idx] = min(int(caps[e.stream_idx]), int(cap))
            for k in range(span):
                row_indices[e.mask_row + k] = e.logits_row + k

        self._caps = caps
        self._n_rows = n_rows
        self._pending = [self._pool.submit(one, e) for e in entries]

    def gather(self, compute_stream: Any = None) -> MaskHandle | None:
        if self._pending is None:
            return None
        futs, self._pending = self._pending, None
        errs = [e for e in (f.exception() for f in futs) if e is not None]
        if errs:
            raise MaskPipelineError(f"{len(errs)} producer loi; dau tien: {errs[0]!r}") from errs[0]
        n_rows = self._n_rows
        if n_rows == 0:
            return None
        g, rg, ready = self._arena.upload(n_rows, self._copy_stream)
        if compute_stream is not None:
            self._backend.wait(compute_stream, ready)
        return MaskHandle(
            masks=g,
            row_indices=rg,
            n_rows=n_rows,
            vocab_size=self._arena.vocab_size,
            spec_caps=self._caps,
        )

    def mark_consumed(self, compute_stream: Any = None) -> None:
        self._arena.mark_consumed(compute_stream)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)
        self._backend.destroy_stream(self._copy_stream)
