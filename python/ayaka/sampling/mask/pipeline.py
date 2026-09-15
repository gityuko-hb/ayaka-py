"""MaskPipeline -- trai tim cua thiet ke.

A4b — đa producer per logits row:
  * MỘT entry per (slot, producer). Entry thứ tự khai báo 0 là PRIMARY
    (window chính, ghi row_indices); các entry sau là SCRATCH: pipeline cấp
    window riêng sau toàn bộ primary block, producer emit vào đó rồi gather
    INTERSECT (AND) vào window chính trước upload.
  * Producers khai Cap.COMMUTATIVE chạy SONG SONG qua thread-pool (mỗi
    producer một future). Slot có producer thiếu COMMUTATIVE chạy CHUỖI tuần
    tự theo thứ tự khai báo — một future duy nhất, giữ an toàn state.
  * draft acceptance của slot = min qua các producer (mỗi producer tự
    rollback phần accepted của mình nên không cần rollback chéo).
  * Caps của MaskHandle = AND caps của mọi producer; accepted_per_stream =
    min accepted count per stream.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ayaka.caps import Cap
from ayaka.device.backend import DeviceBackend, get_backend
from ayaka.sampling.mask.arena import BitmaskArena, MaskHandle
from ayaka.sampling.mask.producer import MaskProducer, MaskRows

_SENTINEL_CAP = 1 << 30


@dataclass(slots=True)
class MaskEntry:
    producer: MaskProducer
    logits_row: int
    mask_row: int
    stream_idx: int = 0
    propose_step: int = 0
    draft: tuple[int, ...] = ()
    commit_tokens: tuple[int, ...] = ()
    # A4b — entry scratch: window riêng, AND vào window chính ở gather().
    scratch: bool = False


class MaskPipelineError(RuntimeError):
    pass


class MaskPipeline:
    __slots__ = (
        "_arena",
        "_backend",
        "_caps",
        "_closed",
        "_copy_stream",
        "_handle_caps",
        "_lock",
        "_n_rows",
        "_pending",
        "_pool",
        "_rows_np",
        "_scratch_groups",
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
        self._handle_caps = Cap.NONE
        self._rows_np: MaskRows | None = None
        self._scratch_groups: list[tuple[int, int, list[int]]] | None = None
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
            self._scratch_groups = None
            return

        rows, row_indices = self._arena.begin_step()
        primary = [e for e in entries if not e.scratch]
        n_rows = sum(e.propose_step + 1 for e in primary)
        scratch_total = sum(e.propose_step + 1 for e in entries if e.scratch)
        if n_rows + scratch_total > self._arena.max_rows:
            raise MaskPipelineError(
                f"can {n_rows + scratch_total} row (kem scratch), "
                f"arena chi co {self._arena.max_rows}"
            )

        caps = np.full(max(n_streams, 1), _SENTINEL_CAP, dtype=np.int32)
        for e in entries:
            if not 0 <= e.stream_idx < caps.size:
                raise MaskPipelineError(
                    f"stream_idx {e.stream_idx} ngoai {caps.size} cap (n_streams={n_streams})"
                )

        handle_caps = Cap.ARGMAX_INVARIANT | Cap.SPEC_VERIFIABLE | Cap.COMMUTATIVE
        for e in entries:
            handle_caps &= getattr(e.producer, "caps", Cap.NONE)

        # Nhóm theo logits_row, giữ THỨ TỰ KHAI BÁO (insertion order của dict
        # + sort stable primary-trước).
        groups: dict[int, list[MaskEntry]] = {}
        for e in entries:
            groups.setdefault(e.logits_row, []).append(e)

        scratch_groups: list[tuple[int, int, list[int]]] = []
        futures: list[Future] = []
        for slot_entries in groups.values():
            ordered = sorted(slot_entries, key=lambda e: e.scratch)
            commutative = all(
                getattr(e.producer, "caps", Cap.NONE) & Cap.COMMUTATIVE for e in ordered
            )
            if commutative:
                for e in ordered:
                    futures.append(
                        self._pool.submit(
                            self._emit_entry, e, rows, row_indices, caps, self._lock
                        )
                    )
            else:
                futures.append(
                    self._pool.submit(
                        self._emit_chain, ordered, rows, row_indices, caps, self._lock
                    )
                )
            if len(ordered) > 1:
                scratch_groups.append(
                    (
                        ordered[0].mask_row,
                        ordered[0].propose_step + 1,
                        [e.mask_row for e in ordered[1:]],
                    )
                )

        self._caps = caps
        self._handle_caps = handle_caps
        self._n_rows = n_rows
        self._rows_np = rows
        self._scratch_groups = scratch_groups or None
        self._pending = futures

    @staticmethod
    def _emit_entry(
        e: MaskEntry,
        rows: MaskRows,
        row_indices: np.ndarray,
        caps: np.ndarray,
        lock: threading.Lock,
    ) -> None:
        if e.commit_tokens:
            e.producer.commit(e.commit_tokens)
        span = e.propose_step + 1
        cap = e.producer.emit(e.draft, rows.window(e.mask_row, span))
        with lock:  # min read-modify-write — 2 producer cùng stream có race
            caps[e.stream_idx] = min(int(caps[e.stream_idx]), int(cap))
        if not e.scratch:
            for k in range(span):
                row_indices[e.mask_row + k] = e.logits_row + k

    @classmethod
    def _emit_chain(
        cls,
        entries: list[MaskEntry],
        rows: MaskRows,
        row_indices: np.ndarray,
        caps: np.ndarray,
        lock: threading.Lock,
    ) -> None:
        """Chuỗi tuần tự (thứ tự khai báo); acceptance = min qua producers."""
        accepted_min = _SENTINEL_CAP
        for e in entries:
            if e.commit_tokens:
                e.producer.commit(e.commit_tokens)
            span = e.propose_step + 1
            cap = e.producer.emit(e.draft, rows.window(e.mask_row, span))
            accepted_min = min(accepted_min, int(cap))
        primary = entries[0]
        with lock:  # min read-modify-write nhất quán với _emit_entry
            caps[primary.stream_idx] = min(int(caps[primary.stream_idx]), accepted_min)
        for k in range(primary.propose_step + 1):
            row_indices[primary.mask_row + k] = primary.logits_row + k

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
        # A4b — intersect scratch windows vào window chính (AND) TRƯỚC upload;
        # cùng parity buffer với launch — không gọi begin_step lần hai.
        rows_ref = self._rows_np
        if self._scratch_groups and rows_ref is not None:
            for primary_base, span, scratch_bases in self._scratch_groups:
                for k in range(span):
                    acc = rows_ref.raw(primary_base + k)
                    for base in scratch_bases:
                        acc &= rows_ref.raw(base + k)
        g, rg, ready = self._arena.upload(n_rows, self._copy_stream)
        if compute_stream is not None:
            self._backend.wait(compute_stream, ready)
        return MaskHandle(
            masks=g,
            row_indices=rg,
            n_rows=n_rows,
            vocab_size=self._arena.vocab_size,
            caps=self._handle_caps,
            accepted_per_stream=self._caps,
        )

    def mark_consumed(self, compute_stream: Any = None) -> None:
        self._arena.mark_consumed(compute_stream)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)
        self._backend.destroy_stream(self._copy_stream)
