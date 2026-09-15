"""Multi-producer mask pipeline with parallel emit and AND-gather.

This module is the core of the Tier-1 design and supports multiple producers
per logits row:

* One entry per ``(slot, producer)``. The entry in declaration order 0 is the
  PRIMARY: it owns the main window and writes ``row_indices``. Later entries
  are SCRATCH: the pipeline assigns them private windows after the whole
  primary block, each producer emits there, and :meth:`MaskPipeline.gather`
  intersects (AND) scratch windows into the primary window before upload.
* Producers advertising :attr:`~ayaka.caps.Cap.COMMUTATIVE` run in parallel
  through a thread pool (one future per producer). A slot with any
  non-commutative producer runs sequentially in declaration order as a single
  future, keeping stateful backends safe.
* Per-slot draft acceptance is the minimum over its producers. Each producer
  rolls back its own accepted prefix, so no cross-producer rollback is needed.
* The :class:`~ayaka.sampling.mask.arena.MaskHandle` caps are the AND of all
  producer caps; ``accepted_per_stream`` is the per-stream minimum accepted
  count.
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
    """One producer's work item for a single decode step.

    Attributes:
        producer: Constraint source that fills the assigned window.
        logits_row: Logits row (slot) this entry constrains.
        mask_row: First arena row of the assigned window (primary or
            private scratch window).
        stream_idx: Stream index used for the ``accepted_per_stream``
            minimum aggregation.
        propose_step: Number of speculative draft tokens; the window span
            is always ``propose_step + 1`` (draft rows plus bonus row).
        draft: Draft token ids to verify.
        commit_tokens: Already-sampled tokens to commit before emitting.
        scratch: True for non-primary entries. Scratch entries emit into a
            private window that :meth:`MaskPipeline.gather` ANDs into the
            primary window.
    """

    producer: MaskProducer
    logits_row: int
    mask_row: int
    stream_idx: int = 0
    propose_step: int = 0
    draft: tuple[int, ...] = ()
    commit_tokens: tuple[int, ...] = ()
    scratch: bool = False


class MaskPipelineError(RuntimeError):
    """Raised for pipeline misuse or producer failures.

    Covers shutdown reuse, double ``launch()`` without ``gather()``,
    capacity overflow, out-of-range ``stream_idx``, and aggregated
    producer exceptions from :meth:`MaskPipeline.gather`.
    """


class MaskPipeline:
    """Emit masks in parallel on CPU, then upload once to GPU.

    The typical cycle per step is ``launch(entries)`` -> ``gather()`` ->
    ``mark_consumed()``. ``launch`` only submits work to the thread pool;
    ``gather`` blocks on it, intersects scratch windows, and uploads.
    """

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
        """Create a pipeline bound to one arena.

        Args:
            arena: Bitmask arena owning the staging buffers.
            n_workers: Thread-pool size for parallel producer emit.
            backend: Device backend for streams/events. Defaults to the
                global backend.
        """
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
        """Begin a step by submitting producer work to the pool.

        Groups entries by ``logits_row`` in declaration order, sorts each
        group primary-first (stable), and submits either one future per
        producer (all commutative) or a single sequential-chain future
        (any non-commutative producer). Computes the AND of all producer
        caps for the resulting handle.

        Args:
            entries: Work items for this step. May be empty, which resets
                the pending state.
            n_streams: Number of streams sized into the
                ``accepted_per_stream`` accumulator.

        Raises:
            MaskPipelineError: If the pipeline is shut down, if a previous
                ``launch()`` has no matching ``gather()``, if primary plus
                scratch rows exceed arena capacity, or if any
                ``stream_idx`` is outside ``[0, max(n_streams, 1))``.
        """
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

        # Group by logits row while preserving declaration order (dict
        # insertion order plus a stable primary-first sort).
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
                        self._pool.submit(self._emit_entry, e, rows, row_indices, caps, self._lock)
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
        """Emit one entry: commit, verify draft, fold acceptance, map rows.

        Args:
            e: Entry to emit.
            rows: Arena CPU view for the active parity buffer.
            row_indices: CPU row-index array to fill for primary entries.
            caps: Per-stream accepted-count accumulator (min-reduced).
            lock: Guards the read-modify-write min update, which races
                when two producers share a stream.
        """
        if e.commit_tokens:
            e.producer.commit(e.commit_tokens)
        span = e.propose_step + 1
        cap = e.producer.emit(e.draft, rows.window(e.mask_row, span))
        with lock:
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
        """Emit a non-commutative slot sequentially in declaration order.

        Args:
            entries: Primary-first ordered entries of one slot.
            rows: Arena CPU view for the active parity buffer.
            row_indices: CPU row-index array; only the primary window is
                mapped.
            caps: Per-stream accepted-count accumulator (min-reduced).
            lock: Guards the min update for consistency with
                :meth:`_emit_entry`.

        Acceptance for the slot is the minimum over its producers.
        """
        accepted_min = _SENTINEL_CAP
        for e in entries:
            if e.commit_tokens:
                e.producer.commit(e.commit_tokens)
            span = e.propose_step + 1
            cap = e.producer.emit(e.draft, rows.window(e.mask_row, span))
            accepted_min = min(accepted_min, int(cap))
        primary = entries[0]
        with lock:
            caps[primary.stream_idx] = min(int(caps[primary.stream_idx]), accepted_min)
        for k in range(primary.propose_step + 1):
            row_indices[primary.mask_row + k] = primary.logits_row + k

    def gather(self, compute_stream: Any = None) -> MaskHandle | None:
        """Wait for pending producers, intersect scratch, and upload.

        Scratch windows are ANDed into their primary window on the same
        parity buffer handed out by ``launch`` (no second ``begin_step``),
        then the primary rows are uploaded on the copy stream.

        Args:
            compute_stream: Optional compute stream that must wait on the
                upload readiness event before consuming the handle.

        Returns:
            A :class:`MaskHandle` with GPU tensors, aggregate caps, and
            per-stream acceptance; None when nothing was launched or the
            step has zero rows.

        Raises:
            MaskPipelineError: If any producer future raised; the first
                error is chained as the cause.
        """
        if self._pending is None:
            return None
        futs, self._pending = self._pending, None
        errs = [e for e in (f.exception() for f in futs) if e is not None]
        if errs:
            raise MaskPipelineError(f"{len(errs)} producer loi; dau tien: {errs[0]!r}") from errs[0]
        n_rows = self._n_rows
        if n_rows == 0:
            return None
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
        """Forward the consumed marker to the arena.

        Args:
            compute_stream: Stream whose completion releases the active
                parity buffer. None selects the default.
        """
        self._arena.mark_consumed(compute_stream)

    def shutdown(self) -> None:
        """Shut down the worker pool and destroy the copy stream.

        Idempotent: repeated calls are no-ops.
        """
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)
        self._backend.destroy_stream(self._copy_stream)
