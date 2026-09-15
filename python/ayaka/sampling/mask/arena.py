"""Double-buffered pinned arena for bitmask staging and upload.

The arena owns two parity buffers so the CPU producer path and the GPU
consumer path never race:

* :meth:`BitmaskArena.begin_step` flips parity, waits for the consumer to
  release that buffer, and hands out writable NumPy views.
* Producers (usually via :class:`~ayaka.sampling.mask.pipeline.MaskPipeline`)
  fill bitmask rows plus a ``row_indices`` mapping on the CPU side.
* :meth:`BitmaskArena.upload` copies the active slice to GPU on a dedicated
  copy stream and returns a readiness event.
* :meth:`BitmaskArena.mark_consumed` records when the compute stream is done
  with the buffer so the next step may reuse it.

Capacity (``max_rows``) is fixed by the upper layer at the batch boundary;
the arena never grows inside the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ayaka.caps import Cap
from ayaka.device.backend import DeviceBackend, get_backend
from ayaka.sampling.mask.producer import MaskRows
from ayaka.sampling.metadata import SamplingFootprint, measure
from ayaka.utils.torch_memory import pinned_empty
from ayaka.utils.torch_utils import resolve_device


def bitmask_words(vocab_size: int) -> int:
    """Return the number of ``uint32`` words needed for ``vocab_size`` bits.

    Args:
        vocab_size: Vocabulary size (must be positive in practice; the
            formula also holds for zero).

    Returns:
        ``ceil(vocab_size / 32)``.
    """
    return (vocab_size + 31) // 32


@dataclass(slots=True)
class MaskHandle:
    """GPU-ready masks for one decode step.

    Attributes:
        masks: GPU bitmask tensor of shape ``(n_rows, words)``.
        row_indices: GPU int32 vector mapping each mask row to its logits
            row (``mask_row -> logits_row + offset`` for speculative rows).
        n_rows: Number of valid rows in ``masks`` and ``row_indices``.
        vocab_size: Vocabulary size the bitmask was built for.
        caps: Aggregate capability (bitwise AND) of every producer that
            wrote the masks. Consumers such as the greedy fast-path and
            the speculative verifier read this to pick a fast path.
        accepted_per_stream: Minimum accepted draft-token count per
            stream; used as the speculative verification gate. None when
            the step carries no draft information.
    """

    masks: torch.Tensor
    row_indices: torch.Tensor
    n_rows: int
    vocab_size: int
    caps: Cap = Cap.NONE
    accepted_per_stream: np.ndarray | None = None


class BitmaskArena:
    """Pinned, grow-only, double-buffered staging area for bitmasks.

    Attributes:
        max_rows: Maximum mask rows (including scratch rows) per step.
        vocab_size: Vocabulary size the arena was sized for.
        words: Number of ``uint32`` words per row.
        device: GPU device holding the upload buffers.
    """

    __slots__ = (
        "_backend",
        "_consumed",
        "_cpu",
        "_gpu",
        "_np",
        "_parity",
        "_rows_cpu",
        "_rows_gpu",
        "_rows_np",
        "device",
        "max_rows",
        "vocab_size",
        "words",
    )

    def __init__(
        self,
        max_rows: int,
        vocab_size: int,
        device: torch.device | None = None,
        *,
        backend: DeviceBackend | None = None,
    ):
        """Allocate both parity buffers.

        Args:
            max_rows: Maximum rows per step. Must be > 0.
            vocab_size: Vocabulary size. Must be > 0.
            device: GPU target for uploads. Defaults to the resolved
                current device.
            backend: Device backend for streams/events. Defaults to the
                global backend.

        Raises:
            ValueError: If ``max_rows`` or ``vocab_size`` is not positive.
        """
        if max_rows <= 0 or vocab_size <= 0:
            raise ValueError("max_rows va vocab_size phai > 0")
        self.max_rows = max_rows
        self.vocab_size = vocab_size
        self.words = bitmask_words(vocab_size)
        self.device = resolve_device(device)
        self._backend = backend if backend is not None else get_backend()

        self._cpu, self._gpu, self._np = [], [], []
        self._rows_cpu, self._rows_gpu, self._rows_np = [], [], []
        for _ in range(2):
            c, _ = pinned_empty((max_rows, self.words), torch.int32)
            g = torch.empty((max_rows, self.words), dtype=torch.int32, device=self.device)
            rc, _ = pinned_empty((max_rows,), torch.int32)
            rg = torch.empty(max_rows, dtype=torch.int32, device=self.device)
            self._cpu.append(c)
            self._gpu.append(g)
            self._rows_cpu.append(rc)
            self._rows_gpu.append(rg)
            self._np.append(c.numpy().view(np.uint32))
            self._rows_np.append(rc.numpy())

        self._consumed = [self._backend.create_event(timing=False) for _ in range(2)]
        self._parity = 0

    def footprint(self) -> SamplingFootprint:
        """Measure device and host memory held by both parity buffers.

        Returns:
            Footprint covering the pinned CPU masks, GPU masks, and both
            row-index buffers.
        """
        return measure([*self._cpu, *self._gpu, *self._rows_cpu, *self._rows_gpu])

    def begin_step(self):
        """Flip parity and hand out writable CPU views for a new step.

        Waits for the consumer to release the newly active buffer, so a
        slow GPU cannot be overwritten by the next CPU fill.

        Returns:
            Tuple of the writable :class:`MaskRows` view and the writable
            ``row_indices`` NumPy array for the active parity buffer.
        """
        self._parity ^= 1
        p = self._parity
        self._backend.synchronize_event(self._consumed[p])
        return MaskRows(self._np[p], self.vocab_size), self._rows_np[p]

    def upload(self, n_rows: int, stream: Any = None):
        """Copy the first ``n_rows`` to GPU on the given copy stream.

        Only the active parity slice is copied. The copy runs under the
        backend stream context and a readiness event is recorded so the
        compute stream can wait on it.

        Args:
            n_rows: Rows to upload. Must not exceed ``max_rows``.
            stream: Copy stream to run on. None selects the default.

        Returns:
            Tuple ``(gpu_masks, gpu_row_indices, ready_event)`` where the
            first two tensors are narrowed to ``n_rows``.

        Raises:
            ValueError: If ``n_rows`` exceeds ``max_rows``. Capacity is
                decided by the upper layer at the batch boundary; the
                arena never grows inside the hot path.
        """
        if n_rows > self.max_rows:
            raise ValueError(
                f"n_rows {n_rows} > max_rows {self.max_rows}; capacity do L2 quyet o "
                "batch boundary, arena khong tu grow trong hot path"
            )
        p = self._parity
        non_blocking = self.device.type == "cuda"
        with self._backend.stream_context(stream):
            g = self._gpu[p].narrow(0, 0, n_rows)
            g.copy_(self._cpu[p].narrow(0, 0, n_rows), non_blocking=non_blocking)
            rg = self._rows_gpu[p].narrow(0, 0, n_rows)
            rg.copy_(self._rows_cpu[p].narrow(0, 0, n_rows), non_blocking=non_blocking)
            ready = self._backend.create_event(timing=False)
            self._backend.record(ready, stream)
        return g, rg, ready

    def mark_consumed(self, stream: Any = None) -> None:
        """Mark the active parity buffer as consumed on ``stream``.

        Records the per-parity consumed event so the next
        :meth:`begin_step` for this parity blocks until the consumer is
        done.

        Args:
            stream: Stream whose completion releases the buffer. None
                selects the default.
        """
        self._backend.record(self._consumed[self._parity], stream)

    def data_ptrs(self) -> list[int]:
        """Return data pointers of all backing tensors.

        Returns:
            List of ``data_ptr()`` values for the CPU/GPU mask and
            row-index buffers of both parities. Intended for debugging
            buffer reuse and aliasing.
        """
        return [t.data_ptr() for t in (*self._cpu, *self._gpu, *self._rows_cpu, *self._rows_gpu)]
