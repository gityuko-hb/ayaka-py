"""BitmaskArena -- pinned, grow-only, double-buffered."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ayaka.caps import Cap
from ayaka.device.backend import DeviceBackend, get_backend
from ayaka.sampling.footprint import SamplingFootprint, measure
from ayaka.sampling.mask.producer import MaskRows
from ayaka.utils.torch_memory import pinned_empty
from ayaka.utils.torch_utils import resolve_device


def bitmask_words(vocab_size: int) -> int:
    return (vocab_size + 31) // 32


@dataclass(slots=True)
class MaskHandle:
    masks: torch.Tensor
    row_indices: torch.Tensor
    n_rows: int
    vocab_size: int
    # A4 — aggregate Cap (AND) của mọi producer đã ghi mask; consumer
    # (Sampler greedy fast-path, spec verifier) đọc để quyết định fast path.
    caps: Cap = Cap.NONE
    # Số token accepted tối thiểu per stream (speculative verification gate)
    # — từng là spec_caps, đổi tên cho đúng nghĩa.
    accepted_per_stream: np.ndarray | None = None


class BitmaskArena:
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
        return measure([*self._cpu, *self._gpu, *self._rows_cpu, *self._rows_gpu])

    def begin_step(self):
        self._parity ^= 1
        p = self._parity
        self._backend.synchronize_event(self._consumed[p])
        return MaskRows(self._np[p], self.vocab_size), self._rows_np[p]

    def upload(self, n_rows: int, stream: Any = None):
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
        self._backend.record(self._consumed[self._parity], stream)

    def data_ptrs(self) -> list[int]:
        return [t.data_ptr() for t in (*self._cpu, *self._gpu, *self._rows_cpu, *self._rows_gpu)]
