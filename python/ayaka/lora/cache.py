"""Stable device slot storage, without adapter identities or eviction policy."""

from __future__ import annotations

import torch

from ayaka.lora.config import LoRAConfig
from ayaka.lora.mapping import ProjectionTarget
from ayaka.lora.weights import AdapterWeights


class DeviceAdapterPool:
    def __init__(
        self,
        targets: tuple[ProjectionTarget, ...],
        config: LoRAConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.targets, self.config, self.device, self.dtype = targets, config, device, dtype
        size = torch.empty((), dtype=dtype).element_size()
        rank_size = torch.empty((), dtype=torch.int32).element_size()
        self.bytes = sum(
            (config.max_adapters + 1)
            * (config.max_rank * (t.local_input_size + t.local_output_size) * size + rank_size)
            for t in targets
        )
        if self.bytes > config.memory_bytes:
            raise MemoryError("LoRA stable slots exceed memory reservation")
        self.weights = {
            t.key: (
                torch.zeros(
                    config.max_adapters + 1,
                    config.max_rank,
                    t.local_input_size,
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros(
                    config.max_adapters + 1,
                    t.local_output_size,
                    config.max_rank,
                    device=device,
                    dtype=dtype,
                ),
            )
            for t in targets
        }
        #: Actual rank per ``(target, slot)``; slot 0 and missing slices are 0.
        #: Written together with A/B under the manager lock, never separately.
        self.ranks = {
            t.key: torch.zeros(config.max_adapters + 1, dtype=torch.int32, device=device)
            for t in targets
        }
        self.quarantined: set[int] = set()
        self._read_events: dict[int, torch.cuda.Event] = {}

    def record_read(self) -> None:
        """Record enqueued reads under the manager lock, including graph replay.

        The reference backend reads every slot before masking. A management copy
        therefore waits for all earlier flight streams, even for an unused slot.
        Capture setup is private to bootstrap; its event is recorded by the
        outer execution guard after capture, never inserted into a graph.
        """
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        stream = torch.cuda.current_stream(self.device)
        event = self._read_events.get(stream.cuda_stream)
        if event is None:
            event = torch.cuda.Event()
            self._read_events[stream.cuda_stream] = event
        event.record(stream)

    def wait_reads(self) -> None:
        """Drain only LoRA reader streams before mutating or freeing storage."""
        for event in self._read_events.values():
            event.synchronize()

    def batch_completion(self) -> torch.cuda.Event | None:
        """Independent, query-only completion for management observability."""
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    @property
    def storage_identity(self) -> tuple:
        return tuple(
            (name, a.data_ptr(), b.data_ptr(), self.ranks[name].data_ptr())
            for name, (a, b) in self.weights.items()
        )

    @torch.inference_mode()
    def copy(self, slot: int, weights: AdapterWeights) -> None:
        """Blocking P0 transfer; never publishes a slot before stream completion.

        All conversion/localization is finished on CPU before the first slot write.
        A/B and actual-rank metadata are written in one transaction so kernels can
        never observe a rank without its tensors. Only manager may invoke this,
        with an unleased/unpinned victim.
        """
        if not 1 <= slot <= self.config.max_adapters or slot in self.quarantined:
            raise ValueError("base, invalid or quarantined adapter slot")
        by_key = {layer.target: layer for layer in weights.layers}
        copies: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        for target in self.targets:
            layer = by_key.get(target.key)
            if layer is not None:
                if not 1 <= layer.rank <= self.config.max_rank:
                    raise ValueError("adapter rank is outside the bank capacity")
                a, b = target.localize(layer.a, layer.b)
                copies[target.key] = (
                    a.to(self.dtype).contiguous(),
                    (b.float() * layer.scale).to(self.dtype).contiguous(),
                    layer.rank,
                )
        self.wait_reads()
        for key, (a, b) in self.weights.items():
            rank_row = self.ranks[key]
            a[slot].zero_()
            b[slot].zero_()
            rank_row[slot] = 0
            entry = copies.get(key)
            if entry is not None:
                source_a, source_b, rank = entry
                a[slot, :rank].copy_(source_a)
                b[slot, :, :rank].copy_(source_b)
                rank_row[slot] = rank
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()
