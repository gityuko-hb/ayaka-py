"""One persistent routing allocation per ticket-owned flight.

``token_slot_ids`` is the canonical row mapping read by the reference backend.
The grouped tables (counts, exclusive offsets, stable permutation and inverse)
are the SGMV contract: tokens are grouped by adapter slot while preserving the
scheduler's token order, and every table is rebuilt on each ``stage`` so a
smaller replay never observes stale padding metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ayaka.utils.validation import require_int


class BatchLoRAMapping:
    def __init__(self, tokens: int, sequences: int, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.token_slot_ids = torch.zeros(tokens, dtype=torch.long, device=device)
        self.sequence_slot_ids = torch.zeros(sequences, dtype=torch.long, device=device)
        self.segment_offsets = torch.zeros(sequences + 1, dtype=torch.long, device=device)
        self.active_slots = torch.zeros(capacity + 1, dtype=torch.bool, device=device)
        #: Tokens per slot, indexed by slot id (0..capacity).
        self.slot_counts = torch.zeros(capacity + 1, dtype=torch.long, device=device)
        #: Exclusive start of each slot block inside ``token_permutation``.
        self.adapter_offsets = torch.zeros(capacity + 1, dtype=torch.long, device=device)
        #: Token positions grouped by ascending slot id, order-stable within a slot.
        self.token_permutation = torch.zeros(tokens, dtype=torch.long, device=device)
        #: Inverse of ``token_permutation``: grouped position of each token.
        self.token_inverse = torch.zeros(tokens, dtype=torch.long, device=device)
        self._token_index = torch.arange(tokens, dtype=torch.long, device=device)
        self._ones = torch.ones(tokens, dtype=torch.long, device=device)
        self._cumsum = torch.zeros(capacity + 1, dtype=torch.long, device=device)
        self._sort_values = torch.zeros(tokens, dtype=torch.long, device=device)
        self.generations: tuple[int, ...] = ()
        #: Algorithm phase staged by the runner: ``decode`` selects BGMV,
        #: ``prefill``/``mixed`` select SGMV when grouped tables are present.
        self.phase = "reference"

    def stage(
        self,
        slots: Sequence[int],
        counts: Sequence[int],
        generations: Sequence[int],
        *,
        max_loras_per_batch: int,
    ) -> None:
        if len(slots) != len(counts) or len(slots) != len(generations):
            raise ValueError("batch LoRA slot/count/generation lengths differ")
        if len(slots) > self.sequence_slot_ids.numel():
            raise ValueError("batch LoRA sequence capacity exceeded")
        for slot, count in zip(slots, counts, strict=True):
            require_int(slot, "adapter slot")
            require_int(count, "query count", minimum=1)
            if slot > self.capacity:
                raise ValueError("adapter slot exceeds capacity")
        if sum(counts) > self.token_slot_ids.numel():
            raise ValueError("batch LoRA token capacity exceeded")
        if len(set(slots) - {0}) > max_loras_per_batch:
            raise ValueError("batch exceeds max_loras_per_batch; scheduler must defer adapters")
        self.token_slot_ids.zero_()
        self.sequence_slot_ids.zero_()
        self.segment_offsets.zero_()
        self.active_slots.zero_()
        offset = 0
        for i, (slot, count) in enumerate(zip(slots, counts, strict=True)):
            self.sequence_slot_ids[i] = slot
            self.segment_offsets[i] = offset
            self.token_slot_ids[offset : offset + count].fill_(slot)
            self.active_slots[slot] = True
            offset += count
        self.segment_offsets[len(slots)] = offset
        self.generations = tuple(generations)
        self._stage_groups(offset)

    def _stage_groups(self, count: int) -> None:
        """Rebuild grouped routing for the first ``count`` tokens, on device."""
        self.slot_counts.zero_()
        if count:
            self.slot_counts.scatter_add_(0, self.token_slot_ids[:count], self._ones[:count])
        # Exclusive prefix sums: adapter_offsets[s] = tokens in slots before s.
        torch.cumsum(self.slot_counts, 0, out=self._cumsum)
        self.adapter_offsets[0] = 0
        self.adapter_offsets[1:].copy_(self._cumsum[:-1])
        if count:
            torch.sort(
                self.token_slot_ids[:count],
                stable=True,
                out=(self._sort_values[:count], self.token_permutation[:count]),
            )
            self.token_inverse[:count].scatter_(
                0, self.token_permutation[:count], self._token_index[:count]
            )
        self.token_permutation[count:].zero_()
        self.token_inverse[count:].zero_()
        self._sort_values[count:].zero_()

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.token_slot_ids,
                self.sequence_slot_ids,
                self.segment_offsets,
                self.active_slots,
                self.slot_counts,
                self.adapter_offsets,
                self.token_permutation,
                self.token_inverse,
                self._token_index,
                self._ones,
                self._cumsum,
                self._sort_values,
            )
        )

    @property
    def storage_identity(self) -> tuple[int, ...]:
        return tuple(
            t.data_ptr()
            for t in (
                self.token_slot_ids,
                self.sequence_slot_ids,
                self.segment_offsets,
                self.active_slots,
                self.slot_counts,
                self.adapter_offsets,
                self.token_permutation,
                self.token_inverse,
            )
        )
