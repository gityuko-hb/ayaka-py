"""Whole-model padding arithmetic for pure-decode graph replay (R08).

The captured graph runs a fixed number of token lanes. Rows beyond the real
request count are *dummy decode lanes*: semantically inert — zero query
contribution, zero KV length, a padding-page table row and a padding slot — so
they cannot read, write or publish anything a live request owns.

Everything here is host-only integer arithmetic; it is deliberately separated
from the runner so the padding contract is testable without a device, and so
the ``slot_mapping >= 0`` rule for ``index_copy_``-based stores is enforced in
one place (a ``-1`` sentinel is only legal for kernels that mask it).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["PaddedDecodeAddressing", "pad_decode_addressing"]


@dataclass(frozen=True, slots=True)
class PaddedDecodeAddressing:
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    computed: tuple[int, ...]
    tables: tuple[tuple[int, ...], ...]
    slots: tuple[int, ...]

    @property
    def num_rows(self) -> int:
        return len(self.lengths)


def pad_decode_addressing(
    *,
    real_starts: Sequence[int],
    real_lengths: Sequence[int],
    real_computed: Sequence[int],
    real_tables: Sequence[Sequence[int]],
    real_slots: Sequence[int],
    width: int,
    bucket: int,
    padding_page: int,
    padding_slot: int,
) -> PaddedDecodeAddressing:
    """Extend one pure-decode step's addressing to the captured ``bucket``.

    Every padded lane contributes exactly one dummy query (``query_start_loc``
    continues by one) while its ``seq_lens`` and ``computed_lens`` stay zero, so
    the attention kernels do no work for it. The padding page must be the
    allocator's reserved, never-live page and the padding slot a non-negative
    flat address: ``index_copy_`` and friends must never see a sentinel.
    """
    if bucket < 1:
        raise ValueError("bucket must be positive")
    if width < 1:
        raise ValueError("block-table width must be positive")
    if padding_page < 0 or padding_slot < 0:
        raise ValueError("padding page and slot must be non-negative")
    real = len(real_lengths)
    if real > bucket:
        raise ValueError(f"{real} real rows exceed the captured bucket {bucket}")
    if len(real_tables) != real or len(real_computed) != real:
        raise ValueError("per-row metadata lengths must agree")
    if len(real_starts) != real + 1:
        raise ValueError("query_start_loc must carry one more entry than seq_lens")
    if real and real_starts[-1] != real:
        raise ValueError("pure-decode staging requires one query per sliced request")
    if len(real_slots) > bucket:
        raise ValueError("real query tokens exceed the captured bucket")
    for table in real_tables:
        if len(table) != width:
            raise ValueError("every block-table row must be padded to the staged width")
    padded_tables = [tuple(row) for row in real_tables]
    padded_tables += [tuple([padding_page] * width) for _ in range(bucket - real)]
    slots = tuple(real_slots) + (padding_slot,) * (bucket - len(real_slots))
    if any(slot < 0 for slot in slots):
        raise ValueError("graph staging must not hand a negative slot to an index_copy_ store")
    starts = tuple(real_starts) + tuple(real + offset for offset in range(1, bucket - real + 1))
    lengths = tuple(real_lengths) + (0,) * (bucket - real)
    computed = tuple(real_computed) + (0,) * (bucket - real)
    return PaddedDecodeAddressing(
        starts=starts,
        lengths=lengths,
        computed=computed,
        tables=tuple(padded_tables),
        slots=slots,
    )
