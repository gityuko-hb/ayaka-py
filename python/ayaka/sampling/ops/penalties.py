"""ayaka/sampling/ops — logit adjustment before mask/temperature/sample.

The four sections below follow pipeline execution order:

    FUSED_STATS (shared primitive, never runs on its own — only called)
        ↓
    BIAS        — "custom ops → BIAS → penalties → mask → ..."
        ↓
    PENALTIES   — "... → BIAS → penalties → mask → ..."
        ↓
    DRY         — n-gram repetition penalty, in the same "logit adjustment"
                  group as penalties (except it follows HISTORY ORDER, not
                  frequency) — NO canonical bit-exact reference yet, see the
                  original note in the DRY section.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch

from ayaka.sampling.metadata import SamplingMetadata

try:  # pragma: no cover
    from ayaka.kernel.triton.sampling.fused import _HAS_TRITON_FUSED, row_logsumexp_gpu
except Exception:  # pragma: no cover
    _HAS_TRITON_FUSED = False
    row_logsumexp_gpu = None

__all__ = [
    "row_logsumexp",
    "BiasState",
    "apply_bias_",
    "apply_bias_padded_",
    "PenaltyState",
    "apply_penalties_",
    "apply_penalties_padded_",
    "dry_bias",
    "apply_dry_",
]


# row_logsumexp — ONE source for per-row penalty stats
# Dispatch: Triton (kernel/triton/sampling/fused.py) on CUDA, torch fallback
# otherwise. Both paths share semantics:
#   * row_gate=None: logsumexp over all rows;
#   * row_gate given: only rows with gate > 0 are read and get nonzero lse;
#     remaining rows get lse = 0 — AVOIDS the old nonzero + index_select copy.
#
# fp32 accumulation regardless of input logits dtype (matches old version:
# sub.to(float32)).


def _row_logsumexp_torch(logits: torch.Tensor, row_gate: torch.Tensor | None) -> torch.Tensor:
    """Pure-torch fallback — oracle for the Triton path, no host sync.

    Args:
        logits: [n_rows, vocab] logits of any float dtype.
        row_gate: Optional [n_rows] gate; only rows with gate > 0 are
            computed, the rest get lse = 0. None computes every row.

    Returns:
        Float32 [n_rows] per-row logsumexp (0 for gated-off rows).
    """
    n = logits.size(0)
    if row_gate is None:
        return torch.logsumexp(logits.to(torch.float32), dim=-1)
    gate = row_gate != 0
    lse = torch.zeros(n, dtype=torch.float32, device=logits.device)
    if bool(gate.any()):
        lse[gate] = torch.logsumexp(logits[gate].to(torch.float32), dim=-1)
    return lse


def row_logsumexp(logits: torch.Tensor, *, row_gate: torch.Tensor | None = None) -> torch.Tensor:
    """Per-row logsumexp with optional gating — see the module docstring.

    Args:
        logits: [n_rows, vocab] logits of any float dtype.
        row_gate: [n_rows] gate (bool or numeric) — only rows with gate > 0
            are computed; None computes all rows.

    Returns:
        Float32 [n_rows]; lse is 0 for gated-off rows.

    Raises:
        ValueError: If ``logits`` is not 2-D or ``row_gate`` length
            mismatches the row count.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D [n, V], got shape {tuple(logits.shape)}")
    if row_gate is not None and row_gate.size(0) != logits.size(0):
        raise ValueError(f"row_gate size {row_gate.size(0)} != logits rows {logits.size(0)}")
    if _HAS_TRITON_FUSED and row_logsumexp_gpu is not None and logits.is_cuda:  # pragma: no cover
        result: torch.Tensor = row_logsumexp_gpu(logits, row_gate)
        return result
    return _row_logsumexp_torch(logits, row_gate)


class BiasState:
    """Per-slot (token, bias) table, device [B, cap] table plus host pending.

    Set-once semantics: :meth:`set` overwrites the whole slot entry (no
    accumulated delta). :meth:`move`/:meth:`reset` mirror PenaltyState for
    the slot lifecycle.

    Attributes:
        max_batch_size: Maximum slot count (device table rows).
        per_slot_cap: Maximum bias entries per slot (device table columns).
        vocab_size: Optional vocabulary size used to validate token ids.
    """

    __slots__ = (
        "_cols",
        "_device",
        "_pending",
        "_vals",
        "max_batch_size",
        "per_slot_cap",
        "vocab_size",
    )

    def __init__(
        self,
        max_batch_size: int,
        *,
        vocab_size: int | None = None,
        per_slot_cap: int = 128,
        device: torch.device | None = None,
    ) -> None:
        """Create an empty bias table.

        Args:
            max_batch_size: Number of slots (device table rows).
            vocab_size: Optional vocabulary size for token-id validation.
            per_slot_cap: Maximum bias entries per slot.
            device: Device holding the tables. Defaults to CPU.

        Raises:
            ValueError: If ``max_batch_size`` or ``per_slot_cap`` is not > 0.
        """
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if per_slot_cap <= 0:
            raise ValueError("per_slot_cap must be > 0")
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.per_slot_cap = per_slot_cap
        self._device = (
            torch.device(device) if isinstance(device, str) else device
        ) or torch.device("cpu")
        self._cols = torch.full(
            (max_batch_size, per_slot_cap), -1, dtype=torch.int64, device=self._device
        )
        self._vals = torch.zeros(
            max_batch_size, per_slot_cap, dtype=torch.float32, device=self._device
        )
        self._pending: dict[str, list[tuple]] = {}

    # ------------------------------------------------------------------
    # Structural contract
    # ------------------------------------------------------------------
    def set(self, slot: int, bias: Mapping[int, float]) -> None:
        """Record ``(token, bias)`` entries for one slot.

        Overwrites the slot's previous entry.

        Args:
            slot: Slot index to overwrite.
            bias: Mapping of token id to additive bias.

        Raises:
            IndexError: If ``slot`` is out of range.
            ValueError: If the entry count exceeds ``per_slot_cap``, a token
                id is outside ``vocab_size``, or a bias is non-finite.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        items = sorted((int(token), float(value)) for token, value in bias.items())
        if len(items) > self.per_slot_cap:
            raise ValueError(
                f"logit_bias has {len(items)} tokens exceeding per_slot_cap="
                f"{self.per_slot_cap} — raise per_slot_cap when constructing BiasState"
            )
        for token, value in items:
            if self.vocab_size is not None and not 0 <= token < self.vocab_size:
                raise ValueError(f"token id {token} out of range [0, vocab_size={self.vocab_size})")
            if not torch.isfinite(torch.tensor(value)).item():
                raise ValueError(f"bias for token {token} must be finite")
        self._pending.setdefault("ops", []).append(("set", slot, items))

    def reset(self, slot: int) -> None:
        """Clear all bias entries for one slot (deferred until flush).

        Args:
            slot: Slot index to clear.

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._pending.setdefault("ops", []).append(("reset", slot))

    def move(self, src: int, dst: int) -> None:
        """Move the bias entry from ``src`` slot to ``dst`` slot.

        Args:
            src: Source slot index.
            dst: Destination slot index.

        Raises:
            IndexError: If either slot is out of range.
        """
        if not (0 <= src < self.max_batch_size and 0 <= dst < self.max_batch_size):
            raise IndexError("move slot out of range")
        if src != dst:
            self._pending.setdefault("ops", []).append(("move", src, dst))

    def count(self, slot: int) -> int:
        """Return the biased-token count for one slot.

        Flushes pending ops before reading.

        Args:
            slot: Slot index to query.

        Returns:
            Number of bias entries currently stored for the slot.

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        self._flush()
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        return int((self._cols[slot] >= 0).sum().item())

    # ------------------------------------------------------------------
    # Coords
    # ------------------------------------------------------------------
    def bias_coords_padded(
        self, active_slots: Sequence[int] | int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return static [n_active, cap] coords (packed rows, slot-space cols).

        Mirrors :meth:`PenaltyState.coords_padded`: rows are PACKED; entries
        are gathered by ``active_slots`` (or int n = identity). Padding uses
        cols=0, vals=0 — apply forces padding delta to 0 via ``valid``.

        Args:
            active_slots: Slot ids in packed order, or int n for identity
                slots ``[0, n)``.
            device: Device for the returned tensors.

        Returns:
            Tuple ``(rows, cols, vals, valid)`` of static [n_active, cap]
            tensors; ``valid`` marks real entries.
        """
        self._flush()
        if isinstance(active_slots, int):
            slots = torch.arange(active_slots, dtype=torch.long, device=self._device)
        else:
            slots = torch.tensor(
                [int(s) for s in active_slots], dtype=torch.long, device=self._device
            )
        n = slots.numel()
        cap = self.per_slot_cap
        rows = torch.arange(n, device=self._device, dtype=torch.long).unsqueeze(1).expand(n, cap)
        if n:
            cols = self._cols.index_select(0, slots)
            vals = self._vals.index_select(0, slots)
        else:
            cols = self._cols.narrow(0, 0, 0)
            vals = self._vals.narrow(0, 0, 0)
        valid = (cols >= 0) if n else torch.zeros((0, cap), dtype=torch.bool, device=self._device)
        if device != self._device:
            rows = rows.to(device)
            cols = cols.to(device)
            vals = vals.to(device)
            valid = valid.to(device)
        return rows, cols, vals, valid

    # ------------------------------------------------------------------
    # Flush pipeline (pending host → device tables)
    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Flush queued host ops into the device tables."""
        ops = self._pending.pop("ops", None)
        if not ops:
            return
        for op in ops:
            if op[0] == "set":
                _, slot, items = op
                self._cols[slot].fill_(-1)
                self._vals[slot].zero_()
                if items:
                    toks = torch.tensor(
                        [t for t, _ in items], dtype=torch.long, device=self._device
                    )
                    bias = torch.tensor(
                        [b for _, b in items], dtype=torch.float32, device=self._device
                    )
                    self._cols[slot, : len(items)] = toks
                    self._vals[slot, : len(items)] = bias
            elif op[0] == "reset":
                _, slot = op
                self._cols[slot].fill_(-1)
                self._vals[slot].zero_()
            else:  # move
                _, src, dst = op
                self._cols[dst] = self._cols[src]
                self._vals[dst] = self._vals[src]
                self._cols[src].fill_(-1)
                self._vals[src].zero_()


def apply_bias_(logits: torch.Tensor, md: SamplingMetadata, state: BiasState) -> torch.Tensor:
    """Add bias into logits in place. Runs BEFORE mask and BEFORE penalties.

    Delta-through-index_put_(accumulate=True) — same pattern as penalties:
    padding (valid=False) forces delta 0 so aliased padding (row, col) pairs
    are harmless; real (row, col) pairs are unique per set so accumulate never
    double-counts.

    Rows are PACKED; entries are per SLOT via
    ``bias_coords_padded(md.active_rows)``.

    Args:
        logits: [n_active, V] logits to adjust in place.
        md: Sampling metadata giving the active batch.
        state: Bias table to apply.

    Returns:
        The same ``logits`` tensor, adjusted in place.

    Raises:
        ValueError: If ``logits`` rows do not match ``md.n_active``.
    """
    n = md.n_active
    if logits.size(0) != n:
        raise ValueError(
            f"logits.size(0)={logits.size(0)} != md.n_active={n} — logits must "
            "match the metadata batch (packed sampling order)"
        )
    if n == 0:
        return logits
    rows, cols, vals, valid = state.bias_coords_padded(md.active_rows or n, logits.device)
    if not bool(valid.any().item()):
        return logits
    delta = torch.where(valid, vals, torch.zeros_like(vals))
    flat_rows = rows.reshape(-1)
    flat_cols = cols.reshape(-1)
    flat_delta = delta.reshape(-1).to(logits.dtype)
    if flat_rows.numel():
        logits.index_put_((flat_rows, flat_cols), flat_delta, accumulate=True)
    return logits


def apply_bias_padded_(
    logits: torch.Tensor,
    md: SamplingMetadata,
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Graph-capturable variant — takes ``bias_coords_padded`` output directly.

    Accepts the static [B, cap] output of :meth:`BiasState.bias_coords_padded`
    directly instead of gathering coords inside the captured region.

    Args:
        logits: [n_active, V] logits to adjust in place.
        md: Sampling metadata giving the active batch.
        rows: Static [B, cap] packed row indices.
        cols: Static [B, cap] token columns.
        vals: Static [B, cap] bias values.
        valid: Static [B, cap] mask marking real entries.

    Returns:
        The same ``logits`` tensor, adjusted in place.

    Raises:
        ValueError: If ``logits`` rows do not match ``md.n_active``.
    """
    if logits.size(0) != md.n_active:
        raise ValueError("logits rows must match md.n_active")
    delta = torch.where(valid, vals, torch.zeros_like(vals)).reshape(-1)
    flat_rows = rows.reshape(-1)
    flat_cols = cols.reshape(-1)
    if flat_rows.numel():
        logits.index_put_((flat_rows, flat_cols), delta.to(logits.dtype), accumulate=True)
    return logits


# Repetition / frequency / presence penalty over a sparse (row, token, count)
# table: per-step cost scales with O(unique generated tokens), not O(B x V).
#
# Design notes:
#
# 1. `unrecord()` rolls back optimistic `record()` calls from speculative
#    decoding. Unrecording a never-recorded token is a caller bug and raises
#    fail-closed; counts never go negative.
#
# 2. The repetition penalty (sign branch) runs on normalized log-softmax, not
#    raw logits: subtract the row logsumexp before the sign compare /
#    multiply-divide, then add it back to preserve the scale of untouched
#    logits. The logsumexp comes from `row_logsumexp` in the FUSED_STATS
#    section above.
#
# 3. Delta-flush with no per-slot Python dict: `record` / `unrecord` /
#    `reset` / `move` only enqueue host deltas — O(delta), no lookup.
#    `coords()` / `coords_padded()` call `_flush()` once per step to merge
#    deltas by (slot, token), update the device tables (`cols` [B, cap] int64,
#    `cnts` [B, cap] fp32, `n_used` [B] int64; live entries packed in
#    [0, n_used), unordered), and compact entries whose count reaches zero.
#    "Unrecord a never-recorded token" raises at flush time, and only for a
#    net-negative delta on an absent token — (record t; unrecord t) within one
#    step nets to zero and is a no-op.
#
# 4. `coords_padded()` returns a static [n_active, cap] shape plus a `valid`
#    mask for CUDA graph capture; the device table itself is the bound.
#
# 5. Dense promotion: a slot whose occupancy reaches `per_slot_cap` (default
#    `promote_fraction x vocab`) is scattered to a dense row of
#    `[max_dense_slots, V]`; later deltas apply straight to dense. `coords()`
#    / `coords_padded()` exclude promoted rows from the sparse part. A full
#    pool raises fail-closed.
#
# 6. Single formula source: `apply_penalties_` builds coords and enters the
#    same `_apply_penalties_core` as `apply_penalties_padded_` — the
#    rep/freq/pres formula is written exactly once, on the graph-capturable
#    path.


class PenaltyState:
    """Generated-token counts, per slot, device [B, cap] tables plus host delta.

    The structural contract (record/unrecord/reset/move) is unchanged from the
    port; the internal state is device tables — per-step host work scales with
    |delta|, not with the accumulated unique-token count.

    Attributes:
        max_batch_size: Maximum slot count.
        per_slot_cap: Maximum sparse entries per slot before dense promotion.
        promote_fraction: Fraction of ``vocab_size`` used as the default
            ``per_slot_cap``.
        vocab_size: Optional vocabulary size for dense-pool allocation.
    """

    __slots__ = (
        "_cols",
        "_cnts",
        "_dense_cnts",
        "_dense_members",
        "_dense_slot_ids",
        "_dense_used",
        "_device",
        "_n_used",
        "_pending",
        "max_batch_size",
        "per_slot_cap",
        "promote_fraction",
        "vocab_size",
    )

    def __init__(
        self,
        max_batch_size: int,
        *,
        vocab_size: int | None = None,
        per_slot_cap: int | None = None,
        promote_fraction: float = 0.25,
        max_dense_slots: int = 4,
        device: torch.device | None = None,
    ) -> None:
        """Create an empty penalty table.

        Args:
            max_batch_size: Number of slots.
            vocab_size: Optional vocabulary size; required to allocate the
                dense pool on promotion.
            per_slot_cap: Maximum sparse entries per slot. Defaults to
                ``ceil(promote_fraction * vocab_size)`` (or 512 without
                ``vocab_size``).
            promote_fraction: Fraction of vocab used for the default cap.
            max_dense_slots: Dense-pool rows; 0 disables dense promotion.
            device: Device holding the tables. Defaults to CPU.

        Raises:
            ValueError: If ``max_batch_size`` is not > 0, ``vocab_size`` or
                ``per_slot_cap`` is not > 0 when given, or ``max_dense_slots``
                is negative.
        """
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if vocab_size is not None and vocab_size <= 0:
            raise ValueError("vocab_size must be > 0 when given")
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.promote_fraction = promote_fraction
        self._device = (
            torch.device(device) if isinstance(device, str) else device
        ) or torch.device("cpu")
        if per_slot_cap is None:
            per_slot_cap = max(1, math.ceil(promote_fraction * vocab_size)) if vocab_size else 512
        if per_slot_cap <= 0:
            raise ValueError("per_slot_cap must be > 0")
        self.per_slot_cap = per_slot_cap
        if max_dense_slots < 0:
            raise ValueError("max_dense_slots must be >= 0")

        cap = per_slot_cap
        self._cols = torch.zeros(max_batch_size, cap, dtype=torch.int64, device=self._device)
        self._cnts = torch.zeros(max_batch_size, cap, dtype=torch.float32, device=self._device)
        self._n_used = torch.zeros(max_batch_size, dtype=torch.int64, device=self._device)
        self._dense_cnts: torch.Tensor | None = None
        self._dense_slot_ids = torch.full(
            (max_dense_slots,), -1, dtype=torch.int64, device=self._device
        )
        self._dense_used = 0
        self._dense_members: set[int] = set()
        self._pending: list[tuple] = []

    # ------------------------------------------------------------------
    # Structural contract — enqueue host deltas, O(delta), no lookup
    # ------------------------------------------------------------------
    def record(self, slot: int, tokens: Iterable[int]) -> None:
        """Record generated tokens for one slot (deferred until flush).

        Args:
            slot: Slot index.
            tokens: Token ids to count (+1 each).

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        pending = self._pending
        for token in tokens:
            pending.append(("delta", slot, int(token), 1))

    def unrecord(self, slot: int, tokens: Iterable[int]) -> None:
        """Undo :meth:`record` — speculative-decode rollback.

        A -1 delta is enqueued; a missing token (net-negative delta for an
        absent token) raises at :meth:`_flush` time — see note 3 in the
        PENALTIES section header.

        Args:
            slot: Slot index.
            tokens: Token ids to un-count (-1 each).

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        pending = self._pending
        for token in tokens:
            pending.append(("delta", slot, int(token), -1))

    def reset(self, slot: int) -> None:
        """Clear all counts for one slot, including any dense row.

        Args:
            slot: Slot index to clear.

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._pending.append(("reset", slot))

    def move(self, src: int, dst: int) -> None:
        """Move counts from ``src`` slot to ``dst`` slot.

        Keeps dense ownership: a promoted row changes ownership to ``dst``
        with counts preserved.

        Args:
            src: Source slot index.
            dst: Destination slot index.

        Raises:
            IndexError: If either slot is out of range.
        """
        if not (0 <= src < self.max_batch_size and 0 <= dst < self.max_batch_size):
            raise IndexError("move slot out of range")
        if src != dst:
            self._pending.append(("move", src, dst))

    def unique_tokens(self, n_active: int) -> int:
        """Return the unique counted tokens of the first ``n_active`` rows.

        Covers both sparse and dense parts.

        Args:
            n_active: Number of leading rows to count.

        Returns:
            Total unique-token count over those rows.
        """
        if n_active <= 0:
            return 0
        self._flush()
        sparse = int(self._keep_mask(n_active).sum().item())
        dense = 0
        if self._dense_cnts is not None:
            ids = self._dense_slot_ids.tolist()
            for i, slot in enumerate(ids):
                if 0 <= slot < n_active:
                    dense += int((self._dense_cnts[i] > 0).sum().item())
        return sparse + dense

    def count(self, slot: int, token: int) -> int:
        """Return occurrences of ``token`` recorded for ``slot``.

        Covers prompt plus output tokens.

        Args:
            slot: Slot index to query.
            token: Token id to look up.

        Returns:
            Recorded count (0 when absent).

        Raises:
            IndexError: If ``slot`` is out of range.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._flush()
        d_idx = self._dense_index_of(slot)
        if d_idx is not None:
            assert self._dense_cnts is not None
            return int(self._dense_cnts[d_idx, token].item())
        used = int(self._n_used[slot].item())
        if used == 0:
            return 0
        hit = (self._cols[slot, :used] == token).nonzero()
        if hit.numel() == 0:
            return 0
        return int(self._cnts[slot, int(hit[0])].item())

    # ------------------------------------------------------------------
    # Coords — ONE flush/step, boolean-index on device
    # ------------------------------------------------------------------
    @staticmethod
    def _as_slots(active_slots: Sequence[int] | int) -> torch.Tensor:
        """Normalize ``active_slots`` to a 1-D slot tensor.

        Args:
            active_slots: Slot ids in packed order, or int n for identity
                slots ``[0, n)``.

        Returns:
            Long tensor of slot ids.

        Raises:
            ValueError: If any slot id is negative.
        """
        if isinstance(active_slots, int):
            return torch.arange(active_slots, dtype=torch.long)
        slots = torch.tensor([int(s) for s in active_slots], dtype=torch.long)
        if bool(torch.any(slots < 0).item()):
            raise ValueError("active_slots must be >= 0")
        return slots

    def coords(
        self, active_slots: Sequence[int] | int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return compact [N] coords for the SPARSE part.

        Promoted rows yield empty coords (dense rows apply separately via the
        dense branch). N = total sparse uniques.

        ``active_slots`` is a SLOT list in packed order (or int n = identity
        slots [0, n)). Returned rows are PACKED indices (0..n-1) — callers
        index straight into packed logits/md.active. State is stored per slot;
        LIFO/mixed-batch slot reuse makes packed != slot, hence this list.

        Args:
            active_slots: Slot ids in packed order, or int n for identity.
            device: Device for the returned tensors.

        Returns:
            Tuple ``(rows, cols, cnts)`` of compact 1-D coords.
        """
        self._flush()
        slots = self._as_slots(active_slots)
        n = slots.numel()
        if n <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            cnts = torch.empty(0, dtype=torch.float32, device=device)
            return empty, empty, cnts
        cols_sel = self._cols.index_select(0, slots)
        cnts_sel = self._cnts.index_select(0, slots)
        used = self._n_used.index_select(0, slots)
        keep = (
            torch.arange(self.per_slot_cap, device=self._device).unsqueeze(0) < used.unsqueeze(1)
        ) & (cnts_sel > 0)
        rows = keep.nonzero(as_tuple=True)[0].to(torch.long)
        cols = cols_sel[keep]
        cnts = cnts_sel[keep]
        if device != self._device:
            rows = rows.to(device)
            cols = cols.to(device)
            cnts = cnts.to(device)
        return rows, cols, cnts

    def coords_padded(
        self, active_slots: Sequence[int] | int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return static [n_active, cap] coords + valid for CUDA graph capture.

        Rows are PACKED indices (0..n-1); entries are gathered by
        ``active_slots`` (slot space — uses index_select, not narrow, so slot
        reuse keeps working). ``valid`` [n_active, cap] bool marks real
        entries. Padding uses cols=0/cnts=0; apply_penalties_padded_ forces
        padding delta to 0 via ``valid`` instead of relying on where padding
        (row, col) points. No max_unique needed: the device table itself is
        the bound — one slot's occupancy cannot exceed per_slot_cap
        (promote/fail-closed happens first).

        Args:
            active_slots: Slot ids in packed order, or int n for identity.
            device: Device for the returned tensors.

        Returns:
            Tuple ``(rows, cols, cnts, valid)`` of static [n_active, cap]
            tensors.
        """
        self._flush()
        slots = self._as_slots(active_slots)
        n = slots.numel()
        rows = (
            torch.arange(max(n, 0), device=self._device, dtype=torch.long)
            .unsqueeze(1)
            .expand(max(n, 0), self.per_slot_cap)
        )
        cols = self._cols.index_select(0, slots) if n else self._cols.narrow(0, 0, 0)
        cnts = self._cnts.index_select(0, slots) if n else self._cnts.narrow(0, 0, 0)
        valid = torch.arange(self.per_slot_cap, device=self._device).unsqueeze(0) < (
            self._n_used.index_select(0, slots)
            if n
            else torch.empty(0, dtype=torch.long, device=self._device)
        ).unsqueeze(1)
        if device != self._device:
            rows = rows.to(device)
            cols = cols.to(device)
            cnts = cnts.to(device)
            valid = valid.to(device)
        return rows, cols, cnts, valid

    # ------------------------------------------------------------------
    # Dense pool
    # ------------------------------------------------------------------
    @property
    def dense_slot_ids(self) -> torch.Tensor:
        """[max_dense_slots] int64 — logits row of each dense slot, -1 = free."""
        return self._dense_slot_ids

    @property
    def dense_counts(self) -> torch.Tensor | None:
        """[max_dense_slots, vocab] fp32, or None when the pool is unallocated."""
        return self._dense_cnts

    def _dense_index_of(self, slot: int) -> int | None:
        """Return the dense-pool row for ``slot``.

        Args:
            slot: Slot index to look up.

        Returns:
            Dense-pool index, or None when the slot is still sparse.
        """
        for i in range(self._dense_slot_ids.size(0)):
            if int(self._dense_slot_ids[i].item()) == slot:
                return i
        return None

    def _ensure_dense_pool(self) -> torch.Tensor:
        """Allocate the dense pool on first promotion and return it.

        Returns:
            The ``[max_dense_slots, vocab]`` dense-count table.

        Raises:
            ValueError: If the state was built without ``vocab_size`` so a
                dense row cannot be represented.
        """
        if self._dense_cnts is None:
            if not self.vocab_size:
                raise ValueError(
                    "PenaltyState was created without vocab_size so it cannot "
                    "promote to dense — raise per_slot_cap or pass "
                    "vocab_size when constructing the state."
                )
            self._dense_cnts = torch.zeros(
                self._dense_slot_ids.size(0),
                self.vocab_size,
                dtype=torch.float32,
                device=self._device,
            )
        return self._dense_cnts

    def _promote(self, slot: int) -> int:
        """Scatter one slot's sparse table into a dense row.

        Args:
            slot: Slot index to promote.

        Returns:
            Dense-pool index assigned to the slot.

        Raises:
            ValueError: If the dense pool is full (fail-closed) or a token id
                lies outside ``[0, vocab_size)``.
        """
        if self._dense_used >= self._dense_slot_ids.size(0):
            raise ValueError(
                f"dense pool is full ({self._dense_used} slots) — fail-closed, "
                "refusing to silently drop penalties; raise max_dense_slots."
            )
        dense = self._ensure_dense_pool()
        d_idx = self._dense_used
        used = int(self._n_used[slot].item())
        if used:
            toks = self._cols[slot, :used]
            if int(toks.max().item()) >= (self.vocab_size or 0) or int(toks.min().item()) < 0:
                raise ValueError(
                    "token id out of [0, vocab_size) during promote — the state "
                    "cannot represent this token densely"
                )
            dense[d_idx].index_add_(0, toks, self._cnts[slot, :used])
        self._n_used[slot] = 0
        self._dense_slot_ids[d_idx] = slot
        self._dense_used += 1
        self._dense_members.add(slot)
        return d_idx

    # ------------------------------------------------------------------
    # Flush pipeline
    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Flush queued host deltas, resets, and moves into device tables.

        Deltas are batched between structural ops; compaction runs once at
        the end.
        """
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        batch: list[tuple[int, int, int]] = []

        def apply_batch() -> None:
            """Flush the accumulated delta batch into the device tables."""
            if batch:
                self._apply_delta_batch(batch)
                batch.clear()

        for item in pending:
            if item[0] == "delta":
                batch.append((item[1], item[2], item[3]))
            else:
                apply_batch()
                if item[0] == "reset":
                    self._apply_reset(item[1])
                else:
                    self._apply_move(item[1], item[2])
        apply_batch()
        self._compact()

    def _apply_reset(self, slot: int) -> None:
        """Apply one queued reset for ``slot``.

        Frees any dense row owned by the slot.

        Args:
            slot: Slot index to reset.
        """
        self._n_used[slot] = 0
        self._cnts[slot].zero_()
        d_idx = self._dense_index_of(slot)
        if d_idx is not None:
            assert self._dense_cnts is not None
            self._dense_cnts[d_idx].zero_()
            self._dense_slot_ids[d_idx] = -1
            self._dense_used -= 1
            self._dense_members.discard(slot)

    def _apply_move(self, src: int, dst: int) -> None:
        """Apply one queued move from ``src`` to ``dst``.

        A dense row (if any) transfers ownership to ``dst`` with counts kept.

        Args:
            src: Source slot index.
            dst: Destination slot index.
        """
        d_idx = self._dense_index_of(src)
        self._cols[dst].copy_(self._cols[src])
        self._cnts[dst].copy_(self._cnts[src])
        self._n_used[dst] = self._n_used[src]
        self._n_used[src] = 0
        self._cnts[src].zero_()
        if d_idx is not None:
            # A dense row transfers OWNERSHIP to dst — counts kept, not cleared.
            self._dense_slot_ids[d_idx] = dst
            self._dense_members.discard(src)
            self._dense_members.add(dst)

    def _keep_mask(self, n_active: int) -> torch.Tensor:
        """Return the ``[n_active, cap]`` live-entry mask of the sparse part.

        Args:
            n_active: Number of leading rows to mask.

        Returns:
            Bool mask marking live sparse entries.
        """
        return (
            torch.arange(self.per_slot_cap, device=self._device).unsqueeze(0)
            < self._n_used.narrow(0, 0, n_active).unsqueeze(1)
        ) & (self._cnts.narrow(0, 0, n_active) > 0)

    def _apply_delta_batch(self, batch: list[tuple[int, int, int]]) -> None:
        """Merge deltas by (slot, token), look up on the device table, update.

        Syncs ONLY on net-negative (unrecord) deltas or when the allocation
        bound hits cap — the pure-record steady-state path never syncs.
        Lookup is pure-device comparison ([U, cap]); U = merged |delta| of the
        step, hence O(delta).

        Args:
            batch: List of ``(slot, token, count)`` deltas.

        Raises:
            KeyError: If an unrecord delta exceeds the stored count.
            ValueError: If promotion fails (propagated from :meth:`_promote`).
        """
        net: dict[tuple[int, int], int] = {}
        for slot, token, c in batch:
            key = (slot, token)
            net[key] = net.get(key, 0) + c
        pairs = [(s, t, n) for (s, t), n in net.items() if n != 0]
        if not pairs:
            return

        slots = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=self._device)
        toks = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=self._device)
        nets = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=self._device)

        # Lookup: is the token in the slot's live entries? (pure device)
        rows = self._cols.index_select(0, slots)
        in_used = torch.arange(self.per_slot_cap, device=self._device).unsqueeze(
            0
        ) < self._n_used.index_select(0, slots).unsqueeze(1)
        match = (rows == toks.unsqueeze(1)) & in_used
        found = match.any(dim=1)
        pos = match.to(torch.long).argmax(dim=1)

        # Overflow: a host bound (every net>0 pair may need a fresh entry)
        # hitting cap triggers the real check; slots truly needing more
        # entries → dense promotion.
        alloc_bound = torch.bincount(slots[nets > 0], minlength=self.max_batch_size)
        pressure = (self._n_used + alloc_bound) > self.per_slot_cap
        if bool(pressure.any().item()):
            for slot_id in torch.nonzero(pressure).flatten().tolist():
                self._promote(int(slot_id))

        s_list = slots.tolist()
        t_list = toks.tolist()
        n_list = nets.tolist()
        f_list = found.tolist()
        p_list = pos.tolist()
        is_dense = [int(s) in self._dense_members for s in s_list]
        dense_pairs = [
            (int(s_list[i]), int(t_list[i]), int(n_list[i]))
            for i in range(len(pairs))
            if is_dense[i]
        ]

        # Sparse-side validation (dense validates itself in _apply_dense_pairs):
        # net-negative deltas must fit the stored count, never go negative.
        if bool((nets < 0).any().item()):
            old = torch.zeros(len(pairs), dtype=torch.float32, device=self._device)
            fi = [i for i in range(len(pairs)) if (not is_dense[i]) and f_list[i]]
            if fi:
                f_idx = torch.tensor(fi, dtype=torch.long, device=self._device)
                s_idx = torch.tensor(
                    [int(s_list[i]) for i in fi], dtype=torch.long, device=self._device
                )
                p_idx = torch.tensor([p_list[i] for i in fi], dtype=torch.long, device=self._device)
                old[f_idx] = self._cnts[s_idx, p_idx]
            bad = (old + nets.to(torch.float32)) < 0
            if bool(bad.any().item()):
                i = int(torch.nonzero(bad)[0])
                raise KeyError(
                    f"unrecord token {int(t_list[i])} at slot {int(s_list[i])} "
                    "with insufficient count in the table (never recorded or "
                    "fully unrecorded) — likely a caller bug."
                )

        # Sparse: update old entries (positions unique per slot).
        upd = [
            i for i in range(len(pairs)) if (not is_dense[i]) and f_list[i] and int(n_list[i]) != 0
        ]
        if upd:
            u_slot = torch.tensor(
                [int(s_list[i]) for i in upd], dtype=torch.long, device=self._device
            )
            u_pos = torch.tensor([p_list[i] for i in upd], dtype=torch.long, device=self._device)
            u_new = self._cnts[u_slot, u_pos] + torch.tensor(
                [float(n_list[i]) for i in upd], device=self._device
            )
            self._cnts.index_put_((u_slot, u_pos), u_new, accumulate=True)

        # Sparse: allocate new entries for unseen tokens — position = n_used +
        # rank within the same-slot group (stable sort), never colliding with
        # old entries.
        alloc = [
            i
            for i in range(len(pairs))
            if (not is_dense[i]) and (not f_list[i]) and int(n_list[i]) > 0
        ]
        if alloc:
            a_slot = torch.tensor(
                [int(s_list[i]) for i in alloc], dtype=torch.long, device=self._device
            )
            a_tok = torch.tensor(
                [int(t_list[i]) for i in alloc], dtype=torch.long, device=self._device
            )
            a_net = torch.tensor(
                [float(n_list[i]) for i in alloc],
                dtype=torch.float32,
                device=self._device,
            )
            order = torch.argsort(a_slot, stable=True)
            s_sorted = a_slot[order]
            boundary = torch.cat(
                [
                    torch.ones(1, dtype=torch.bool, device=self._device),
                    s_sorted[1:] != s_sorted[:-1],
                ]
            )
            idx = torch.arange(order.numel(), device=self._device)
            run_start = torch.cummax(
                torch.where(boundary, idx, -torch.ones_like(idx)), dim=0
            ).values
            rank = idx - run_start
            base = self._n_used.index_select(0, s_sorted)
            positions = base + rank
            self._cols.index_put_((s_sorted, positions), a_tok[order])
            self._cnts.index_put_((s_sorted, positions), a_net[order])
            self._n_used.index_add_(0, s_sorted, torch.ones_like(s_sorted, dtype=torch.int64))

        if dense_pairs:
            self._apply_dense_pairs(dense_pairs)

    def _apply_dense_pairs(self, dense_pairs: list[tuple[int, int, int]]) -> None:
        """Apply deltas for promoted slots via plain index_put_.

        Pairs are unique per (slot, token), so no accumulation is needed.

        Args:
            dense_pairs: List of ``(slot, token, net)`` deltas for dense rows.

        Raises:
            KeyError: If a delta would drive a dense count negative.
        """
        dense = self._ensure_dense_pool()
        ids = self._dense_slot_ids.tolist()
        d_rows = [ids.index(s) for s, _, _ in dense_pairs]
        d_idx = torch.tensor(d_rows, dtype=torch.long, device=self._device)
        toks = torch.tensor([t for _, t, _ in dense_pairs], dtype=torch.long, device=self._device)
        nets = torch.tensor(
            [float(n) for _, _, n in dense_pairs], dtype=torch.float32, device=self._device
        )
        old = dense[d_idx, toks]
        new = old + nets
        if bool((new < 0).any().item()):
            i = int(torch.nonzero(new < 0)[0])
            raise KeyError(
                f"unrecord token {int(toks[i])} at slot {int(self._dense_slot_ids[d_idx[i]])} "
                "with insufficient count in the dense table — likely a caller bug."
            )
        dense.index_put_((d_idx, toks), new)

    def _compact(self) -> None:
        """Compact per slot: drop zero-count entries, pack survivors to front.

        Generalizes swap-remove to multiple deletions in one flush — stable
        sort by a "live-first" key keeps relative order; fully on device, no
        host sync.
        """
        b, cap = self._cols.shape
        if b == 0 or cap == 0:
            return
        j = torch.arange(cap, device=self._device).unsqueeze(0)
        keep = (j < self._n_used.unsqueeze(1)) & (self._cnts > 0)
        n_keep = keep.sum(dim=1)
        key = torch.where(keep, j, j + cap)
        order = key.sort(dim=1, stable=True).indices
        cols_new = self._cols.gather(1, order)
        cnts_new = self._cnts.gather(1, order)
        tail = j >= n_keep.unsqueeze(1)
        cols_new = torch.where(tail, torch.zeros_like(cols_new), cols_new)
        cnts_new = torch.where(tail, torch.zeros_like(cnts_new), cnts_new)
        self._cols.copy_(cols_new)
        self._cnts.copy_(cnts_new)
        self._n_used.copy_(n_keep)


def _canonicalize_rep_logsumexp(logits: torch.Tensor, rep_full: torch.Tensor) -> torch.Tensor:
    """Per-row logsumexp for rows with rep_penalty > 0, 0 for the rest.

    ONE read per row: `row_logsumexp` in the FUSED_STATS section above takes
    row_gate directly on device — NO torch.nonzero (implicit host sync) and NO
    index_select materializing an [active x V] copy like the old version.

    Args:
        logits: [n_rows, V] logits.
        rep_full: [n_rows] per-row repetition penalties used as the gate.

    Returns:
        Float32 [n_rows] logsumexp (0 for rows without repetition penalty).
    """
    return row_logsumexp(logits, row_gate=rep_full)


def _slot_map(md: SamplingMetadata) -> torch.Tensor | None:
    """Return the slot → packed-row map [max_batch_size], or None when identity.

    BiasState/PenaltyState store state per SLOT; logits/md.active are per
    PACKED row. When ``set_active_rows`` is non-identity (LIFO slot reuse,
    mixed batch), this map is required — otherwise penalty/bias hits the
    wrong row.

    Args:
        md: Sampling metadata holding the active-row mapping.

    Returns:
        Long slot→packed map, or None for the identity mapping.
    """
    active = md.active_rows
    if active is None:
        return None
    return torch.tensor(active, dtype=torch.long, device=md.device)


def _apply_penalties_core(
    logits: torch.Tensor,
    md: SamplingMetadata,
    rows: torch.Tensor,
    cols: torch.Tensor,
    cnt: torch.Tensor,
    valid: torch.Tensor,
    dense_cnts: torch.Tensor | None,
    dense_slot_ids: torch.Tensor | None,
    slot_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalty formula — ONE single copy shared by both paths.

    rows/cols/cnt: [N] or [B, S]; valid has the same shape. Padding
    (valid=False) may alias a real entry → write DELTA (new - old) via
    index_put_(accumulate=True): ordered accumulation for duplicate indices,
    padding delta forced to 0 — adding 0 anywhere is safe.

    Args:
        logits: [n_active, V] logits adjusted in place.
        md: Sampling metadata with per-row penalty weights.
        rows: Packed row indices ([N] or [B, S]).
        cols: Token columns matching ``rows``.
        cnt: Per-coord generation counts.
        valid: Same-shape mask marking real entries.
        dense_cnts: Optional dense-pool counts for promoted slots.
        dense_slot_ids: Optional dense-pool slot ids.
        slot_map: Optional slot → packed map for dense ids.

    Returns:
        The same ``logits`` tensor, adjusted in place.
    """
    rep_full = md.active("rep_penalty")
    freq_full = md.active("freq_penalty")
    pres_full = md.active("pres_penalty")

    lse = _canonicalize_rep_logsumexp(logits, rep_full)

    rows_f = rows.reshape(-1)
    cols_f = cols.reshape(-1)
    cnt_f = cnt.reshape(-1).to(torch.float32)
    valid_f = valid.reshape(-1).to(torch.bool)

    rep_row = rep_full.index_select(0, rows_f)
    freq = freq_full.index_select(0, rows_f)
    pres = pres_full.index_select(0, rows_f)
    lse_row = lse.index_select(0, rows_f)

    old_vals = logits[rows_f, cols_f].to(torch.float32)
    shifted = old_vals - lse_row
    penalized = torch.where(shifted < 0, shifted * rep_row, shifted / rep_row)
    new_vals = torch.where(rep_row > 0, penalized + lse_row, old_vals)
    new_vals = new_vals - freq * cnt_f - pres * (cnt_f > 0).to(new_vals.dtype)

    # inf/nan may appear in the UNTAKEN branch — safe because torch.where does
    # not propagate NaN/Inf from the untaken branch, and delta is forced to 0.
    delta = torch.where(valid_f, new_vals - old_vals, torch.zeros_like(new_vals))
    if rows_f.numel():
        logits.index_put_((rows_f, cols_f), delta.to(logits.dtype), accumulate=True)

    # dense branch: promoted rows apply directly over the full [V].
    if dense_cnts is not None and dense_slot_ids is not None:
        d = dense_slot_ids.size(0)
        n_rows = logits.size(0)
        if d and n_rows:
            sid = dense_slot_ids.clamp_min(0)
            if slot_map is not None:
                # Dense ids are SLOTs — map to packed logits rows.
                sid = slot_map.index_select(0, sid.clamp_max(slot_map.size(0) - 1))
            act = (dense_slot_ids >= 0) & (
                dense_slot_ids < (slot_map.size(0) if slot_map is not None else n_rows)
            )
            rep = rep_full.index_select(0, sid) * act.to(rep_full.dtype)
            freq_d = freq_full.index_select(0, sid) * act.to(freq_full.dtype)
            pres_d = pres_full.index_select(0, sid) * act.to(pres_full.dtype)
            cnt_d = dense_cnts * act.to(dense_cnts.dtype).unsqueeze(1)
            v_old = logits.index_select(0, sid).to(torch.float32)
            lse_d = lse.index_select(0, sid).unsqueeze(1)
            shifted_d = v_old - lse_d
            penalized_d = torch.where(
                shifted_d < 0, shifted_d * rep.unsqueeze(1), shifted_d / rep.unsqueeze(1)
            )
            # The rep branch ONLY touches counted tokens (cnt > 0) — matches
            # sparse: coords exist only for tokens with count.
            counted = (cnt_d > 0) & (rep.unsqueeze(1) > 0)
            v_new = torch.where(counted, penalized_d + lse_d, v_old)
            v_new = (
                v_new
                - freq_d.unsqueeze(1) * cnt_d
                - pres_d.unsqueeze(1) * (cnt_d > 0).to(v_new.dtype)
            )
            delta_d = torch.where(act.unsqueeze(1), v_new - v_old, torch.zeros_like(v_new))
            v_sz = logits.size(1)
            col_idx = (
                torch.arange(v_sz, device=logits.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(d, v_sz)
            )
            logits.index_put_(
                (sid.unsqueeze(1).expand(d, v_sz), col_idx),
                delta_d.to(logits.dtype),
                accumulate=True,
            )
    return logits


def apply_penalties_(
    logits: torch.Tensor, md: SamplingMetadata, state: PenaltyState
) -> torch.Tensor:
    """Apply penalties in place. Runs BEFORE mask — see NEG_INF in ops/sampling
    (BITMASK section).

    rep: sign-branch on normalized log-softmax (see the module docstring).
    freq/pres: unchanged additive/subtractive form from the original.

    Builds compact 1-D coords (flushed) then enters _apply_penalties_core —
    the same path as apply_penalties_padded_. State coords are in SLOT space;
    md.active is PACKED — `_slot_map` translates slot → packed when the
    active-row map is non-identity.

    Args:
        logits: [n_active, V] logits adjusted in place.
        md: Sampling metadata with penalty weights.
        state: Penalty counts to apply.

    Returns:
        The same ``logits`` tensor, adjusted in place.

    Raises:
        ValueError: If ``logits`` rows do not match ``md.n_active``.
    """
    n = md.n_active
    if logits.size(0) != n:
        raise ValueError(
            f"logits.size(0)={logits.size(0)} != md.n_active={n} — caller passed "
            "mismatched logits for the metadata batch (e.g. sliced logits without "
            "slicing md/state over the same row set). Fail-closed instead of a "
            "confusing downstream IndexError."
        )
    if n == 0 or not md.any_penalty:
        return logits
    rows, cols, cnt = state.coords(md.active_rows or n, logits.device)
    dense_cnts = state.dense_counts if state._dense_used else None
    dense_ids = state.dense_slot_ids if state._dense_used else None
    if rows.numel() == 0 and dense_cnts is None:
        return logits
    valid = torch.ones(rows.numel(), dtype=torch.bool, device=rows.device)
    slot_map = _slot_map(md)
    return _apply_penalties_core(
        logits, md, rows, cols, cnt, valid, dense_cnts, dense_ids, slot_map
    )


def apply_penalties_padded_(
    logits: torch.Tensor,
    md: SamplingMetadata,
    rows: torch.Tensor,
    cols: torch.Tensor,
    cnt: torch.Tensor,
    valid: torch.Tensor,
    *,
    dense_cnts: torch.Tensor | None = None,
    dense_slot_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Graph-capturable variant of apply_penalties_ — takes padded coords directly.

    Takes the static 2-D [B, S] output of :meth:`PenaltyState.coords_padded`
    instead of calling ``coords()`` (``coords()`` is dynamic-shape and must
    not run inside a graph).

    Slots promoted to dense do NOT appear in coords_padded — the caller MUST
    pass dense_cnts/dense_slot_ids (same static shape) so the dense part still
    applies inside the capture region; omitting them silently drops dense-row
    penalties — with apply_penalties_ that cannot happen (it always passes
    them).

    Args:
        logits: [n_active, V] logits adjusted in place.
        md: Sampling metadata with penalty weights.
        rows: Static packed row indices.
        cols: Static token columns.
        cnt: Static per-coord counts.
        valid: Static mask marking real entries.
        dense_cnts: Optional static dense-pool counts.
        dense_slot_ids: Optional static dense-pool slot ids.

    Returns:
        The same ``logits`` tensor, adjusted in place.
    """
    return _apply_penalties_core(logits, md, rows, cols, cnt, valid, dense_cnts, dense_slot_ids)


# DRY ("Don't Repeat Yourself") — penalizes the token that would EXTEND a
# repeated n-gram, unlike the PENALTIES section above (which penalizes by
# token FREQUENCY, ignoring order/pattern).
#
# NO canonical reference for bit-exact comparison yet (Ayaka has no Rust or
# reference DRY build — unlike the GUMBEL/MIROSTAT sections of ops/sampling,
# which are ports from ayaka-sampling). The (multiplier, base, allowed_length) parameters follow the
# common llama.cpp/koboldcpp convention — RE-VERIFY if exact parity with a
# specific implementation is required.
#
# Algorithm: for each history position i (0 <= i < T-1), measure the backward
# match length k between history[i-k:i] and history[T-k:T] (the current tail).
# When k reaches allowed_length, token history[i] (the token that once
# "followed" that matched pattern) gets
# penalty = multiplier * base^(k - allowed_length). When several positions
# penalize one token, take the MAX (no accumulation).
#
# DELIBERATELY written as a Python loop (O(B x T x max_ngram), not vectorized)
# — this is a reference oracle, not the hot path. Unlike PenaltyState (which
# already delta-flushes on device tables), DRY needs HISTORY ORDER, so it keeps
# the host loop; it is a later optimization/kernelization candidate only if
# profiling says so, not now.


def dry_bias(
    history: list[list[int]],
    vocab_size: int,
    multiplier: torch.Tensor,
    base: torch.Tensor,
    allowed_length: torch.Tensor,
    *,
    max_ngram: int = 32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the [B, V] DRY bias (<= 0) to ADD into logits.

    Unlike `apply_penalties_` this is never a sign-branch — DRY is always
    additive, so it has no gauge-dependence concern.

    Args:
        history: Per-request generated token ids, in EXACT ORDER (unlike
            PenaltyState — DRY needs order, not just frequency).
        vocab_size: Vocabulary size (bias width).
        multiplier: [B] per-row penalty scale.
        base: [B] per-row exponential base.
        allowed_length: [B] per-row match length before penalizing.
        max_ngram: Maximum backward match length to consider.
        device: Device for the returned bias. Defaults to CPU.

    Returns:
        ``[B, V]`` non-positive bias to add into logits.
    """
    b = len(history)
    bias = torch.zeros(b, vocab_size, device=device)
    mult = multiplier.tolist()
    bse = base.tolist()
    allow = allowed_length.tolist()

    for row, hist in enumerate(history):
        t = len(hist)
        if t < 2:
            continue
        cap = min(max_ngram, t - 1)
        al = int(allow[row])
        for i in range(t - 1):
            k = 0
            while (
                k < cap
                and i - 1 - k >= 0
                and (t - 1 - k) >= 0
                and hist[i - 1 - k] == hist[t - 1 - k]
            ):
                k += 1
            if k >= al:
                tok = hist[i]
                penalty = mult[row] * (bse[row] ** (k - al))
                if -penalty < bias[row, tok]:
                    bias[row, tok] = -penalty
    return bias


def apply_dry_(
    logits: torch.Tensor,
    history: list[list[int]],
    multiplier: torch.Tensor,
    base: torch.Tensor,
    allowed_length: torch.Tensor,
    *,
    max_ngram: int = 32,
) -> torch.Tensor:
    """Apply DRY penalties in place, additively.

    Needs no canonicalization like the rep branch of apply_penalties_ because
    this is not a sign-branch on raw logits.

    Args:
        logits: [B, V] logits adjusted in place.
        history: Per-request generated token ids in exact order.
        multiplier: [B] per-row penalty scale.
        base: [B] per-row exponential base.
        allowed_length: [B] per-row match length before penalizing.
        max_ngram: Maximum backward match length to consider.

    Returns:
        The same ``logits`` tensor, adjusted in place.
    """
    bias = dry_bias(
        history,
        logits.size(1),
        multiplier,
        base,
        allowed_length,
        max_ngram=max_ngram,
        device=logits.device,
    )
    logits += bias
    return logits
