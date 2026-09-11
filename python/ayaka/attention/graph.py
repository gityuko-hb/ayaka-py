from __future__ import annotations

from dataclasses import dataclass

import torch

from ayaka.attention.metadata import CommonAttentionMetadata
from ayaka.types import AttentionCudaGraphSupport, ForwardMode

from .errors import AttentionErrorCode, GraphStateError

__all__ = ["GraphBuffers"]


@dataclass
class GraphBuffers:
    """Address-stable device buffers for one attention group's captured decode path.

    Sizing rules, and why each one is what it is:

    * ``max_batch_size`` = max of the capture batch-size list.
    * ``max_seq_len`` = the engine's live sequence ceiling in TOKENS (admission control),
      the bound a kernel schedules against. Not ``max_columns``, which is that ceiling
      measured in pages.
    * ``max_columns`` = the block-table width a replay may need, derived from the engine's
      live sequence ceiling, NOT from whatever the capture batch happened to have. A capture
      bakes the width; a replay with a wider table has nowhere to put the extra columns and
      must be refused loudly (``GRAPH_STAGING_WIDTH_EXCEEDED``) rather than silently
      truncated. FreeToken truncates -- ``w = min(src.shape[1], cap.full_snap.shape[1])``
      then copies ``[:, :w]`` -- which turns an admission-control bug into wrong attention
      output with no error.
    * ``max_tokens`` = ``max_batch_size * max_query_len``; for pure decode that is just
      ``max_batch_size``, for TARGET_VERIFY it is the speculation width times that.

    Fill values matter too. ``block_table`` is filled with 0 rather than -1 because page 0 is
    the reserved null page: a warmup or padded row that reads it gets valid memory and
    ``seq_lens == 0`` stops the kernel before it uses the value. ``seq_lens`` starts at 0 so
    a padded row does zero work -- this is what makes padding to a captured batch size cheap
    instead of a full-length dummy attention.
    """

    seq_lens: torch.Tensor  # [max_bs] int32
    computed_lens: torch.Tensor  # [max_bs] int32
    query_start_loc: torch.Tensor  # [max_bs + 1] int32
    block_table: torch.Tensor  # [max_bs, max_columns] int32
    slot_mapping: torch.Tensor  # [max_tokens] int32
    positions: torch.Tensor  # [max_tokens] int32

    max_batch_size: int
    max_columns: int
    max_seq_len: int
    max_tokens: int
    max_query_len: int
    device: torch.device

    @classmethod
    def create(
        cls,
        *,
        max_batch_size: int,
        max_seq_len: int,
        max_columns: int,
        device: torch.device,
        max_query_len: int = 1,
    ) -> GraphBuffers:
        max_tokens = max_batch_size * max_query_len
        return cls(
            seq_lens=torch.zeros(max_batch_size, dtype=torch.int32, device=device),
            computed_lens=torch.zeros(max_batch_size, dtype=torch.int32, device=device),
            # A padded row must have query_start_loc[i+1] == query_start_loc[i] so it
            # contributes no query tokens; arange gives exactly that for query_len 1 and is
            # overwritten wholesale on every stage anyway.
            query_start_loc=torch.arange(max_batch_size + 1, dtype=torch.int32, device=device),
            block_table=torch.zeros(
                (max_batch_size, max_columns), dtype=torch.int32, device=device
            ),
            slot_mapping=torch.zeros(max_tokens, dtype=torch.int32, device=device),
            positions=torch.zeros(max_tokens, dtype=torch.int32, device=device),
            max_batch_size=max_batch_size,
            max_columns=max_columns,
            # Token ceiling, unlike ``max_columns`` which counts pages. A kernel that sizes a
            # grid or a scheduling bound from tokens (flash-attn's ``max_seqlen_k``) must get
            # tokens; handing it pages silently shrinks the bound by ``page_size``.
            max_seq_len=max_seq_len,
            max_tokens=max_tokens,
            max_query_len=max_query_len,
            device=device,
        )

    def stage(
        self, common: CommonAttentionMetadata, padded_batch_size: int
    ) -> CommonAttentionMetadata:
        """Copy ``common`` into the persistent buffers; return a view-bound equivalent.

        Runs on the engine stream immediately before ``graph.replay()``, so the copies are
        ordered against both the replay that reads them and the next batch's allocation.

        Rows in ``[num_reqs, padded_batch_size)`` are padding: their ``seq_lens`` is zeroed
        so the kernel exits immediately, and their ``query_start_loc`` is flattened onto the
        last real offset so they own no query tokens.
        """
        bs = common.num_reqs
        if padded_batch_size > self.max_batch_size:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_BATCH_SIZE_UNSUPPORTED,
                "padded batch size exceeds the captured maximum",
                padded_batch_size=padded_batch_size,
                max_batch_size=self.max_batch_size,
            )
        width = common.num_blocks_per_req
        if width > self.max_columns:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_STAGING_WIDTH_EXCEEDED,
                "block table is wider than the captured staging buffer; the engine's "
                "sequence ceiling and the capture width have diverged",
                width=width,
                max_columns=self.max_columns,
            )
        if common.num_tokens > self.max_tokens:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_STAGING_WIDTH_EXCEEDED,
                "token count exceeds the captured staging buffer",
                num_tokens=common.num_tokens,
                max_tokens=self.max_tokens,
            )

        self.seq_lens[:bs].copy_(common.seq_lens, non_blocking=True)
        self.seq_lens[bs:padded_batch_size].zero_()
        self.computed_lens[:bs].copy_(common.computed_lens, non_blocking=True)
        self.computed_lens[bs:padded_batch_size].zero_()
        self.query_start_loc[: bs + 1].copy_(common.query_start_loc, non_blocking=True)
        if padded_batch_size > bs:
            # Flatten the tail onto the final offset: zero query tokens for padded rows.
            self.query_start_loc[bs + 1 : padded_batch_size + 1].fill_(int(common.num_tokens))
        self.block_table[:bs, :width].copy_(common.block_table, non_blocking=True)
        if width < self.max_columns:
            self.block_table[:bs, width:].zero_()
        self.block_table[bs:padded_batch_size].zero_()
        self.slot_mapping[: common.num_tokens].copy_(common.slot_mapping, non_blocking=True)
        self.positions[: common.num_tokens].copy_(common.positions, non_blocking=True)

        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: padded_batch_size + 1],
            seq_lens=self.seq_lens[:padded_batch_size],
            computed_lens=self.computed_lens[:padded_batch_size],
            block_table=self.block_table[:padded_batch_size],
            slot_mapping=self.slot_mapping[: common.num_tokens],
            positions=self.positions[: common.num_tokens],
            query_start_loc_cpu=common.query_start_loc_cpu,
            seq_lens_cpu=common.seq_lens_cpu,
            num_reqs=padded_batch_size,
            num_tokens=common.num_tokens,
            # Both of these are frozen at the capture ceiling, not carried from the live
            # batch, and for the same reason: a backend may legitimately size a GRID from
            # either (the extend kernel's M dimension is cdiv(max_query_len, BLOCK_M)), and
            # a grid dimension that moves between capture and replay is exactly the bug this
            # whole module exists to prevent.
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            mode=common.mode,
        )

    def capture_placeholder(
        self,
        padded_batch_size: int,
        *,
        mode: ForwardMode = ForwardMode.DECODE,
        query_len: int = 1,
    ) -> CommonAttentionMetadata:
        """A metadata object pointing at the buffers, for the capture pass itself.

        The capture batch is dummy rows. What matters is not their contents but that every
        tensor the captured kernels touch is one of these persistent buffers -- so we hand
        back full-width views and let the caller fill them however it likes.
        """
        num_tokens = padded_batch_size * query_len
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: padded_batch_size + 1],
            seq_lens=self.seq_lens[:padded_batch_size],
            computed_lens=self.computed_lens[:padded_batch_size],
            block_table=self.block_table[:padded_batch_size],
            slot_mapping=self.slot_mapping[:num_tokens],
            positions=self.positions[:num_tokens],
            query_start_loc_cpu=torch.zeros(padded_batch_size + 1, dtype=torch.int32),
            seq_lens_cpu=torch.ones(padded_batch_size, dtype=torch.int32),
            num_reqs=padded_batch_size,
            num_tokens=num_tokens,
            max_query_len=query_len,
            max_seq_len=self.max_seq_len,
            mode=mode,
        )


def check_graph_eligible(
    mode: ForwardMode,
    support: int,
    batch_size: int,
    capture_sizes: list[int],
) -> None:
    """Raise unless this batch may be served from a captured graph.

    Declarative, not a probe: the runner asks the backend's declared support level instead
    of trying a capture and catching an exception.
    """

    if support == AttentionCudaGraphSupport.NEVER:
        raise GraphStateError(
            AttentionErrorCode.GRAPH_MODE_UNSUPPORTED,
            "backend declares no CUDA-graph support",
        )
    if mode is ForwardMode.DECODE:
        required = AttentionCudaGraphSupport.PURE_DECODE
    elif mode is ForwardMode.TARGET_VERIFY:
        required = AttentionCudaGraphSupport.UNIFORM_QUERY
    else:
        required = AttentionCudaGraphSupport.ALWAYS
    if support < required:
        raise GraphStateError(
            AttentionErrorCode.GRAPH_MODE_UNSUPPORTED,
            "backend cannot capture this forward mode",
            mode=mode.name,
            support=AttentionCudaGraphSupport(support).name,
            required=AttentionCudaGraphSupport(required).name,
        )
    if batch_size not in capture_sizes:
        raise GraphStateError(
            AttentionErrorCode.GRAPH_BATCH_SIZE_UNSUPPORTED,
            "no graph was captured for this padded batch size",
            batch_size=batch_size,
            capture_sizes=capture_sizes,
        )
