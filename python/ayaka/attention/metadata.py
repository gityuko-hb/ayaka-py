from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, TypeVarTuple

from ayaka.types import ForwardMode

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CommonAttentionMetadata:
    """Backend-agnostic per-group addressing for one forward.


    Built ONCE per (forward, group) handed to the group's metadata builder. This
    is the piece FreeToken does not have: its fa/fi/triton/trtllm backends each rebuild the
    same ``cu_seqlens`` cumsum and the same page-table gather in their own
    ``prepare_metadata``, so the identical host loop runs once per backend.

    Tensor discipline
    -----------------
    * ``*_cpu`` fields are PINNED host mirrors, for planners that genuinely need host values
      (FlashInfer's ``plan()`` reads indptr on the host). Nothing else may read them; a
      kernel bound taken from here breaks graph replay.
    * ``block_table`` is a SNAPSHOT, never a view of the live page table -- the next batch's
      allocation mutates that table while this batch's graph is still replaying.
    * ``seq_lens`` / ``query_start_loc`` are int32 on device and ARE the kernel bounds.

    Shapes (R = num_reqs, T = num_tokens):
      query_start_loc  [R+1] int32   cumulative query lengths, [0] == 0
      seq_lens         [R]   int32   total KV length after this forward's writes
      computed_lens    [R]   int32   cached prefix length (seq_lens - query_len)
      block_table      [R,C] int32   page ids; C >= ceil(max_seq_len / page_size)
      slot_mapping     [T]   int32   flat write destination for this forward's K/V
      positions        [T]   int32   absolute position of each query token
      spec_tree_mask   [T]   int64   TARGET_VERIFY only; bit j == may attend request-local
                                     draft token j. None for a linear (chain) draft, where
                                     ordinary causal masking is already correct.
    """

    # device (kernel-visible)
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    computed_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    positions: torch.Tensor

    # pinned host mirrors (planner-only)
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor

    # host scalars: SHAPE HINTS ONLY, never a kernel bound
    num_reqs: int
    num_tokens: int
    max_query_len: int
    max_seq_len: int
    mode: ForwardMode

    # speculative decoding
    spec_tree_mask: torch.Tensor | None = None

    @property
    def num_blocks_per_req(self) -> int:
        """Staged width of the block table. A capture bakes this, so every replay must fit."""
        return int(self.block_table.shape[1])

    @property
    def has_tree_mask(self) -> bool:
        return self.spec_tree_mask is not None

class BaseAttentionMetadata(ABC):
    """Marker base for a backend's concrete per-forward metadata.
 
    Deliberately empty. FreeToken puts ``get_last_indices(bs)`` on its base, which forces
    every backend -- including ones whose query indexing has nothing to do with the LM head
    -- to carry an LM-head concern. That index is derivable from
    ``CommonAttnMetadata.query_start_loc`` by the sampler, so it belongs there, not here.
    """
 
    __slots__ = ()
    
M = TypeVar("M", bound=BaseAttentionMetadata)
