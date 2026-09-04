from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ayaka.types import AttentionCudaGraphSupport, ForwardMode

if TYPE_CHECKING:
    import torch

    from ayaka.attention.ports import PagedKVCache
    from ayaka.attention.spec import AttentionGroupSpec


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


class BaseAttentionMetadata:
    """Marker base for a backend's concrete per-forward metadata.

    Deliberately empty. FreeToken puts ``get_last_indices(bs)`` on its base, which forces
    every backend -- including ones whose query indexing has nothing to do with the LM head
    -- to carry an LM-head concern. That index is derivable from
    ``CommonAttnMetadata.query_start_loc`` by the sampler, so it belongs there, not here.
    """

    __slots__ = ()


class BaseAttentionMetadataBuilder[M: BaseAttentionMetadata](ABC):
    """Per-group metadata construction and CUDA-graph buffer ownership.

    Lifecycle, in order:

        __init__(group, kv_cache, device)
        init_graph_state(...)            once, if this group is capture-eligible
        build(common)                    every eager forward
        build_for_capture(common, bs)    once per captured batch size
        build_for_replay(common, bs)     every replayed forward -- must be COPY ONLY
        reset_graph_state()              on pool rebuild

    ``build_for_replay`` running on the hot path is why it must not allocate, must not
    synchronize, and must not call anything that plans. If a backend's planner cannot meet
    that (FlashInfer's ``plan()`` reads host indptr and can allocate), the backend declares
    a lower ``cudagraph_support`` rather than doing it anyway.
    """

    #: How much of a batch this builder's backend can serve from a captured graph.
    #:
    #: This is deliberately an ordinary class default rather than a ``ClassVar``: each
    #: instance copies it and may lower the effective support after inspecting its group.
    cudagraph_support: AttentionCudaGraphSupport = AttentionCudaGraphSupport.NEVER

    #: Whether ``build`` needs host-side sequence lengths (blocks overlap scheduling on the
    #: H2D copy issued by PinnedStaging). FlashInfer needs them; Triton and FA do not.
    reads_host_lens: ClassVar[bool] = False

    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
    ) -> None:
        self.group = group
        self.spec = group.spec
        self.kv_cache = kv_cache
        self.device = device
        self.capture_sizes: list[int] = []
        self.max_graph_batch_size: int | None = None
        self.max_graph_seq_len: int | None = None
        self.max_graph_query_len: int | None = None
        # Instance copy so a builder can LOWER its declared support after inspecting the
        # group (e.g. a spec-decode-capable kernel serving a group that never verifies).
        # Raising it above the class value is never correct -- the registry advertised the
        # class value to the selector, and a promise made at selection time must hold.
        self.cudagraph_support = type(self).cudagraph_support

    @abstractmethod
    def build(self, common: CommonAttentionMetadata) -> M:
        """Eager path. May allocate; runs on the scheduler stream under overlap."""

    def init_graph_state(
        self,
        *,
        max_batch_size: int,
        max_seq_len: int,
        capture_sizes: list[int],
        max_query_len: int = 1,
    ) -> None:
        """Allocate every persistent buffer this builder will hand to a capture.

        ``max_seq_len`` is the engine's live ceiling -- ``min(model max positions, KV token
        budget)`` -- not the longest sequence seen so far. Capture width is a promise about
        every future replay, so it must come from admission control, not from history.
        """
        self.capture_sizes = sorted(capture_sizes)
        self.max_graph_batch_size = max_batch_size
        self.max_graph_seq_len = max_seq_len
        self.max_graph_query_len = max_query_len

    def build_for_capture(self, common: CommonAttentionMetadata, padded_batch_size: int) -> M:
        """Metadata bound to the persistent buffers, for the capture pass."""
        raise NotImplementedError(
            f"{type(self).__name__} declares cudagraph_support="
            f"{self.cudagraph_support.name} but does not implement build_for_capture "
            f"for mode={common.mode.name}, padded_batch_size={padded_batch_size}"
        )

    def build_for_replay(self, common: CommonAttentionMetadata, padded_batch_size: int) -> M:
        """Restage this step into the persistent buffers. Copy only; no allocation."""
        raise NotImplementedError(
            f"{type(self).__name__} declares cudagraph_support="
            f"{self.cudagraph_support.name} but does not implement build_for_replay "
            f"for mode={common.mode.name}, padded_batch_size={padded_batch_size}"
        )

    def reset_graph_state(self) -> None:
        """Drop capture scratch so ``init_graph_state`` can run again after a rebuild."""
        self.capture_sizes = []
        self.max_graph_batch_size = None
        self.max_graph_seq_len = None
        self.max_graph_query_len = None
