from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from ayaka.attention.base import BaseAttentionBackend
from ayaka.attention.errors import (
    AttentionErrorCode,
    BackendCapabilityError,
    GraphStateError,
)
from ayaka.attention.metadata import (
    BaseAttentionMetadata,
    BaseAttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from ayaka.attention.ports import PagedKVCache
from ayaka.attention.spec import AttentionGroupSpec
from ayaka.types import AttentionCudaGraphSupport, AttentionType, KVLayoutKind, MaskKind
from ayaka.utils.import_utils import CapabilityError, require_module

__all__ = [
    "FlashInferBackend",
    "FlashInferMetadata",
    "FlashInferMetadataBuilder",
]

_MIN_WORKSPACE = 256 * 1024 * 1024
_WORKSPACE_SLACK = 32 * 1024 * 1024
_MAX_HEAD_DIM = 256

#: FlashInfer spells its layouts in uppercase; ``KVLayoutKind`` values are lowercase.
_FLASHINFER_LAYOUT: dict[KVLayoutKind, str] = {
    KVLayoutKind.NHD: "NHD",
    KVLayoutKind.HND: "HND",
}

_FLASHINFER_REMEDY = 'pip install "flashinfer-python>=0.6,<0.7"'


def _require_flashinfer() -> ModuleType:
    """Import flashinfer or refuse with a remedy.

    Lazy on purpose: flashinfer JIT-compiles kernels on first use, and a process that
    selects a Triton or torch backend must not pay that cost. The ImportError is normalized
    to ``MISSING_DEPENDENCY`` so selection can record why this candidate lost.
    """

    try:
        return require_module("flashinfer", capability="flashinfer", remedy=_FLASHINFER_REMEDY)
    except CapabilityError as exc:
        raise BackendCapabilityError(
            AttentionErrorCode.MISSING_DEPENDENCY,
            f"flashinfer is required but not importable: {exc.detail}",
            remedy=exc.remedy,
        ) from exc


def _check_group_supported(group: AttentionGroupSpec) -> None:
    """Refuse at construction what no forward could serve correctly.

    All conditions are static properties of the group, so they cost nothing per step and run
    before the optional import.
    """

    spec = group.spec
    if group.attn_type not in (AttentionType.FULL, AttentionType.SWA):
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_ATTN_TYPE,
            f"flashinfer serves FULL and SWA groups, not {group.attn_type.value}",
            group_id=group.group_id,
        )
    if spec.has_sinks:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_SINKS,
            "flashinfer has no attention-sink term",
            group_id=group.group_id,
        )
    if group.kv_layout is KVLayoutKind.NLD:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_KV_LAYOUT,
            "flashinfer addresses the pool as NHD or HND, not as an MLA latent plane",
            group_id=group.group_id,
            kv_layout=group.kv_layout.value,
        )
    if group.kv_cache_dtype.is_quantized:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_KV_CACHE_DTYPE,
            "the quantized-KV path needs per-layer device scales threaded through plan/run; "
            "not wired up yet",
            group_id=group.group_id,
            kv_cache_dtype=group.kv_cache_dtype.value,
        )
    if spec.head_dim_qk > _MAX_HEAD_DIM or spec.head_dim_qk % 8:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_HEAD_DIM,
            f"flashinfer needs head_dim_qk <= {_MAX_HEAD_DIM} and a multiple of 8",
            group_id=group.group_id,
            head_dim=spec.head_dim_qk,
        )
    if spec.head_dim_vo > _MAX_HEAD_DIM or spec.head_dim_vo % 8:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_HEAD_DIM,
            f"flashinfer needs head_dim_vo <= {_MAX_HEAD_DIM} and a multiple of 8",
            group_id=group.group_id,
            head_dim=spec.head_dim_vo,
        )


def _derive_workspace_bytes(spec, device: torch.device) -> int:
    try:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:  # pragma: no cover
        sm_count = 128
    cta_tile_q = 64 if spec.head_dim_qk >= 256 else 128
    padded_batch = -(-2 * sm_count // max(1, spec.num_kv_heads))
    tmp_v = spec.num_qo_heads * padded_batch * cta_tile_q * spec.head_dim_qk * 4
    return max(_MIN_WORKSPACE, tmp_v + _WORKSPACE_SLACK)


@dataclass
class FlashInferMetadata(BaseAttentionMetadata):
    __slots__ = ("common", "wrapper", "output")

    common: CommonAttentionMetadata
    wrapper: Any  # BatchPrefill/BatchDecodeWithPagedKVCacheWrapper
    output: torch.Tensor | None


class FlashInferMetadataBuilder(BaseAttentionMetadataBuilder):
    """FlashInfer plan/run pair, split across the graph boundary exactly where it must be.

    ``plan()`` reads host scheduling metadata, so it can never live inside a capture. The
    builder therefore does two things at capture time: it runs the first plan against the
    persistent indptr/indices buffers so the captured kernel binds those addresses, and on
    every replay it re-runs the plan OUTSIDE the graph so the buffers hold this step's
    addressing before ``replay()`` reads them.
    """

    cudagraph_support = AttentionCudaGraphSupport.PURE_DECODE
    reads_host_lens = True

    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
        flashinfer: ModuleType,
    ) -> None:
        super().__init__(group, kv_cache, device)
        self._flashinfer = flashinfer
        spec = group.spec
        self._float_workspace = torch.empty(
            _derive_workspace_bytes(spec, device), dtype=torch.uint8, device=device
        )
        # Separate int workspaces: sharing one through private attributes saved 8 MiB and
        # coupled two wrappers that the graph path must be able to own independently.
        self._prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._float_workspace, kv_layout=_FLASHINFER_LAYOUT[group.kv_layout], backend="fa2"
        )
        self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._float_workspace,
            use_tensor_cores=spec.gqa_group_size >= 4,
            kv_layout=_FLASHINFER_LAYOUT[group.kv_layout],
            backend="fa2",
        )

        self._graph_wrappers: dict[int, Any] = {}
        self._graph_indptr: torch.Tensor | None = None
        self._graph_indices: torch.Tensor | None = None
        self._graph_last_page: torch.Tensor | None = None
        self._graph_output: torch.Tensor | None = None
        self._max_columns = 0
        # plan() launches an async H2D from a pinned staging buffer it reuses. Waiting on
        # this event before the next plan keeps the previous copy from being overwritten
        # mid-flight -- a corruption that only shows under load.
        self._plan_done = torch.cuda.Event()
        self._plan_done.record()

    def _ragged(self, common: CommonAttentionMetadata):
        """(indptr, indices, last_page_len) on host, derived from the block table.

        The block table is rectangular and the page counts come from ``seq_lens_cpu``, so the
        flatten is one host-side prefix sum plus one device gather -- not R separate slices.

        Rows in ``[len(seq_lens_cpu), num_reqs)`` are padding: they get zero pages and a
        sentinel ``last_page_len`` of 1. FlashInfer's CUDA-graph planner requires the arrays
        to have the captured batch size, and replaying a partial batch must still hand it
        that size -- the trailing rows simply do no work.
        """
        page = self.group.page_size
        seq_lens_cpu = common.seq_lens_cpu[: common.num_reqs]
        if seq_lens_cpu.numel() < common.num_reqs:
            pad = common.num_reqs - seq_lens_cpu.numel()
            seq_lens_cpu = torch.cat((seq_lens_cpu, torch.zeros(pad, dtype=seq_lens_cpu.dtype)))
        num_pages = ((seq_lens_cpu.to(torch.int64) + page - 1) // page).to(torch.int32)
        indptr = torch.zeros(common.num_reqs + 1, dtype=torch.int32)
        indptr[1:] = torch.cumsum(num_pages, 0, dtype=torch.int32)
        last_page = seq_lens_cpu - (num_pages - 1) * page
        last_page = torch.where(seq_lens_cpu > 0, last_page, torch.ones_like(last_page))

        # Flatten the rectangular table to the ragged list with one masked select on device.
        cols = torch.arange(common.num_blocks_per_req, device=self.device, dtype=torch.int32)
        keep = cols.unsqueeze(0) < num_pages.to(self.device).unsqueeze(1)
        indices = common.block_table[: common.num_reqs].masked_select(keep)
        return indptr, indices, last_page.to(torch.int32)

    def _plan(self, wrapper, common: CommonAttentionMetadata, decode: bool) -> None:
        spec = self.spec
        indptr, indices, last_page = self._ragged(common)
        self._plan_done.synchronize()
        common_kwargs = dict(
            num_qo_heads=spec.num_qo_heads,
            num_kv_heads=spec.num_kv_heads,
            page_size=self.group.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.kv_cache.dtype,
            kv_data_type=self.kv_cache.dtype,
            non_blocking=True,
        )
        window = spec.sliding_window if spec.mask is MaskKind.SLIDING else None
        if decode:
            wrapper.plan(
                indptr=indptr,
                indices=indices,
                last_page_len=last_page,
                head_dim=spec.head_dim_qk,
                window_left=-1 if window is None else window - 1,
                logits_soft_cap=spec.logits_soft_cap,
                # sm_scale belongs to the plan: run() takes no such argument, and passing it
                # there is a TypeError, not a silently ignored keyword.
                sm_scale=spec.sm_scale,
                **common_kwargs,
            )
        else:
            wrapper.plan(
                qo_indptr=common.query_start_loc_cpu[: common.num_reqs + 1],
                paged_kv_indptr=indptr,
                paged_kv_indices=indices,
                paged_kv_last_page_len=last_page,
                head_dim_qk=spec.head_dim_qk,
                head_dim_vo=spec.head_dim_vo,
                causal=spec.mask is not MaskKind.FULL,
                window_left=-1 if window is None else window - 1,
                logits_soft_cap=spec.logits_soft_cap,
                sm_scale=spec.sm_scale,
                **common_kwargs,
            )
        self._plan_done.record()

    # ----- eager ---------------------------------------------------------------------------
    def build(self, common: CommonAttentionMetadata) -> FlashInferMetadata:
        decode = common.mode.is_decode_like and common.max_query_len == 1
        wrapper = self._decode if decode else self._prefill
        self._plan(wrapper, common, decode)
        return FlashInferMetadata(common, wrapper, None)

    # ----- cuda graph -----------------------------------------------------------------------
    def init_graph_state(
        self,
        *,
        max_batch_size: int,
        max_seq_len: int,
        capture_sizes: list[int],
        max_query_len: int = 1,
    ) -> None:
        super().init_graph_state(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            capture_sizes=capture_sizes,
            max_query_len=max_query_len,
        )
        if max_query_len != 1:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_MODE_UNSUPPORTED,
                "flashinfer captures pure decode only; a uniform multi-token query batch "
                "needs the prefill wrapper, whose plan() cannot run inside a graph",
                max_query_len=max_query_len,
            )
        page = self.group.page_size
        self._max_columns = (max_seq_len + page - 1) // page
        spec = self.spec
        # indices is the flattened worst case: every request at the full column width.
        self._graph_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)
        self._graph_indices = torch.zeros(
            max_batch_size * self._max_columns, dtype=torch.int32, device=self.device
        )
        self._graph_last_page = torch.ones(max_batch_size, dtype=torch.int32, device=self.device)
        self._graph_output = torch.empty(
            (max_batch_size, spec.num_qo_heads, spec.head_dim_vo),
            dtype=self.kv_cache.dtype,
            device=self.device,
        )

    def _graph_wrapper(self, bs: int):
        wrapper = self._graph_wrappers.get(bs)
        if wrapper is None:
            assert self._graph_indptr is not None
            assert self._graph_last_page is not None
            # The plain wrapper with use_cuda_graph=True is the public CUDA-graph entry
            # point; it takes the backend explicitly, so no post-construction private
            # mutation of _backend is needed to pin the kernel family.
            wrapper = self._flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._float_workspace,
                kv_layout=_FLASHINFER_LAYOUT[self.group.kv_layout],
                use_cuda_graph=True,
                use_tensor_cores=self.spec.gqa_group_size >= 4,
                paged_kv_indptr_buffer=self._graph_indptr[: bs + 1],
                paged_kv_indices_buffer=self._graph_indices,
                paged_kv_last_page_len_buffer=self._graph_last_page[:bs],
                backend="fa2",
            )
            self._graph_wrappers[bs] = wrapper
        return wrapper

    def build_for_capture(
        self, common: CommonAttentionMetadata, padded_batch_size: int
    ) -> FlashInferMetadata:
        wrapper = self._graph_wrapper(padded_batch_size)
        self._plan(wrapper, common, decode=True)
        assert self._graph_output is not None
        return FlashInferMetadata(common, wrapper, self._graph_output[:padded_batch_size])

    def build_for_replay(
        self, common: CommonAttentionMetadata, padded_batch_size: int
    ) -> FlashInferMetadata:
        # Runs on the engine stream immediately before replay(), OUTSIDE the graph: the
        # wrapper writes its scheduling metadata into the buffers the capture already
        # recorded, so replay picks it up without the plan itself being captured.
        wrapper = self._graph_wrappers.get(padded_batch_size)
        if wrapper is None:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_BATCH_SIZE_UNSUPPORTED,
                "no flashinfer graph wrapper was built for this batch size",
                batch_size=padded_batch_size,
                available=sorted(self._graph_wrappers),
            )
        self._plan(wrapper, common, decode=True)
        assert self._graph_output is not None
        return FlashInferMetadata(common, wrapper, self._graph_output[:padded_batch_size])

    def reset_graph_state(self) -> None:
        super().reset_graph_state()
        # The wrappers alias indptr/indices scratch that is about to be freed; dropping them
        # is what lets init_graph_state run again after a pool rebuild. The float/int
        # workspaces are long-lived and deliberately survive.
        self._graph_wrappers = {}
        self._graph_indptr = None
        self._graph_indices = None
        self._graph_last_page = None
        self._graph_output = None


class FlashInferBackend(BaseAttentionBackend):
    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
    ) -> None:
        super().__init__(group, kv_cache, device)
        major, minor = torch.cuda.get_device_capability(device.index or 0)
        if major < 8:
            raise BackendCapabilityError(
                AttentionErrorCode.UNSUPPORTED_ARCH,
                "flashinfer's fa2 kernels require sm80 or newer",
                compute_capability=f"{major}.{minor}",
            )
        _check_group_supported(group)
        self._flashinfer = _require_flashinfer()

    def build_metadata_builder(self) -> FlashInferMetadataBuilder:
        return FlashInferMetadataBuilder(self.group, self.kv_cache, self.device, self._flashinfer)

    def forward(
        self,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: BaseAttentionMetadata,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(metadata, FlashInferMetadata)
        spec = self.spec
        common = metadata.common
        if common.has_tree_mask:
            raise BackendCapabilityError(
                AttentionErrorCode.UNSUPPORTED_TREE_MASK,
                "flashinfer paged attention applies a dense causal mask; a speculative tree "
                "mask has no encoding in plan/run",
                group_id=self.group.group_id,
            )
        key = key.view(-1, spec.num_kv_heads, spec.head_dim_qk)
        value = value.view(-1, spec.num_kv_heads, spec.head_dim_vo)
        self.kv_cache.store_kv(key, value, common.slot_mapping, layer_id)

        paged = (
            self.kv_cache.key_cache(layer_id),
            self.kv_cache.value_cache(layer_id),
        )
        out = output if output is not None else metadata.output
        # sm_scale lives in plan(); run() takes q/paged_kv_cache/out only.
        return metadata.wrapper.run(
            q=query.view(-1, spec.num_qo_heads, spec.head_dim_qk),
            paged_kv_cache=paged,
            out=out,
        )
