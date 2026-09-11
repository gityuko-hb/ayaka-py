from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ayaka.attention.base import BaseAttentionBackend
from ayaka.attention.errors import (
    AttentionErrorCode,
    BackendCapabilityError,
    GraphStateError,
)
from ayaka.attention.graph import GraphBuffers
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
    "FlashAttentionBackend",
    "FlashAttentionMetadata",
    "FlashAttentionMetadataBuilder",
]

#: flash-attn's forward kernels top out here; a deeper head is padded by the library itself,
#: which changes the output width and is never what the model asked for.
_MAX_HEAD_DIM = 256

#: FA2's paged KV kernel indexes a `block_table` only in whole 256-token pages. FA3 lifted
#: this restriction, which is one more reason the two versions are separate code paths.
_FA2_PAGE_MULTIPLE = 256

_FA2_REMEDY = (
    "pip install flash-attn (source build; there is no binary wheel on PyPI). FA2 covers sm80-sm89"
)
_FA3_REMEDY = (
    "build and install the hopper/ directory of flash-attention, which provides the "
    "flash_attn_interface module; FA3 requires sm90 (Hopper)"
)


@dataclass
class FlashAttentionMetadata(BaseAttentionMetadata):
    __slots__ = ("common", "output")

    common: CommonAttentionMetadata
    output: torch.Tensor | None


def _require_attention_fn(version: int) -> Callable[..., torch.Tensor]:
    """Import the kernel entry point for `version`, or refuse with a remedy.

    Deliberately lazy: importing flash-attn compiles/loads CUDA extensions, and a process
    that selects a Triton backend must not pay for that. The failure is normalized to
    ``MISSING_DEPENDENCY`` so backend selection can report *why* this candidate lost
    instead of leaking a raw ImportError.
    """

    if version == 3:
        module_name = "flash_attn_interface"
        fn_name = "flash_attn_with_kvcache"
        remedy = _FA3_REMEDY
    else:
        module_name = "flash_attn"
        fn_name = "flash_attn_varlen_func"
        remedy = _FA2_REMEDY
    try:
        module = require_module(module_name, capability="flash_attention", remedy=remedy)
    except CapabilityError as exc:
        raise BackendCapabilityError(
            AttentionErrorCode.MISSING_DEPENDENCY,
            f"flash-attn v{version} requires the {module_name!r} module: {exc.detail}",
            remedy=exc.remedy,
        ) from exc
    return getattr(module, fn_name)


def _check_group_supported(group: AttentionGroupSpec, version: int) -> None:
    """Refuse at construction what no forward could serve correctly.

    Every condition here is provable from the static group spec, so this runs once instead of
    being rediscovered per call, and it runs before the optional import so an unsupported
    device never needs flash-attn installed to be told so.
    """

    spec = group.spec
    if group.attn_type not in (AttentionType.FULL, AttentionType.SWA):
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_ATTN_TYPE,
            f"flash-attn serves FULL and SWA groups, not {group.attn_type.value}",
            group_id=group.group_id,
        )
    if spec.has_sinks:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_SINKS,
            "flash-attn has no attention-sink term",
            group_id=group.group_id,
        )
    if group.kv_layout is not KVLayoutKind.NHD:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_KV_LAYOUT,
            "flash-attn's paged kernel addresses the pool as [page, page_size, heads, dim]",
            group_id=group.group_id,
            kv_layout=group.kv_layout.value,
        )
    if group.kv_cache_dtype.is_quantized:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_KV_CACHE_DTYPE,
            "flash-attn reads the KV pool at compute precision, not quantized storage",
            group_id=group.group_id,
            kv_cache_dtype=group.kv_cache_dtype.value,
        )
    if spec.head_dim_qk != spec.head_dim_vo:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_HEAD_DIM,
            "flash-attn v2/v3 write V at Q's head width; asymmetric QK/VO needs an MLA backend",
            group_id=group.group_id,
            head_dim_qk=spec.head_dim_qk,
            head_dim_vo=spec.head_dim_vo,
        )
    if spec.head_dim_qk > _MAX_HEAD_DIM or spec.head_dim_qk % 8:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_HEAD_DIM,
            f"flash-attn needs head_dim <= {_MAX_HEAD_DIM} and a multiple of 8",
            group_id=group.group_id,
            head_dim=spec.head_dim_qk,
        )
    if version == 2 and group.page_size % _FA2_PAGE_MULTIPLE:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_PAGE_SIZE,
            "FA2's paged kernel requires a page size that is a multiple of "
            f"{_FA2_PAGE_MULTIPLE} (FA3 lifts this)",
            group_id=group.group_id,
            page_size=group.page_size,
        )


class FlashAttentionMetadataBuilder(BaseAttentionMetadataBuilder):
    cudagraph_support = AttentionCudaGraphSupport.UNIFORM_QUERY
    reads_host_lens = False

    def __init__(
        self, group: AttentionGroupSpec, kv_cache: PagedKVCache, device: torch.device
    ) -> None:
        super().__init__(group, kv_cache, device)
        self._buffers: GraphBuffers | None = None
        self._graph_output: torch.Tensor | None = None

    def build(self, common: CommonAttentionMetadata) -> FlashAttentionMetadata:
        return FlashAttentionMetadata(common, None)

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
        page = self.group.page_size
        self._buffers = GraphBuffers.create(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_columns=(max_seq_len + page - 1) // page,
            device=self.device,
            max_query_len=max_query_len,
        )
        spec = self.spec
        self._graph_output = torch.empty(
            (max_batch_size * max_query_len, spec.num_qo_heads, spec.head_dim_vo),
            dtype=self.kv_cache.dtype,
            device=self.device,
        )

    def _bind(self, common: CommonAttentionMetadata, bs: int) -> FlashAttentionMetadata:
        if self._buffers is None:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_NOT_INITIALIZED,
                "init_graph_state has not run for this group",
                group_id=self.group.group_id,
            )
        staged = self._buffers.stage(common, bs)
        assert self._graph_output is not None
        return FlashAttentionMetadata(staged, self._graph_output[: staged.num_tokens])

    def build_for_capture(self, common, padded_batch_size: int) -> FlashAttentionMetadata:
        return self._bind(common, padded_batch_size)

    def build_for_replay(self, common, padded_batch_size: int) -> FlashAttentionMetadata:
        return self._bind(common, padded_batch_size)

    def reset_graph_state(self) -> None:
        super().reset_graph_state()
        self._buffers = None
        self._graph_output = None


class FlashAttentionBackend(BaseAttentionBackend):
    """Paged attention through flash-attn, dispatched FA3/FA2 by device capability.

    The two libraries are genuinely different APIs, not two versions of one signature:
    FA3 takes a varlen query plus ``page_table``/``cu_seqlens_q``, FA2 takes a flat varlen
    query plus ``block_table`` and derives per-request KV lengths from ``cu_seqlens_k``.
    Calling one with the other's keywords is a TypeError, which is exactly what this class
    existed to get wrong before -- so the dispatch is data (``_version``) and each branch is
    the call its own library documents.

    Neither library exposes an ``out`` parameter on its public entry point, so the
    caller-owned output contract is met with a copy. That copy is capture-safe: the kernel's
    internal allocation belongs to the graph's private pool, and the ``copy_`` is replayed.
    """

    def __init__(
        self, group: AttentionGroupSpec, kv_cache: PagedKVCache, device: torch.device
    ) -> None:
        super().__init__(group, kv_cache, device)
        major, minor = torch.cuda.get_device_capability(device.index or 0)
        if major == 9:
            self._version = 3
        elif major == 8:
            self._version = 2
        else:
            raise BackendCapabilityError(
                AttentionErrorCode.UNSUPPORTED_ARCH,
                "flash-attn serves sm80-sm89 (FA2) and sm90 (FA3)",
                compute_capability=f"{major}.{minor}",
            )
        _check_group_supported(group, self._version)
        self._attention: Callable[..., torch.Tensor] = _require_attention_fn(self._version)
        spec = group.spec
        window = spec.sliding_window if spec.mask is MaskKind.SLIDING else None
        # flash-attn counts the window as tokens strictly LEFT of the query, so an inclusive
        # window of W is (W - 1, 0). Off-by-one here is silent: it attends one token too few
        # and the output is merely slightly wrong.
        self._window: tuple[int, int] = (-1, -1) if window is None else (window - 1, 0)

    def build_metadata_builder(self) -> FlashAttentionMetadataBuilder:
        return FlashAttentionMetadataBuilder(self.group, self.kv_cache, self.device)

    def _cu_seqlens_k(self, common: CommonAttentionMetadata) -> torch.Tensor:
        """FA2's per-request KV bounds, derived on device from the staged sequence lengths.

        Device-derived, not a host read: this runs inside graph capture, where a `.cpu()` or
        a Python loop over the batch would either sync or bake the wrong lengths in.
        """
        seq_lens = common.seq_lens
        zero = torch.zeros(1, dtype=seq_lens.dtype, device=seq_lens.device)
        return torch.cat((zero, seq_lens.cumsum(0, dtype=torch.int32)))

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
        assert isinstance(metadata, FlashAttentionMetadata)
        spec = self.spec
        common = metadata.common
        if common.has_tree_mask:
            raise BackendCapabilityError(
                AttentionErrorCode.UNSUPPORTED_TREE_MASK,
                "flash-attn applies a dense causal mask; a speculative tree mask has no "
                "flash-attn encoding",
                group_id=self.group.group_id,
            )

        key = key.view(-1, spec.num_kv_heads, spec.head_dim_qk)
        value = value.view(-1, spec.num_kv_heads, spec.head_dim_vo)
        self.kv_cache.store_kv(key, value, common.slot_mapping, layer_id)

        q = query.view(-1, spec.num_qo_heads, spec.head_dim_qk)
        causal = spec.mask is not MaskKind.FULL
        if self._version == 3:
            result = self._attention(
                q=q,
                k_cache=self.kv_cache.key_cache(layer_id),
                v_cache=self.kv_cache.value_cache(layer_id),
                page_table=common.block_table,
                cu_seqlens_q=common.query_start_loc,
                max_seqlen_q=common.max_query_len,
                cache_seqlens=common.seq_lens,
                softmax_scale=spec.sm_scale,
                causal=causal,
                window_size=self._window,
                softcap=spec.logits_soft_cap,
            )
        else:
            result = self._attention(
                q=q,
                k=self.kv_cache.key_cache(layer_id),
                v=self.kv_cache.value_cache(layer_id),
                cu_seqlens_q=common.query_start_loc,
                cu_seqlens_k=self._cu_seqlens_k(common),
                max_seqlen_q=common.max_query_len,
                max_seqlen_k=common.max_seq_len,
                causal=causal,
                window_size=self._window,
                softcap=spec.logits_soft_cap,
                softmax_scale=spec.sm_scale,
                block_table=common.block_table,
            )

        out = output if output is not None else metadata.output
        if out is None:
            return result
        out.copy_(result)
        return out
