from __future__ import annotations

from dataclasses import dataclass

import torch

from ayaka.attention.base import BaseAttentionBackend
from ayaka.attention.errors import (
    AttentionErrorCode,
    AttentionMetadataError,
    BackendCapabilityError,
)
from ayaka.attention.metadata import (
    BaseAttentionMetadata,
    BaseAttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from ayaka.attention.ports import PagedKVCache
from ayaka.attention.spec import AttentionGroupSpec
from ayaka.kernel.triton.paged_attention import (
    compute_max_num_partitions,
    decode_paged_attention_op,
    paged_attention_op,
)
from ayaka.types import (
    AttentionCudaGraphSupport,
    AttentionType,
    ForwardMode,
    KVLayoutKind,
    MaskKind,
)

__all__ = [
    "TritonAttentionBackend",
    "TritonAttentionMetadata",
    "TritonAttentionMetadataBuilder",
]

_MAX_HEAD_DIM = 256
_DEFAULT_KV_PARTITION_SIZE = 512


def _check_group_supported(
    group: AttentionGroupSpec,
    kv_cache: PagedKVCache,
    device: torch.device,
) -> None:
    spec = group.spec
    if device.type != "cuda":
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_ARCH,
            "Triton paged attention requires a CUDA device",
            group_id=group.group_id,
            device=str(device),
        )
    if torch.device(kv_cache.device) != device:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_ARCH,
            "the KV cache must live on the backend CUDA device",
            group_id=group.group_id,
            backend_device=str(device),
            cache_device=str(kv_cache.device),
        )
    if group.attn_type not in (AttentionType.FULL, AttentionType.SWA):
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_ATTN_TYPE,
            f"Triton paged attention serves FULL and SWA groups, not {group.attn_type.value}",
            group_id=group.group_id,
        )
    if group.kv_layout is not KVLayoutKind.NHD:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_KV_LAYOUT,
            "Triton paged attention addresses [page, page_size, head, dim] NHD caches",
            group_id=group.group_id,
            kv_layout=group.kv_layout.value,
        )
    if spec.mask is MaskKind.FULL:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_NON_CAUSAL,
            "the Triton paged-attention kernels apply a causal mask",
            group_id=group.group_id,
        )
    if spec.head_dim_qk != spec.head_dim_vo:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_HEAD_DIM,
            "Triton paged attention requires equal QK and value head dimensions",
            group_id=group.group_id,
            head_dim_qk=spec.head_dim_qk,
            head_dim_vo=spec.head_dim_vo,
        )
    if not 16 <= spec.head_dim_qk <= _MAX_HEAD_DIM:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_HEAD_DIM,
            f"Triton paged attention needs 16 <= head_dim <= {_MAX_HEAD_DIM}",
            group_id=group.group_id,
            head_dim=spec.head_dim_qk,
        )
    if spec.logits_soft_cap != 0.0:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_LOGITS_SOFT_CAP,
            "the Triton paged-attention kernels do not apply logit soft-capping",
            group_id=group.group_id,
            logits_soft_cap=spec.logits_soft_cap,
        )
    if spec.has_sinks:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_SINKS,
            "the backend contract does not yet provide per-layer attention-sink tensors",
            group_id=group.group_id,
        )
    if kv_cache.dtype not in (torch.float16, torch.bfloat16):
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_DTYPE,
            "Triton paged attention computes in float16 or bfloat16",
            group_id=group.group_id,
            dtype=str(kv_cache.dtype),
        )
    if kv_cache.page_size != group.page_size or kv_cache.kv_layout is not group.kv_layout:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_KV_LAYOUT,
            "the KV cache geometry does not match the attention group",
            group_id=group.group_id,
            group_page_size=group.page_size,
            cache_page_size=kv_cache.page_size,
            group_layout=group.kv_layout.value,
            cache_layout=kv_cache.kv_layout.value,
        )
    if kv_cache.kv_cache_dtype is not group.kv_cache_dtype:
        raise BackendCapabilityError(
            AttentionErrorCode.UNSUPPORTED_KV_CACHE_DTYPE,
            "the KV cache storage dtype does not match the attention group",
            group_id=group.group_id,
            group_dtype=group.kv_cache_dtype.value,
            cache_dtype=kv_cache.kv_cache_dtype.value,
        )


@dataclass
class TritonAttentionMetadata(BaseAttentionMetadata):
    """CSR slot addressing and per-forward split-KV scratch."""

    __slots__ = (
        "common",
        "indptr",
        "indices",
        "query_to_request",
        "attn_logits",
        "attn_lse",
        "max_context_len_bucket",
        "decode",
        "output",
    )

    common: CommonAttentionMetadata
    indptr: torch.Tensor
    indices: torch.Tensor
    query_to_request: torch.Tensor
    attn_logits: torch.Tensor | None
    attn_lse: torch.Tensor | None
    max_context_len_bucket: int
    decode: bool
    output: torch.Tensor | None


class TritonAttentionMetadataBuilder(BaseAttentionMetadataBuilder[TritonAttentionMetadata]):
    """Build exact ragged slot lists for the Triton kernels.

    The output width of the CSR gather is ``sum(seq_lens)``. That value comes from the
    scheduler-maintained CPU mirror, while all page-table indexing stays on the CUDA device.
    Because eager construction allocates tensors with that dynamic width, this first backend
    integration deliberately makes no CUDA-graph promise.
    """

    cudagraph_support = AttentionCudaGraphSupport.NEVER
    reads_host_lens = True

    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
        *,
        kv_partition_size: int = _DEFAULT_KV_PARTITION_SIZE,
    ) -> None:
        super().__init__(group, kv_cache, device)
        if (
            isinstance(kv_partition_size, bool)
            or not isinstance(kv_partition_size, int)
            or kv_partition_size < 32
            or kv_partition_size % 32
        ):
            raise ValueError("kv_partition_size must be a positive multiple of 32")
        self.kv_partition_size = kv_partition_size

    def _validate_common(self, common: CommonAttentionMetadata) -> tuple[int, int]:
        group_id = self.group.group_id
        if common.num_reqs < 0 or common.num_tokens < 0:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "request and token counts must be nonnegative",
                group_id=group_id,
                num_reqs=common.num_reqs,
                num_tokens=common.num_tokens,
            )
        if (
            common.seq_lens_cpu.device.type != "cpu"
            or common.query_start_loc_cpu.device.type != "cpu"
        ):
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_DEVICE_MISMATCH,
                "the Triton metadata builder requires scheduler-owned CPU length mirrors",
                group_id=group_id,
            )
        if common.seq_lens_cpu.numel() < common.num_reqs:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "seq_lens_cpu is shorter than num_reqs",
                group_id=group_id,
            )
        if common.query_start_loc_cpu.numel() < common.num_reqs + 1:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "query_start_loc_cpu is shorter than num_reqs + 1",
                group_id=group_id,
            )

        device_tensors = {
            "seq_lens": (common.seq_lens, 1),
            "query_start_loc": (common.query_start_loc, 1),
            "block_table": (common.block_table, 2),
            "positions": (common.positions, 1),
        }
        for name, (tensor, rank) in device_tensors.items():
            if tensor.device != self.device:
                raise AttentionMetadataError(
                    AttentionErrorCode.METADATA_DEVICE_MISMATCH,
                    f"{name} must be on the backend device",
                    group_id=group_id,
                    expected=str(self.device),
                    actual=str(tensor.device),
                )
            if tensor.dtype != torch.int32 or tensor.dim() != rank:
                raise AttentionMetadataError(
                    AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                    f"{name} must be a rank-{rank} int32 tensor",
                    group_id=group_id,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )

        if common.seq_lens.numel() < common.num_reqs:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "seq_lens is shorter than num_reqs",
                group_id=group_id,
            )
        if common.query_start_loc.numel() < common.num_reqs + 1:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "query_start_loc is shorter than num_reqs + 1",
                group_id=group_id,
            )
        if common.block_table.shape[0] < common.num_reqs:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "block_table has fewer rows than num_reqs",
                group_id=group_id,
            )
        if common.positions.numel() < common.num_tokens:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "positions is shorter than num_tokens",
                group_id=group_id,
            )

        seq_lens_cpu = common.seq_lens_cpu[: common.num_reqs].to(torch.int64)
        if bool((seq_lens_cpu < 0).any().item()):
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "sequence lengths must be nonnegative",
                group_id=group_id,
            )
        total_kv_tokens = int(seq_lens_cpu.sum().item())
        actual_max_seq_len = int(seq_lens_cpu.max().item()) if common.num_reqs else 0
        required_columns = -(-actual_max_seq_len // self.group.page_size)
        if common.block_table.shape[1] < required_columns:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "block_table is too narrow for the staged sequence lengths",
                group_id=group_id,
                required_columns=required_columns,
                actual_columns=common.block_table.shape[1],
            )

        query_start_cpu = common.query_start_loc_cpu[: common.num_reqs + 1].to(torch.int64)
        if query_start_cpu.numel() and (
            int(query_start_cpu[0].item()) != 0
            or bool((query_start_cpu[1:] < query_start_cpu[:-1]).any().item())
            or int(query_start_cpu[-1].item()) != common.num_tokens
        ):
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "query_start_loc_cpu must be monotonic from zero to num_tokens",
                group_id=group_id,
            )
        return total_kv_tokens, actual_max_seq_len

    def _build_slot_csr(
        self,
        common: CommonAttentionMetadata,
        total_kv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_lens = common.seq_lens[: common.num_reqs]
        indptr = torch.zeros(common.num_reqs + 1, dtype=torch.int32, device=self.device)
        indptr[1:] = seq_lens.cumsum(0, dtype=torch.int32)
        if total_kv_tokens == 0:
            return indptr, torch.empty(0, dtype=torch.int32, device=self.device)

        request_ids = torch.repeat_interleave(
            torch.arange(common.num_reqs, dtype=torch.int64, device=self.device),
            seq_lens,
            output_size=total_kv_tokens,
        )
        logical_positions = torch.arange(total_kv_tokens, dtype=torch.int64, device=self.device)
        logical_positions -= indptr[request_ids].to(torch.int64)
        page_columns = torch.div(logical_positions, self.group.page_size, rounding_mode="floor")
        physical_pages = common.block_table[request_ids, page_columns].to(torch.int64)
        indices = physical_pages * self.group.page_size
        indices += logical_positions.remainder(self.group.page_size)
        return indptr, indices.to(torch.int32)

    def _build_query_to_request(self, common: CommonAttentionMetadata) -> torch.Tensor:
        query_lengths = (
            common.query_start_loc[1 : common.num_reqs + 1]
            - common.query_start_loc[: common.num_reqs]
        )
        return torch.repeat_interleave(
            torch.arange(common.num_reqs, dtype=torch.int32, device=self.device),
            query_lengths,
            output_size=common.num_tokens,
        )

    def build(self, common: CommonAttentionMetadata) -> TritonAttentionMetadata:
        total_kv_tokens, actual_max_seq_len = self._validate_common(common)
        indptr, indices = self._build_slot_csr(common, total_kv_tokens)
        query_to_request = self._build_query_to_request(common)
        decode = common.mode is ForwardMode.DECODE
        if decode and (common.max_query_len != 1 or common.num_tokens != common.num_reqs):
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "DECODE requires exactly one query token per request",
                group_id=self.group.group_id,
                num_reqs=common.num_reqs,
                num_tokens=common.num_tokens,
                max_query_len=common.max_query_len,
            )

        max_context_len_bucket = max(1, common.max_seq_len, actual_max_seq_len)
        attn_logits = None
        attn_lse = None
        if decode:
            num_partitions = compute_max_num_partitions(
                max_context_len_bucket,
                self.kv_partition_size,
                self.spec.sliding_window,
            )
            shape = (
                common.num_reqs,
                self.spec.num_qo_heads,
                num_partitions,
                self.spec.head_dim_vo,
            )
            attn_logits = torch.empty(shape, dtype=torch.float32, device=self.device)
            attn_lse = torch.empty(shape[:-1], dtype=torch.float32, device=self.device)

        return TritonAttentionMetadata(
            common=common,
            indptr=indptr,
            indices=indices,
            query_to_request=query_to_request,
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            max_context_len_bucket=max_context_len_bucket,
            decode=decode,
            output=None,
        )


class TritonAttentionBackend(BaseAttentionBackend):
    """Ayaka attention backend backed by the in-tree Triton paged kernels."""

    name = "triton"

    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
        *,
        kv_partition_size: int = _DEFAULT_KV_PARTITION_SIZE,
    ) -> None:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        super().__init__(group, kv_cache, resolved_device)
        _check_group_supported(group, kv_cache, resolved_device)
        if (
            isinstance(kv_partition_size, bool)
            or not isinstance(kv_partition_size, int)
            or kv_partition_size < 32
            or kv_partition_size % 32
        ):
            raise ValueError("kv_partition_size must be a positive multiple of 32")
        self.kv_partition_size = kv_partition_size

    def build_metadata_builder(self) -> TritonAttentionMetadataBuilder:
        return TritonAttentionMetadataBuilder(
            self.group,
            self.kv_cache,
            self.device,
            kv_partition_size=self.kv_partition_size,
        )

    def _scales(self, layer_id: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.group.kv_cache_dtype.is_quantized:
            return None, None
        key_scale = self.kv_cache.k_scale(layer_id)
        value_scale = self.kv_cache.v_scale(layer_id)
        if key_scale is None or value_scale is None:
            raise AttentionMetadataError(
                AttentionErrorCode.MISSING_KV_SCALE,
                "an FP8 KV cache requires device-resident key and value scales",
                group_id=self.group.group_id,
                layer_id=layer_id,
            )
        return key_scale, value_scale

    def _output(
        self,
        query: torch.Tensor,
        metadata: TritonAttentionMetadata,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        out = output if output is not None else metadata.output
        if out is None:
            return torch.empty_like(query)
        if out.shape != query.shape or out.dtype != query.dtype or out.device != query.device:
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "output must match query shape, dtype, and device",
                group_id=self.group.group_id,
                query_shape=tuple(query.shape),
                output_shape=tuple(out.shape),
                query_dtype=str(query.dtype),
                output_dtype=str(out.dtype),
                query_device=str(query.device),
                output_device=str(out.device),
            )
        return out

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
        if not isinstance(metadata, TritonAttentionMetadata):
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_TYPE_MISMATCH,
                "TritonAttentionBackend requires TritonAttentionMetadata",
                group_id=self.group.group_id,
                actual_type=type(metadata).__name__,
            )
        common = metadata.common
        if common.has_tree_mask:
            raise BackendCapabilityError(
                AttentionErrorCode.UNSUPPORTED_TREE_MASK,
                "Triton paged attention applies a dense causal mask",
                group_id=self.group.group_id,
            )

        spec = self.spec
        q = query.reshape(-1, spec.num_qo_heads, spec.head_dim_qk)
        k = key.reshape(-1, spec.num_kv_heads, spec.head_dim_qk)
        v = value.reshape(-1, spec.num_kv_heads, spec.head_dim_vo)
        out = self._output(q, metadata, output)
        key_cache = self.kv_cache.key_cache(layer_id)
        value_cache = self.kv_cache.value_cache(layer_id)
        key_scale, value_scale = self._scales(layer_id)

        # The just-produced K/V belongs to the same attention view. Queue its scatter first
        # on the current stream so the paged read observes those tokens.
        self.kv_cache.store_kv(k, v, common.slot_mapping, layer_id)

        sliding_window = spec.sliding_window or 0
        if metadata.decode:
            if metadata.attn_logits is None or metadata.attn_lse is None:
                raise AttentionMetadataError(
                    AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                    "decode metadata is missing split-KV workspace",
                    group_id=self.group.group_id,
                )
            decode_paged_attention_op(
                q,
                key_cache,
                value_cache,
                metadata.indptr,
                metadata.indices,
                common.positions[: common.num_tokens],
                metadata.attn_logits,
                metadata.attn_lse,
                metadata.max_context_len_bucket,
                spec.sm_scale,
                self.kv_partition_size,
                sliding_window,
                None,
                key_scale,
                value_scale,
                out,
            )
        else:
            paged_attention_op(
                q,
                key_cache,
                value_cache,
                metadata.indptr,
                metadata.indices,
                metadata.query_to_request,
                common.positions[: common.num_tokens],
                spec.sm_scale,
                sliding_window,
                None,
                key_scale,
                value_scale,
                out,
            )
        return out
