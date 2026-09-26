from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from ayaka.attention.base import BaseAttentionBackend
from ayaka.attention.errors import (
    AttentionErrorCode,
    AttentionMetadataError,
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
from ayaka.kernel.triton.paged_attention import (
    _prepare_kv_scales,
    _prepare_output,
    _validate_index_tensor,
    _validate_query_and_caches,
    compute_max_num_partitions,
    decode_paged_attention_op,
    paged_attention_op,
    validate_decode_workspace,
)
from ayaka.types import (
    AttentionCudaGraphSupport,
    AttentionType,
    ForwardMode,
    KVLayoutKind,
    MaskKind,
)
from ayaka.utils.math_utils import div_ceil

__all__ = [
    "DEFAULT_KV_PARTITION_SIZE",
    "TritonAttentionBackend",
    "TritonAttentionMetadata",
    "TritonAttentionMetadataBuilder",
]

_MAX_HEAD_DIM = 256
DEFAULT_KV_PARTITION_SIZE = 512


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

    The eager ``build`` path gathers a CSR list whose width is ``sum(seq_lens)``; that
    value comes from the scheduler-maintained CPU mirror, while all page-table indexing
    stays on the CUDA device. Eager construction allocates tensors with that dynamic width,
    so it is not capture-safe by itself.

    Graph state (R08) serves ``PURE_DECODE`` only. The capture path lays each request's
    flat KV slots out in a fixed-width row (``indptr[r] = r * max_slots``), so every
    address the decode kernels see is static and every staging step is a copy into
    persistent buffers — no shape-dependent allocation and no host read on replay. Padded
    rows keep ``seq_lens == 0`` and their whole row points at the reserved padding page;
    the decode kernels bound their KV loop by ``min(indptr_span, position + 1)``, so a
    dummy decode lane reads and writes nothing live.
    """

    cudagraph_support = AttentionCudaGraphSupport.PURE_DECODE
    reads_host_lens = True

    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
        *,
        kv_partition_size: int = DEFAULT_KV_PARTITION_SIZE,
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
        self._graph_max_batch_size = 0
        self._graph_max_seq_len = 0
        self._graph_max_columns = 0
        self._graph_max_slots = 0
        self._graph_padding_slot = 0
        self._graph_indptr: torch.Tensor | None = None
        self._graph_query_to_request: torch.Tensor | None = None
        self._graph_attn_logits: torch.Tensor | None = None
        self._graph_attn_lse: torch.Tensor | None = None
        self._graph_output: torch.Tensor | None = None
        self._graph_token_positions: torch.Tensor | None = None
        self._graph_page_columns: torch.Tensor | None = None
        self._graph_page_offsets: torch.Tensor | None = None
        self._graph_request_offsets: torch.Tensor | None = None
        self._graph_slots_scratch: torch.Tensor | None = None
        self._graph_invalid_scratch: torch.Tensor | None = None
        self._graph_bytes = 0

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
            "computed_lens": (common.computed_lens, 1),
            "query_start_loc": (common.query_start_loc, 1),
            "block_table": (common.block_table, 2),
            "slot_mapping": (common.slot_mapping, 1),
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
            if (
                tensor.dtype != torch.int32
                or tensor.dim() != rank
                or (rank == 1 and not tensor.is_contiguous())
            ):
                raise AttentionMetadataError(
                    AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                    f"{name} must be a rank-{rank} int32 tensor",
                    group_id=group_id,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )

        if min(common.seq_lens.numel(), common.computed_lens.numel()) < common.num_reqs:
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
        if min(common.positions.numel(), common.slot_mapping.numel()) < common.num_tokens:
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
        required_columns = div_ceil(actual_max_seq_len, self.group.page_size)
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
        query_lens = query_start_cpu[1:] - query_start_cpu[:-1]
        # Captured dummy decode rows have one query and zero context. They
        # produce zero output and address the reserved padding page, never live KV.
        padded_decode = (seq_lens_cpu == 0) & (common.mode is ForwardMode.DECODE)
        if bool(((query_lens > seq_lens_cpu) & ~padded_decode).any()) or (
            query_lens.numel() and int(query_lens.max()) > common.max_query_len
        ):
            raise AttentionMetadataError(
                AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                "query lengths exceed the sequence lengths or query shape hint",
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

    # ----- cuda graph (R08, PURE_DECODE) ---------------------------------------------------

    @staticmethod
    def graph_state_layout(
        *,
        max_batch_size: int,
        max_seq_len: int,
        page_size: int,
        num_qo_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        kv_partition_size: int,
        output_dtype: torch.dtype,
        sliding_window: int | None = None,
    ) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        """Shape and dtype of every persistent decode-graph buffer, in one place.

        ``init_graph_state`` allocates exactly this layout and
        :meth:`estimate_graph_state_bytes` prices it without a device, so the
        reserve admitted before capture cannot drift from the real allocation.
        """
        columns = div_ceil(max_seq_len, page_size)
        max_slots = columns * page_size
        partitions = compute_max_num_partitions(
            max(1, max_seq_len),
            kv_partition_size,
            sliding_window,
        )
        return (
            ("indptr", (max_batch_size + 1,), torch.int32),
            ("query_to_request", (max_batch_size,), torch.int32),
            (
                "attn_logits",
                (max_batch_size, num_qo_heads, partitions, head_dim_vo),
                torch.float32,
            ),
            ("attn_lse", (max_batch_size, num_qo_heads, partitions), torch.float32),
            ("output", (max_batch_size, num_qo_heads, head_dim_qk), output_dtype),
            ("token_positions", (max_slots,), torch.int32),
            ("page_columns", (max_slots,), torch.int64),
            ("page_offsets", (max_slots,), torch.int32),
            ("request_offsets", (max_batch_size,), torch.int32),
            ("slots_scratch", (max_batch_size, max_slots), torch.int32),
            ("invalid_scratch", (max_batch_size, max_slots), torch.bool),
        )

    @classmethod
    def estimate_graph_state_bytes(
        cls,
        *,
        max_batch_size: int,
        max_seq_len: int,
        page_size: int,
        num_qo_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        kv_partition_size: int,
        output_dtype: torch.dtype,
        sliding_window: int | None = None,
    ) -> int:
        """Bytes ``init_graph_state`` will allocate, with no device needed."""
        return sum(
            math.prod(shape) * dtype.itemsize
            for _, shape, dtype in cls.graph_state_layout(
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                page_size=page_size,
                num_qo_heads=num_qo_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                kv_partition_size=kv_partition_size,
                output_dtype=output_dtype,
                sliding_window=sliding_window,
            )
        )

    def init_graph_state(
        self,
        *,
        max_batch_size: int,
        max_seq_len: int,
        capture_sizes: list[int],
        max_query_len: int = 1,
        padding_slot: int | None = None,
    ) -> None:
        """Allocate every persistent buffer a decode capture will bind.

        ``padding_slot`` is the reserved padding page's flat slot; padded rows point
        there so a dummy decode lane never touches a live page. ``max_seq_len`` is the
        engine's token ceiling, not the longest sequence seen.
        """
        super().init_graph_state(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            capture_sizes=capture_sizes,
            max_query_len=max_query_len,
        )
        if max_query_len != 1:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_MODE_UNSUPPORTED,
                "Triton captures pure decode only; a uniform multi-token query batch "
                "needs the ragged prefill path",
                max_query_len=max_query_len,
            )
        page = self.group.page_size
        columns = div_ceil(max_seq_len, page)
        max_slots = columns * page
        spec = self.spec
        device = self.device
        self._graph_max_batch_size = max_batch_size
        self._graph_max_seq_len = max(1, max_seq_len)
        self._graph_max_columns = columns
        self._graph_max_slots = max_slots
        self._graph_padding_slot = 0 if padding_slot is None else int(padding_slot)
        if self._graph_padding_slot < 0:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_STAGING_WIDTH_EXCEEDED,
                "the padding slot must be a non-negative flat slot address",
                padding_slot=self._graph_padding_slot,
            )
        buffers = {
            name: torch.zeros(shape, dtype=dtype, device=device)
            for name, shape, dtype in self.graph_state_layout(
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                page_size=page,
                num_qo_heads=spec.num_qo_heads,
                head_dim_qk=spec.head_dim_qk,
                head_dim_vo=spec.head_dim_vo,
                kv_partition_size=self.kv_partition_size,
                output_dtype=self.kv_cache.dtype,
                sliding_window=spec.sliding_window,
            )
        }
        buffers["token_positions"] = torch.arange(max_slots, dtype=torch.int32, device=device)
        buffers["page_columns"] = (buffers["token_positions"] // page).to(torch.int64)
        buffers["page_offsets"] = buffers["token_positions"] % page
        buffers["request_offsets"] = torch.arange(
            1, max_batch_size + 1, dtype=torch.int32, device=device
        )
        buffers["query_to_request"] = torch.arange(max_batch_size, dtype=torch.int32, device=device)
        self._graph_indptr = buffers["indptr"]
        self._graph_indptr[1:].copy_(buffers["request_offsets"])
        self._graph_indptr[1:].mul_(max_slots)
        self._graph_query_to_request = buffers["query_to_request"]
        self._graph_attn_logits = buffers["attn_logits"]
        self._graph_attn_lse = buffers["attn_lse"]
        self._graph_output = buffers["output"]
        self._graph_token_positions = buffers["token_positions"]
        self._graph_page_columns = buffers["page_columns"]
        self._graph_page_offsets = buffers["page_offsets"]
        self._graph_request_offsets = buffers["request_offsets"]
        self._graph_slots_scratch = buffers["slots_scratch"]
        self._graph_invalid_scratch = buffers["invalid_scratch"]
        self._graph_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in buffers.values()
        )

    @property
    def graph_state_bytes(self) -> int:
        """Persistent backend buffers a decode capture depends on, in bytes."""
        return self._graph_bytes

    def graph_binding_digest(self) -> str:
        """Base digest plus the split-KV/launch topology this backend pinned.

        The graph state bakes ``compute_max_num_partitions`` at capture time, so
        a Triton version, partition size, page size, window or head geometry
        change must produce a new binding and force a recapture.
        """
        import triton

        parts = (
            super().graph_binding_digest(),
            f"triton={triton.__version__}",
            f"partition={self.kv_partition_size}",
            f"page={self.group.page_size}",
            f"window={self.spec.sliding_window}",
            f"heads={self.spec.num_qo_heads}/{self.spec.num_kv_heads}",
            f"dims={self.spec.head_dim_qk}/{self.spec.head_dim_vo}",
        )
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]

    def _stage_graph_decode(
        self, common: CommonAttentionMetadata, padded_batch_size: int
    ) -> TritonAttentionMetadata:
        """Copy one padded batch's addressing into the persistent graph buffers.

        Runs OUTSIDE the captured region on the engine stream, immediately before
        ``graph.replay()``. Every op is a static-shape device write into memory the
        capture already bound; the captured kernels read the buffers, so table values,
        sequence lengths and prefix hits may change without invalidating the graph.

        ``indptr`` and ``query_to_request`` are bucket constants written once by
        ``init_graph_state``; only the slot list depends on this step's page table
        and lengths. The slot list aliases ``slots_scratch`` (no second buffer, no
        copy), and an invalid lane is masked to the reserved padding slot in place.
        """
        indptr = self._graph_indptr
        query_to_request = self._graph_query_to_request
        slots = self._graph_slots_scratch
        invalid_scratch = self._graph_invalid_scratch
        logits_scratch = self._graph_attn_logits
        lse_scratch = self._graph_attn_lse
        output = self._graph_output
        if (
            indptr is None
            or query_to_request is None
            or slots is None
            or invalid_scratch is None
            or logits_scratch is None
            or lse_scratch is None
            or output is None
            or self._graph_token_positions is None
            or self._graph_page_columns is None
            or self._graph_page_offsets is None
        ):
            raise GraphStateError(
                AttentionErrorCode.GRAPH_NOT_INITIALIZED,
                "init_graph_state has not run for this group",
                group_id=self.group.group_id,
            )
        bs = padded_batch_size
        if bs < 1 or bs > self._graph_max_batch_size:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_BATCH_SIZE_UNSUPPORTED,
                "padded batch size exceeds the captured maximum",
                padded_batch_size=bs,
                max_batch_size=self._graph_max_batch_size,
            )
        table = common.block_table[:bs]
        if table.shape[1] < self._graph_max_columns:
            raise GraphStateError(
                AttentionErrorCode.GRAPH_STAGING_WIDTH_EXCEEDED,
                "the staged block table is narrower than the captured ceiling",
                width=int(table.shape[1]),
                max_columns=self._graph_max_columns,
            )
        max_slots = self._graph_max_slots
        seq_lens = common.seq_lens[:bs]
        indices = slots[:bs, :max_slots]
        torch.index_select(table, 1, self._graph_page_columns, out=indices)
        indices.mul_(self.group.page_size)
        indices.add_(self._graph_page_offsets)
        invalid = invalid_scratch[:bs, :max_slots]
        torch.ge(self._graph_token_positions.unsqueeze(0), seq_lens.unsqueeze(1), out=invalid)
        indices.masked_fill_(invalid, self._graph_padding_slot)
        return TritonAttentionMetadata(
            common=common,
            indptr=indptr[: bs + 1],
            indices=indices.reshape(-1),
            query_to_request=query_to_request[:bs],
            attn_logits=logits_scratch[:bs],
            attn_lse=lse_scratch[:bs],
            max_context_len_bucket=self._graph_max_seq_len,
            decode=True,
            output=output[:bs],
        )

    def build_for_capture(
        self, common: CommonAttentionMetadata, padded_batch_size: int
    ) -> TritonAttentionMetadata:
        # Validation runs once per capture, outside the recorded region: it performs
        # host reads that must never run on the replay path.
        self._validate_common(common)
        return self._stage_graph_decode(common, padded_batch_size)

    def build_for_replay(
        self, common: CommonAttentionMetadata, padded_batch_size: int
    ) -> TritonAttentionMetadata:
        # Copy-only staging: the planner and paged-input validation already rejected
        # stale leases, so no host read or allocation is needed here.
        return self._stage_graph_decode(common, padded_batch_size)

    def reset_graph_state(self) -> None:
        super().reset_graph_state()
        self._graph_max_batch_size = 0
        self._graph_max_seq_len = 0
        self._graph_max_columns = 0
        self._graph_max_slots = 0
        self._graph_indptr = None
        self._graph_query_to_request = None
        self._graph_attn_logits = None
        self._graph_attn_lse = None
        self._graph_output = None
        self._graph_token_positions = None
        self._graph_page_columns = None
        self._graph_page_offsets = None
        self._graph_request_offsets = None
        self._graph_slots_scratch = None
        self._graph_invalid_scratch = None
        self._graph_bytes = 0

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
    supports_ragged_mixed = True

    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
        *,
        kv_partition_size: int = DEFAULT_KV_PARTITION_SIZE,
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
        # Reuse the kernel's metadata-only validators before the mutating store.
        # Address values and page generations were validated by the lease owner;
        # reading device values here would introduce a synchronization/capture break.
        _validate_query_and_caches(q, key_cache, value_cache)
        _prepare_kv_scales(key_cache, key_scale, value_scale, q)
        _prepare_output(q, out)
        if q.shape[0] != common.num_tokens or k.shape[0] != q.shape[0] or v.shape != k.shape:
            raise ValueError("Q/K/V token counts must match the attention snapshot")
        if any(t.dtype != q.dtype or t.device != q.device for t in (k, v)):
            raise ValueError("Q/K/V must share dtype and device")
        _validate_index_tensor(
            common.slot_mapping, "slot_mapping", q.device, numel=common.num_tokens
        )
        _validate_index_tensor(common.positions, "positions", q.device, numel=common.num_tokens)
        _validate_index_tensor(metadata.indptr, "indptr", q.device, numel=common.num_reqs + 1)
        _validate_index_tensor(metadata.indices, "indices", q.device)
        if metadata.decode != (common.mode is ForwardMode.DECODE):
            raise ValueError("attention phase disagrees with the snapshot")
        if metadata.decode:
            if metadata.attn_logits is None or metadata.attn_lse is None:
                raise AttentionMetadataError(
                    AttentionErrorCode.METADATA_SHAPE_MISMATCH,
                    "decode metadata is missing split-KV workspace",
                    group_id=self.group.group_id,
                )
            if common.num_tokens != common.num_reqs or common.max_query_len != 1:
                raise ValueError("DECODE requires one query per request")
            if metadata.max_context_len_bucket < common.max_seq_len:
                raise ValueError("decode context bucket is smaller than the snapshot")
            validate_decode_workspace(
                q,
                metadata.attn_logits,
                metadata.attn_lse,
                metadata.max_context_len_bucket,
                self.kv_partition_size,
                spec.sliding_window,
            )
        else:
            _validate_index_tensor(
                metadata.query_to_request, "query_to_request", q.device, numel=common.num_tokens
            )

        # Queue the scatter and paged read on the same stream. Never retry a failed launch.
        self.kv_cache.store_kv(k, v, common.slot_mapping, layer_id)
        sliding_window = spec.sliding_window or 0
        if metadata.decode:
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
