"""Resident paged execution after executor COW and before completion.

The runner takes addressing from immutable execution leases. It never allocates
pages, changes mappings, commits progress, or publishes tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ayaka.attention.base import BaseAttentionBackend
from ayaka.attention.metadata import (
    BaseAttentionMetadata,
    BaseAttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from ayaka.attention.spec import AttentionGroupSpec
from ayaka.kvcache.build import MHAStorage
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.storage.ports import KVStorage
from ayaka.memory.views import SequenceExecutionView
from ayaka.plan import GraphMode
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.request.schema import Request
from ayaka.runner.model_runner import ModelRunner
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sched.plan import PreparedStep
from ayaka.types import AttentionType, ForwardMode, KVCacheDtype, KVLayoutKind, MaskKind

if TYPE_CHECKING:
    from ayaka.serving.constraints import GrammarConstraints


class StoragePagedKVCache:
    """Zero-copy attention port over an existing unquantized NHD MHA slab."""

    def __init__(self, storage: KVStorage) -> None:
        if not isinstance(storage, MHAStorage):
            raise TypeError("paged attention requires MHAStorage")
        if storage.torch_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("this adapter requires unquantized floating-point KV")
        self.storage = storage
        self.device = storage.device
        self.dtype = storage.torch_dtype
        self.page_size = storage.page_size
        self.num_layers = storage.num_layers
        self.kv_layout = KVLayoutKind.NHD
        self.kv_cache_dtype = KVCacheDtype.AUTO

    def key_cache(self, layer_id: int) -> torch.Tensor:
        return self.storage.key_buffers[layer_id]

    def value_cache(self, layer_id: int) -> torch.Tensor:
        return self.storage.value_buffers[layer_id]

    def k_scale(self, layer_id: int) -> None:
        return None

    def v_scale(self, layer_id: int) -> None:
        return None

    def store_kv(self, key, value, slot_mapping, layer_id: int) -> None:
        # Slots are host-validated by the lease owner before enqueue.
        for cache, source in (
            (self.key_cache(layer_id), key),
            (self.value_cache(layer_id), value),
        ):
            cache.flatten(0, 1).index_copy_(0, slot_mapping.to(torch.long), source)


@dataclass
class ReferenceMetadata(BaseAttentionMetadata):
    common: CommonAttentionMetadata


class _ReferenceBuilder(BaseAttentionMetadataBuilder[ReferenceMetadata]):
    def build(self, common: CommonAttentionMetadata) -> ReferenceMetadata:
        return ReferenceMetadata(common)


class ReferencePagedAttention(BaseAttentionBackend):
    """Explicit eager SDPA reference with actual paged writes and cached reads."""

    def build_metadata_builder(self) -> _ReferenceBuilder:
        return _ReferenceBuilder(self.group, self.kv_cache, self.device)

    def forward(self, layer_id, query, key, value, metadata, *, output=None):
        if not isinstance(metadata, ReferenceMetadata):
            raise TypeError("reference attention needs ReferenceMetadata")
        common = metadata.common
        self.kv_cache.store_kv(key, value, common.slot_mapping, layer_id)
        pieces = []
        starts = common.query_start_loc_cpu.tolist()
        for row, seq_len in enumerate(common.seq_lens_cpu.tolist()):
            begin, end = starts[row : row + 2]
            offsets = torch.arange(seq_len, device=query.device)
            pages = common.block_table[row].index_select(0, offsets // self.group.page_size)
            slots = pages * self.group.page_size + offsets % self.group.page_size
            k = self.kv_cache.key_cache(layer_id).flatten(0, 1).index_select(0, slots)
            v = self.kv_cache.value_cache(layer_id).flatten(0, 1).index_select(0, slots)
            positions = common.positions[begin:end, None]
            mask = offsets[None, :] <= positions
            if self.spec.sliding_window is not None:
                mask &= offsets[None, :] > positions - self.spec.sliding_window
            out = F.scaled_dot_product_attention(
                query[begin:end].transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                scale=self.spec.sm_scale,
                enable_gqa=True,
            )
            pieces.append(out.squeeze(0).transpose(0, 1))
        result = torch.cat(pieces)
        if output is not None:
            output.copy_(result)
            return output
        return result


class PagedModelRunner(ModelRunner):
    """Forward every scheduled query against resident KV, then sample last rows.

    groups maps manager names to static attention geometry. Every model layer
    occurs exactly once. Backend selection is fixed at construction. Call through
    ResidentKVExecutor, which orders COW and owns the stream and completion fence.
    """

    def __init__(
        self,
        coordinator: SamplingCoordinator,
        model,
        kv: LogicalKVManager,
        groups: Mapping[str, AttentionGroupSpec],
        *,
        backend: str = "triton",
        force_reference: bool = False,
    ) -> None:
        super().__init__(coordinator, model, force_reference=force_reference)
        if backend not in ("triton", "reference", "flash_attention", "flashinfer"):
            raise ValueError("unknown paged attention backend")
        if set(groups) != set(kv.group_names):
            raise ValueError("attention groups must exactly match logical KV groups")
        parameter = next(model.parameters())
        self.device = parameter.device
        self.kv = kv
        self.backends: dict[str, BaseAttentionBackend] = {}
        self.builders: dict[str, BaseAttentionMetadataBuilder] = {}
        self.layers: dict[int, tuple[str, int]] = {}
        self._closed = False
        self._valid_token_mask: torch.Tensor | None = None
        self._request_source: Callable[[str], RequestLifecycle] | None = None
        self._request_ir: dict[str, Request] = {}
        self.constraints: GrammarConstraints | None = None
        for name in kv.group_names:
            group = groups[name]
            cache = StoragePagedKVCache(kv.storages[name].storage)
            if cache.device != self.device or cache.dtype != parameter.dtype:
                raise ValueError("model and KV must share device and dtype")
            if group.attn_type not in (AttentionType.FULL, AttentionType.SWA):
                raise ValueError("paged runner requires MHA/GQA attention groups")
            if group.spec.mask is MaskKind.FULL:
                raise ValueError("causal model runner requires a causal attention mask")
            if group.page_size != cache.page_size or group.kv_layout is not cache.kv_layout:
                raise ValueError("attention geometry disagrees with KV layout")
            if group.kv_cache_dtype is not cache.kv_cache_dtype:
                raise ValueError("attention dtype disagrees with KV dtype")
            if len(group.layer_ids) != cache.num_layers:
                raise ValueError("attention layer count disagrees with KV storage")
            if (
                cache.key_cache(0).shape[2:] != (group.spec.num_kv_heads, group.spec.head_dim_qk)
                or group.spec.head_dim_vo != group.spec.head_dim_qk
            ):
                raise ValueError("attention head geometry disagrees with KV storage")
            for local, layer in enumerate(group.layer_ids):
                if layer in self.layers:
                    raise ValueError("model layer belongs to multiple KV groups")
                self.layers[layer] = name, local
            if backend == "reference":
                if group.spec.logits_soft_cap or group.spec.has_sinks:
                    raise ValueError("reference backend does not support softcap or sinks")
                implementation = ReferencePagedAttention(group, cache, self.device)
            elif backend == "triton":
                from ayaka.attention.backend.triton_backend import TritonAttentionBackend

                implementation = TritonAttentionBackend(group, cache, self.device)
            elif backend == "flash_attention":
                from ayaka.attention.backend.flash_attention_backend import FlashAttentionBackend

                implementation = FlashAttentionBackend(group, cache, self.device)
            else:
                from ayaka.attention.backend.flashinfer_backend import FlashInferBackend

                implementation = FlashInferBackend(group, cache, self.device)
            self.backends[name] = implementation
            self.builders[name] = implementation.build_metadata_builder()
        if set(self.layers) != set(range(model.config.num_hidden_layers)):
            raise ValueError("every model layer must have one attention group")

    def set_valid_token_ids(self, token_ids) -> None:
        """Exclude padded or missing tokenizer IDs from every sampling row."""
        ids = tuple(token_ids)
        if not ids or any(
            type(t) is not int or t < 0 or t >= self._model.config.vocab_size for t in ids
        ):
            raise ValueError("invalid tokenizer vocabulary for model")
        mask = torch.zeros(self._model.config.vocab_size, dtype=torch.bool, device=self.device)
        mask[list(ids)] = True
        self._valid_token_mask = mask

    def set_request_source(self, source, *, constraints=None) -> None:
        """Bind the serialized lifecycle owner for committed grammar state."""
        self._request_source = source
        self.constraints = constraints

    def prepare_request(self, request) -> None:
        """Validate model features before scheduler admission."""
        if str(request.request_id) in self._request_ir:
            raise ValueError("duplicate runner request")
        if request.multimodal:
            for embedding in request.multimodal:
                if any(len(row) != self._model.config.hidden_size for row in embedding.rows):
                    raise ValueError("embedding width disagrees with model")
        if request.constraint is not None:
            if self.constraints is None or self._request_source is None:
                raise ValueError("structured decoding requires a bound grammar compiler")
            self.constraints.register(str(request.request_id), request.constraint)
        self._request_ir[str(request.request_id)] = request

    def forget_request(self, request_id: str) -> None:
        self._request_ir.pop(request_id, None)
        if self.constraints is not None:
            self.constraints.forget(request_id)

    def _apply_bans(self, prepared, logits, rows):
        if self._valid_token_mask is not None:
            logits.masked_fill_(~self._valid_token_mask, float("-inf"))
        super()._apply_bans(prepared, logits, rows)
        row = 0
        for scheduled in prepared.step.slices:
            if not scheduled.sample_last_query:
                continue
            request = self._request_ir.get(scheduled.request_id)
            if request is not None and request.constraint is not None:
                if self._request_source is None or self.constraints is None:
                    raise RuntimeError("constraint owners are not bound")
                lifecycle = self._request_source(scheduled.request_id)
                self.constraints.apply(
                    scheduled.request_id, lifecycle.output_token_ids, logits[row]
                )
            row += 1

    def _embeddings(self, step, tokens):
        if not any(
            self._request_ir.get(s.request_id) is not None
            and self._request_ir[s.request_id].multimodal
            for s in step.slices
        ):
            return None
        hidden = self._model.transformer.wte(tokens)
        offset = 0
        for scheduled in step.slices:
            request = self._request_ir.get(scheduled.request_id)
            for embedding in () if request is None else request.multimodal:
                begin = max(scheduled.query_start, embedding.start)
                end = min(scheduled.query_end, embedding.start + len(embedding.rows))
                if end > begin:
                    values = torch.tensor(
                        embedding.rows[begin - embedding.start : end - embedding.start],
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    start = offset + begin - scheduled.query_start
                    hidden[start : start + end - begin].copy_(values)
            offset += scheduled.query_count
        return hidden

    def _common(self, prepared: PreparedStep, name: str) -> CommonAttentionMetadata:
        step = prepared.step
        group = self.backends[name].group
        page_size = group.page_size
        lengths = [s.query_end for s in step.slices]
        starts = [0]
        tables = []
        slots = []
        width = (max(lengths) + page_size - 1) // page_size
        for scheduled, view in zip(step.slices, prepared.memory_view.sequences, strict=True):
            starts.append(starts[-1] + scheduled.query_count)
            if isinstance(view, SequenceExecutionView):
                table = list(view.block_table)
                slots.extend(slot.flat_slot for slot in view.write_slots)
                table += [0] * (width - len(table))
            else:
                selected = next(g for g in view.groups if g.group_name == name)
                if selected.layer_ids != group.layer_ids or selected.page_size != page_size:
                    raise ValueError("lease group geometry disagrees with attention group")
                table = [selected.padding_page] * width
                for logical, physical in zip(
                    selected.logical_blocks, selected.block_table, strict=True
                ):
                    table[logical] = physical
                slots.extend(slot.flat_slot for slot in selected.write_slots)
            tables.append(table)

        def host(values):
            return torch.tensor(values, dtype=torch.int32, pin_memory=self.device.type == "cuda")

        def device(values):
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        starts_cpu, lens_cpu = host(starts), host(lengths)
        mode = (
            ForwardMode.DECODE
            if step.is_pure_decode
            else ForwardMode.EXTEND
            if any(s.query_start for s in step.slices)
            else ForwardMode.PREFILL
        )
        return CommonAttentionMetadata(
            query_start_loc=starts_cpu.to(self.device, non_blocking=True),
            seq_lens=lens_cpu.to(self.device, non_blocking=True),
            computed_lens=device([s.query_start for s in step.slices]),
            block_table=device(tables),
            slot_mapping=device(slots),
            positions=device(step.positions),
            query_start_loc_cpu=starts_cpu,
            seq_lens_cpu=lens_cpu,
            num_reqs=len(step.slices),
            num_tokens=step.num_tokens,
            max_query_len=max(s.query_count for s in step.slices),
            max_seq_len=max(lengths),
            mode=mode,
        )

    def _forward_prepared(self, prepared: PreparedStep):
        if self._closed:
            raise RuntimeError("paged runner is closed")
        prepared.validate()
        # The executor validated allocator generations before marking IN_FLIGHT.
        step = prepared.step
        if step.graph.mode is not GraphMode.EAGER:
            raise ValueError("paged serving runner currently requires eager execution")
        if step.prompt_logprobs:
            raise ValueError("paged prompt logprobs require chunk-boundary hidden-state carry")
        metadata = {
            name: builder.build(self._common(prepared, name))
            for name, builder in self.builders.items()
        }

        def attention(layer, query, key, value):
            name, local = self.layers[layer]
            return self.backends[name].forward(local, query, key, value, metadata[name])

        tokens = torch.tensor(step.token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(step.positions, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            embeddings = self._embeddings(step, tokens)
            if embeddings is None:
                hidden = self._model.forward_hidden(tokens, positions, attention)
            else:
                hidden = self._model.forward_hidden(
                    tokens, positions, attention, inputs_embeds=embeddings
                )
        rows = torch.tensor(step.sampling_rows, dtype=torch.long, device=self.device)
        return hidden.index_select(0, rows), len(step.sampling_rows), [], [], []

    def close(self) -> None:
        for request_id in tuple(self._request_ir):
            self.forget_request(request_id)
        self._request_source = None
        self._closed = True
