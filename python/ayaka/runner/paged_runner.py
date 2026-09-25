"""Resident paged execution after executor COW and before completion.

The runner takes addressing from immutable execution leases. It never allocates
pages, changes mappings, commits progress, or publishes tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ayaka.attention.base import BaseAttentionBackend
from ayaka.attention.metadata import (
    BaseAttentionMetadata,
    BaseAttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from ayaka.attention.spec import AttentionGroupSpec
from ayaka.kernel.triton.cache.cache_ops import write_kv
from ayaka.kvcache.build import MHAStorage
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.storage.ports import KVStorage
from ayaka.memory.views import ExecutionMemoryView, SequenceExecutionView
from ayaka.plan import GraphMode
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.request.schema import Request
from ayaka.runner.buffers import RunnerBufferLease, RunnerBuffers, StagedGroup
from ayaka.runner.graph.padding import pad_decode_addressing
from ayaka.runner.graph.pool import CaptureProgram, DecodeGraphConfig, DecodeGraphPool
from ayaka.runner.model_runner import ModelRunner, _ForwardResult
from ayaka.runner.paged_inputs import PagedGroupBinding, validate_paged_inputs
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sched.plan import BatchMode, Phase, PreparedStep
from ayaka.types import (
    AttentionCudaGraphSupport,
    AttentionType,
    ForwardMode,
    KVCacheDtype,
    KVLayoutKind,
    MaskKind,
)
from ayaka.utils.math_utils import div_ceil

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
        # Slots are host-validated by the lease owner before enqueue. The
        # semantic op declares both mutations and remains capture-safe.
        write_kv(
            key,
            value,
            self.key_cache(layer_id),
            self.value_cache(layer_id),
            slot_mapping.to(torch.long),
            1.0,
            1.0,
        )


@dataclass
class ReferenceMetadata(BaseAttentionMetadata):
    common: CommonAttentionMetadata


class _ReferenceBuilder(BaseAttentionMetadataBuilder[ReferenceMetadata]):
    def build(self, common: CommonAttentionMetadata) -> ReferenceMetadata:
        return ReferenceMetadata(common)


class ReferencePagedAttention(BaseAttentionBackend):
    """Explicit eager SDPA reference with actual paged writes and cached reads."""

    supports_ragged_mixed = True

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


@dataclass(frozen=True, slots=True)
class PagedForwardTrace:
    """Host-only record of one model invocation, not a completion/commit proof.

    Query ranges follow packed request order. Failed model invocations count;
    dense oracle and startup profiling do not pass through this boundary.

    ``graph_bucket`` is set only when the model work was a real CUDA-graph
    replay for that padded bucket; ``graph_key`` carries the captured resource
    identity. Together they are the replay evidence a counter cannot provide.
    """

    step_id: int
    request_ids: tuple[str, ...]
    phases: tuple[Phase, ...]
    semantic_mode: BatchMode
    forward_mode: ForwardMode
    query_ranges: tuple[tuple[int, int], ...]
    sampling_rows: tuple[int, ...]
    num_tokens: int
    graph_bucket: int | None = None
    graph_key: str | None = None


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
        buffers: RunnerBuffers | None = None,
    ) -> None:
        super().__init__(coordinator, model, force_reference=force_reference)
        if backend not in ("triton", "reference", "flash_attention", "flashinfer"):
            raise ValueError("unknown paged attention backend")
        if set(groups) != set(kv.group_names):
            raise ValueError("attention groups must exactly match logical KV groups")
        if buffers is not None:
            configured = {name for name, _ in buffers.spec.group_columns}
            if configured != set(groups):
                raise ValueError("runner buffer groups must exactly match attention groups")
            # Buffer/budget coverage follows the group layout: each group's
            # block-table columns must fit its own page geometry, not a
            # single shared page size.
            max_sequence_tokens = getattr(kv.backend, "max_sequence_tokens", None)
            if isinstance(max_sequence_tokens, int):
                for name, group in groups.items():
                    columns = buffers.spec.columns_for(name)
                    required = (max_sequence_tokens + group.page_size - 1) // group.page_size
                    if columns < required:
                        raise ValueError(
                            f"runner buffer group {name!r} covers {columns} blocks but its "
                            f"page geometry needs {required}"
                        )
        self.buffers = buffers
        self.kv = kv
        self._backend_name = backend
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self._bindings: dict[str, PagedGroupBinding] = {}
        self.forward_calls = 0
        self.forward_tokens = 0
        self.forward_observer: Callable[[PagedForwardTrace], None] | None = None
        self.backends: dict[str, BaseAttentionBackend] = {}
        self.builders: dict[str, BaseAttentionMetadataBuilder] = {}
        self.layers: dict[int, tuple[str, int]] = {}
        self._closed = False
        self._valid_token_mask: torch.Tensor | None = None
        self._request_source: Callable[[str], RequestLifecycle] | None = None
        self._request_ir: dict[str, Request] = {}
        self.constraints: GrammarConstraints | None = None
        self._graph_pool: DecodeGraphPool | None = None
        self._graph_config: DecodeGraphConfig | None = None
        self._graph_capture_leases: dict[int, RunnerBufferLease] = {}
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
            self._bindings[name] = PagedGroupBinding(group, cache.storage.spec)
        self.supports_mixed_batches = all(
            implementation.supports_ragged_mixed for implementation in self.backends.values()
        )
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
        if request.sampling.prompt_logprobs is not None:
            raise ValueError("paged prompt logprobs require chunk-boundary hidden-state carry")
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

    # ── decode CUDA graphs ─────────────────────────────────────────────

    @property
    def graph_pool(self):
        """Planner seam consumed by ``KVStepRuntime`` (``plan``/``validate``)."""
        return None if self._graph_pool is None else self._graph_pool.planner

    def graph_stats(self):
        """Observability snapshot, or None while graphs are not enabled."""
        return None if self._graph_pool is None else self._graph_pool.planner.stats()

    def graph_unavailable_reason(self) -> str | None:
        """Static reason this runner cannot capture decode graphs, or ``None``.

        Declared, not discovered: ``enable_graph`` enforces exactly the reason
        reported here, so callers can gate on it without attempting a capture.
        Multi-group grouped KV stays fail-closed: every attention group would
        need its own captured metadata, and the graph identity and budget must
        cover the whole group tuple -- a combination with no capture/replay
        certification yet. A single coalesced group over every layer keeps the
        R08/R10 certification.
        """
        if self.device.type != "cuda":
            return "decode graphs require a CUDA device"
        head = getattr(self._model, "lm_head", None)
        if getattr(head, "requires_gather", False):
            return (
                "decode graphs do not support a vocabulary-parallel LM head, which "
                "all-gathers logits across the TP group"
            )
        if len(self.backends) != 1:
            return (
                "decode graphs are certified for exactly one attention group; grouped KV "
                "stays eager until a group-layout combination passes capture/replay"
            )
        builder = next(iter(self.builders.values()))
        support = builder.cudagraph_support
        if support < AttentionCudaGraphSupport.PURE_DECODE:
            return f"attention backend does not support pure-decode CUDA graphs: {support.name}"
        return None

    def enable_graph(self, config: DecodeGraphConfig, *, stream=None) -> int:
        """Warm up, capture and reconcile pure-decode graphs before admission.

        Every configured bucket is captured for every flight slot, because a
        capture binds the exact backing pointers of one slot. Any failure tears
        the partial pool down and re-raises: serving must not start with a
        half-captured graph path.
        """
        if self._closed:
            raise RuntimeError("paged runner is closed")
        if self._graph_pool is not None:
            raise RuntimeError("decode graphs are already enabled for this runner")
        if self.buffers is None:
            raise ValueError("decode graphs require the persistent runner buffer pool")
        reason = self.graph_unavailable_reason()
        if reason is not None:
            raise ValueError(reason)
        spec = self.buffers.spec
        ceiling = min(spec.max_num_seqs, spec.max_num_batched_tokens)
        buckets = tuple(bucket for bucket in config.buckets if bucket <= ceiling)
        if not buckets:
            raise ValueError("no decode graph bucket fits the runner buffer ceilings")
        resolved = replace(config, buckets=buckets)
        support = next(iter(self.builders.values())).cudagraph_support
        self._graph_config = resolved
        pool = DecodeGraphPool(
            device=self.device,
            backend=self._backend_name,
            dtype=str(self.dtype).removeprefix("torch."),
            buckets=buckets,
            support=support,
            slots=tuple(range(spec.max_inflight)),
            generation=lambda: self.kv.generation if self.kv is not None else None,
            barrier_fn=resolved.barrier_fn,
            eager_only_reason=self._graph_eager_only_reason,
            enable_gc_freeze=resolved.enable_gc_freeze,
        )
        try:
            for metadata_builder in self.builders.values():
                metadata_builder.init_graph_state(
                    max_batch_size=buckets[-1],
                    max_seq_len=resolved.max_model_len,
                    capture_sizes=list(buckets),
                    max_query_len=1,
                    padding_slot=resolved.padding_slot,
                )
            self._acquire_capture_leases()
            actual = pool.capture(
                CaptureProgram(prepare=self._stage_capture),
                stream=stream,
                persistent_bytes=sum(
                    int(getattr(metadata_builder, "graph_state_bytes", 0))
                    for metadata_builder in self.builders.values()
                ),
            )
            pool.reconcile(resolved.reserve_bytes)
        except BaseException:
            self._release_capture_leases()
            pool.cleanup()
            for metadata_builder in self.builders.values():
                metadata_builder.reset_graph_state()
            self._graph_config = None
            raise
        self._release_capture_leases()
        self._graph_pool = pool
        return actual

    def on_workspace_growth(self) -> None:
        """Executor hook: a step grew the shared workspace, drop captured graphs."""
        if self._graph_pool is not None:
            self._graph_pool.note_workspace_growth()

    def _acquire_capture_leases(self) -> None:
        assert self.buffers is not None
        try:
            for _ in range(self.buffers.spec.max_inflight):
                lease = self.buffers.acquire(step_id=0)
                self._graph_capture_leases[lease.index] = lease
        except BaseException:
            self._release_capture_leases()
            raise

    def _release_capture_leases(self) -> None:
        for lease in tuple(self._graph_capture_leases.values()):
            lease.release()
        self._graph_capture_leases.clear()

    def _graph_eager_only_reason(self, step) -> str | None:
        """Request-scoped features whose eager path must not be captured."""
        for scheduled in step.slices:
            request = self._request_ir.get(scheduled.request_id)
            if request is None:
                continue
            if request.constraint is not None:
                return "grammar_constraint"
            if request.multimodal:
                return "multimodal_embedding"
        return None

    def _model_forward(self, tokens, positions, metadata) -> Callable[[], Any]:
        """Pure device forward + LM head for the padded bucket, no host branch."""

        def attention(layer, query, key, value):
            name, local = self.layers[layer]
            return self.backends[name].forward(local, query, key, value, metadata[name])

        def forward():
            with torch.inference_mode():
                hidden = self._model.forward_hidden(tokens, positions, attention)
                return self._model.logits_from_hidden(hidden)

        return forward

    def _stage_capture(self, slot: int, bucket: int) -> Callable[[], Any]:
        """Stage capture-placeholder input into one slot, outside the graph."""
        lease = self._graph_capture_leases.get(slot)
        if lease is None:
            raise RuntimeError(f"no capture lease held for flight slot {slot}")
        buffers = lease.buffers
        tokens, positions = buffers.stage_tokens([0] * bucket, [0] * bucket)
        metadata = {}
        for name, metadata_builder in self.builders.items():
            common = self._capture_common(buffers, name, bucket)
            metadata[name] = metadata_builder.build_for_capture(common, bucket)
        return self._model_forward(tokens, positions, metadata)

    def _capture_common(self, buffers, name: str, bucket: int) -> CommonAttentionMetadata:
        """Padded placeholder metadata: every dummy lane is zero-length."""
        config = self._graph_config
        if config is None:
            raise RuntimeError("decode graph config is not bound")
        columns = buffers.spec.columns_for(name)
        staged = buffers.stage_group(
            name,
            starts=list(range(bucket + 1)),
            lengths=[0] * bucket,
            computed=[0] * bucket,
            tables=[[config.padding_page] * columns for _ in range(bucket)],
            width=columns,
            slots=[config.padding_slot] * bucket,
            positions=[0] * bucket,
        )
        return self._staged_graph_metadata(staged, bucket, max_seq_len=config.max_model_len)

    def _stage_graph_step(
        self, prepared: PreparedStep, lease: RunnerBufferLease, bucket: int
    ) -> Callable[[], Any]:
        """Stage one live decode step into its captured slot backing."""
        step = prepared.step
        buffers = lease.buffers
        tokens = list(step.token_ids)
        positions = list(step.positions)
        if len(tokens) != len(positions):
            raise ValueError("token ids and positions must have the same length")
        real = len(tokens)
        if real > bucket:
            raise ValueError(f"{real} real tokens exceed the captured bucket {bucket}")
        tokens += [0] * (bucket - real)
        positions += [0] * (bucket - real)
        tokens_dev, positions_dev = buffers.stage_tokens(tokens, positions)
        metadata = {}
        for name, metadata_builder in self.builders.items():
            common = self._common_graph(prepared, name, bucket)
            metadata[name] = metadata_builder.build_for_replay(common, bucket)
        return self._model_forward(tokens_dev, positions_dev, metadata)

    def _common_graph(
        self, prepared: PreparedStep, name: str, bucket: int
    ) -> CommonAttentionMetadata:
        """Padded attention metadata for one graph-replayed step.

        Real rows keep their packed order; the tail is dummy decode lanes with
        ``seq_lens == 0``, a flat-offset continuation of ``query_start_loc``,
        padding-page table rows and padding slots, so a padded lane neither
        reads nor writes a live page and can never be sampled.
        """
        step = prepared.step
        lease = prepared.buffers
        if lease is None:
            raise ValueError("decode graph staging requires a runner buffer lease")
        columns = lease.buffers.spec.columns_for(name)
        tables, slots, padding_page, padding_slot = self._real_addressing(prepared, name, columns)
        padded = pad_decode_addressing(
            real_starts=tuple(step.query_start_loc),
            real_lengths=tuple(scheduled.query_end for scheduled in step.slices),
            real_computed=tuple(scheduled.query_start for scheduled in step.slices),
            real_tables=tuple(tables),
            real_slots=tuple(slots),
            width=columns,
            bucket=bucket,
            padding_page=padding_page,
            padding_slot=padding_slot,
        )
        positions = list(step.positions)
        if len(positions) > bucket:
            raise ValueError("real positions exceed the captured bucket")
        positions += [0] * (bucket - len(positions))
        staged = lease.buffers.stage_group(
            name,
            starts=list(padded.starts),
            lengths=list(padded.lengths),
            computed=list(padded.computed),
            tables=list(padded.tables),
            width=columns,
            slots=list(padded.slots),
            positions=positions,
        )
        max_seq_len = max(padded.lengths) if padded.lengths else 0
        return self._staged_graph_metadata(staged, bucket, max_seq_len=max_seq_len)

    def _real_addressing(
        self, prepared: PreparedStep, name: str, columns: int
    ) -> tuple[list[list[int]], list[int], int, int]:
        """(tables, slots, padding_page, padding_slot) for the real packed rows."""
        memory = prepared.memory_view
        group = self.backends[name].group
        tables: list[list[int]] = []
        slots: list[int] = []
        padding_page = 0
        padding_slot = 0
        for view in memory.sequences:
            if isinstance(view, SequenceExecutionView):
                if not isinstance(memory, ExecutionMemoryView):
                    raise ValueError("homogeneous sequence view requires a homogeneous lease")
                table = list(view.block_table)
                slots.extend(slot.flat_slot for slot in view.write_slots)
                table += [0] * (columns - len(table))
                padding_page = memory.padding_page
                padding_slot = memory.padding_slot
            else:
                selected = next(g for g in view.groups if g.group_name == name)
                if selected.layer_ids != group.layer_ids or selected.page_size != group.page_size:
                    raise ValueError("lease group geometry disagrees with attention group")
                table = [selected.padding_page] * columns
                for logical, physical in zip(
                    selected.logical_blocks, selected.block_table, strict=True
                ):
                    table[logical] = physical
                slots.extend(slot.flat_slot for slot in selected.write_slots)
                padding_page = selected.padding_page
                padding_slot = selected.padding_slot
            tables.append(table)
        return tables, slots, padding_page, padding_slot

    @staticmethod
    def _staged_graph_metadata(
        staged: StagedGroup, bucket: int, *, max_seq_len: int
    ) -> CommonAttentionMetadata:
        return CommonAttentionMetadata(
            query_start_loc=staged.query_start_loc,
            seq_lens=staged.seq_lens,
            computed_lens=staged.computed_lens,
            block_table=staged.block_table,
            slot_mapping=staged.slot_mapping,
            positions=staged.positions,
            query_start_loc_cpu=staged.query_start_loc_cpu,
            seq_lens_cpu=staged.seq_lens_cpu,
            num_reqs=bucket,
            num_tokens=bucket,
            # The capture bakes one query per lane and the engine's token
            # ceiling; both are constants of the graph, never of this step.
            max_query_len=1,
            max_seq_len=max_seq_len,
            mode=ForwardMode.DECODE,
        )

    def _forward_tail(self, prepared: PreparedStep) -> _ForwardResult:
        pool = self._graph_pool
        if pool is not None and prepared.step.graph.mode is GraphMode.REPLAY:
            lease = prepared.buffers
            if lease is not None and pool.should_replay(prepared.step, lease.index):
                return self._graph_tail(prepared, lease)
        return super()._forward_tail(prepared)

    def _graph_tail(self, prepared: PreparedStep, lease: RunnerBufferLease) -> _ForwardResult:
        step = prepared.step
        bucket = step.graph.bucket
        pool = self._graph_pool
        if pool is None:
            raise RuntimeError("decode graph pool is not bound")
        rows = len(step.slices)

        def stage() -> Callable[[], Any]:
            return self._stage_graph_step(prepared, lease, bucket)

        logits = pool.execute(step, slot=lease.index, rows=rows, stage=stage)
        self._observe_forward(
            step,
            num_tokens=step.num_tokens,
            graph_bucket=bucket,
            graph_key=step.graph.graph_key,
        )
        return _ForwardResult(logits=logits, sampling_count=len(step.sampling_rows))

    def _observe_forward(
        self,
        step,
        *,
        num_tokens: int,
        graph_bucket: int | None = None,
        graph_key: str | None = None,
    ) -> None:
        if self.forward_observer is None:
            return
        self.forward_observer(
            PagedForwardTrace(
                step_id=step.step_id,
                request_ids=step.request_order,
                phases=tuple(s.phase for s in step.slices),
                semantic_mode=step.semantic_mode,
                forward_mode=step.forward_mode,
                query_ranges=tuple((s.query_start, s.query_end) for s in step.slices),
                sampling_rows=step.sampling_rows,
                num_tokens=num_tokens,
                graph_bucket=graph_bucket,
                graph_key=graph_key,
            )
        )

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
        starts = list(step.query_start_loc)
        tables = []
        slots = []
        width = div_ceil(max(lengths), page_size)
        for view in prepared.memory_view.sequences:
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

        lease = prepared.buffers
        if lease is not None:
            staged = lease.buffers.stage_group(
                name,
                starts=starts,
                lengths=lengths,
                computed=[s.query_start for s in step.slices],
                tables=tables,
                width=width,
                slots=slots,
                positions=step.positions,
            )
            return self._staged_metadata(step, staged, lengths)

        def host(values):
            return torch.tensor(values, dtype=torch.int32, pin_memory=self.device.type == "cuda")

        def device(values):
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        starts_cpu, lens_cpu = host(starts), host(lengths)
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
            mode=step.forward_mode,
        )

    @staticmethod
    def _staged_metadata(step, staged: StagedGroup, lengths: list[int]) -> CommonAttentionMetadata:
        """Metadata whose device tensors are persistent slot views, not copies.

        The ragged R06 contract is preserved exactly: real ``num_tokens``, packed
        request order and the live forward mode. Only the backing storage moved.
        """
        return CommonAttentionMetadata(
            query_start_loc=staged.query_start_loc,
            seq_lens=staged.seq_lens,
            computed_lens=staged.computed_lens,
            block_table=staged.block_table,
            slot_mapping=staged.slot_mapping,
            positions=staged.positions,
            query_start_loc_cpu=staged.query_start_loc_cpu,
            seq_lens_cpu=staged.seq_lens_cpu,
            num_reqs=len(step.slices),
            num_tokens=step.num_tokens,
            max_query_len=max(s.query_count for s in step.slices),
            max_seq_len=max(lengths),
            mode=step.forward_mode,
        )

    def _validate_prepared(self, prepared: PreparedStep) -> None:
        if self._closed:
            raise RuntimeError("paged runner is closed")
        prepared.validate()
        # The executor validated allocator generations before marking IN_FLIGHT.
        compute = prepared.execution.compute
        if compute.dtype.torch_dtype != self.dtype or compute.kv_dtype.torch_dtype != self.dtype:
            raise ValueError("execution dtype disagrees with model/KV binding")
        if compute.layer_range != (0, self._model.config.num_hidden_layers):
            raise ValueError("execution layer range disagrees with model binding")
        step = prepared.step
        if step.graph.mode not in (GraphMode.EAGER, GraphMode.REPLAY):
            raise ValueError("paged serving runner supports eager or graph replay execution")
        if step.graph.mode is GraphMode.REPLAY:
            if self._graph_pool is None:
                raise ValueError("graph replay step requires an enabled decode graph pool")
            if self.buffers is None:
                raise ValueError("graph replay step requires the persistent runner buffer pool")
            limits = (self.buffers.spec.max_num_seqs, self.buffers.spec.max_num_batched_tokens)
            if not 1 <= step.graph.bucket <= min(limits):
                raise ValueError("graph bucket exceeds the runner buffer ceilings")
        if self.buffers is not None:
            lease = prepared.buffers
            if lease is None:
                raise ValueError("paged runner requires a runner buffer lease for this step")
            if lease.generation != self.buffers.generation:
                raise ValueError("runner buffer lease belongs to a replaced pool generation")
        elif prepared.buffers is not None:
            raise ValueError("prepared step carries a lease but the runner has no pool binding")
        if step.prompt_logprobs:
            raise ValueError("paged prompt logprobs require chunk-boundary hidden-state carry")
        validate_paged_inputs(prepared, self._bindings, self._model.config.max_position_embeddings)

    def _forward_prepared(self, prepared: PreparedStep):
        step = prepared.step
        metadata = {
            name: builder.build(self._common(prepared, name))
            for name, builder in self.builders.items()
        }

        def attention(layer, query, key, value):
            name, local = self.layers[layer]
            return self.backends[name].forward(local, query, key, value, metadata[name])

        lease = prepared.buffers
        if lease is not None:
            tokens, positions = lease.buffers.stage_tokens(step.token_ids, step.positions)
        else:
            tokens = torch.tensor(step.token_ids, dtype=torch.long, device=self.device)
            positions = torch.tensor(step.positions, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            embeddings = self._embeddings(step, tokens)
            self._observe_forward(step, num_tokens=tokens.numel())
            self.forward_calls += 1
            self.forward_tokens += tokens.numel()
            if embeddings is None:
                hidden = self._model.forward_hidden(tokens, positions, attention)
            else:
                hidden = self._model.forward_hidden(
                    tokens, positions, attention, inputs_embeds=embeddings
                )
        if lease is not None:
            rows = lease.buffers.stage_sampling_rows(step.sampling_rows)
        else:
            rows = torch.tensor(step.sampling_rows, dtype=torch.long, device=self.device)
        return hidden.index_select(0, rows), len(step.sampling_rows), [], [], []

    def close(self) -> None:
        for request_id in tuple(self._request_ir):
            self.forget_request(request_id)
        self._request_source = None
        self.forward_observer = None
        # Graphs bind KV views and flight backing; the executor only reaches
        # close() after every ticket settled, so nothing can still be replaying.
        if self._graph_pool is not None:
            self._release_capture_leases()
            self._graph_pool.cleanup()
            self._graph_pool = None
        self._graph_config = None
        # Attention backends hold views into the KV slab; dropping them here is
        # what lets a runtime resize release the old storage deterministically.
        self.backends.clear()
        self.builders.clear()
        self.kv = None
        self._closed = True
