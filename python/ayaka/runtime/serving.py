"""Composition of Ayaka's native model, resident KV, tokenizer and HTTP service.

KV capacity is an explicit slab budget, not a claim about total free VRAM.
Shutdown releases slabs only after the engine proves every flight settled.
The capacity may be changed while the server runs: :meth:`ServingRuntime.resize`
validates the byte cost against measured device headroom *before* anything is
freed, so a rejected resize keeps the old caches serving.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ayaka.attention.spec import AttentionGroupSpec, AttentionSpec
from ayaka.configs.cache import TieredCacheConfig
from ayaka.configs.memory import MemoryConfig
from ayaka.configs.scheduler import PreemptionMode, SchedulerCapabilities, SchedulerConfig
from ayaka.configs.serving import ServingConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.materialize import KVStorageLease
from ayaka.kvcache.resize import (
    LEDGER_OVERHEAD_BYTES,
    CacheRebuildRejected,
    CacheResizeFatal,
    CacheResizeMode,
    ResizePlan,
    ResizeRejectionReason,
    device_headroom,
    validate_resize,
)
from ayaka.kvcache.status import CacheStatus, build_cache_status
from ayaka.kvcache.storage.geometry import MHAStorageSpec
from ayaka.memory.caching import CachingAllocator
from ayaka.memory.capacity import (
    CapacityFreeze,
    CapacitySnapshot,
)
from ayaka.memory.host.host_policy import HostMemoryPolicy, MirrorDecision
from ayaka.memory.ledger import MemoryLedger
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.tiering import TieringConfig
from ayaka.memory.workspace import WorkspaceManager
from ayaka.obs import runtime_event
from ayaka.plan import ComputePlan, ExecutionPlan, GraphMode, MemoryPlan
from ayaka.prefix.identity import build_prefix_context
from ayaka.runner.buffers import RunnerBuffers, RunnerBufferSpec
from ayaka.runner.paged_runner import PagedModelRunner
from ayaka.runtime.output import OutputProcessor
from ayaka.runtime.resident import ResidentKVEngine
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.serving.constraints import GrammarConstraints
from ayaka.serving.http import create_app
from ayaka.serving.prepare import RequestProcessor
from ayaka.serving.service import ServingService
from ayaka.tokenizers.service import TokenizerService
from ayaka.types import AttentionType, DType, MemoryTier
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.torch_memory import empty_cache
from ayaka.utils.validation import require_int
from ayaka.worker.local import LocalWorker
from ayaka.worker.resources import WorkerResourcePlan, WorkerResources

#: Fixed bookkeeping bytes the ledger reserves on top of the aligned slab.
_LEDGER_OVERHEAD_BYTES = LEDGER_OVERHEAD_BYTES

#: Default reserve kept free across a resize; covers allocator fragmentation
#: and activations that are not part of the KV slab.
DEFAULT_RESIZE_SAFETY_BYTES = 256 << 20


def _validate_serving_tiering(tiering: TieredCacheConfig | None) -> None:
    """R12B scope guard for the serving boundary's host tier policy.

    The homogeneous full-retention host tier is the only certified
    combination: NVMe and swap tiers belong to a later phase with their own
    contracts, and a host tier without capacity is a configuration error.
    """

    if tiering is None:
        return
    if tiering.nvme_bytes or tiering.swap_bytes:
        raise ValueError(
            "R12B tiering certifies the host tier only; nvme_bytes and "
            "swap_bytes need a separate phase"
        )
    if tiering.host_bytes < 1:
        raise ValueError("tiering.host_bytes must be positive when tiering is enabled")


@dataclass(slots=True)
class _KVTail:
    """Every owner that is replaced together when the cache is rebuilt."""

    resources: WorkerResources
    ledger: MemoryLedger
    storage: KVStorageLease
    manager: RuntimeMemoryManager
    kv: LogicalKVManager
    workspace: WorkspaceManager
    workspace_allocator: CachingAllocator
    capacity: CapacitySnapshot
    buffers: RunnerBuffers
    runner: PagedModelRunner
    engine: ResidentKVEngine


class ServingRuntime:
    """Single-device, native serving; caller owns the supplied model.

    KV capacity is an explicit slab budget, not a claim about total free VRAM.
    Shutdown releases slabs only after the engine proves every flight settled.
    Runtime resize is serialized on the engine thread and is refused while any
    request is active, so the old cache stays consistent until the swap.
    """

    def __init__(
        self,
        model,
        tokenizer_path: str | Path,
        *,
        config: ServingConfig | None = None,
        model_revision="local",
        weights_revision="local",
        pages=1024,
        page_size=16,
        max_requests=32,
        max_pending_requests: int | None = None,
        max_inflight: int = 1,
        batch_tokens=256,
        prefill_chunk=128,
        max_model_len: int | None = None,
        tokenizer_workers=2,
        backend="triton",
        resize_safety_bytes: int = DEFAULT_RESIZE_SAFETY_BYTES,
        resize_headroom_bytes: int | Callable[[], int] | None = None,
        memory_config: MemoryConfig | None = None,
        activation_bytes: int | None = None,
        weights_bytes: int | None = None,
        workspace_ceiling_bytes: int = 0,
        graph_pool_bytes: int = 0,
        staging_bytes: int = 0,
        device_total_bytes: int | None = None,
        tiering: TieredCacheConfig | None = None,
    ):
        """Build model bindings and materialize the initial KV slab.

        Args:
            model: Loaded native model; this runtime never loads weights.
            tokenizer_path: Tokenizer directory.
            config: Serving policy; defaults to ``ServingConfig()``.
            model_revision: Revision tag used in execution and prefix identity.
            weights_revision: Revision tag used in prefix identity.
            pages: Initial KV page capacity; at least one full context plus the
                padding page must fit.
            page_size: Tokens per KV page.
            max_requests: Scheduler sequence ceiling.
            max_pending_requests: Host lifecycle ceiling, including waiting and
                running requests. Defaults to ``max_requests``; increasing it
                reserves logical handles and sampling slots, but does not
                increase KV pages or per-step metadata capacity.
            max_inflight: Concurrent ticket ceiling. Sizes worker, workspace,
                metadata and graph resources before capacity is frozen.
            batch_tokens: Maximum batched tokens per step.
            prefill_chunk: Chunked-prefill cap.
            max_model_len: Served context ceiling; may reserve less than the
                checkpoint's ``max_position_embeddings`` but never more.  An
                oversized request is rejected before admission on the effective
                value.
            tokenizer_workers: Encode worker count.
            backend: ``triton``, ``reference``, ``flash_attention`` or
                ``flashinfer``; CPU requires ``reference``.
            resize_safety_bytes: Reserve that must stay free across a resize.
            resize_headroom_bytes: Measured headroom for the fit-check. An int
                is a fixed budget, a callable is sampled per check, and None
                queries the CUDA device (CPU diagnostics then cannot resize).
            memory_config: Device-budget policy for the frozen capacity plan.
            activation_bytes: Measured activation peak; profiled on CUDA when
                omitted, otherwise the config reserve is used.
            weights_bytes: Measured weight bytes; computed from the model when
                omitted.
            workspace_ceiling_bytes: Frozen per-step workspace ceiling.
            graph_pool_bytes: Reserved graph-private charge (capture arrives in
                R08; the charge keeps the budget honest until then).
            staging_bytes: Host staging claim, page-locked on CUDA. Raised to
                the persistent runner-buffer staging requirement when smaller.
            device_total_bytes: Device size for the policy budget; queried from
                CUDA when omitted.
            tiering: Opt-in host KV tier policy (R12B). ``host_bytes`` sizes a
                frozen host mirror; ``max_inflight_bytes`` bounds in-flight
                staging. NVMe/swap tiers are rejected. Tiering moves block
                residency only: the device slab capacity stays frozen.

        Decode-graph policy comes from ``config.decode_graph`` /
        ``config.graph_buckets``; ``graph_pool_bytes`` overrides the auto-sized
        reserve. Capture failures fail initialization instead of serving a
        half-captured graph path.
        """
        self.config = config or ServingConfig()
        self._closed = False
        self._decode_graph = bool(self.config.decode_graph)
        configured_buckets = self.config.graph_buckets
        if self._decode_graph and configured_buckets is not None:
            resolved_buckets = tuple(int(bucket) for bucket in configured_buckets)
        else:
            resolved_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self._graph_buckets = resolved_buckets
        parameter = next(model.parameters())
        device, dtype, model_config = parameter.device, parameter.dtype, model.config
        model_ceiling = model_config.max_position_embeddings
        if max_model_len is None:
            max_seq = model_ceiling
        else:
            if not isinstance(max_model_len, int) or isinstance(max_model_len, bool):
                raise TypeError("max_model_len must be an integer or None")
            if not 1 <= max_model_len <= model_ceiling:
                raise ValueError(
                    f"max_model_len must be in [1, {model_ceiling}], the checkpoint context"
                )
            max_seq = max_model_len
        if max_requests < 1 or pages < 2 or not 1 <= prefill_chunk <= batch_tokens:
            raise ValueError("invalid serving capacity")
        require_int(max_inflight, "max_inflight", minimum=1)
        if max_pending_requests is None:
            max_pending_requests = max_requests
        require_int(max_pending_requests, "max_pending_requests", minimum=1)
        if (pages - 1) * page_size < max_seq:
            raise ValueError("KV slab must fit at least one full model context plus padding page")
        if device.type == "cpu" and backend != "reference":
            raise ValueError("CPU serving requires the explicit reference attention backend")
        if device.type == "cpu" and self._decode_graph:
            raise ValueError("decode CUDA graphs require a CUDA device")
        if resize_safety_bytes < 0:
            raise ValueError("resize_safety_bytes must be non-negative")

        self._model = model
        self._device = device
        self._dtype = dtype
        self._model_config = model_config
        self._max_seq = max_seq
        self._backend_name = backend
        self._page_size = page_size
        self._max_requests = max_requests
        self._max_inflight = max_inflight
        self._max_pending_requests = max_pending_requests
        self._batch_tokens = batch_tokens
        self._prefill_chunk = prefill_chunk
        self._tokenizer_path = Path(tokenizer_path)
        self._model_revision = model_revision
        self._weights_revision = weights_revision
        self._resize_safety_bytes = resize_safety_bytes
        self._resize_headroom_source = resize_headroom_bytes
        self._memory_config = memory_config or MemoryConfig()
        for name, value in (
            ("workspace_ceiling_bytes", workspace_ceiling_bytes),
            ("graph_pool_bytes", graph_pool_bytes),
            ("staging_bytes", staging_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._workspace_ceiling_bytes = workspace_ceiling_bytes
        self._graph_bytes = graph_pool_bytes
        self._staging_bytes = staging_bytes
        self._device_total_bytes = device_total_bytes
        # Host tiering is opt-in at the serving boundary (R12B): only the
        # homogeneous full-retention tier is certified, so an NVMe or swap
        # configuration is rejected here instead of being half-wired.
        if tiering is not None and not isinstance(tiering, TieredCacheConfig):
            raise TypeError("tiering must be a TieredCacheConfig or None")
        _validate_serving_tiering(tiering)
        self._tiering = tiering
        # The operator's pin ceiling plus one probe of the machine. The mirror
        # decision is taken per build, because a SWAP resize holds both mirrors
        # and a starved host is a moment, not a configuration.
        self._host_policy = (
            HostMemoryPolicy.from_limits(tiering.host_bytes, self._memory_config.host)
            if tiering is not None
            else None
        )
        self._weights_bytes = (
            self._measure_weights(model) if weights_bytes is None else weights_bytes
        )
        if self._weights_bytes < 0:
            raise ValueError("weights_bytes must be non-negative")
        self._activation_bytes = self._resolve_activation_bytes(activation_bytes)
        self._freeze: CapacityFreeze | None = None
        self._constraints: GrammarConstraints | None = None
        self._tail: _KVTail | None = None
        self._mirror_bytes = 0
        self._mirror_pinned = False

        self.tokenizer = self.kv = self.storage = self.engine = self.service = None
        self.manager = self.workspace = self.workspace_allocator = self.runner_buffers = None
        self._sampling = SamplingCoordinator(
            max_pending_requests, device=device, vocab_size=model_config.vocab_size
        )
        dtype_name = str(dtype).removeprefix("torch.")
        kv_heads = getattr(model_config, "num_kv_heads", model_config.num_attention_heads)
        if not isinstance(kv_heads, int) or isinstance(kv_heads, bool) or kv_heads < 1:
            raise ValueError("model KV head count must be a positive integer")
        if model_config.num_attention_heads % kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads for grouped KV")
        self._kv_heads = kv_heads
        self._group = AttentionGroupSpec(
            0,
            tuple(range(model_config.num_hidden_layers)),
            AttentionType.FULL,
            AttentionSpec(
                model_config.num_attention_heads,
                kv_heads,
                model_config.head_dim,
                model_config.head_dim,
                model_config.scaling,
            ),
            page_size,
        )
        if self._decode_graph and self._graph_bytes == 0:
            self._graph_bytes = self._estimate_graph_bytes()
        self._execution = ExecutionPlan(
            "serving",
            self.config.model,
            model_revision,
            weights_revision,
            ComputePlan(
                dtype=DType.from_str(dtype_name),
                kv_dtype=DType.from_str(dtype_name),
                layer_range=(0, model_config.num_hidden_layers),
                max_num_batched_tokens=batch_tokens,
                enable_chunked_prefill=True,
            ),
        )
        self._scheduler_config = SchedulerConfig(
            preemption_mode=PreemptionMode.RECOMPUTE,
            max_num_seqs=max_requests,
            max_num_requests=max_pending_requests,
            max_inflight=max_inflight,
            max_num_batched_tokens=batch_tokens,
            max_prefill_chunk_tokens=prefill_chunk,
        )
        self._capabilities = SchedulerCapabilities(
            max_num_seqs=max_requests,
            chunked_prefill=True,
            prefix_cache=True,
            recompute_preemption=True,
            graph_mode=GraphMode.REPLAY if self._decode_graph else GraphMode.EAGER,
        )
        # R07: persistent per-flight metadata buffers. The device footprint is
        # reserved through the frozen capacity plan; pinned staging mirrors the
        # same tensors and must fit the staging claim, which is raised to the
        # pool's requirement when the caller left it below that.
        self._buffer_spec = RunnerBufferSpec.create(
            max_num_seqs=max_requests,
            max_num_batched_tokens=batch_tokens,
            max_inflight=self._scheduler_config.max_inflight,
            group_columns={"default": div_ceil(max_seq, page_size)},
        )
        self._runner_buffer_bytes = self._buffer_spec.device_bytes
        self._runner_staging_bytes = self._buffer_spec.staging_bytes if device.type == "cuda" else 0
        self._staging_bytes = max(staging_bytes, self._runner_staging_bytes)
        try:
            self.tokenizer = TokenizerService(
                TokenizerConfig(self._tokenizer_path, encode_pool_workers=tokenizer_workers),
                max_model_len=max_seq,
                model_vocab_size=model_config.vocab_size,
            )
            assert self.tokenizer.tokenizer is not None
            self._valid_token_ids = tuple(self.tokenizer.tokenizer.get_vocab().values())
            self._output = OutputProcessor(
                self.tokenizer, eos_token_ids=self.tokenizer.eos_token_ids
            )
            self._prefix_context = self._prefix_identity
            if self.config.structured_outputs:
                import xgrammar as xgr
                from transformers import AutoTokenizer

                raw = AutoTokenizer.from_pretrained(
                    str(self._tokenizer_path), trust_remote_code=False
                )
                self._constraints = GrammarConstraints(
                    xgr.TokenizerInfo.from_huggingface(raw, vocab_size=model_config.vocab_size)
                )
            tail = self._build_tail(pages)
            self._bind_tail(tail)
            self.processor = RequestProcessor(self.tokenizer, self.config)
            self.service = ServingService(
                self.engine,
                self.config,
                constraints=self._constraints,
                controller=self,
            )
        except BaseException:
            try:
                self._teardown_tail()
            finally:
                if self.tokenizer is not None:
                    self.tokenizer.close()
            raise

    # ------------------------------------------------------------------
    # cache geometry and rebuild helpers
    # ------------------------------------------------------------------

    def _storage_spec(self, pages: int) -> MHAStorageSpec:
        """Geometry of the homogeneous slab at ``pages`` capacity."""
        return MHAStorageSpec(
            num_layers=self._model_config.num_hidden_layers,
            num_kv_heads_local=self._kv_heads,
            head_dim=self._model_config.head_dim,
            page_size=self._page_size,
            capacity_pages=pages,
            dtype=str(self._dtype).removeprefix("torch."),
        )

    @staticmethod
    def _measure_weights(model) -> int:
        """Exact parameter bytes; the one number the checkpoint cannot lie about."""
        return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    def _resolve_activation_bytes(self, explicit: int | None) -> int:
        """Measured peak from profiling, or the configured reserve as fallback."""
        if explicit is not None:
            if not isinstance(explicit, int) or isinstance(explicit, bool) or explicit < 0:
                raise ValueError("activation_bytes must be a non-negative integer")
            return explicit
        if self._device.type == "cuda":
            return self._profile_activation()
        return self._memory_config.activation_reserve_bytes

    def _profile_activation(self) -> int:
        """Peak allocator usage of a dense forward on the supported shapes.

        The profiling pass runs before the KV slab exists; a dummy dense
        sequence is a conservative upper bound for the paged path (it holds the
        attention working set that later lives in the slab), which is the safe
        direction for the KV budget that is derived from it.
        """
        import torch

        from ayaka.utils.torch_memory import peak_memory_bytes

        shapes = sorted(
            {
                max(1, min(self._batch_tokens, self._max_seq)),
                max(1, min(self._max_requests, self._max_seq)),
            }
        )
        peak = 0
        with torch.inference_mode():
            for tokens in shapes:
                ids = torch.zeros(tokens, dtype=torch.int64, device=self._device)
                try:
                    baseline = torch.cuda.memory_allocated(self._device)
                    with peak_memory_bytes(self._device) as observed:
                        self._model.forward_dense(ids)
                finally:
                    del ids
                # peak_memory_bytes reports the process-wide absolute peak.
                # Existing weights and unrelated live runtimes are already
                # charged to their owners; only this forward's delta is ours.
                peak = max(peak, max(0, observed[0] - baseline))
        return peak

    def _resident_pinned_mirror_bytes(self) -> int:
        """Pinned mirror bytes still resident from the currently bound tail.

        A SWAP resize allocates the replacement mirror while the old one is
        still pinned, so the old bytes count against the same ceiling. The
        ledger cannot answer this: it charges staging and mirror to the same
        owner account. The value is tracked on the bound tail, not read from a
        closed manager, so a REBUILD (which frees first) does not over-count.
        """
        return self._mirror_bytes if self._mirror_pinned else 0

    def _check_host_mirror(self, pages: int) -> MirrorDecision | None:
        """Validate the mirror at ``pages`` against the host pin ceiling.

        Read-only and raisable; used by :meth:`plan_resize` so a refused resize
        never reaches a destructive free. The allocation path repeats the check
        with the same inputs, so both agree.
        """
        if self._tiering is None or self._host_policy is None:
            return None
        spec = self._storage_spec(pages)
        tiering_config, _ = self._tiering_config(spec)
        assert tiering_config is not None
        mirror_bytes = tiering_config.host_capacity_pages * spec.bytes_per_page
        decision = self._host_policy.decide_mirror(
            mirror_bytes,
            requested_tier=(
                MemoryTier.HOST_PINNED if self._device.type == "cuda" else MemoryTier.HOST_PAGEABLE
            ),
            already_pinned_bytes=self._staging_bytes + self._resident_pinned_mirror_bytes(),
        )
        if not decision:
            raise CacheRebuildRejected(
                f"host KV mirror refused by host memory policy: {decision.reason}",
                reason=ResizeRejectionReason.BUDGET,
                requested_pages=tiering_config.host_capacity_pages,
                need_bytes=mirror_bytes,
                available_bytes=decision.headroom_bytes,
            )
        return decision

    def _build_kv(self, pages: int) -> WorkerResources:
        """Delegate allocation, reconciliation and freeze to the Worker owner."""
        spec = self._storage_spec(pages)
        tiering_config, max_inflight_bytes = self._tiering_config(spec)
        return WorkerResourcePlan(
            device=self._device,
            storage_spec=self._storage_spec(pages),
            buffer_spec=self._buffer_spec,
            memory_config=self._memory_config,
            device_total_bytes=self._device_total_bytes,
            model_id=self.config.model,
            model_revision=self._model_revision,
            weights_revision=self._weights_revision,
            max_model_len=self._max_seq,
            max_num_seqs=self._max_requests,
            max_num_requests=self._max_pending_requests,
            max_num_batched_tokens=self._batch_tokens,
            max_inflight=self._scheduler_config.max_inflight,
            weights_bytes=self._weights_bytes,
            activation_bytes=self._activation_bytes,
            workspace_ceiling_bytes=self._workspace_ceiling_bytes,
            graph_bytes=self._graph_bytes,
            staging_bytes=self._staging_bytes,
            tiering=tiering_config,
            max_inflight_bytes=max_inflight_bytes,
            host_policy=self._host_policy if tiering_config is not None else None,
            already_pinned_host_bytes=self._resident_pinned_mirror_bytes(),
        ).build()

    def _tiering_config(self, spec: MHAStorageSpec) -> tuple[TieringConfig | None, int | None]:
        """Convert the serving tiering policy into the runtime tier config.

        The mirror is page-for-page with the device slab, so its capacity in
        pages is ``host_bytes // bytes_per_page``; a host_bytes that cannot
        hold one page is a configuration error, not a silent empty tier.
        """

        if self._tiering is None:
            return None, None
        host_pages, remainder = divmod(self._tiering.host_bytes, spec.bytes_per_page)
        if host_pages < 1:
            raise ValueError(
                f"tiering.host_bytes ({self._tiering.host_bytes}) must cover at least one "
                f"KV page ({spec.bytes_per_page} B at page_size={spec.page_size})"
            )
        return TieringConfig(host_capacity_pages=host_pages), self._tiering.max_inflight_bytes

    def _build_runner(self, kv: LogicalKVManager, buffers: RunnerBuffers) -> PagedModelRunner:
        """Bind a fresh paged runner to the slab, this model and its buffers."""
        runner = PagedModelRunner(
            self._sampling,
            self._model,
            kv,
            {"default": self._group},
            backend=self._backend_name,
            force_reference=self._device.type == "cpu",
            buffers=buffers,
        )
        runner.set_valid_token_ids(self._valid_token_ids)
        return runner

    def _build_engine(
        self,
        kv: LogicalKVManager,
        runner: PagedModelRunner,
        ledger: MemoryLedger,
        workspace: WorkspaceManager,
        buffers: RunnerBuffers,
        resources: WorkerResources,
    ) -> ResidentKVEngine:
        """Resolve a fresh scheduler plan against the new ledger and wire it."""
        plan = self._scheduler_config.resolve(
            self._max_seq,
            execution=self._execution,
            memory=ledger.snapshot(),
            workspace=MemoryPlan(activation_bytes=self._activation_bytes),
            capabilities=self._capabilities,
        )
        engine = ResidentKVEngine(
            plan,
            kv,
            runner,
            sampling=self._sampling,
            output=self._output,
            prefix_context=self._prefix_context,
            workspace=workspace,
            buffers=buffers,
            worker=LocalWorker(kv, runner, max_inflight=plan.max_inflight, resources=resources),
        )
        if self._constraints is not None:
            runner.set_request_source(engine.requests.get, constraints=self._constraints)
        return engine

    def _prefix_identity(self, _request: object):
        """Canonical prefix identity; the KV storage geometry is authoritative.

        The request contributes its tenant salt downstream (the preparer pairs
        both namespaces); this provider carries the execution identity only.
        """
        if self.kv is None:
            raise RuntimeError("prefix identity requested before the KV slab is bound")
        return build_prefix_context(
            model_id=self.config.model,
            model_revision=f"{self._model_revision}:{self._weights_revision}",
            config=self._model_config,
            storage_spec=self.kv.storages["default"].storage.spec,
        )

    def _build_tail(self, pages: int) -> _KVTail:
        """Materialize every owner for one capacity, cleaning up on failure."""
        resources = self._build_kv(pages)
        ledger, kv = resources.ledger, resources.kv
        storage = resources.storages[0]
        workspace, allocator = resources.workspace, resources.allocator
        capacity, buffers = resources.capacity, resources.buffers
        assert workspace is not None and allocator is not None and buffers is not None
        manager = kv.backend
        assert isinstance(manager, RuntimeMemoryManager)
        runner = None
        engine = None
        try:
            runner = self._build_runner(kv, buffers)
            if self._decode_graph:
                runner.enable_graph(self._decode_graph_config(kv))
            engine = self._build_engine(kv, runner, ledger, workspace, buffers, resources)
            if self._decode_graph and runner.graph_pool is not None:
                runner.graph_pool.attach_metrics(engine.executor.metrics)
        except BaseException:
            try:
                if engine is not None:
                    engine.close()
                elif runner is not None:
                    runner.close()
            finally:
                if not resources.closed:
                    resources.close()
            raise
        return _KVTail(
            resources=resources,
            ledger=ledger,
            storage=storage,
            manager=manager,
            kv=kv,
            workspace=workspace,
            workspace_allocator=allocator,
            capacity=capacity,
            buffers=buffers,
            runner=runner,
            engine=engine,
        )

    def _decode_graph_config(self, kv: LogicalKVManager):
        """Bootstrap capture config: buckets, ceilings, padding address, budget."""
        from ayaka.runner.graph.pool import DecodeGraphConfig

        return DecodeGraphConfig(
            buckets=self._graph_buckets,
            max_model_len=self._max_seq,
            padding_page=kv.padding_page,
            padding_slot=kv.padding_slot,
            reserve_bytes=self._graph_bytes,
        )

    def _estimate_graph_bytes(self) -> int:
        """Reserve for the persistent graph state plus the capture memory pool.

        The Triton builder prices its persistent buffers without a device; the
        measured activation peak is the upper bound for what capture records
        into the graph pool. Any other backend must state ``graph_pool_bytes``
        explicitly — guessing another backend's footprint would under-reserve
        silently.
        """
        if self._backend_name != "triton":
            raise ValueError("graph_pool_bytes must be set explicitly for non-Triton decode graphs")
        ceiling = min(self._max_requests, self._batch_tokens)
        largest = max((bucket for bucket in self._graph_buckets if bucket <= ceiling), default=0)
        if largest < 1:
            raise ValueError("no decode graph bucket fits the scheduler ceilings")
        from ayaka.attention.backend.triton_backend import (
            DEFAULT_KV_PARTITION_SIZE,
            TritonAttentionMetadataBuilder,
        )

        spec = self._group.spec
        persistent = TritonAttentionMetadataBuilder.estimate_graph_state_bytes(
            max_batch_size=largest,
            max_seq_len=self._max_seq,
            page_size=self._page_size,
            num_qo_heads=spec.num_qo_heads,
            head_dim_qk=spec.head_dim_qk,
            head_dim_vo=spec.head_dim_vo,
            kv_partition_size=DEFAULT_KV_PARTITION_SIZE,
            output_dtype=self._dtype,
            sliding_window=spec.sliding_window,
        )
        return (persistent + self._activation_bytes) * self._max_inflight

    def _bind_tail(self, tail: _KVTail) -> None:
        """Point every public attribute at the new owner set."""
        if self._freeze is None:
            self._freeze = tail.resources.freeze
        else:
            if (
                tail.capacity.generation.owner_incarnation
                <= self._freeze.current.generation.owner_incarnation
            ):
                raise ValueError("rebuild must advance owner_incarnation")
            self._freeze = tail.resources.freeze
        self._tail = tail
        self.storage = tail.storage
        self.manager = tail.manager
        self.kv = tail.kv
        self.workspace = tail.workspace
        self.workspace_allocator = tail.workspace_allocator
        self.runner_buffers = tail.buffers
        self.engine = tail.engine
        mirror = getattr(tail.manager, "host_storage", None)
        self._mirror_bytes = 0 if mirror is None else int(mirror.total_bytes)
        self._mirror_pinned = bool(mirror is not None and mirror.pinned)

    def _teardown_tail(self) -> None:
        """Drain and release the currently bound owners, if any.

        Ordering matters: the engine drains every launched ticket and closes the
        runner (which drops attention views into the slab), then the per-flight
        buffers (every lease must be released by retirement), then the workspace
        arenas (no lease may remain). The logical manager close drops
        prefix-cache ownership and its slab pins last, and only then is the
        storage freed.
        """
        engine = self.engine
        if engine is not None:
            if not engine.close().closed:
                raise RuntimeError("engine still owns unsettled work; tail retained")
        if self._tail is not None:
            self._tail.resources.close()
        # The mirror died with the tail; a following build must not count it.
        self._mirror_bytes = 0
        self._mirror_pinned = False

    def _resolve_headroom(self) -> int | None:
        """Measured resizable headroom, or None when the device cannot report."""
        source = self._resize_headroom_source
        if callable(source):
            value = source()
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("resize headroom probe must return an integer")
            if value < 0:
                raise ValueError("resize headroom probe must be non-negative")
            return value
        if source is not None:
            if not isinstance(source, int) or isinstance(source, bool):
                raise TypeError("resize_headroom_bytes must be an int, callable or None")
            if source < 0:
                raise ValueError("resize_headroom_bytes must be non-negative")
            return source
        if self._device.type == "cuda":
            return device_headroom(self._device)
        return None

    # ------------------------------------------------------------------
    # resize API
    # ------------------------------------------------------------------

    def _frozen_budget(self) -> int | None:
        """KV budget frozen at bootstrap; only enforced where the budget is policy."""
        if self._freeze is None or self._device.type != "cuda":
            return None
        return self._freeze.current.kv_budget_bytes

    def plan_resize(self, pages: int) -> ResizePlan:
        """Validate a capacity without freeing or allocating anything.

        Raises:
            CacheRebuildRejected: ``INVALID_PAGES`` below the context floor,
                ``UNSUPPORTED`` when headroom cannot be measured, or ``BUDGET``
                when the frozen KV budget or the transient headroom cannot fit
                the request. The old cache is untouched in every case.
        """
        self._require_live()
        storage = self.storage
        assert storage is not None
        plan = validate_resize(
            spec=storage.storage.spec,
            requested_pages=pages,
            page_size=self._page_size,
            max_sequence_tokens=self._max_seq,
            headroom_bytes=self._resolve_headroom(),
            device=self._device,
            safety_bytes=self._resize_safety_bytes,
            kv_budget_bytes=self._frozen_budget(),
        )
        # The host mirror is rebuilt alongside the slab; its pin ceiling is
        # part of the fit and must reject here, before any free.
        self._check_host_mirror(pages)
        return plan

    def cache_status(self) -> CacheStatus:
        """Return current capacity and the largest admissible resize for a UI."""
        self._require_live()
        storage = self.storage
        manager = self.manager
        assert storage is not None and manager is not None
        return build_cache_status(
            name="default",
            spec=storage.storage.spec,
            snapshot=manager.snapshot(),
            max_sequence_tokens=self._max_seq,
            headroom_bytes=self._resolve_headroom(),
            safety_bytes=self._resize_safety_bytes,
            budget_bytes=self._frozen_budget(),
        )

    def resize(self, pages: int, *, timeout: float = 30.0) -> CacheStatus:
        """Request a capacity change through the engine owner thread.

        Args:
            pages: Requested page capacity.
            timeout: Seconds to wait for the engine thread to answer.

        Returns:
            The cache status after a successful resize.

        Raises:
            EngineUnavailableError: When the engine is closing or the request
                timed out.
            CacheRebuildRejected: When the engine is busy or the budget does
                not fit; the old cache keeps serving.
            CacheResizeFatal: When recovery after a destructive free failed and
                the engine must restart.
        """
        self._require_live()
        if self.service is None:
            raise RuntimeError("runtime has no serving service")
        return self.service.resize(pages, timeout=timeout)

    def apply_resize(self, pages: int) -> CacheStatus:
        """Execute a validated resize; only called on the engine owner thread.

        The busy gate and the budget fit-check run before any destructive free.
        A ``SWAP`` plan materializes the new slab alongside the old one, so a
        failure leaves the old cache serving. A ``REBUILD`` plan frees first and
        restores the previous capacity if the new materialization fails.
        """
        self._require_live()
        runtime_event(
            "resize",
            owner_id="default",
            resource_generation=getattr(self.kv, "generation", None),
            status="requested",
            detail=f"pages={pages}",
        )
        engine = self.engine
        assert engine is not None
        if engine.has_unfinished or engine.executor.pending_completion or engine.executor.tickets:
            runtime_event("resize", owner_id="default", status="rejected_busy")
            raise CacheRebuildRejected(
                "engine is busy; resize requires an idle server; old cache kept",
                reason=ResizeRejectionReason.BUSY,
                requested_pages=pages,
            )
        try:
            plan = self.plan_resize(pages)
        except CacheRebuildRejected as exc:
            runtime_event("resize", owner_id="default", status="rejected", detail=str(exc))
            raise
        if plan.pages == plan.current_pages:
            runtime_event("resize", owner_id="default", status="unchanged")
            return self.cache_status()
        if plan.mode is CacheResizeMode.SWAP:
            tail = self._build_tail(plan.pages)
            try:
                self._teardown_tail()
            except BaseException as exc:
                self._bind_tail(tail)
                runtime_event("resize", owner_id="default", status="failed", detail=str(exc))
                raise
            self._bind_tail(tail)
            runtime_event(
                "resize",
                owner_id="default",
                resource_generation=getattr(self.kv, "generation", None),
                status="applied_swap",
            )
            return self.cache_status()

        try:
            self._teardown_tail()
        except BaseException as exc:
            runtime_event("resize", owner_id="default", status="fatal", detail=str(exc))
            raise CacheResizeFatal(
                "cache teardown failed before the rebuild; engine must restart"
            ) from exc
        empty_cache()
        try:
            tail = self._build_tail(plan.pages)
        except BaseException as exc:
            try:
                restored = self._build_tail(plan.current_pages)
            except BaseException as restore_exc:
                runtime_event("resize", owner_id="default", status="fatal", detail=str(restore_exc))
                raise CacheResizeFatal(
                    f"cache resize to {plan.pages} pages failed and the "
                    f"{plan.current_pages}-page cache could not be restored; engine must restart"
                ) from restore_exc
            self._bind_tail(restored)
            runtime_event("resize", owner_id="default", status="rejected_restored", detail=str(exc))
            raise CacheRebuildRejected(
                f"cache allocation for {plan.pages} pages failed after the old slab was "
                f"freed; {plan.current_pages}-page cache restored",
                reason=ResizeRejectionReason.ALLOCATION_FAILED,
                requested_pages=plan.pages,
                need_bytes=plan.need_bytes,
                available_bytes=plan.headroom_bytes + plan.old_bytes - plan.safety_bytes,
                old_bytes=plan.old_bytes,
            ) from exc
        self._bind_tail(tail)
        runtime_event(
            "resize",
            owner_id="default",
            resource_generation=getattr(self.kv, "generation", None),
            status="applied_rebuild",
        )
        return self.cache_status()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _require_live(self) -> None:
        """Reject cache operations after shutdown has started."""
        if self._closed:
            raise RuntimeError("runtime is closed")
        if self.storage is None or self.manager is None or self.engine is None:
            raise RuntimeError("runtime cache is not available")

    def app(self):
        if self.service is None or self._closed:
            raise RuntimeError("runtime is not available")
        return create_app(self.service, self.processor, close=self.close, admin=self)

    def close(self, timeout=10.0) -> bool:
        if self._closed:
            return True
        if self.service is not None and not self.service.close(timeout):
            return False
        self._teardown_tail()
        if self.tokenizer is not None:
            self.tokenizer.close()
        self._closed = True
        return True
