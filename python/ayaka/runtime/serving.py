"""Composition of Ayaka's native model, resident KV, tokenizer and HTTP service.

KV capacity is an explicit slab budget, not a claim about total free VRAM.
Shutdown releases slabs only after the engine proves every flight settled.
The capacity may be changed while the server runs: :meth:`ServingRuntime.resize`
validates the byte cost against measured device headroom *before* anything is
freed, so a rejected resize keeps the old caches serving.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ayaka.attention.spec import AttentionGroupSpec, AttentionSpec
from ayaka.configs.memory import MemoryConfig, MemoryProfile, plan_device_memory
from ayaka.configs.scheduler import PreemptionMode, SchedulerCapabilities, SchedulerConfig
from ayaka.configs.serving import ServingConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.materialize import KVStorageLease, materialize_kv_storage
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
    MemoryLane,
    build_capacity_snapshot,
    claim_tier,
    mint_generation,
    reconcile_actual_usage,
)
from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.source import TorchDeviceSource, TorchHostByteSource
from ayaka.memory.workspace import WorkspaceManager
from ayaka.plan import ComputePlan, ExecutionPlan, MemoryPlan
from ayaka.prefix.identity import build_prefix_context
from ayaka.runner.paged_runner import PagedModelRunner
from ayaka.runtime.output import OutputProcessor
from ayaka.runtime.resident import ResidentKVEngine
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.serving.constraints import GrammarConstraints
from ayaka.serving.http import create_app
from ayaka.serving.prepare import RequestProcessor
from ayaka.serving.service import ServingService
from ayaka.tokenizers.service import TokenizerService
from ayaka.types import AttentionType, DType, MemoryOwner, MemoryTier
from ayaka.utils.torch_memory import device_memory, empty_cache

#: Fixed bookkeeping bytes the ledger reserves on top of the aligned slab.
_LEDGER_OVERHEAD_BYTES = LEDGER_OVERHEAD_BYTES

#: Default reserve kept free across a resize; covers allocator fragmentation
#: and activations that are not part of the KV slab.
DEFAULT_RESIZE_SAFETY_BYTES = 256 << 20


@dataclass(slots=True)
class _KVTail:
    """Every owner that is replaced together when the cache is rebuilt."""

    ledger: MemoryLedger
    storage: KVStorageLease
    manager: RuntimeMemoryManager
    kv: LogicalKVManager
    workspace: WorkspaceManager
    workspace_allocator: CachingAllocator
    capacity: CapacitySnapshot
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
        batch_tokens=256,
        prefill_chunk=128,
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
            batch_tokens: Maximum batched tokens per step.
            prefill_chunk: Chunked-prefill cap.
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
            staging_bytes: Host staging claim, page-locked on CUDA.
            device_total_bytes: Device size for the policy budget; queried from
                CUDA when omitted.
        """
        self.config = config or ServingConfig()
        self._closed = False
        parameter = next(model.parameters())
        device, dtype, model_config = parameter.device, parameter.dtype, model.config
        max_seq = model_config.max_position_embeddings
        if max_requests < 1 or pages < 2 or not 1 <= prefill_chunk <= batch_tokens:
            raise ValueError("invalid serving capacity")
        if (pages - 1) * page_size < max_seq:
            raise ValueError("KV slab must fit at least one full model context plus padding page")
        if device.type == "cpu" and backend != "reference":
            raise ValueError("CPU serving requires the explicit reference attention backend")
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
        self._weights_bytes = (
            self._measure_weights(model) if weights_bytes is None else weights_bytes
        )
        if self._weights_bytes < 0:
            raise ValueError("weights_bytes must be non-negative")
        self._activation_bytes = self._resolve_activation_bytes(activation_bytes)
        self._freeze: CapacityFreeze | None = None
        self._constraints: GrammarConstraints | None = None
        self._tail: _KVTail | None = None

        self.tokenizer = self.kv = self.storage = self.engine = self.service = None
        self.manager = self.workspace = self.workspace_allocator = None
        self._sampling = SamplingCoordinator(
            max_requests, device=device, vocab_size=model_config.vocab_size
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
            max_num_requests=max_requests,
            max_num_batched_tokens=batch_tokens,
            max_prefill_chunk_tokens=prefill_chunk,
        )
        self._capabilities = SchedulerCapabilities(
            max_num_seqs=max_requests,
            chunked_prefill=True,
            prefix_cache=True,
            recompute_preemption=True,
        )
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
            try:
                return self._profile_activation()
            except Exception:
                # Profiling is best-effort; the config reserve is the fallback.
                pass
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
                    with peak_memory_bytes(self._device) as observed:
                        self._model.forward_dense(ids)
                finally:
                    del ids
                peak = max(peak, observed[0])
        return peak

    def _resolve_budget(self, pages: int) -> tuple[int, int, int]:
        """Resolve (budget, kv_budget, device_total) for one tail geometry.

        On CUDA the policy budget is the device budget for every owner and the
        KV slab is the remainder after the measured non-KV profile; on CPU the
        diagnostic budget is derived from what the process owns (host memory is
        not policy-bounded) and every claim stays on HOST_PAGEABLE.
        """
        spec = self._storage_spec(pages)
        slab = spec.aligned_total_bytes(256) + _LEDGER_OVERHEAD_BYTES
        if self._device.type == "cuda":
            total = self._device_total_bytes
            if total is None:
                total = device_memory(self._device).driver_total
            profile = MemoryProfile(
                weights_bytes=self._weights_bytes,
                activation_bytes=self._activation_bytes,
                kernel_workspace_bytes=self._workspace_ceiling_bytes,
                graph_pool_bytes=self._graph_bytes,
                measured=True,
            )
            plan = plan_device_memory(total, self._memory_config, profile)
            if slab > plan.kv_cache_bytes:
                raise CacheRebuildRejected(
                    f"the {pages}-page KV slab needs {slab} bytes but only "
                    f"{plan.kv_cache_bytes} bytes remain after non-KV allocations; "
                    "lower pages or raise gpu_memory_utilization",
                    reason=ResizeRejectionReason.BUDGET,
                    requested_pages=pages,
                    need_bytes=slab,
                    available_bytes=plan.kv_cache_bytes,
                )
            return plan.policy_budget_bytes, plan.kv_cache_bytes, total
        # Host diagnostics are not policy-bounded; the budget covers what the
        # process owns plus allocator slack so no claim can exceed capacity.
        owned = (
            slab
            + self._weights_bytes
            + self._activation_bytes
            + self._workspace_ceiling_bytes
            + self._graph_bytes
            + self._staging_bytes
        )
        overhead = max(4 << 20, owned // 16)
        return owned + overhead, slab, owned + overhead

    def _build_ledger(self, pages: int) -> tuple[MemoryLedger, int, int]:
        """Create the full-budget ledger and admit the fixed non-KV claims."""
        budget, kv_budget, total = self._resolve_budget(pages)
        lane = MemoryLane.CUDA if self._device.type == "cuda" else MemoryLane.CPU
        device_index = self._device.index or 0
        if lane is MemoryLane.CPU:
            ledger = MemoryLedger.for_device(
                device_budget_bytes=budget,
                device_total_bytes=total,
                host_pageable_bytes=budget,
                host_total_bytes=total,
                host_pinned_bytes=self._staging_bytes,
                device_index=device_index,
            )
        else:
            ledger = MemoryLedger.for_device(
                device_budget_bytes=budget,
                device_total_bytes=total,
                host_pinned_bytes=self._staging_bytes,
                device_index=device_index,
            )
        for owner, label, nbytes in (
            (MemoryOwner.WEIGHT, "serving.weights", self._weights_bytes),
            (MemoryOwner.COMPILE, "serving.graph", self._graph_bytes),
        ):
            if not nbytes:
                continue
            ledger.admit(
                Reservation.backed(
                    owner,
                    label,
                    nbytes,
                    tier=claim_tier(owner, lane=lane),
                    device_index=device_index,
                )
            )
        if self._staging_bytes:
            ledger.admit(
                Reservation.backed(
                    MemoryOwner.WORKSPACE,
                    "serving.staging",
                    self._staging_bytes,
                    tier=claim_tier(MemoryOwner.WORKSPACE, lane=lane, pinned_host=True),
                    device_index=device_index,
                )
            )
        return ledger, budget, kv_budget

    def _build_workspace(self, ledger: MemoryLedger) -> tuple[WorkspaceManager, CachingAllocator]:
        """Allocate the activation arena and freeze the workspace ceiling."""
        if self._device.type == "cuda":
            source = TorchDeviceSource(device_index=self._device.index or 0)
            tier = MemoryTier.DEVICE
        else:
            source = TorchHostByteSource(pinned=False)
            tier = MemoryTier.HOST_PAGEABLE
        allocator = CachingAllocator(
            source,
            ledger=ledger,
            tier=tier,
            device_index=self._device.index or 0,
            ledger_label="serving.workspace",
        )
        manager = WorkspaceManager(allocator, workspace_ceiling_bytes=self._workspace_ceiling_bytes)
        manager.initialize(self._activation_bytes)
        return manager, allocator

    def _build_kv(
        self, pages: int
    ) -> tuple[
        MemoryLedger,
        KVStorageLease,
        RuntimeMemoryManager,
        LogicalKVManager,
        WorkspaceManager,
        CachingAllocator,
        CapacitySnapshot,
    ]:
        """Materialize one charged slab set plus its workspace and snapshot.

        Order matters: ledger with fixed claims, KV slab, workspace/activation
        arena, then the frozen capacity snapshot that names all of them. Any
        failure closes owners in reverse before re-raising.
        """
        ledger, budget, kv_budget = self._build_ledger(pages)
        spec = self._storage_spec(pages)
        storage = materialize_kv_storage(
            spec,
            ledger=ledger,
            label="serving.kv",
            device=str(self._device),
            zero_initialize=True,
        )
        workspace: WorkspaceManager | None = None
        allocator: CachingAllocator | None = None
        try:
            manager = RuntimeMemoryManager(
                total_pages=pages,
                page_size=self._page_size,
                max_sequences=self._max_requests,
                max_sequence_tokens=self._max_seq,
                storage=storage.storage,
            )
            kv = LogicalKVManager(manager, {"default": storage})
        except BaseException:
            storage.close()
            with suppress(Exception):
                ledger.release_owner(MemoryOwner.WEIGHT)
            with suppress(Exception):
                ledger.release_owner(MemoryOwner.COMPILE)
            raise
        try:
            workspace, allocator = self._build_workspace(ledger)
            lane = MemoryLane.CUDA if self._device.type == "cuda" else MemoryLane.CPU
            # Freeze the materialized activation capacity (pooled class bytes),
            # not the profiling figure: the claim must cover what the pool holds.
            activation_claim = allocator.bytes_by_owner().get(MemoryOwner.ACTIVATION, 0)
            generation = mint_generation(
                model_id=self.config.model,
                model_revision=self._model_revision,
                weights_revision=self._weights_revision,
                kv_storage=kv.fingerprint,
                backend=type(kv.backend).__name__,
                workspace=workspace.workspace_generation,
            )
            snapshot = build_capacity_snapshot(
                generation=generation,
                lane=lane,
                dtype=str(self._dtype).removeprefix("torch."),
                kv_dtype=str(self._dtype).removeprefix("torch."),
                page_size=self._page_size,
                group_pages={"default": pages},
                max_model_len=self._max_seq,
                max_num_seqs=self._max_requests,
                max_num_batched_tokens=self._batch_tokens,
                max_inflight=self._scheduler_config.max_inflight,
                activation_bytes=activation_claim,
                workspace_ceiling_bytes=self._workspace_ceiling_bytes,
                graph_bytes=self._graph_bytes,
                staging_bytes=self._staging_bytes,
                budget_bytes=budget,
                kv_budget_bytes=kv_budget,
                weights_bytes=self._weights_bytes,
                ledger=ledger,
                staging_pinned=lane is MemoryLane.CUDA,
            )
            kv.bind_capacity(snapshot)
            reconcile_actual_usage(ledger, snapshot)
        except BaseException:
            if workspace is not None and allocator is not None:
                self._close_workspace(workspace, allocator)
            kv.close()
            storage.close()
            with suppress(Exception):
                ledger.release_owner(MemoryOwner.WEIGHT)
            with suppress(Exception):
                ledger.release_owner(MemoryOwner.WORKSPACE)
            with suppress(Exception):
                ledger.release_owner(MemoryOwner.COMPILE)
            raise
        return ledger, storage, manager, kv, workspace, allocator, snapshot

    @staticmethod
    def _close_workspace(workspace: WorkspaceManager, allocator: CachingAllocator) -> None:
        with suppress(Exception):
            workspace.close()
        with suppress(Exception):
            allocator.close()

    def _build_runner(self, kv: LogicalKVManager) -> PagedModelRunner:
        """Bind a fresh paged runner to the slab and this model."""
        runner = PagedModelRunner(
            self._sampling,
            self._model,
            kv,
            {"default": self._group},
            backend=self._backend_name,
            force_reference=self._device.type == "cpu",
        )
        runner.set_valid_token_ids(self._valid_token_ids)
        return runner

    def _build_engine(
        self,
        kv: LogicalKVManager,
        runner: PagedModelRunner,
        ledger: MemoryLedger,
        workspace: WorkspaceManager,
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
        ledger, storage, manager, kv, workspace, allocator, capacity = self._build_kv(pages)
        runner = None
        try:
            runner = self._build_runner(kv)
            engine = self._build_engine(kv, runner, ledger, workspace)
        except BaseException:
            if runner is not None:
                runner.close()
            kv.close()
            self._close_workspace(workspace, allocator)
            storage.close()
            raise
        return _KVTail(
            ledger=ledger,
            storage=storage,
            manager=manager,
            kv=kv,
            workspace=workspace,
            workspace_allocator=allocator,
            capacity=capacity,
            runner=runner,
            engine=engine,
        )

    def _bind_tail(self, tail: _KVTail) -> None:
        """Point every public attribute at the new owner set."""
        if self._freeze is None:
            self._freeze = CapacityFreeze(tail.capacity)
        else:
            self._freeze.replace(tail.capacity)
        self._tail = tail
        self.storage = tail.storage
        self.manager = tail.manager
        self.kv = tail.kv
        self.workspace = tail.workspace
        self.workspace_allocator = tail.workspace_allocator
        self.engine = tail.engine

    def _teardown_tail(self) -> None:
        """Drain and release the currently bound owners, if any.

        Ordering matters: the engine drains every launched ticket and closes the
        runner (which drops attention views into the slab), then the workspace
        arenas are freed (no lease may remain). The logical manager close drops
        prefix-cache ownership and its slab pins last, and only then is the
        storage freed.
        """
        engine = self.engine
        if engine is not None:
            engine.close()
        if self.workspace is not None and self.workspace_allocator is not None:
            self.workspace.close()
            self.workspace_allocator.close()
        if self.kv is not None and not self.kv.closed:
            self.kv.close()
        if self.storage is not None and not self.storage.closed:
            self.storage.close()

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
        return validate_resize(
            spec=storage.storage.spec,
            requested_pages=pages,
            page_size=self._page_size,
            max_sequence_tokens=self._max_seq,
            headroom_bytes=self._resolve_headroom(),
            device=self._device,
            safety_bytes=self._resize_safety_bytes,
            kv_budget_bytes=self._frozen_budget(),
        )

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
        engine = self.engine
        assert engine is not None
        if engine.has_unfinished or engine.executor.pending_completion or engine.executor.tickets:
            raise CacheRebuildRejected(
                "engine is busy; resize requires an idle server; old cache kept",
                reason=ResizeRejectionReason.BUSY,
                requested_pages=pages,
            )
        plan = self.plan_resize(pages)
        if plan.pages == plan.current_pages:
            return self.cache_status()
        if plan.mode is CacheResizeMode.SWAP:
            tail = self._build_tail(plan.pages)
            try:
                self._teardown_tail()
            except BaseException:
                self._bind_tail(tail)
                raise
            self._bind_tail(tail)
            return self.cache_status()

        try:
            self._teardown_tail()
        except BaseException as exc:
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
                raise CacheResizeFatal(
                    f"cache resize to {plan.pages} pages failed and the "
                    f"{plan.current_pages}-page cache could not be restored; engine must restart"
                ) from restore_exc
            self._bind_tail(restored)
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
