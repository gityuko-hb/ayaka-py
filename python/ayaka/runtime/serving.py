"""Composition of Ayaka's native model, resident KV, tokenizer and HTTP service."""

from __future__ import annotations

from pathlib import Path

from ayaka.attention.spec import AttentionGroupSpec, AttentionSpec
from ayaka.configs.scheduler import PreemptionMode, SchedulerCapabilities, SchedulerConfig
from ayaka.configs.serving import ServingConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.kvcache.materialize import materialize_kv_storage
from ayaka.kvcache.storage.geometry import MHAStorageSpec
from ayaka.memory.ledger import MemoryLedger
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.plan import ComputePlan, ExecutionPlan, MemoryPlan
from ayaka.prefix.identity import PrefixCacheContext
from ayaka.runner.paged_runner import PagedModelRunner
from ayaka.runtime.output import OutputProcessor
from ayaka.runtime.resident import ResidentKVEngine
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.serving.constraints import GrammarConstraints
from ayaka.serving.http import create_app
from ayaka.serving.prepare import RequestProcessor
from ayaka.serving.service import ServingService
from ayaka.tokenizers.service import TokenizerService
from ayaka.types import AttentionType, DType


class ServingRuntime:
    """Single-device, native serving; caller owns the supplied model.

    KV capacity is an explicit slab budget, not a claim about total free VRAM.
    Shutdown releases slabs only after the engine proves every flight settled.
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
    ):
        self.config = config or ServingConfig()
        parameter = next(model.parameters())
        device, dtype, model_config = parameter.device, parameter.dtype, model.config
        max_seq = model_config.max_position_embeddings
        if max_requests < 1 or pages < 2 or not 1 <= prefill_chunk <= batch_tokens:
            raise ValueError("invalid serving capacity")
        if (pages - 1) * page_size < max_seq:
            raise ValueError("KV slab must fit at least one full model context plus padding page")
        if device.type == "cpu" and backend != "reference":
            raise ValueError("CPU serving requires the explicit reference attention backend")
        storage_spec = MHAStorageSpec(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads_local=model_config.num_attention_heads,
            head_dim=model_config.head_dim,
            page_size=page_size,
            capacity_pages=pages,
            dtype=str(dtype).removeprefix("torch."),
        )
        budget = storage_spec.aligned_total_bytes(256) + (1 << 20)
        ledger = MemoryLedger.for_device(
            device_budget_bytes=budget,
            device_total_bytes=budget,
            host_pageable_bytes=budget,
            device_index=device.index or 0,
        )
        self.tokenizer = self.kv = self.storage = self.engine = self.service = None
        self._closed = False
        runner = None
        try:
            self.storage = materialize_kv_storage(
                storage_spec,
                ledger=ledger,
                label="serving.kv",
                device=str(device),
                zero_initialize=True,
            )
            self.manager = RuntimeMemoryManager(
                total_pages=pages,
                page_size=page_size,
                max_sequences=max_requests,
                max_sequence_tokens=max_seq,
                storage=self.storage.storage,
            )
            self.kv = LogicalKVManager(self.manager, {"default": self.storage})
            group = AttentionGroupSpec(
                0,
                tuple(range(model_config.num_hidden_layers)),
                AttentionType.FULL,
                AttentionSpec(
                    model_config.num_attention_heads,
                    model_config.num_attention_heads,
                    model_config.head_dim,
                    model_config.head_dim,
                    model_config.scaling,
                ),
                page_size,
            )
            sampling = SamplingCoordinator(
                max_requests, device=device, vocab_size=model_config.vocab_size
            )
            runner = PagedModelRunner(
                sampling,
                model,
                self.kv,
                {"default": group},
                backend=backend,
                force_reference=device.type == "cpu",
            )
            self.tokenizer = TokenizerService(
                TokenizerConfig(tokenizer_path, encode_pool_workers=tokenizer_workers),
                max_model_len=max_seq,
                model_vocab_size=model_config.vocab_size,
            )
            assert self.tokenizer.tokenizer is not None
            runner.set_valid_token_ids(self.tokenizer.tokenizer.get_vocab().values())
            constraints = None
            if self.config.structured_outputs:
                import xgrammar as xgr
                from transformers import AutoTokenizer

                raw = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)
                constraints = GrammarConstraints(
                    xgr.TokenizerInfo.from_huggingface(raw, vocab_size=model_config.vocab_size)
                )
            execution = ExecutionPlan(
                "serving",
                self.config.model,
                model_revision,
                weights_revision,
                ComputePlan(
                    dtype=DType.from_str(str(dtype).removeprefix("torch.")),
                    kv_dtype=DType.from_str(str(dtype).removeprefix("torch.")),
                    layer_range=(0, model_config.num_hidden_layers),
                    max_num_batched_tokens=batch_tokens,
                    enable_chunked_prefill=True,
                ),
            )
            plan = SchedulerConfig(
                preemption_mode=PreemptionMode.RECOMPUTE,
                max_num_seqs=max_requests,
                max_num_requests=max_requests,
                max_num_batched_tokens=batch_tokens,
                max_prefill_chunk_tokens=prefill_chunk,
            ).resolve(
                max_seq,
                execution=execution,
                memory=ledger.snapshot(),
                workspace=MemoryPlan(),
                capabilities=SchedulerCapabilities(
                    max_num_seqs=max_requests,
                    chunked_prefill=True,
                    prefix_cache=True,
                    recompute_preemption=True,
                ),
            )
            self.engine = ResidentKVEngine(
                plan,
                self.kv,
                runner,
                sampling=sampling,
                output=OutputProcessor(self.tokenizer, eos_token_ids=self.tokenizer.eos_token_ids),
                prefix_context=lambda _: PrefixCacheContext(
                    self.config.model,
                    model_revision=f"{model_revision}:{weights_revision}",
                    cache_dtype=storage_spec.dtype,
                ),
            )
            self.processor = RequestProcessor(self.tokenizer, self.config)
            self.service = ServingService(self.engine, self.config, constraints=constraints)
        except BaseException:
            if self.engine is not None:
                result = self.engine.close()
                if not result.closed:
                    raise RuntimeError(
                        "initialization failed with engine work still in flight"
                    ) from None
            elif runner is not None:
                runner.close()
            if self.tokenizer is not None:
                self.tokenizer.close()
            if self.kv is not None:
                self.kv.close()
            if self.storage is not None:
                self.storage.close()
            raise

    def app(self):
        if self.service is None or self._closed:
            raise RuntimeError("runtime is not available")
        return create_app(self.service, self.processor, close=self.close)

    def close(self, timeout=10.0) -> bool:
        if self._closed:
            return True
        if self.service is not None and not self.service.close(timeout):
            return False
        if self.kv is not None:
            self.manager.clear_prefix_cache()
            self.manager.reclaim_deferred()
            self.kv.close()
        if self.storage is not None:
            self.storage.close()
        if self.tokenizer is not None:
            self.tokenizer.close()
        self._closed = True
        return True
