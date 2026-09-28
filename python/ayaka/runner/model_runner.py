"""Canonical EP0 facade, preserving PagedModelRunner's sampling and ownership API."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from enum import StrEnum

import torch

from ayaka.attention.spec import AttentionGroupSpec
from ayaka.configs.phase import PhaseConfig
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.model_loader.readiness import (
    adopt_model_weights,
    require_ready_weights,
    weight_binding,
)
from ayaka.plan import GraphMode
from ayaka.runner.buffers import RunnerBuffers
from ayaka.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from ayaka.runner.eager_runner import EagerRunner
from ayaka.runner.execution_batch import ExecutionBatch
from ayaka.runner.execution_result import ForwardResult
from ayaka.runner.graph.pool import DecodeGraphConfig
from ayaka.runner.input_buffers import TensorBinding
from ayaka.runner.paged_runner import PagedModelRunner
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sched.plan import PreparedStep
from ayaka.worker.base import WorkerStep


class BootstrapStage(StrEnum):
    MODEL_LOADED = "model_loaded"
    WEIGHTS_FINALIZED = "weights_finalized"
    RESOURCES_BOUND = "resources_bound"
    CAPTURE_READY = "capture_ready"
    WORKER_READY = "worker_ready"
    FAILED = "failed"
    CLOSED = "closed"


class ModelRunner(PagedModelRunner):
    """Route one prepared batch through typed eager/decode phase adapters.

    Inherits the compatibility surface rather than wrapping another stateful
    paged runner. KV, buffers, graph pool and sampling each retain one owner.
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
        max_model_len: int | None = None,
        lora=None,
    ) -> None:
        self.bootstrap_stage = BootstrapStage.MODEL_LOADED
        adopt_model_weights(model)
        self.bootstrap_stage = BootstrapStage.WEIGHTS_FINALIZED
        super().__init__(
            coordinator,
            model,
            kv,
            groups,
            backend=backend,
            force_reference=force_reference,
            buffers=buffers,
            max_model_len=max_model_len,
        )
        self.bootstrap_stage = BootstrapStage.RESOURCES_BOUND
        self.eager_runner = EagerRunner(self)
        self.decode_runner = DecodeCudaGraphRunner(self)
        self.prefill_runner = None
        self.decode_max_padding_ratio = float("inf")
        self.lora = lora
        if lora is not None and buffers is not None:
            if lora.working_bytes(buffers.spec.max_num_batched_tokens) > lora.config.memory_bytes:
                raise MemoryError("LoRA working set exceeds memory reservation")
        self.speculative_runner = None
        self.plain_greedy = False
        self._adapter_leases = {}
        self._lora_rows = (
            []
            if lora is None or buffers is None
            else [
                torch.zeros(
                    buffers.spec.max_num_batched_tokens, dtype=torch.long, device=self.device
                )
                for _ in range(buffers.spec.max_inflight)
            ]
        )

    def prepare_request(self, request) -> None:
        if self.speculative_runner is not None or self.plain_greedy:
            from ayaka.runner.speculative_runner import require_plain_greedy

            require_plain_greedy(request)
            if request.adapter is not None:
                raise ValueError("speculative x LoRA is not certified")
            if (
                self.speculative_runner is not None
                and request.max_total_len
                > self.speculative_runner.draft.model.config.max_position_embeddings
            ):
                raise ValueError("request exceeds the draft model context")
        lease = None
        if request.adapter is not None:
            if self.lora is None:
                raise ValueError("LoRA is disabled")
            lease = self.lora.acquire(request.adapter)
        retained = False
        try:
            if self.lora is not None:
                self.lora.retain_request()
                retained = True
            super().prepare_request(request)
        except BaseException:
            if lease is not None:
                lease.close()
            if retained and self.lora is not None:
                self.lora.release_request()
            raise
        if lease is not None:
            self._adapter_leases[str(request.request_id)] = lease

    def forget_request(self, request_id: str) -> None:
        registered = request_id in self._request_ir
        super().forget_request(request_id)
        lease = self._adapter_leases.pop(request_id, None)
        if lease is not None:
            lease.close()
        if self.speculative_runner is not None:
            self.speculative_runner.forget(request_id)
        if registered and self.lora is not None:
            self.lora.release_request()

    def enable_speculative(self, draft, config) -> None:
        from ayaka.runner.speculative_runner import SpeculativeRunner

        if self.lora is not None or self.speculative_runner is not None:
            raise ValueError("incompatible or already enabled speculative runner")
        if self.buffers is None or self.buffers.live:
            raise RuntimeError("speculative bootstrap requires drained static buffers")
        self.speculative_runner = SpeculativeRunner(self, draft, config)

    def _stage_lora(self, prepared: PreparedStep) -> None:
        if self.lora is None:
            return
        if prepared.buffers is None:
            raise ValueError("LoRA requires a ticket-owned flight buffer")
        values = []
        for scheduled in prepared.step.slices:
            if scheduled.request_id not in self._request_ir:
                raise ValueError("LoRA request was not admitted under an adapter lease")
            lease = self._adapter_leases.get(scheduled.request_id)
            if lease is not None:
                lease.validate()
            values.extend([0 if lease is None else lease.binding.slot] * scheduled.query_count)
        rows = self._lora_rows[prepared.buffers.index]
        rows.zero_()
        rows[: len(values)].copy_(torch.tensor(values, dtype=torch.long, device=self.device))

    def model_variant_context(self, tokens):
        if self.lora is None:
            return nullcontext()
        if self.buffers is None:
            raise ValueError("LoRA requires static buffers")
        index = next(
            (
                i
                for i, pointers in enumerate(self.buffers.pointers())
                if pointers[0] == tokens.data_ptr()
            ),
            None,
        )
        if index is None:
            raise ValueError("LoRA inputs are outside the flight backing")
        return self.lora.select(self._lora_rows[index][: tokens.numel()])

    def _model_forward(self, tokens, positions, metadata):
        forward = super()._model_forward(tokens, positions, metadata)

        def run():
            with self.model_variant_context(tokens):
                return forward()

        return run

    def _execute_eager_inputs(self, prepared, inputs):
        with self.model_variant_context(inputs.tokens):
            return super()._execute_eager_inputs(prepared, inputs)

    def enable_prefill_graph(self, config: PhaseConfig) -> None:
        from ayaka.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner

        if self.prefill_runner is not None:
            raise RuntimeError("prefill graphs already enabled; drain and rebuild")
        self.prefill_runner = PrefillCudaGraphRunner(self, config)
        try:
            self.prefill_runner.capture()
        except BaseException:
            self.bootstrap_stage = BootstrapStage.FAILED
            raise

    def on_workspace_growth(self) -> None:
        super().on_workspace_growth()
        if self.prefill_runner is not None:
            self.prefill_runner.invalidated = True

    def enable_graph(self, config: DecodeGraphConfig, *, stream=None) -> int:
        if self.bootstrap_stage is BootstrapStage.FAILED:
            raise RuntimeError("failed bootstrap requires a new execution owner")
        require_ready_weights(self._model)
        self.bootstrap_stage = BootstrapStage.CAPTURE_READY
        try:
            return super().enable_graph(config, stream=stream)
        except BaseException:
            self.bootstrap_stage = BootstrapStage.FAILED
            raise

    def _graph_binding_digest(self) -> str:
        """Bind model, KV backing, static inputs and backend workspaces together."""
        proof = getattr(self._model, "_ayaka_weight_readiness", None)
        parts = [
            "output=logits",
            repr(self._graph_config),
            str(getattr(proof, "revision", None)),
            weight_binding(self._model),
            super()._graph_binding_digest(),
        ]
        if self.lora is not None:
            parts.extend(
                (
                    repr(self.lora.storage_identity),
                    repr(tuple(row.data_ptr() for row in self._lora_rows)),
                )
            )
        if self.buffers is not None:
            parts.append(repr(self.buffers.pointers()))
        for name, backend in self.backends.items():
            cache = backend.kv_cache
            parts.append(repr((name, id(backend), id(self.builders[name]))))
            for layer in range(cache.num_layers):
                parts.append(repr(TensorBinding.of(cache.key_cache(layer))))
                parts.append(repr(TensorBinding.of(cache.value_cache(layer))))
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]

    def graph_unavailable_reason(self) -> str | None:
        reason = super().graph_unavailable_reason()
        if reason is not None:
            return reason
        if self._backend_name != "triton":
            return "EP0 decode graphs are qualified for Triton attention only"
        return None

    def _forward_tail(self, prepared: PreparedStep) -> ForwardResult:
        if self.kv is None:
            raise RuntimeError("execution runner is closed")
        self._stage_lora(prepared)
        if self.speculative_runner is not None and prepared.step.is_pure_decode:
            return self.speculative_runner.execute(prepared)
        batch = ExecutionBatch.from_prepared(prepared, generation=self.kv.generation)
        phase = self.eager_runner
        if self.prefill_runner is not None and not prepared.step.is_pure_decode:
            support = self.prefill_runner.supports(batch)
            if support.supported:
                phase = self.prefill_runner
            else:
                self.prefill_runner.fallbacks[str(support.reason)] += 1
        pool = self._graph_pool
        lease = prepared.buffers
        # The canonical planner records fallback exactly once, before staging.
        if (
            pool is not None
            and lease is not None
            and prepared.step.graph.mode is GraphMode.REPLAY
            and pool.should_replay(prepared.step, lease.index)
            and self.decode_runner.supports(batch).supported
            and prepared.step.graph.bucket / batch.token_count <= self.decode_max_padding_ratio
        ):
            phase = self.decode_runner
        invocation = phase.load_batch(batch)
        return phase.execute(invocation).forward

    def prepare_execution(self, step: WorkerStep) -> ExecutionBatch:
        """Worker adapter validates before COW or any other mutating enqueue."""
        batch = ExecutionBatch.from_worker(step)
        self._validate_prepared(batch.prepared)
        if self.lora is not None:
            adapters = []
            for scheduled in batch.prepared.step.slices:
                if scheduled.request_id not in self._request_ir:
                    raise ValueError("LoRA request was not admitted")
                lease = self._adapter_leases.get(scheduled.request_id)
                if lease is not None:
                    lease.validate()
                adapters.append(None if lease is None else lease.binding)
            batch = replace(batch, variant=self.lora.variant, adapters=tuple(adapters))
        if self.speculative_runner is not None:
            self.speculative_runner.validate_binding()
        return batch

    def execute_batch(self, batch: ExecutionBatch):
        batch.validate()
        if self.lora is not None:
            if batch.variant != self.lora.variant:
                raise ValueError("execution variant changed before enqueue")
            for scheduled, binding in zip(batch.prepared.step.slices, batch.adapters, strict=True):
                lease = self._adapter_leases.get(scheduled.request_id)
                if (None if lease is None else lease.binding) != binding:
                    raise ValueError("adapter request binding changed before enqueue")
                if binding is not None:
                    self.lora.validate(binding)
        if self.plain_greedy:
            from ayaka.executor.ticket import SampleOutputs

            forward = self._forward_tail(batch.prepared)
            if not forward.sampling_count:
                return SampleOutputs(torch.empty(0, dtype=torch.long, device=self.device))
            self._apply_bans(batch.prepared, forward.logits, forward.sampling_count)
            return SampleOutputs(forward.logits.argmax(-1))
        return self(batch.prepared)

    def mark_worker_ready(self) -> None:
        if self.bootstrap_stage in (BootstrapStage.FAILED, BootstrapStage.CLOSED):
            raise RuntimeError("failed/closed execution bootstrap cannot become ready")
        self.bootstrap_stage = BootstrapStage.WORKER_READY

    def close(self) -> None:
        if self._closed:
            return
        if self.buffers is not None and self.buffers.live:
            raise RuntimeError("cannot close execution while flight leases remain live")
        if self.prefill_runner is not None:
            self.prefill_runner.close()
        if self.speculative_runner is not None:
            self.speculative_runner.close()
        super().close()
        self.eager_runner.close()
        self.decode_runner.close()
        self.bootstrap_stage = BootstrapStage.CLOSED
