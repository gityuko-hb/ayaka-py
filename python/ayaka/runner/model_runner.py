"""Canonical EP0 facade, preserving PagedModelRunner's sampling and ownership API."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
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
        self.speculative_runner = None
        self.plain_greedy = False

    def prepare_request(self, request) -> None:
        if self.speculative_runner is not None or self.plain_greedy:
            from ayaka.runner.speculative_runner import require_plain_greedy

            require_plain_greedy(request)
            if (
                self.speculative_runner is not None
                and request.max_total_len
                > self.speculative_runner.draft.model.config.max_position_embeddings
            ):
                raise ValueError("request exceeds the draft model context")
        super().prepare_request(request)

    def forget_request(self, request_id: str) -> None:
        super().forget_request(request_id)
        if self.speculative_runner is not None:
            self.speculative_runner.forget(request_id)

    def enable_speculative_decoding(self) -> None:
        """Enable paged verification of scheduler-planned draft rows.

        The dense reference runner and this path are mutually exclusive.
        """
        if self.speculative_runner is not None or self.plain_greedy:
            raise ValueError("speculative decoding is incompatible with this runner's features")
        super().enable_speculative_decoding()

    def enable_speculative(self, draft, config) -> None:
        from ayaka.runner.speculative_runner import SpeculativeRunner

        if self.speculative_runner is not None or self.speculative_decoding:
            raise ValueError("incompatible or already enabled speculative runner")
        if self.buffers is None or self.buffers.live:
            raise RuntimeError("speculative bootstrap requires drained static buffers")
        self.speculative_runner = SpeculativeRunner(self, draft, config)

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
        if self.speculative_runner is not None:
            self.speculative_runner.validate_binding()
        return batch

    def execute_batch(self, batch: ExecutionBatch):
        batch.validate()
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
