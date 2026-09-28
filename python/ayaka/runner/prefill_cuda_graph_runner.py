"""Opt-in ragged prefill graphs over ticket-owned inputs and canonical KV.

Each flight/bucket has isolated metadata and graph scratch. A zero-context
sentinel request absorbs padded tokens; masked stores never touch live KV.
The CSR is packed during staging, so prefix lengths remain runtime data.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ayaka.attention.backend.triton_backend import TritonAttentionMetadata
from ayaka.attention.metadata import CommonAttentionMetadata
from ayaka.configs.phase import PhaseConfig
from ayaka.runner.base_runner import (
    FallbackReason,
    PreparedInvocation,
    RunnerSupport,
    claim_invocation,
    load_invocation,
)
from ayaka.runner.buffers import _Stage1D
from ayaka.runner.execution_batch import ExecutionBatch
from ayaka.runner.execution_result import ExecutionResult, ForwardResult, OutputLifetime
from ayaka.runner.graph.backend import BreakableGraphBackend, FullGraphBackend
from ayaka.runner.graph.graph import GraphBackend, ShapeKey
from ayaka.runner.graph.pool import execution_enqueue_gate
from ayaka.runner.graph.program import compiler_report, model_program
from ayaka.runner.graph.runner import select_bucket
from ayaka.runner.input_buffers import TensorBinding
from ayaka.sched.plan import Phase, PreparedStep
from ayaka.types import ForwardMode

if TYPE_CHECKING:
    from ayaka.runner.paged_runner import PagedModelRunner


class PrefillMetadata:
    """Typed CSR metadata with fixed capacity and a final empty sentinel row."""

    def __init__(self, owner: PagedModelRunner, bucket: int, requests: int, context: int) -> None:
        self.bucket, self.requests, self.context = bucket, requests, context
        self.stages: dict[str, _Stage1D] = {}
        for name, size in (
            ("starts", requests + 2),
            ("lengths", requests + 1),
            ("computed", requests + 1),
            ("slots", bucket),
            ("positions", bucket),
            ("indptr", requests + 2),
            ("indices", requests * context),
            ("mapping", bucket),
        ):
            self.stages[name] = _Stage1D(size, dtype=torch.int32, device=owner.device, pin=True)
        # The CSR is the actual reader addressing; common.block_table is diagnostic.
        self.table = torch.zeros((requests + 1, 1), dtype=torch.int32, device=owner.device)
        tensors = self.stages
        self.metadata = TritonAttentionMetadata(
            common=CommonAttentionMetadata(
                query_start_loc=tensors["starts"].device,
                seq_lens=tensors["lengths"].device,
                computed_lens=tensors["computed"].device,
                block_table=self.table,
                slot_mapping=tensors["slots"].device,
                positions=tensors["positions"].device,
                query_start_loc_cpu=tensors["starts"].host,
                seq_lens_cpu=tensors["lengths"].host,
                num_reqs=requests + 1,
                num_tokens=bucket,
                max_query_len=bucket,
                max_seq_len=context,
                mode=ForwardMode.EXTEND,
            ),
            indptr=tensors["indptr"].device,
            indices=tensors["indices"].device,
            query_to_request=tensors["mapping"].device,
            attn_logits=None,
            attn_lse=None,
            max_context_len_bucket=context,
            decode=False,
            output=None,
        )

    def stage(self, owner: PagedModelRunner, prepared: PreparedStep | None) -> None:
        name = next(iter(owner.backends))
        starts, lengths, computed, slots, positions, indices, mapping, indptr = (
            [0],
            [],
            [],
            [],
            [],
            [],
            [],
            [0],
        )
        if prepared is not None:
            assert prepared.buffers is not None
            columns = prepared.buffers.buffers.spec.columns_for(name)
            tables, slots, _, _ = owner._real_addressing(prepared, name, columns)
            page = owner.backends[name].group.page_size
            for row, (scheduled, table) in enumerate(
                zip(prepared.step.slices, tables, strict=True)
            ):
                lengths.append(scheduled.query_end)
                computed.append(scheduled.query_start)
                starts.append(starts[-1] + scheduled.query_count)
                positions.extend(range(scheduled.query_start, scheduled.query_end))
                mapping.extend([row] * scheduled.query_count)
                indices.extend(
                    table[i // page] * page + i % page for i in range(scheduled.query_end)
                )
                indptr.append(len(indices))
        real = len(positions)
        while len(lengths) < self.requests:
            lengths.append(0)
            computed.append(0)
            starts.append(real)
            indptr.append(len(indices))
        lengths.append(0)
        computed.append(0)
        starts.append(self.bucket)
        indptr.append(len(indices))
        slots.extend([-1] * (self.bucket - real))
        positions.extend([0] * (self.bucket - real))
        mapping.extend([self.requests] * (self.bucket - real))
        for key, values in (
            ("starts", starts),
            ("lengths", lengths),
            ("computed", computed),
            ("slots", slots),
            ("positions", positions),
            ("indptr", indptr),
            ("indices", indices),
            ("mapping", mapping),
        ):
            self.stages[key].fill(values)


@dataclass
class PrefillInstance:
    metadata: PrefillMetadata
    backend: GraphBackend
    program: Callable[[], Any]


class PrefillCudaGraphRunner:
    def __init__(self, owner: PagedModelRunner, config: PhaseConfig) -> None:
        self.owner, self.config = owner, config
        self.instances: dict[tuple[int, int], PrefillInstance] = {}
        self.binding = ""
        self.generation = None
        self.closed = False
        self.invalidated = False
        self.hits = 0
        self.fallbacks: Counter[str] = Counter()
        self.capture_bytes = self.peak_bytes = 0
        self.capture_seconds = 0.0

    def _resource_binding(self) -> str:
        parts = [self.owner._graph_binding_digest(), repr(self.config)]
        for key, instance in sorted(self.instances.items()):
            parts.append(repr(key))
            parts.extend(
                repr(TensorBinding.of(stage.device)) for stage in instance.metadata.stages.values()
            )
            parts.append(repr(TensorBinding.of(instance.metadata.table)))
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()

    def compilation_report(self) -> tuple[dict, ...]:
        return tuple(
            {"slot": slot, "bucket": bucket, **report}
            for (slot, bucket), instance in self.instances.items()
            if (report := compiler_report(instance.program)) is not None
        )

    def capture(self) -> None:
        owner = self.owner
        if (
            owner.device.type != "cuda"
            or owner._backend_name != "triton"
            or len(owner.backends) != 1
        ):
            raise ValueError("prefill graphs require CUDA and one Triton attention group")
        if owner.buffers is None or owner.buffers.live or self.instances or self.closed:
            raise RuntimeError("prefill capture requires a fresh, drained buffer owner")
        from ayaka.model_loader.readiness import require_ready_weights

        require_ready_weights(owner._model)
        with execution_enqueue_gate(), torch.random.fork_rng(devices=[owner.device.index or 0]):
            torch.cuda.synchronize(owner.device)
            before = torch.cuda.memory_reserved(owner.device)
            allocated = torch.cuda.memory_allocated(owner.device)
            torch.cuda.reset_peak_memory_stats(owner.device)
            started = time.monotonic()
            stream = torch.cuda.Stream(device=owner.device)
            assert owner.kv is not None
            self.generation = owner.kv.generation
            owner_binding = owner._graph_binding_digest()
            try:
                owner._acquire_capture_leases()
                with torch.cuda.stream(stream), torch.inference_mode():
                    for slot, lease in owner._graph_capture_leases.items():
                        for bucket in reversed(self.config.buckets):
                            metadata = PrefillMetadata(
                                owner,
                                bucket,
                                self.config.max_requests,
                                owner._max_model_len or owner._model.config.max_position_embeddings,
                            )
                            metadata.stage(owner, None)
                            tokens, positions = lease.buffers.stage_tokens(
                                [0] * bucket, [0] * bucket
                            )
                            backend_type = (
                                FullGraphBackend
                                if self.config.backend == "full"
                                else BreakableGraphBackend
                            )
                            backend = backend_type(device=owner.device)
                            program = model_program(
                                owner,
                                tokens,
                                positions,
                                {next(iter(owner.backends)): metadata.metadata},
                                backend=self.config.backend,
                                compile_seconds=self.config.compile_seconds,
                            )
                            self.instances[slot, bucket] = PrefillInstance(
                                metadata, backend, program
                            )
                            with backend.capture_session(stream):
                                backend.capture_one(ShapeKey(bucket), program)
                torch.cuda.synchronize(owner.device)
                self.capture_bytes = max(
                    torch.cuda.memory_reserved(owner.device) - before,
                    torch.cuda.memory_allocated(owner.device) - allocated,
                    0,
                )
                self.peak_bytes = max(
                    torch.cuda.max_memory_reserved(owner.device) - before,
                    torch.cuda.max_memory_allocated(owner.device) - allocated,
                    0,
                )
                if max(self.capture_bytes, self.peak_bytes) > self.config.memory_bytes:
                    raise MemoryError("prefill capture exceeds its phase memory reservation")
                if (
                    owner_binding != owner._graph_binding_digest()
                    or self.generation != owner.kv.generation
                ):
                    raise RuntimeError("prefill resource binding changed during capture")
                self.capture_seconds = time.monotonic() - started
                self.binding = self._resource_binding()
            except BaseException:
                torch.cuda.synchronize(owner.device)
                owner._release_capture_leases()
                self.close()
                raise
            finally:
                owner._release_capture_leases()

    def supports(self, batch: ExecutionBatch) -> RunnerSupport:
        batch.validate()
        if self.closed:
            raise RuntimeError("prefill runner is closed")
        step = batch.prepared.step
        # EXTEND may mix decode/prefill; never split or reorder that step.
        if not step.slices or any(s.phase is not Phase.PREFILL for s in step.slices):
            return RunnerSupport(FallbackReason.WRONG_PHASE)
        if step.prompt_logprobs or self.owner._graph_eager_only_reason(step):
            return RunnerSupport(FallbackReason.OUTPUT_MODE)
        if len(step.slices) > self.config.max_requests:
            return RunnerSupport(FallbackReason.BUCKET_CEILING)
        bucket = select_bucket(step.num_tokens, self.config.buckets)
        if bucket is None or bucket / step.num_tokens > self.config.max_padding_ratio:
            return RunnerSupport(FallbackReason.BUCKET_CEILING)
        lease = batch.prepared.buffers
        if lease is None or (lease.index, bucket) not in self.instances:
            return RunnerSupport(FallbackReason.SLOT_UNAVAILABLE)
        if (
            self.invalidated
            or batch.generation != self.generation
            or self.binding != self._resource_binding()
        ):
            return RunnerSupport(FallbackReason.STALE_GENERATION)
        return RunnerSupport()

    def load_batch(self, batch: ExecutionBatch) -> PreparedInvocation:
        support = self.supports(batch)
        if not support.supported:
            raise ValueError(f"prefill graph is ineligible: {support.reason}")
        owner = self.owner
        owner._validate_prepared(batch.prepared)
        with load_invocation(owner, self, batch) as invocation:
            lease = batch.prepared.buffers
            assert lease is not None
            step = batch.prepared.step
            bucket = select_bucket(step.num_tokens, self.config.buckets)
            assert bucket is not None
            invocation.binding = self.binding
            lease.buffers.stage_tokens(
                [*step.token_ids, *([0] * (bucket - step.num_tokens))],
                [*step.positions, *([0] * (bucket - step.num_tokens))],
            )
            self.instances[lease.index, bucket].metadata.stage(owner, batch.prepared)
            return invocation

    def execute(self, invocation: PreparedInvocation) -> ExecutionResult:
        claim_invocation(self.owner, self, invocation)
        batch = invocation.batch
        if not self.supports(batch).supported or invocation.binding != self.binding:
            raise RuntimeError("prefill binding changed after load; drain before recovery")
        lease = batch.prepared.buffers
        assert lease is not None
        step = batch.prepared.step
        bucket = select_bucket(step.num_tokens, self.config.buckets)
        assert bucket is not None
        instance = self.instances[lease.index, bucket]
        with torch.inference_mode(), instance.backend.replay_session():
            raw = instance.backend.replay(ShapeKey(bucket), forward_fn=instance.program)
            rows = lease.buffers.stage_sampling_rows(step.sampling_rows)
            logits = raw.index_select(0, rows)
        self.hits += 1
        self.owner._observe_forward(
            step,
            num_tokens=step.num_tokens,
            graph_bucket=bucket,
            graph_key=f"prefill:{self.config.backend}:{bucket}:{lease.index}",
        )
        return ExecutionResult(
            ForwardResult(logits=logits, sampling_count=len(step.sampling_rows)),
            OutputLifetime.OWNED,
            lease.index,
            bucket,
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.owner.buffers is not None and self.owner.buffers.live:
            raise RuntimeError("drain flight leases before closing prefill graphs")
        for instance in self.instances.values():
            instance.backend.cleanup()
        self.instances.clear()
        self.closed = True
