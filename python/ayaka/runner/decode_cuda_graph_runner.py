"""Serving decode adapter; DecodeGraphPool remains the graph/resource owner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.plan import GraphMode
from ayaka.runner.base_runner import (
    FallbackReason,
    PreparedInvocation,
    RunnerSupport,
    claim_invocation,
    load_invocation,
)
from ayaka.runner.execution_batch import ExecutionBatch
from ayaka.runner.execution_result import ExecutionResult, OutputLifetime

if TYPE_CHECKING:
    from ayaka.runner.paged_runner import PagedModelRunner


class DecodeCudaGraphRunner:
    def __init__(self, owner: PagedModelRunner) -> None:
        self.owner = owner
        self.closed = False

    def supports(self, batch: ExecutionBatch) -> RunnerSupport:
        """Pure eligibility query: no planning, counter mutation or device work."""
        batch.validate()
        if self.closed:
            raise RuntimeError("decode runner is closed")
        step = batch.prepared.step
        pool = self.owner._graph_pool
        if pool is None:
            return RunnerSupport(FallbackReason.DISABLED)
        if not step.is_pure_decode:
            return RunnerSupport(FallbackReason.WRONG_PHASE)
        if step.prompt_logprobs:
            return RunnerSupport(FallbackReason.OUTPUT_MODE)
        if step.num_tokens > pool.buckets[-1]:
            return RunnerSupport(FallbackReason.BUCKET_CEILING)
        if step.graph.mode is not GraphMode.REPLAY:
            return RunnerSupport(FallbackReason.MISSING_GRAPH)
        lease = batch.prepared.buffers
        if lease is None or lease.index not in pool.slots:
            return RunnerSupport(FallbackReason.SLOT_UNAVAILABLE)
        planner = pool.planner
        if planner.invalidated or planner.captured_generation != batch.generation:
            return RunnerSupport(FallbackReason.STALE_GENERATION)
        if planner.captured_binding != self.owner._graph_binding_digest():
            return RunnerSupport(FallbackReason.STALE_GENERATION)
        if not pool.has_instance(step.graph.bucket, lease.index):
            return RunnerSupport(FallbackReason.MISSING_GRAPH)
        return RunnerSupport()

    def load_batch(self, batch: ExecutionBatch) -> PreparedInvocation:
        support = self.supports(batch)
        if not support.supported:
            raise ValueError(f"decode graph is ineligible: {support.reason}")
        self.owner._validate_prepared(batch.prepared)
        lease = batch.prepared.buffers
        assert lease is not None
        with load_invocation(self.owner, self, batch) as invocation:
            invocation.binding = self.owner._graph_binding_digest()
            self.owner._stage_graph_step(batch.prepared, lease, batch.prepared.step.graph.bucket)
            return invocation

    def execute(self, invocation: PreparedInvocation) -> ExecutionResult:
        if self.closed:
            raise RuntimeError("decode runner is closed")
        claim_invocation(self.owner, self, invocation)
        self.owner._validate_prepared(invocation.batch.prepared)
        lease = invocation.batch.prepared.buffers
        assert lease is not None
        if invocation.binding != self.owner._graph_binding_digest():
            raise ValueError("graph binding changed after load_batch; drain before recovery")
        # Already staged on the worker stream. No planning/copy runs inside
        # replay and any error propagates to the whole-ticket drain path.
        result = self.owner._graph_tail(invocation.batch.prepared, lease, staged=True)
        return ExecutionResult(
            result,
            OutputLifetime.FLIGHT_LEASE,
            lease.index,
            invocation.batch.prepared.step.graph.bucket,
        )

    def close(self) -> None:
        self.closed = True
