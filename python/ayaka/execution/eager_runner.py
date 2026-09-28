"""CPU/CUDA eager adapter over the existing model/logits implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.execution.base_runner import (
    PreparedInvocation,
    RunnerSupport,
    claim_invocation,
    load_invocation,
)
from ayaka.execution.execution_batch import ExecutionBatch
from ayaka.execution.execution_result import ExecutionResult, OutputLifetime

if TYPE_CHECKING:
    from ayaka.runner.paged_runner import PagedModelRunner


class EagerRunner:
    def __init__(self, owner: PagedModelRunner) -> None:
        self.owner = owner
        self.closed = False

    def supports(self, batch: ExecutionBatch) -> RunnerSupport:
        batch.validate()
        if self.closed:
            raise RuntimeError("eager runner is closed")
        return RunnerSupport()

    def load_batch(self, batch: ExecutionBatch) -> PreparedInvocation:
        self.supports(batch)
        self.owner._validate_prepared(batch.prepared)
        with load_invocation(self.owner, self, batch) as invocation:
            invocation.eager_inputs = self.owner._prepare_eager_inputs(batch.prepared)
            return invocation

    def execute(self, invocation: PreparedInvocation) -> ExecutionResult:
        if self.closed:
            raise RuntimeError("eager runner is closed")
        claim_invocation(self.owner, self, invocation)
        self.owner._validate_prepared(invocation.batch.prepared)
        if invocation.eager_inputs is None:
            raise ValueError("eager invocation has no staged inputs")
        result = self.owner._project_forward_result(
            self.owner._execute_eager_inputs(invocation.batch.prepared, invocation.eager_inputs)
        )
        return ExecutionResult(result, OutputLifetime.OWNED, None)

    def close(self) -> None:
        # The canonical owner releases resources after the worker drains.
        self.closed = True
