"""Execution contracts over the canonical worker, KV and buffer owners.

Import concrete runners from ``model_runner`` or ``runner_factory``. Importing
these contracts never initializes CUDA or requires an optional GPU library.
"""

from ayaka.execution.execution_batch import ExecutionBatch
from ayaka.execution.execution_result import ExecutionResult, OutputLifetime
from ayaka.execution.forward_context import ForwardContext, forward_context, get_forward_context

__all__ = [
    "ExecutionBatch",
    "ExecutionResult",
    "ForwardContext",
    "OutputLifetime",
    "forward_context",
    "get_forward_context",
]
