"""Custom operator execution pipeline for logits transformation.

Applies registered custom operators declared in `plan.custom_ops` sequentially
to raw logits prior to subsequent sampling stages, matching the canonical pipeline
order:

    custom ops -> penalties -> mask -> temperature -> stats -> filter/sample

This positioning corresponds to `_preprocess_logits` in SGLang and logit processors
in vLLM, ensuring custom operators execute before any standard penalty, masking,
or sampling transformations.

Operator Execution Contracts:
    - In-place mutation: If `mutates_args` contains `"logits"`, the operator modifies
      the tensor in place and returns `None` (or returns the same tensor instance).
    - Functional transformation: If the operator returns a new tensor, the executor
      rebinds the active logits reference. The returned tensor must strictly preserve
      the original shape, device, and dtype.

All validations fail closed: unregistered operator names or contract violations
(e.g., shape or device mismatch) immediately raise exceptions rather than degrading
silently.
"""

from __future__ import annotations

import torch

from ayaka.plan import SamplingPlan
from ayaka.utils.import_utils import CapabilityError

__all__ = ["apply_custom_ops"]


def apply_custom_ops(logits: torch.Tensor, plan: SamplingPlan) -> torch.Tensor:
    """Execute plan custom operators sequentially on logits.

    Args:
        logits: Float logits tensor of shape `[batch_size, vocab_size]`.
        plan: Sampling plan declaring custom operator names to execute.

    Returns:
        Resulting logits tensor after all custom operators have been applied.

    Raises:
        CapabilityError: If a declared custom operator is not found in the registry.
        ValueError: If a functional operator returns an invalid type or fails to
            preserve tensor shape, device, or dtype.
    """
    from ayaka.kernel.ops import registered_ops

    registry = registered_ops()
    for name in plan.custom_ops:
        handle = registry.get(name)
        # Fail closed if a planned custom op is no longer present in the registry.
        if handle is None:
            raise CapabilityError(
                "sampling_custom_ops",
                detail=(f"custom op {name!r} is no longer registered at execution time"),
                remedy="register the op via ayaka.kernel.ops.custom_op before stepping",
            )
        # Dispatch based on declared mutation contract.
        if "logits" in handle.mutates_args:
            # In-place operator modifying logits directly.
            handle(logits)
        else:
            # Functional operator returning a potentially new tensor.
            out = handle(logits)
            if out is not logits:
                # Validate output type contract.
                if not isinstance(out, torch.Tensor):
                    raise ValueError(
                        f"custom op {name!r} must return a tensor or None, got {type(out).__name__}"
                    )
                # Ensure functional operator preserves shape, device, and dtype invariants.
                if (
                    out.shape != logits.shape
                    or out.device != logits.device
                    or out.dtype != logits.dtype
                ):
                    raise ValueError(
                        f"custom op {name!r} must preserve logits shape/device/dtype "
                        f"(got shape={tuple(out.shape)} device={out.device} "
                        f"dtype={out.dtype})"
                    )
                logits = out
    return logits
