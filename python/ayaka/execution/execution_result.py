"""Model payload lifetime; completion remains the worker fence's authority."""

from dataclasses import dataclass
from enum import StrEnum

from ayaka.runner.model_runner import _ForwardResult


class OutputLifetime(StrEnum):
    OWNED = "owned"
    FLIGHT_LEASE = "flight_lease"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Logits consumed by the existing sampler before the worker records its fence.

    Graph logits are borrowed until that flight retires. Only sampled token
    and requested logprob payloads escape through SampleOutputs; callers that
    retain full logits must copy them on the producing stream before reuse.
    Returning this object never proves device completion.
    """

    forward: _ForwardResult
    lifetime: OutputLifetime
    slot_index: int | None
    graph_bucket: int | None = None
