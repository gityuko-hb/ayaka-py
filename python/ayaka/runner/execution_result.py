"""Forward payload contract; completion remains the worker fence's authority."""

from dataclasses import dataclass
from enum import StrEnum

import torch

from ayaka.sampling.logprobs import PromptLogprobSliceReport


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Logits for one step plus the reporting metadata that produced them.

    ``logits`` rows follow the packed order: sampling rows first, then
    prompt-scored rows. A graph runner returns logits whose projection already
    happened inside the captured graph, so the base class must not re-project.
    """

    logits: torch.Tensor
    sampling_count: int
    prompt_descriptors: tuple[PromptLogprobSliceReport, ...] = ()
    prompt_targets: tuple[int, ...] = ()
    prompt_ks: tuple[int, ...] = ()


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

    forward: ForwardResult
    lifetime: OutputLifetime
    slot_index: int | None
    graph_bucket: int | None = None
