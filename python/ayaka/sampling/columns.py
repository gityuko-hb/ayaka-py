"""Column definitions and serialization schemas for sampling parameter metadata.

Defines the column layout, supported data types, and mapping logic used to serialize
high-level `SamplingParams` into contiguous host-staged and device-resident tensor
buffers managed by `SamplingMetadata`.
"""

from __future__ import annotations

from typing import Final

import torch

from ayaka.sampling.logprobs import LOGPROBS_DISABLED, MODE_ORDINALS
from ayaka.sampling.params import SamplingParams
from ayaka.sampling.rng import derive_seed as _derive_seed

#: Single-precision floating-point column names.
F32_COLUMNS: Final[tuple[str, ...]] = (
    "temperature",
    "top_p",
    "min_p",
    "rep_penalty",
    "freq_penalty",
    "pres_penalty",
)
#: 32-bit signed integer column names.
I32_COLUMNS: Final[tuple[str, ...]] = (
    "top_k",
    "logprobs_k",
    "prompt_logprobs_k",
    "logprob_mode",
    "return_support",
    "bias_count",
)
#: 64-bit signed integer column names (RNG seed and step offset).
I64_COLUMNS: Final[tuple[str, ...]] = ("seed", "offset")

#: Aggregate sequence of all registered sampling columns.
ALL_COLUMNS: Final[tuple[str, ...]] = F32_COLUMNS + I32_COLUMNS + I64_COLUMNS

#: Mapping from column names to their corresponding PyTorch tensor dtypes.
COLUMN_DTYPES: Final[dict[str, torch.dtype]] = {
    **dict.fromkeys(F32_COLUMNS, torch.float32),
    **dict.fromkeys(I32_COLUMNS, torch.int32),
    **dict.fromkeys(I64_COLUMNS, torch.int64),
}

ColumnValues = dict[str, float | int]

#: Sentinel value representing a disabled top-k filter.
TOP_K_DISABLED: Final[int] = -1


def columns_of(params: SamplingParams, *, request_index: int = 0) -> ColumnValues:
    """Extract and serialize sampling parameter columns for metadata storage.

    Converts high-level `SamplingParams` into primitive float and integer values
    suitable for writing into device/host column buffers. If `params.seed` is unset,
    a deterministic seed is derived from `request_index`.

    Args:
        params: Sampling configuration parameters.
        request_index: Monotonic request counter used for seed derivation if unseeded.

    Returns:
        Dictionary mapping column names to their typed scalar representations.
    """
    # Derive deterministic seed per request index if none was explicitly configured.
    seed = params.seed if params.seed is not None else _derive_seed(request_index)
    return {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "min_p": params.min_p,
        "rep_penalty": params.repetition_penalty,
        "freq_penalty": params.frequency_penalty,
        "pres_penalty": params.presence_penalty,
        "top_k": TOP_K_DISABLED if params.top_k < 0 else params.top_k,
        "logprobs_k": LOGPROBS_DISABLED if params.logprobs is None else params.logprobs,
        "prompt_logprobs_k": (
            LOGPROBS_DISABLED if params.prompt_logprobs is None else params.prompt_logprobs
        ),
        "logprob_mode": MODE_ORDINALS[params.logprob_mode],
        "return_support": 1 if params.return_sampling_support else 0,
        "bias_count": 0 if params.logit_bias is None else len(params.logit_bias),
        "seed": seed,
        "offset": 0,
    }


def column_defaults() -> ColumnValues:
    """Return default column values corresponding to empty `SamplingParams`.

    Returns:
        Dictionary mapping all sampling column names to their default scalar values.
    """
    return columns_of(SamplingParams())


def is_greedy(params: SamplingParams) -> bool:
    """Determine whether the sampling parameters designate a greedy decoding path.

    Args:
        params: Sampling configuration parameters.

    Returns:
        True if temperature is zero or top_k is 1, False otherwise.
    """
    return params.is_greedy
