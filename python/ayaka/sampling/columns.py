from __future__ import annotations

from typing import Final

import torch

from ayaka.sampling.logprobs import LOGPROBS_DISABLED, MODE_ORDINALS
from ayaka.sampling.params import SamplingParams
from ayaka.sampling.rng import derive_seed as _derive_seed

F32_COLUMNS: Final[tuple[str, ...]] = (
    "temperature",
    "top_p",
    "min_p",
    "rep_penalty",
    "freq_penalty",
    "pres_penalty",
)
I32_COLUMNS: Final[tuple[str, ...]] = (
    "top_k",
    "logprobs_k",
    "prompt_logprobs_k",
    "logprob_mode",
)
I64_COLUMNS: Final[tuple[str, ...]] = ("seed", "offset")

ALL_COLUMNS: Final[tuple[str, ...]] = F32_COLUMNS + I32_COLUMNS + I64_COLUMNS

COLUMN_DTYPES: Final[dict[str, torch.dtype]] = {
    **dict.fromkeys(F32_COLUMNS, torch.float32),
    **dict.fromkeys(I32_COLUMNS, torch.int32),
    **dict.fromkeys(I64_COLUMNS, torch.int64),
}

ColumnValues = dict[str, float | int]

TOP_K_DISABLED: Final[int] = -1


def columns_of(params: SamplingParams, *, request_index: int = 0) -> ColumnValues:
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
        "seed": seed,
        "offset": 0,
    }


def column_defaults() -> ColumnValues:
    return columns_of(SamplingParams())


def is_greedy(params: SamplingParams) -> bool:
    return params.is_greedy
