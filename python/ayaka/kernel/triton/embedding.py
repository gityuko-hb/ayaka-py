"""Fused vocabulary-parallel embedding lookup: mask, gather and zero in one pass.

One program handles one token and one block of embedding columns. Out-of-shard
tokens (including the padded base rows and padding between the base and added
embeddings) load from local row zero and are then zeroed, so the caller's group
all-reduce reconstructs the full embedding. The eager reference is the
authoritative semantics: it shares ``masked_vocab_input`` with the layer's CPU
path, so the two cannot drift.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.caps import Cap
from ayaka.kernel.ops import custom_op
from ayaka.utils.torch_utils import compute_torch_dtypes
from ayaka.utils.validation import require_int

_SUPPORTED_DTYPES = compute_torch_dtypes()
_MAX_BLOCK_D = 256
_NUM_WARPS = 4


@triton.jit
def _vocab_parallel_embedding_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(axis=0).to(tl.int64)
    token_id = tl.load(input_ptr + token).to(tl.int64)

    org_mask = (token_id >= org_vocab_start_index) & (token_id < org_vocab_end_index)
    added_mask = (token_id >= added_vocab_start_index) & (token_id < added_vocab_end_index)
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = tl.where(org_mask, org_vocab_start_index, 0) + tl.where(
        added_mask, added_offset, 0
    )
    valid = org_mask | added_mask
    local = (token_id - valid_offset).to(tl.int64)

    cols = tl.program_id(axis=1) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < hidden
    row = tl.load(
        weight_ptr + local * hidden + cols,
        mask=col_mask & valid,
        other=0.0,
    )
    row = tl.where(valid, row, 0.0)
    tl.store(output_ptr + token * hidden + cols, row, mask=col_mask)


def _reference(
    input_: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    from ayaka.layers.embedding import masked_vocab_input

    masked, invalid = masked_vocab_input(
        input_,
        org_vocab_start_index=org_vocab_start_index,
        org_vocab_end_index=org_vocab_end_index,
        num_org_vocab_padding=num_org_vocab_padding,
        added_vocab_start_index=added_vocab_start_index,
        added_vocab_end_index=added_vocab_end_index,
    )
    output = torch.nn.functional.embedding(masked.long(), weight)
    return output.masked_fill(invalid.unsqueeze(-1), 0)


def _fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    del (
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    return torch.empty((input_.numel(), weight.shape[1]), dtype=weight.dtype, device=weight.device)


def _validate(
    input_: torch.Tensor,
    weight: torch.Tensor,
    *indices: int,
) -> None:
    if not isinstance(input_, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("input and weight must be torch.Tensors")
    if input_.dtype not in (torch.int32, torch.int64):
        raise TypeError("input ids must be int32 or int64")
    if input_.ndim != 1 or not input_.is_contiguous():
        raise ValueError("input ids must be a contiguous 1-D tensor")
    if weight.ndim != 2 or weight.dtype not in _SUPPORTED_DTYPES or weight.stride(1) != 1:
        raise ValueError("weight must be a row-contiguous 2-D floating tensor")
    if input_.device != weight.device:
        raise ValueError("input and weight must share a device")
    for value in indices:
        require_int(value, "vocabulary index")


@custom_op(
    namespace="ayaka",
    name="vocab_parallel_embedding",
    fake_impl=_fake,
    reference=_reference,
    dispatch_key="CUDA",
    caps=Cap.CUDAGRAPH_SAFE,
)
def vocab_parallel_embedding(
    input_: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    """Gather this rank's owned token rows, zeroing every out-of-shard token."""
    _validate(
        input_,
        weight,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    num_tokens, hidden = input_.numel(), weight.shape[1]
    output = torch.empty((num_tokens, hidden), dtype=weight.dtype, device=weight.device)
    if num_tokens == 0:
        return output
    block_d = min(_MAX_BLOCK_D, triton.next_power_of_2(hidden))
    grid = (num_tokens, triton.cdiv(hidden, block_d))
    with torch.cuda.device(weight.device):
        cast(Any, _vocab_parallel_embedding_kernel)[grid](
            input_,
            weight,
            output,
            hidden,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            BLOCK_D=block_d,
            num_warps=_NUM_WARPS,
        )
    return output


@triton.jit
def _simple_embedding_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    token_id = tl.load(input_ptr + token).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < hidden
    row = tl.load(weight_ptr + token_id * hidden + cols, mask=col_mask)
    tl.store(output_ptr + token * hidden + cols, row, mask=col_mask)


def _simple_reference(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.embedding(input_.long(), weight)


def _simple_fake(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty((input_.numel(), weight.shape[1]), dtype=weight.dtype, device=weight.device)


def _simple_validate(input_: torch.Tensor, weight: torch.Tensor) -> None:
    if not isinstance(input_, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("input and weight must be torch.Tensors")
    if input_.dtype not in (torch.int32, torch.int64):
        raise TypeError("input ids must be int32 or int64")
    if input_.ndim != 1 or not input_.is_contiguous():
        raise ValueError("input ids must be a contiguous 1-D tensor")
    if weight.ndim != 2 or weight.dtype not in _SUPPORTED_DTYPES or weight.stride(1) != 1:
        raise ValueError("weight must be a row-contiguous 2-D floating tensor")
    if input_.device != weight.device:
        raise ValueError("input and weight must share a device")


@custom_op(
    namespace="ayaka",
    name="embedding_lookup",
    fake_impl=_simple_fake,
    reference=_simple_reference,
    dispatch_key="CUDA",
    caps=Cap.CUDAGRAPH_SAFE,
)
def embedding_lookup(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Gather token rows directly without sharding or padding logic."""
    _simple_validate(input_, weight)
    num_tokens, hidden = input_.numel(), weight.shape[1]
    output = torch.empty((num_tokens, hidden), dtype=weight.dtype, device=weight.device)
    if num_tokens == 0:
        return output
    block_d = min(_MAX_BLOCK_D, triton.next_power_of_2(hidden))
    grid = (num_tokens, triton.cdiv(hidden, block_d))
    with torch.cuda.device(weight.device):
        cast(Any, _simple_embedding_kernel)[grid](
            input_,
            weight,
            output,
            hidden,
            BLOCK_D=block_d,
            num_warps=_NUM_WARPS,
        )
    return output
