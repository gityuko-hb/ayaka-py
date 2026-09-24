"""Plain-torch oracles for the embedding kernels."""

from __future__ import annotations

import torch


def vocab_parallel_embedding_ref(
    input_: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    """Masked gather through ``ayaka.layers.embedding.masked_vocab_input``.

    Shares the helper with the layer's CPU path so the two cannot drift.
    """
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


def embedding_lookup_ref(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Plain gather of token rows."""
    return torch.nn.functional.embedding(input_.long(), weight)
