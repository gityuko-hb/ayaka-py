"""Legacy newline splitting, gated by an explicit adapter equivalence contract.

A newline is only a candidate boundary: normalization, added tokens and BPE
merges can cross it. HF does not advertise chunk support. Special-token
processing always stays with the adapter's whole-prompt encode method.
"""

from __future__ import annotations

from ayaka.tokenizers.ports import TokenizerLike


def safe_split_points(text: str, target_chars: int) -> list[int]:
    """Return candidate newline offsets; the legacy name is not a safety proof."""
    if target_chars <= 0 or len(text) <= target_chars:
        return []
    points: list[int] = []
    last = 0
    while last + target_chars < len(text):
        probe = last + target_chars
        cut = -1
        while probe < len(text):
            newline = text.find("\n", probe)
            if newline == -1 or newline + 1 >= len(text):
                break
            if not text[newline + 1].isspace():
                cut = newline + 1
                break
            probe = newline + 1
        if cut == -1:
            break
        points.append(cut)
        last = cut
    return points


def split_into_chunks(text: str, target_chars: int) -> list[str]:
    """Split at candidate offsets, preserving every character exactly once."""
    points = [0, *safe_split_points(text, target_chars), len(text)]
    return [text[start:end] for start, end in zip(points, points[1:], strict=False)]


def encode_chunked(
    tokenizer: TokenizerLike,
    text: str,
    *,
    chunk_chars: int,
    add_special_tokens: bool = True,
) -> tuple[list[int], int]:
    """Encode whole unless an adapter guarantees these splits and specials are off.

    HF always takes the whole-prompt path. Future adapters opting in must prove
    equivalence for their normalizer, pre-tokenizer, merges and added vocabulary.
    Returns IDs and the number of independently encoded chunks.
    """
    if add_special_tokens or not tokenizer.supports_chunked_encode:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens), 1
    chunks = split_into_chunks(text, chunk_chars)
    if len(chunks) == 1:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens), 1
    parts = tokenizer.encode_batch(chunks, add_special_tokens=False)
    if len(parts) != len(chunks):
        raise ValueError("encode_batch cardinality mismatch")
    return [token for part in parts for token in part], len(chunks)
