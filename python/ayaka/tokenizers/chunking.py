"""Long-prompt chunking for byte-level BPE tokenizers.

Byte-level BPE tokenizers (GPT-2 family, LLaMA, Qwen, Mistral…) are
*chunk-safe*: encoding ``text[:k] + text[k:]`` always produces the same token
sequence as encoding ``text`` directly, provided the split point lands on a
character boundary that cannot be merged across it by the BPE algorithm.
Newline characters immediately before a non-whitespace character are such
boundaries because BPE merges never span newlines.

This module provides:

* :func:`safe_split_points` — discover newline-based split offsets in a string.
* :func:`split_into_chunks` — slice the string at those offsets.
* :func:`encode_chunked` — encode a long prompt by splitting, encoding each
  chunk independently, stitching the token lists, and injecting BOS/EOS.

.. note::
   SentencePiece Unigram tokenizers are **not** chunk-safe.  Callers must
   check ``TokenizerLike.supports_chunked_encode`` before invoking these
   functions.
"""

from __future__ import annotations

from collections.abc import Sequence


def safe_split_points(text: str, target_chars: int) -> list[int]:
    """Return character offsets where *text* can be safely split for chunked encoding.

    A *safe split point* is a position immediately after a ``'\\n'`` character
    where the next character is not whitespace.  Such a boundary can never be
    the middle of a BPE merge, so encoding the two halves independently
    produces the same token IDs as encoding the whole string.

    The algorithm scans forward from ``last_cut + target_chars`` looking for
    the first qualifying newline.  If none is found before the end of the
    string, no further splits are added.

    Args:
        text: The input string to examine.
        target_chars: Desired minimum chunk size in characters.  Splits are
            only searched at or beyond ``last_cut + target_chars`` from the
            previous split.

    Returns:
        A sorted list of split offsets (indices into *text*).  Empty when the
        string is already shorter than *target_chars* or no qualifying newline
        boundary exists.
    """
    if target_chars <= 0 or len(text) <= target_chars:
        return []   # text already fits in one chunk — nothing to split

    points: list[int] = []
    last = 0        # character index of the last accepted cut point
    n = len(text)

    # Outer loop: advance `last` by at least `target_chars` on each iteration.
    while last + target_chars < n:
        # Start probing for a newline no earlier than `target_chars` ahead.
        probe = last + target_chars
        cut = -1

        # Inner loop: walk forward through newlines until we find one
        # whose successor is a non-whitespace character.
        while probe < n:
            nl = text.find("\n", probe)
            if nl == -1 or nl + 1 >= n:
                # No newline found before end of string — no safe cut ahead.
                break
            if not text[nl + 1].isspace():
                # Character after '\n' is non-whitespace: this is a safe BPE
                # boundary.  BPE merges never cross a '\n' followed by a
                # non-space character, so encoding the two halves independently
                # produces identical token IDs to encoding the whole string.
                cut = nl + 1
                break
            # Next char is whitespace (e.g. indented continuation of a code
            # block).  Splitting here could break indented tokens; skip ahead.
            probe = nl + 1

        if cut == -1:
            # No safe boundary found in the remaining text — stop here.
            break
        points.append(cut)
        last = cut   # next search starts from this cut point
    return points


def split_into_chunks(text: str, target_chars: int) -> list[str]:
    """Split *text* into chunks at safe newline boundaries.

    Delegates split-point discovery to :func:`safe_split_points`.  When no
    safe boundary is found (e.g. the text is a single long line), the whole
    string is returned as a single chunk so the caller can fall back to a
    regular ``encode`` call.

    Args:
        text: The string to split.
        target_chars: Target minimum chunk size; forwarded verbatim to
            :func:`safe_split_points`.

    Returns:
        A list of substrings whose concatenation equals *text*.  The list
        always contains at least one element.
    """
    pts = safe_split_points(text, target_chars)
    if not pts:
        return [text]   # no safe cuts found — treat as one chunk
    out: list[str] = []
    prev = 0
    for p in pts:
        out.append(text[prev:p])   # slice [prev, p) is one chunk
        prev = p
    out.append(text[prev:])        # final chunk: from last cut to end
    return out


def encode_chunked(
    tokenizer,
    text: str,
    *,
    chunk_chars: int,
    add_special_tokens: bool = True,
) -> tuple[list[int], int]:
    """Encode a (potentially long) prompt using chunked encoding.

    When ``tokenizer.supports_chunked_encode`` is ``True`` and the prompt
    exceeds *chunk_chars*, the text is split at safe newline boundaries, each
    chunk is encoded independently **without** special tokens, the token lists
    are concatenated, and BOS/EOS are injected around the result when
    *add_special_tokens* is ``True``.

    When chunking is not applicable — either because the tokenizer does not
    support it or the text fits within a single chunk — the tokenizer's
    ``encode`` method is called directly so its own special-token and
    truncation logic applies unchanged.

    Args:
        tokenizer: A ``TokenizerLike`` instance.
        text: The prompt string to tokenize.
        chunk_chars: Target chunk size in characters.  The actual chunk size
            may exceed this value when no qualifying newline boundary exists
            near the target offset; in that case the remainder is encoded as
            one final chunk.
        add_special_tokens: When ``True``, ``tokenizer.bos_token_id`` is
            prepended and ``tokenizer.eos_token_id`` is appended to the
            stitched token list (mirroring what a plain ``encode`` call with
            ``add_special_tokens=True`` would produce).  When ``False``, no
            special tokens are injected.

    Returns:
        A ``(token_ids, n_chunks)`` tuple:

        * ``token_ids`` — the complete encoded sequence.
        * ``n_chunks`` — number of independent chunks that were encoded.
          Always ``1`` when chunking was not applied.
    """
    # Fast path 1: tokenizer does not support chunked encoding (e.g. Unigram/SentencePiece).
    # Fall back directly to atomic single-pass encode.
    if not tokenizer.supports_chunked_encode:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens), 1

    # Attempt to divide the prompt into chunks at safe newline boundaries.
    chunks = split_into_chunks(text, chunk_chars)
    # Fast path 2: string is short enough or has no safe split boundaries (single chunk).
    if len(chunks) == 1:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens), 1

    # Encode all chunks in parallel/batch without special tokens (BOS/EOS).
    # Special tokens must not be injected in the interior boundaries between chunks!
    parts: Sequence[list[int]] = tokenizer.encode_batch(
        chunks, add_special_tokens=False
    )
    # Stitch the individual chunk token lists into a unified sequence.
    ids: list[int] = []
    for p in parts:
        ids.extend(p)

    # Re-inject BOS token at the very beginning and EOS token at the very end
    # to emulate the exact behavior of a single global encode with add_special_tokens=True.
    if add_special_tokens:
        bos = tokenizer.bos_token_id
        if bos is not None:
            ids.insert(0, bos)
        eos = tokenizer.eos_token_id
        if eos is not None:
            ids.append(eos)
    return ids, len(chunks)
