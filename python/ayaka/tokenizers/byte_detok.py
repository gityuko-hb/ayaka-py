"""Byte-table–based incremental detokenizer and verification helpers.

For byte-level BPE tokenizers (GPT-2 family, LLaMA, Qwen…) each token maps to
a fixed sequence of raw bytes.  :class:`ByteIncrementalDetokenizer` exploits
this by looking each token ID up in the pre-built byte table returned by
``HfTokenizer.token_bytes_table()``, feeding the bytes into Python's
incremental UTF-8 codec, and forwarding fully decoded characters to the caller.

This approach is significantly faster than the window-diff
:class:`~ayaka.tokenizers.detokenizer.SlowIncrementalDetokenizer` and avoids
the occasional GIL-contention latency spikes caused by HuggingFace's batch
``decode``.

Public surface:

* :func:`verify_byte_table` — validate that the byte table round-trips
  correctly against the tokenizer's own ``decode`` output, both for a set of
  representative sample strings and for randomly sampled token sequences.
* :class:`ByteIncrementalDetokenizer` — the fast byte-path detokenizer, used
  automatically by
  :meth:`~ayaka.tokenizers.detokenizer.IncrementalDetokenizer.create` when the
  tokenizer passes verification.
"""

from __future__ import annotations

import codecs
from collections.abc import Sequence

from ayaka.configs.tokenizer import DetokenizeParams, DetokUpdate
from ayaka.tokenizers.detokenizer import IncrementalDetokenizer


def verify_byte_table(tokenizer, samples: Sequence[str], *, n_random: int = 512) -> bool:
    """Verify that ``tokenizer.token_bytes_table()`` round-trips correctly.

    For each sample string, the function:

    1. Encodes the string into token IDs (without special tokens).
    2. Reassembles the raw bytes by concatenating table entries for each ID.
    3. Decodes the bytes as UTF-8 (with ``errors="replace"``).
    4. Compares the result against ``tokenizer.decode(ids)``.

    Additionally, ``n_random`` random token sequences are probed to catch
    gaps in the table that the fixed sample strings might not exercise.

    Args:
        tokenizer: A tokenizer that exposes ``token_bytes_table()``,
            ``encode()``, and ``decode()``.
        samples: Representative strings to encode and verify.  Good choices
            cover multi-byte Unicode (e.g. Vietnamese, CJK), emoji (4-byte),
            and mixed whitespace.
        n_random: Number of random token sequences to probe.  A fixed seed
            (``0``) is used so failures are deterministic and reproducible.

    Returns:
        ``True`` if every probe round-trips correctly; ``False`` on the first
        mismatch or if the table is empty (indicating the tokenizer does not
        support the byte path).
    """
    import random

    tbl = tokenizer.token_bytes_table()
    if not tbl:
        # Tokenizer does not support byte table (e.g. not byte-level BPE); fast exit.
        return False

    # Validation closure: concatenates raw byte table entries for the given IDs,
    # decodes the merged byte stream as UTF-8, and compares with the canonical
    # tokenizer.decode() output.
    def ok(ids: list[int]) -> bool:
        got = b"".join(tbl[i] for i in ids if 0 <= i < len(tbl))
        return got.decode("utf-8", errors="replace") == tokenizer.decode(
            ids, skip_special_tokens=False
        )

    # Phase 1: Verify realistic human-language samples (Vietnamese, CJK, emoji, code, whitespace).
    for s in samples:
        if not ok(tokenizer.encode(s, add_special_tokens=False)):
            return False

    # Phase 2: Fuzzing / random sampling.
    # Probe random token sequences with random lengths (1..24) to catch holes,
    # corrupt offsets, or unexpected edge cases in the byte table.
    rng = random.Random(0)
    n = len(tbl)
    for _ in range(n_random):
        ids = [rng.randrange(n) for _ in range(rng.randint(1, 24))]
        if not ok(ids):
            return False
    return True


class ByteIncrementalDetokenizer(IncrementalDetokenizer):
    """Incremental detokenizer using a pre-built byte vocabulary table.

    Rather than calling the tokenizer's ``decode`` method on every step, each
    token ID is looked up in the byte table returned by
    ``tokenizer.token_bytes_table()``.  The raw bytes are fed into Python's
    stateful incremental UTF-8 codec, which emits decoded characters as soon as
    a valid code-point boundary is reached.

    This is the preferred backend when the tokenizer passes
    :func:`verify_byte_table`; it delivers lower and more consistent latency
    than the window-diff fallback.

    Attributes:
        _tbl: Byte-level vocabulary table indexed by token ID; ``tbl[i]`` is
            the raw byte representation of token *i*.
        _dec: Stateful incremental UTF-8 decoder with ``errors="replace"`` so
            that invalid byte sequences do not raise exceptions.
        _skip: Whether to suppress special tokens in the output.
        _special: Frozen set of special-token IDs to suppress (empty when
            ``skip_special_tokens`` is ``False``).
        _n_invalid: Running count of token IDs that fell outside the table
            range.  Used for monitoring.
    """

    __slots__ = (
        "_tbl",
        "_dec",
        "_skip",
        "_special",
        "_n_invalid",
    )

    def __init__(self, tokenizer, params: DetokenizeParams) -> None:
        """Initialise the byte-path detokenizer.

        Args:
            tokenizer: A ``TokenizerLike`` that also satisfies
                ``BytesTokenizerLike`` — i.e. exposes ``token_bytes_table()``
                and ``all_special_ids``.
            params: Per-request detokenization parameters (stop strings,
                special-token handling, text accumulation mode).
        """
        super().__init__(params)
        self._tbl: list[bytes] = tokenizer.token_bytes_table()
        # Incremental UTF-8 decoder buffers incomplete multi-byte sequences
        # until all bytes belonging to a code-point arrive.
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        self._skip = params.skip_special_tokens
        self._special = tokenizer.all_special_ids if params.skip_special_tokens else frozenset()
        self._n_invalid = 0

    def _decode_next(self, token_id: int) -> str:
        """Decode a single token ID to its text fragment via the byte table.

        Looks the token up in ``_tbl``, passes the raw bytes through the
        incremental UTF-8 decoder, and returns whatever the decoder can emit
        immediately.  Returns an empty string when:

        * the token is a special token and ``skip_special_tokens`` is ``True``,
        * the token ID is outside the table range, or
        * the bytes extend an incomplete multi-byte sequence that cannot yet
          be emitted.

        Args:
            token_id: A single integer token ID produced by the model.

        Returns:
            Decoded text fragment, possibly empty.
        """
        # Filter out special tokens (BOS, EOS, PAD) if requested.
        if self._skip and token_id in self._special:
            return ""
        # Bounds check: out-of-range token IDs are tracked and ignored.
        if not (0 <= token_id < len(self._tbl)):
            self._n_invalid += 1
            return ""
        # Lookup pre-computed raw bytes from the table.
        b = self._tbl[token_id]
        # Feed bytes to the incremental decoder:
        # If 'b' completes a UTF-8 character, decode() returns that character.
        # If 'b' is a partial multi-byte sequence (e.g. first 2 bytes of a 4-byte emoji),
        # decode() stores it in internal buffer and returns "".
        return self._dec.decode(b) if b else ""

    def finish(self) -> DetokUpdate:
        """Flush the UTF-8 decoder and finalise the stream.

        Forces the incremental codec to emit any bytes accumulated for an
        incomplete multi-byte sequence (replacing them with U+FFFD), passes
        the tail through the stop checker, and returns a terminal
        :class:`~ayaka.configs.tokenizer.DetokUpdate`.

        Unlike the base-class :meth:`finish`, this override runs the UTF-8
        flush *before* draining the stop-checker buffer so that partial
        multi-byte sequences at the very end of the stream are handled
        correctly.

        Returns:
            A :class:`~ayaka.configs.tokenizer.DetokUpdate` with any remaining
            tail text and the stop-match result.
        """
        # final=True forces the incremental decoder to flush any buffered bytes,
        # converting incomplete multi-byte sequences into replacement chars (\ufffd).
        tail = self._dec.decode(b"", final=True)
        if tail:
            # If any trailing characters were emitted by final flush, feed them
            # through the stop checker to ensure no stop strings are violated.
            emit, matched = (
                self._stop.feed(tail) if self._stop.has_stops else (tail, None)
            )
            if emit:
                self._n_chars += len(emit)
                if self._params.accumulate_text:
                    self._text.append(emit)
            # Return update containing the flushed tail text and stop match if detected.
            return DetokUpdate(delta=emit or "", stop_matched=matched)
        # No trailing decoder bytes; fall back to base-class stop checker flush.
        return super().finish()

    @property
    def n_invalid_ids(self) -> int:
        """Number of out-of-range token IDs received since construction.

        Non-zero values typically indicate that the model produced tokens
        beyond ``max_token_id`` (e.g. speculative-decoding artefacts or a
        misconfigured tokenizer).  Exposed for monitoring and debugging.
        """
        return self._n_invalid
