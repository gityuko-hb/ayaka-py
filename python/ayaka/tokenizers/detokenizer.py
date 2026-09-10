"""Incremental detokenizers for streaming LLM output.

This module provides an abstract base class and two concrete subclasses for
converting a live stream of token IDs — emitted one-by-one by the sampler —
into human-readable text with minimal latency.  A third implementation,
:class:`~ayaka.tokenizers.byte_detok.ByteIncrementalDetokenizer`, lives in
``byte_detok.py`` to avoid a circular import.

Architecture
------------
::

    IncrementalDetokenizer (ABC)          ← this module
    ├── ByteIncrementalDetokenizer        ← byte_detok.py  (fastest)
    ├── FastIncrementalDetokenizer        ← this module    (Rust DecodeStream)
    └── SlowIncrementalDetokenizer        ← this module    (window-diff; universal)

Factory selection
-----------------
:meth:`IncrementalDetokenizer.create` chooses the best available backend in
priority order:

1. **Byte path** — selected when the tokenizer exposes ``token_bytes_table()``
   *and* ``tokenizer.byte_path_verified is True`` (set by
   ``HfTokenizer.verify_byte_path()`` after loading).
2. **Stream path** — selected when the tokenizer provides a Rust-backed
   ``DecodeStream`` (``new_decode_stream()`` returns non-``None``) *and*
   ``params.spaces_between_special_tokens`` is ``True``.
3. **Window-diff path** — universal fallback; works with any
   ``TokenizerLike``.

Stop-string matching is handled by
:class:`~ayaka.tokenizers.stop_checker.StopChecker` transparently across all
backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ayaka.configs.tokenizer import DetokenizeParams, DetokUpdate
from ayaka.tokenizers.ports import TokenizerLike
from ayaka.tokenizers.stop_checker import StopChecker

# Number of prompt-tail tokens used to prime stream / window backends so the
# first generation token decodes in the correct byte context.
_WINDOW = 5

# Minimum number of "extra" tokens kept behind the current read offset in the
# SlowIncrementalDetokenizer window; prevents convert_tokens_to_string from
# producing different output than when the full sequence is given.
_TRIM_SLACK = 64

# How many consecutive stall ticks are tolerated before forcing a flush in the
# window-diff backend (guards against infinite stall on broken tokenizers).
_MAX_STALL = 8


class IncrementalDetokenizer(ABC):
    """Abstract base for streaming token-to-text converters.

    Subclasses implement :meth:`_decode_next` to convert a single token ID to
    its raw text fragment.  The base class handles stop-string matching via
    :class:`~ayaka.tokenizers.stop_checker.StopChecker`, total character
    counting, and optional full-text accumulation.

    Attributes:
        _params: Per-request detokenization parameters.
        _stop: Stop-string checker bound to this request's stop strings.
        _text: Accumulated text fragments; populated only when
            ``params.accumulate_text`` is ``True``.
        _n_chars: Total characters emitted to the client so far.
    """

    __slots__ = ("_params", "_stop", "_text", "_n_chars")

    def __init__(self, params: DetokenizeParams) -> None:
        """Initialise shared detokenizer state.

        Args:
            params: Per-request detokenization parameters (stop strings,
                special-token handling, text accumulation mode).
        """
        self._params = params
        self._stop = StopChecker(params.stop, params.include_stop_str_in_output)
        self._text: list[str] = []
        self._n_chars = 0

    @staticmethod
    def create(
        tokenizer: TokenizerLike,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
        backend: str = "auto",
    ) -> IncrementalDetokenizer:
        """Factory: select and construct the best detokenizer for *tokenizer*.

        Selection priority for ``backend="auto"``:

        1. :class:`~ayaka.tokenizers.byte_detok.ByteIncrementalDetokenizer`
           — when *tokenizer* exposes ``token_bytes_table()`` **and**
           ``tokenizer.byte_path_verified`` is ``True``.
        2. :class:`FastIncrementalDetokenizer` — when ``new_decode_stream()``
           returns a non-``None`` stream **and**
           ``params.spaces_between_special_tokens`` is ``True``.
        3. :class:`SlowIncrementalDetokenizer` — universal fallback.

        Args:
            tokenizer: The tokenizer to wrap.
            params: Per-request detokenization parameters.
            prompt_token_ids: Token IDs of the already-processed prompt
                prefix.  Used to prime the stream / window backends so that
                the first generation token decodes in the correct byte context.
            backend: ``"auto"`` (default), ``"bytes"``, ``"stream"``, or
                ``"window"``.  An explicit value raises :exc:`ValueError` if
                the requested backend is unavailable.

        Returns:
            A ready-to-use :class:`IncrementalDetokenizer` instance.

        Raises:
            ValueError: When an explicit *backend* is requested but the
                required tokenizer capability is absent.
        """
        # Backend Selection Hierarchy:
        # 1. Byte path: Lowest latency, zero GIL contention on decode.
        # Requires token_bytes_table() AND byte_path_verified=True.
        if backend in ("auto", "bytes") and hasattr(tokenizer, "token_bytes_table"):
            if getattr(tokenizer, "byte_path_verified", False):
                from ayaka.tokenizers.byte_detok import ByteIncrementalDetokenizer

                return ByteIncrementalDetokenizer(tokenizer, params)
        if backend == "bytes":
            raise ValueError(
                f"backend='bytes' nhưng {type(tokenizer).__name__} không phơi "
                "token_bytes_table() (hoặc tự kiểm lúc load đã trượt)."
            )
        # 2. Rust DecodeStream path: Low latency, handles byte assembly in Rust.
        # Only valid if spaces_between_special_tokens=True (DecodeStream limitation).
        if backend in ("auto", "stream"):
            if params.spaces_between_special_tokens:
                stream = tokenizer.new_decode_stream(
                    skip_special_tokens=params.skip_special_tokens
                )
                if stream is not None:
                    return FastIncrementalDetokenizer(
                        stream, params, prompt_token_ids=prompt_token_ids
                    )
            if backend == "stream":
                raise ValueError(
                    f"backend='stream' nhưng {type(tokenizer).__name__} không "
                    "cung cấp DecodeStream (hoặc params không tương thích)."
                )
        # 3. Window-diff path: Universal fallback for any tokenizer (SentencePiece, etc.).
        return SlowIncrementalDetokenizer(
            tokenizer, params, prompt_token_ids=prompt_token_ids
        )

    def update(self, new_token_ids: Sequence[int]) -> DetokUpdate:
        """Feed new token IDs and return the text produced this tick.

        Calls :meth:`_decode_next` for each token, concatenates the fragments,
        passes the result through the stop checker, and returns a
        :class:`~ayaka.configs.tokenizer.DetokUpdate`.

        The ``stalled`` flag in the result is ``True`` when at least one token
        produced no text *and* the combined output is empty, indicating that
        the detokenizer is waiting for more tokens before it can emit a
        complete Unicode character.

        Args:
            new_token_ids: One or more token IDs to process in sequence.

        Returns:
            A :class:`~ayaka.configs.tokenizer.DetokUpdate` carrying the
            emittable text delta, stop-match result, and stall indicator.
        """
        n = len(new_token_ids)
        if n == 1:
            # Single-token fast path: common during streaming auto-regressive generation.
            raw = self._decode_next(new_token_ids[0])
            stalled = not raw  # no text emitted means token was partial UTF-8 or special
        else:
            # Multi-token batch path: e.g. speculative decoding acceptance or prefill tokens.
            raw_parts: list[str] = []
            stalled = False
            for tid in new_token_ids:
                piece = self._decode_next(tid)
                if piece:
                    raw_parts.append(piece)
                else:
                    stalled = True
            raw = "".join(raw_parts)

        # Feed the decoded raw string into the stop-string detector.
        # If stop strings are matched, feed() truncates at the stop boundary.
        emit, matched = self._stop.feed(raw) if self._stop.has_stops else (raw, None)
        if emit:
            self._n_chars += len(emit)
            if self._params.accumulate_text:
                self._text.append(emit)
        # 'stalled' indicates that input tokens arrived but produced no immediate visible output.
        return DetokUpdate(delta=emit, stop_matched=matched, stalled=stalled and not raw)

    def finish(self) -> DetokUpdate:
        """Finalise the stream and flush any buffered stop-checker state.

        Called once by the scheduler when the request ends (EOS token, stop
        match, or max-token limit).  Releases any characters held in the
        stop-checker's look-back buffer.

        Subclasses may override this method to flush codec state (e.g.
        :class:`~ayaka.tokenizers.byte_detok.ByteIncrementalDetokenizer` must
        also drain the incremental UTF-8 decoder) before calling ``super()``.

        Returns:
            A terminal :class:`~ayaka.configs.tokenizer.DetokUpdate` with any
            remaining look-back text.  ``stop_matched`` is always ``None``
            here because a genuine stop match would have been signalled by an
            earlier :meth:`update` call.
        """
        emit = self._stop.flush() if self._stop.has_stops else ""
        if emit:
            self._n_chars += len(emit)
            if self._params.accumulate_text:
                self._text.append(emit)
        return DetokUpdate(delta=emit)

    @property
    def output_text(self) -> str:
        """Full accumulated output text (requires ``accumulate_text=True``).

        Joins internal text fragments on first access and caches the result as
        a single string for subsequent calls.

        Raises:
            RuntimeError: When ``params.accumulate_text`` was ``False``.  In
                streaming-delta mode only a sliding window is maintained, not
                the full history.

        Returns:
            The complete decoded output produced so far.
        """
        if not self._params.accumulate_text:
            raise RuntimeError(
                "output_text cần accumulate_text=True; ở chế độ stream-delta "
                "state chỉ giữ cửa sổ, không giữ toàn bộ text."
            )
        if len(self._text) > 1:
            self._text = ["".join(self._text)]
        return self._text[0] if self._text else ""

    @property
    def n_chars(self) -> int:
        """Total number of characters forwarded to the client so far."""
        return self._n_chars

    @abstractmethod
    def _decode_next(self, token_id: int) -> str:
        """Decode a single token ID to its raw text fragment.

        Implemented by each concrete backend.  Returns an empty string when
        the token cannot yet produce output (e.g. mid-sequence multi-byte
        character, out-of-range ID, or suppressed special token).

        Args:
            token_id: A single integer token ID from the model.

        Returns:
            Decoded text fragment, possibly empty.
        """
        ...


class FastIncrementalDetokenizer(IncrementalDetokenizer):
    """Detokenizer backed by the Rust ``DecodeStream`` from the ``tokenizers`` library.

    Feeds token IDs one-by-one into a ``DecodeStreamLike`` object obtained via
    ``tokenizer.new_decode_stream()``.  The Rust implementation handles
    byte-level UTF-8 assembly internally and emits text as soon as a character
    boundary is reached, making this the lowest-latency backend that does not
    require a pre-built byte table.

    When the underlying stream raises an error containing ``"invalid prefix"``
    (which can occur after a sampler reset or beam-search rollback), the stream
    is reset via ``stream.reset()`` and the offending token is re-fed.  All
    other errors propagate to the caller.

    Attributes:
        _stream: The active ``DecodeStreamLike`` instance.
        _reset_count: Number of stream resets triggered by invalid-prefix
            errors.  Non-zero values indicate that the sampler produced tokens
            the tokenizer cannot parse in sequence, which may warrant
            investigation.
    """

    __slots__ = ("_stream", "_reset_count")

    def __init__(
        self,
        stream,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
    ) -> None:
        """Initialise the fast stream detokenizer.

        Primes the stream with the last :data:`_WINDOW` tokens of the prompt
        so that the first generation token decodes in the correct byte context.
        Priming errors (e.g. the stream object is brand-new) are silently
        ignored by breaking out of the priming loop.

        Args:
            stream: A ``DecodeStreamLike`` instance (typically a
                ``_DecodeStreamAdapter`` wrapping ``tokenizers.DecodeStream``).
            params: Per-request detokenization parameters.
            prompt_token_ids: Token IDs of the already-processed prompt prefix.
        """
        super().__init__(params)
        self._stream = stream
        self._reset_count = 0
        # Prime the decoder with the trailing prompt tokens so that multi-byte
        # sequences split across prompt/generation boundary decode correctly.
        for tid in prompt_token_ids[-_WINDOW:]:
            try:
                self._stream.step(tid)
            except Exception:
                break

    def _decode_next(self, token_id: int) -> str:
        """Feed one token to the Rust stream and return any decoded text.

        Recovers automatically from ``"invalid prefix"`` errors by resetting
        the stream via ``stream.reset()`` and re-processing the token.  This
        handles the case where the sampler backtracks and resumes from a
        different position.

        Only :exc:`ValueError` and :exc:`RuntimeError` are intercepted (the
        two most common Python wrappers for Rust errors in the ``tokenizers``
        library); all other exception types propagate unchanged.

        Args:
            token_id: A single token ID from the model.

        Returns:
            Decoded text fragment, or an empty string when the stream is
            accumulating bytes for a multi-byte character.
        """
        try:
            out = self._stream.step(token_id)
        except (ValueError, RuntimeError) as e:
            # Rust tokenizers DecodeStream may throw an error if the fed token cannot
            # be a valid continuation of the current prefix state (e.g. branch switch).
            if "invalid prefix" not in str(e).lower():
                raise
            self._reset_count += 1
            # Reset internal stream state and re-attempt stepping the token freshly.
            self._stream = self._stream.reset()
            out = self._stream.step(token_id)
        return out or ""


class SlowIncrementalDetokenizer(IncrementalDetokenizer):
    """Window-diff detokenizer — universal fallback for any ``TokenizerLike``.

    Maintains a sliding window of decoded token strings.  On each step the
    window slice ``[prefix_offset : read_offset]`` and the slice
    ``[prefix_offset : read_offset + 1]`` are both decoded via
    ``convert_tokens_to_string``; the difference is the new text.  This
    matches the technique used by vLLM and other serving frameworks and handles
    all tokenizers, including SentencePiece Unigram models that are not
    chunk-safe.

    The window is trimmed periodically (every :data:`_TRIM_SLACK` tokens) to
    keep memory usage bounded independent of sequence length.

    A *stall* occurs when the new text is not longer than the prefix text or
    ends with U+FFFD (replacement character for incomplete UTF-8).  Up to
    :data:`_MAX_STALL` consecutive stalls are tolerated before a forced flush,
    which guards against infinite stalling on pathological tokenizer output.

    Attributes:
        _tok: The wrapped tokenizer.
        _skip: Whether to strip special tokens from the decoded output.
        _between: Whether to insert a space between adjacent special tokens.
        _win: Sliding window of token strings, indexed relative to ``_base``.
        _base: Absolute token index corresponding to ``_win[0]``.
        _prefix_offset: Absolute index of the last successfully read position.
        _read_offset: Absolute index up to which tokens have been appended.
        _stall: Consecutive stall counter; reset to zero on every non-stall
            step.
        _added_vocab: Token strings that were added after the base vocabulary
            was built.  Non-empty only for slow (pure-Python) tokenizers;
            these tokens must be handled separately from the BPE merge logic.
        _specials: Set of special-token strings to suppress when
            ``skip_special_tokens`` is ``True``.
    """

    __slots__ = (
        "_tok",
        "_skip",
        "_between",
        "_win",
        "_base",
        "_prefix_offset",
        "_read_offset",
        "_stall",
        "_added_vocab",
        "_specials",
    )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
    ) -> None:
        """Initialise the window-diff detokenizer.

        Seeds the sliding window with the last ``_WINDOW + 2`` prompt tokens
        so that the boundary between the prompt and the first generation token
        is decoded correctly.

        Args:
            tokenizer: The ``TokenizerLike`` to use for decoding.
            params: Per-request detokenization parameters.
            prompt_token_ids: Token IDs of the already-processed prompt prefix.
        """
        super().__init__(params)
        self._tok = tokenizer
        self._skip = params.skip_special_tokens
        self._between = params.spaces_between_special_tokens
        self._stall = 0

        # Prime the sliding window with prompt tokens so that token boundaries
        # at the prompt-completion transition are merged correctly by BPE.
        tail = list(prompt_token_ids[-(_WINDOW + 2) :])
        toks = tokenizer.convert_ids_to_tokens(tail, skip_special_tokens=self._skip)
        self._win: list[str] = [t if t is not None else "" for t in toks]
        self._base = 0                        # absolute token index of _win[0]
        self._read_offset = len(self._win)    # absolute index up to which prompt is primed
        self._prefix_offset = max(self._read_offset - _WINDOW, 0)  # look-back boundary for diffing

        # Slow (Python) tokenizers require separate handling of added tokens.
        self._added_vocab = (
            frozenset(tokenizer.get_added_vocab())
            if not tokenizer.is_fast
            else frozenset()
        )
        self._specials = (
            frozenset(tokenizer.all_special_tokens) if self._skip else frozenset()
        )

    def _to_string(self, toks: list[str]) -> str:
        """Merge a window slice of token strings into readable text.

        For fast tokenizers, delegates directly to
        ``tokenizer.convert_tokens_to_string``.  For slow tokenizers, added-
        vocabulary tokens are extracted and handled separately (they must not
        be passed through the byte-merging logic), with optional inter-special-
        token spaces injected.

        Args:
            toks: Token strings from the current window slice.

        Returns:
            The merged text.
        """
        # Fast tokenizer path: C++/Rust backend handles added tokens and byte-merging natively.
        if not self._added_vocab:
            return self._tok.convert_tokens_to_string(toks)

        # Slow tokenizer fallback: Segment tokens by added-vocab boundaries.
        # Added tokens must not be merged with regular BPE byte sequences.
        subs: list[str] = []
        cur: list[str] = []
        for t in toks:
            if t in self._specials:
                continue
            if t in self._added_vocab:
                if cur:
                    subs.append(self._tok.convert_tokens_to_string(cur))
                    cur = []
                subs.append(t)
            else:
                cur.append(t)
        if cur:
            subs.append(self._tok.convert_tokens_to_string(cur))
        # Join segments with or without spaces based on configuration.
        return (" " if self._between else "").join(subs)

    def _decode_next(self, token_id: int) -> str:
        """Decode one token ID using the window-diff algorithm.

        Appends the new token string to the window, re-decodes the prefix
        slice and the prefix-plus-new-token slice, and returns the difference.
        Stalls up to :data:`_MAX_STALL` times when the output does not grow
        (e.g. a multi-byte sequence is incomplete), then forces a flush to
        avoid an infinite stall.

        Args:
            token_id: A single token ID from the model.

        Returns:
            New text characters produced by this token, or an empty string
            when stalled.
        """
        if not (0 <= token_id <= self._tok.max_token_id):
            return ""

        # 1. Convert token ID to token string and append to the sliding window.
        new = self._tok.convert_ids_to_tokens(
            [token_id], skip_special_tokens=self._skip
        )
        if not new:
            return ""
        self._win.append(new[0] if new[0] is not None else "")

        # 2. Compute slice indices relative to the sliding window base (_base).
        n_abs = self._base + len(self._win)
        p = self._prefix_offset - self._base
        r = self._read_offset - self._base

        # 3. Decode the reference prefix slice vs the expanded slice containing the new token.
        prefix_text = self._to_string(self._win[p:r])
        new_text = self._to_string(self._win[p:])

        # 4. Check for stalls:
        # If output length didn't grow or ends with UTF-8 replacement char \ufffd,
        # the token likely contributed an incomplete byte of a multi-byte sequence.
        if len(new_text) <= len(prefix_text) or new_text.endswith("\ufffd"):
            self._stall += 1
            # Stall tolerated up to _MAX_STALL ticks before forcing progression.
            if self._stall <= _MAX_STALL:
                return ""
        self._stall = 0

        # 5. Extract delta text by stripping the prefix portion.
        delta = new_text[len(prefix_text) :]
        # Advance offsets forward.
        self._prefix_offset = self._read_offset
        self._read_offset = n_abs
        # 6. Trim old entries from the window to maintain bounded memory.
        self._trim()
        return delta

    def _trim(self) -> None:
        """Drop old window entries that are no longer needed.

        Retains :data:`_TRIM_SLACK` tokens of look-back behind the current
        prefix offset (sufficient for ``convert_tokens_to_string`` to produce
        correct output at any boundary) and advances ``_base`` accordingly.
        """
        # Trim entries that are older than prefix_offset - _TRIM_SLACK.
        # This keeps the window size bounded regardless of sequence length.
        drop = self._prefix_offset - self._base - _TRIM_SLACK
        if drop > 0:
            del self._win[:drop]
            self._base += drop
