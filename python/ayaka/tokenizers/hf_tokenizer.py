"""HuggingFace tokenizer adapter for the Ayaka serving engine.

This module provides :class:`HfTokenizer`, a concrete implementation of the
:class:`~ayaka.tokenizers.ports.TokenizerLike` protocol that wraps any
HuggingFace ``PreTrainedTokenizer`` or ``PreTrainedTokenizerFast`` (including
``AutoTokenizer``).  It also implements
:class:`~ayaka.tokenizers.ports.BytesTokenizerLike` for tokenizers whose
pre-tokenizer is byte-level BPE, enabling the fast
:class:`~ayaka.tokenizers.byte_detok.ByteIncrementalDetokenizer` backend.

The companion ``_DecodeStreamAdapter`` bridges HuggingFace's Rust
``tokenizers.decoders.DecodeStream`` to the
:class:`~ayaka.tokenizers.ports.DecodeStreamLike` protocol, so the fast
:class:`~ayaka.tokenizers.detokenizer.FastIncrementalDetokenizer` backend can
use it without knowing about its concrete type.

Supported tokenizer modes
--------------------------
* ``"auto"`` / ``"hf"`` → ``AutoTokenizer.from_pretrained(..., use_fast=True)``
* ``"slow"`` → ``AutoTokenizer.from_pretrained(..., use_fast=False)``
* ``"tiktoken"`` / ``"mistral"`` → not handled here; those require separate
  adapters that also satisfy ``TokenizerLike``.

Byte-path verification
-----------------------
After constructing a :class:`HfTokenizer`, call :meth:`HfTokenizer.verify_byte_path`
to validate that the byte table round-trips correctly.  Only then will
:meth:`~ayaka.tokenizers.detokenizer.IncrementalDetokenizer.create` select the
fast byte backend.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.tokenizers.ports import DecodeStreamLike

# Pre-tokenizer class names that indicate a byte-level BPE tokenizer.
# These are the only tokenizers for which ``token_bytes_table`` is valid and
# chunked encoding is safe.
_BYTE_LEVEL_TYPES = {"ByteLevel"}

# Test pattern for byte-level path checking.
_DEFAULT_PROBES: tuple[str, ...] = (
    # Basic ASCII & rules for handling leading,
    # trailing, and double whitespace (check for GPT-2 'Ġ' or SentencePiece ' ')
    "Hello world",
    "  Leading,   multiple   spaces, and trailing.  ",

    # Source code formatting, mixed tabs/spaces indentation, and line endings (\r\n, \n)
    "def foo(x: int = 42) -> str:\n\t\"\"\"Docstring.\"\"\"\n\r\n    return f'{x}\\n'\n",

    # Vietnamese & Latin Extended (2–3 bytes/character)
    # Check combinations of tone marks and accented vowels
    "Tiếng Việt: Xin chào thế giới! Phở bò, đường xá, ệ, ữ, ỗ, ậ, ỷ, ỹ.",

    # CJK languages ​​do not use spaces (3 bytes; testing word boundary segmentation capabilities)
    "日本語テキスト（漢字・ひらがな・カタカナ）。中文测试，自然语言处理。한국어 테스트.",

    # RTL writing systems and other character sets (Arabic, Cyrillic, Greek)
    "مرحبا بالعالم! Привет мир! Ελληνικά.",

    # 4-byte UTF-8 characters, single emojis, and complex combinations
    # (ZWJ sequences, Variation Selectors)
    "Emoji 4-byte: 🚀 🦄 🔥 👨‍👩‍👧‍👦 🏳️‍🌈 → ⚡",

    # Arithmetic, mathematical symbols, currency, and contractions
    # (Contractions / Regex split rules)
    "Prices: $1,234.56 or €99.99! They're, couldn't, isn't (π ≈ 3.14159 >= 0).",

    # 8. Markup-style delimiter / simulated special token
    # (handling the issue of special characters being swallowed)
    "<|im_start|>user\n<tag attr=\"val\">content</tag><|endoftext|>",

    # Zero-Width Space, Non-Breaking Space
    "Zero\u200bWidth\u200cJoiner\u00a0NBSP and\u202fNarrowNBSP.",
)
class _DecodeStreamAdapter:
    """Adapter bridging HuggingFace's Rust ``DecodeStream`` to ``DecodeStreamLike``.

    ``tokenizers.decoders.DecodeStream`` is an internal Rust object that is not
    importable as a standalone class.  This adapter wraps it with the
    :class:`~ayaka.tokenizers.ports.DecodeStreamLike` interface expected by
    :class:`~ayaka.tokenizers.detokenizer.FastIncrementalDetokenizer`.

    The ``reset()`` method creates a fresh stream with the same configuration,
    used by ``FastIncrementalDetokenizer`` to recover from ``"invalid prefix"``
    errors without requiring a reference back to the owning tokenizer.

    Attributes:
        _raw: The underlying ``tokenizers.Tokenizer`` object (Rust backend),
            required by ``DecodeStream.step``.
        _stream: The active ``DecodeStream`` instance.
        _skip: Whether special tokens should be suppressed in output.
    """

    __slots__ = (
        "_raw",
        "_stream",
        "_skip",
    )

    def __init__(self, raw_tokenizer, skip_special_tokens: bool) -> None:
        """Wrap a Rust ``DecodeStream`` for the given tokenizer backend.

        Args:
            raw_tokenizer: The ``tokenizers.Tokenizer`` Rust object (accessible
                via ``PreTrainedTokenizerFast._tokenizer`` or
                ``PreTrainedTokenizerFast.backend_tokenizer``).
            skip_special_tokens: Whether special tokens should be stripped from
                the decoded output.
        """
        from tokenizers.decoders import DecodeStream

        self._raw = raw_tokenizer
        self._skip = skip_special_tokens
        self._stream = DecodeStream(skip_special_tokens=skip_special_tokens)

    def step(self, token_id: int) -> str | None:
        """Feed one token ID and return any newly decodable text.

        Delegates to the Rust ``DecodeStream.step`` which maintains internal
        byte-level state.

        Args:
            token_id: A single integer token ID from the model.

        Returns:
            A string fragment ready for output, or ``None`` when the
            accumulated bytes do not yet form a complete character boundary.
        """
        return self._stream.step(self._raw, token_id)

    def reset(self) -> _DecodeStreamAdapter:
        """Create a fresh stream with the same configuration.

        Called by
        :class:`~ayaka.tokenizers.detokenizer.FastIncrementalDetokenizer`
        after an ``"invalid prefix"`` error to discard corrupted byte-level
        state and resume decoding from the current token.

        Returns:
            A new :class:`_DecodeStreamAdapter` with the same tokenizer
            backend and ``skip_special_tokens`` setting.
        """
        return _DecodeStreamAdapter(self._raw, self._skip)


class HfTokenizer:
    """HuggingFace tokenizer adapter implementing ``TokenizerLike`` and ``BytesTokenizerLike``.

    Wraps any ``transformers.PreTrainedTokenizer`` or
    ``transformers.PreTrainedTokenizerFast`` (including those loaded via
    ``AutoTokenizer``) and exposes the unified interface required by the Ayaka
    serving engine.

    Key behaviours:

    * **Vocabulary snapshot** — ``get_vocab()`` and ``get_added_vocab()``
      results are captured at construction time and cached as plain
      dictionaries; subsequent calls are O(1).
    * **Fingerprinting** — a BLAKE2b-16 hash of the tokenizer's serialised
      JSON (or the sorted vocabulary when serialisation is unavailable)
      provides a stable cache key across process restarts.
    * **Byte table** — for byte-level BPE tokenizers, ``token_bytes_table()``
      builds and caches the raw-byte representation of every token ID.  Callers
      should invoke :meth:`verify_byte_path` after loading to confirm the table
      is correct before activating the byte detokenization backend.
    * **Chunked encoding** — ``supports_chunked_encode`` is ``True`` when the
      pre-tokenizer is byte-level BPE.

    Attributes:
        _tok: The wrapped HuggingFace tokenizer object.
        _raw: The underlying Rust tokenizer backend, or ``None`` for slow
            (pure-Python) tokenizers.
        _name: ``name_or_path`` string captured at construction.
        _fp: BLAKE2b-16 fingerprint of the tokenizer state.
        _eos: EOS token ID, or ``None``.
        _bos: BOS token ID, or ``None``.
        _pad: PAD token ID, or ``None``.
        _special_ids: Frozen set of all special-token IDs.
        _special_toks: Tuple of all special-token strings.
        _vocab: Full token-string → token-ID mapping (snapshot).
        _added: Added-vocabulary token-string → token-ID mapping (snapshot).
        _vocab_size: Logical vocabulary size from the model config.
        _max_token_id: Highest token ID in the vocabulary (≥ vocab_size − 1).
        _max_chars: Maximum string length across all vocabulary entries; used
            to size look-back windows.
        _is_fast: Whether the underlying tokenizer is Rust-backed.
        _chunkable: Whether the tokenizer supports chunked encoding.
        _decoded_vocab: Cached ``get_decoded_vocab()`` result, or ``None``
            before first access.
        _trunc_side: Truncation side (``"left"`` or ``"right"``).
        _byte_table: Cached ``token_bytes_table()`` result, or ``None`` before
            first access.
        byte_path_verified: Set to ``True`` by :meth:`verify_byte_path` when
            the byte table passes all probes; ``False`` initially.
    """

    __slots__ = (
        "_tok", "_raw", "_name", "_fp", "_eos", "_bos", "_pad",
        "_special_ids", "_special_toks", "_vocab", "_added", "_vocab_size",
        "_max_token_id", "_max_chars", "_is_fast", "_chunkable", "_decoded_vocab",
        "_trunc_side", "_byte_table", "byte_path_verified",
    )

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> HfTokenizer:
        """Construct :class:`HfTokenizer` from a :class:`~ayaka.configs.tokenizer.TokenizerConfig`.

        Calls ``AutoTokenizer.from_pretrained`` with the parameters from
        *config* and wraps the result.  The ``mode`` field maps to
        ``use_fast``: any value other than ``"slow"`` enables the Rust-backed
        fast tokenizer.

        Args:
            config: Tokenizer configuration specifying the hub ID or local
                path, revision, trust policy, cache directory, truncation side,
                and tokenizer mode.

        Returns:
            A fully initialised :class:`HfTokenizer` ready for use.
        """
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            config.tokenizer,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.download_dir,
            use_fast=config.mode != "slow",
        )
        return cls(tok, truncation_side=config.truncation_side)

    def __init__(self, tok, *, truncation_side: str = "left") -> None:
        """Wrap an existing HuggingFace tokenizer.

        Captures a snapshot of the vocabulary, computes the fingerprint,
        detects byte-level BPE capability, and initialises all cached fields.

        Args:
            tok: A ``transformers.PreTrainedTokenizer`` or
                ``transformers.PreTrainedTokenizerFast`` instance.
            truncation_side: Which end to truncate on overflow.  ``"left"``
                (default) keeps the most recent context; ``"right"`` keeps the
                prompt start.
        """
        self._tok = tok
        self._trunc_side = truncation_side
        self._name = str(getattr(tok, "name_or_path", "<unknown>"))
        self._is_fast = bool(getattr(tok, "is_fast", False))
        self._raw = getattr(tok, "_tokenizer", None) or getattr(
            tok, "backend_tokenizer", None
        )
        if self._raw is not None:
            try:
                self._raw.no_truncation()
                self._raw.no_padding()
            except Exception:
                pass

        self._vocab: dict[str, int] = dict(tok.get_vocab())
        self._added: dict[str, int] = dict(tok.get_added_vocab())

        # Use the tokenizer's own vocab_size attribute (logical size from the
        # model config) rather than len(get_vocab()), which includes added
        # tokens and would give an inflated count.
        self._vocab_size: int = getattr(tok, "vocab_size", len(self._vocab))

        # Guard against -1 when both dicts are empty (e.g. stub tokenizer).
        self._max_token_id: int = max(
            0,
            max(self._vocab.values(), default=-1),
            max(self._added.values(), default=-1),
        )
        self._max_chars = max((len(t) for t in self._vocab), default=1)

        self._eos = getattr(tok, "eos_token_id", None)
        self._bos = getattr(tok, "bos_token_id", None)
        self._pad = getattr(tok, "pad_token_id", None)
        self._special_toks = tuple(getattr(tok, "all_special_tokens", ()) or ())
        self._special_ids = frozenset(
            i for i in (getattr(tok, "all_special_ids", ()) or ()) if i is not None
        )

        self._chunkable = self._detect_bytelevel()
        self._decoded_vocab: list[bytes] | None = None
        self._byte_table: list[bytes] | None = None
        self.byte_path_verified = False
        self._fp = self._compute_fingerprint()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_bytelevel(self) -> bool:
        """Return ``True`` when the pre-tokenizer is (or contains) ``ByteLevel``.

        Inspects the Rust backend's pre-tokenizer.  Handles both a bare
        ``ByteLevel`` pre-tokenizer and a ``Sequence`` pre-tokenizer that
        contains at least one ``ByteLevel`` step.

        Returns:
            ``True`` when chunked encoding is safe for this tokenizer.
        """
        # Slow / Python tokenizer: cannot inspect Rust pre_tokenizer.
        if self._raw is None:
            return False
        pt = getattr(self._raw, "pre_tokenizer", None)
        if pt is None:
            return False
        name = type(pt).__name__
        # Case 1: Standalone ByteLevel pre-tokenizer (e.g. GPT-2, LLaMA-3).
        if name in _BYTE_LEVEL_TYPES:
            return True
        # Case 2: Sequence of pre-tokenizers (e.g. WhitespaceSplit + ByteLevel).
        if name == "Sequence":
            try:
                return any(
                    type(sub).__name__ in _BYTE_LEVEL_TYPES for sub in pt  # type: ignore[union-attr]
                )
            except TypeError:
                return False
        return False

    def _compute_fingerprint(self) -> str:
        """Compute a BLAKE2b-16 fingerprint of the tokenizer state.

        Uses the tokenizer's serialised JSON string when available (fast path
        via the Rust backend).  Falls back to hashing the sorted vocabulary
        entries when serialisation is not supported.

        Returns:
            A 32-character lowercase hex string.
        """
        h = hashlib.blake2b(digest_size=16)
        # Fast path: Hash complete serialized JSON representation from the Rust backend.
        if self._raw is not None:
            try:
                h.update(self._raw.to_str().encode())
                return h.hexdigest()
            except Exception:  # noqa: BLE001
                pass
        # Fallback path: Deterministically iterate through sorted vocab pairs (id, token_str).
        # Ensures cross-platform and cross-run stability for pure-Python tokenizers.
        for t, i in sorted(self._vocab.items(), key=lambda kv: kv[1]):
            h.update(f"{i}\0{t}\0".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # TokenizerLike — metadata properties
    # ------------------------------------------------------------------

    @property
    def name_or_path(self) -> str:
        """Hub ID or local directory the tokenizer was loaded from."""
        return self._name

    def fingerprint(self) -> str:
        """Content-derived hash that changes when the tokenizer data changes.

        Used by caching layers to detect tokenizer updates across process
        restarts without re-loading the full tokenizer.

        Returns:
            A 32-character BLAKE2b-16 hex digest.
        """
        return self._fp

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID, or ``None`` if unset."""
        return self._eos

    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token ID, or ``None`` if unset."""
        return self._bos

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID, or ``None`` if unset."""
        return self._pad

    @property
    def all_special_ids(self) -> frozenset[int]:
        """Frozen set of all registered special-token IDs."""
        return self._special_ids

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        """Tuple of all registered special-token strings."""
        return self._special_toks

    @property
    def vocab_size(self) -> int:
        """Logical vocabulary size from the model config.

        This is the value reported by ``tokenizer.vocab_size`` — the size of
        the *base* vocabulary, **not** including added tokens.  Use
        :attr:`max_token_id` to find the highest producible token ID.
        """
        return self._vocab_size

    @property
    def max_token_id(self) -> int:
        """Highest token ID the tokenizer can produce (≥ vocab_size − 1).

        May exceed ``vocab_size - 1`` when added tokens have IDs beyond the
        base vocabulary range.  Guaranteed to be ≥ 0 even for stub tokenizers
        with empty vocabularies.
        """
        return self._max_token_id

    @property
    def max_chars_per_token(self) -> int:
        """Upper bound on the string length of a single token.

        Used to pre-allocate look-back windows in
        :class:`~ayaka.tokenizers.detokenizer.SlowIncrementalDetokenizer` and
        to bound stop-string matching buffers.
        """
        return self._max_chars

    @property
    def is_fast(self) -> bool:
        """``True`` when the underlying tokenizer is Rust-backed (fast)."""
        return self._is_fast

    @property
    def supports_chunked_encode(self) -> bool:
        """``True`` when the tokenizer's pre-tokenizer is byte-level BPE.

        When ``True``, :func:`~ayaka.tokenizers.chunking.encode_chunked` may
        split long prompts at newline boundaries and encode each chunk
        independently.
        """
        return self._chunkable

    # ------------------------------------------------------------------
    # TokenizerLike — encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode a single string into token IDs.

        Args:
            text: The input string to tokenize.
            add_special_tokens: Whether to prepend/append BOS/EOS tokens as
                configured by the model's ``tokenizer_config.json``.

        Returns:
            A list of integer token IDs.
        """
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode multiple strings in a single call.

        Uses the Rust backend's ``encode_batch`` path when available (fast
        tokenizer), falling back to sequential ``encode`` calls for slow
        tokenizers.

        Args:
            texts: A sequence of input strings.
            add_special_tokens: Whether to prepend/append special tokens.

        Returns:
            A list of token-ID lists, one per input string.
        """
        if self._raw is not None:
            encs = self._raw.encode_batch(
                list(texts), add_special_tokens=add_special_tokens
            )
            return [e.ids for e in encs]
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def truncate(self, ids: list[int], max_length: int | None) -> tuple[list[int], bool]:
        """Truncate a token-ID list to *max_length*.

        Args:
            ids: Token IDs to truncate.
            max_length: Maximum allowed length.  ``None`` disables truncation.

        Returns:
            ``(truncated_ids, was_truncated)`` where ``was_truncated`` is
            ``True`` when the list was shortened.
        """
        if max_length is None or len(ids) <= max_length:
            return ids, False
        return (
            ids[-max_length:] if self._trunc_side == "left" else ids[:max_length]
        ), True

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Render a chat conversation into a model-ready prompt string.

        Args:
            messages: Chat messages, each a mapping with at least ``"role"``
                and ``"content"`` keys.
            tools: Optional tool/function definitions for function-calling
                models.
            add_generation_prompt: Whether to append the assistant-turn prefix
                so the model begins generating immediately.
            chat_template: Override the default Jinja2 chat template embedded
                in the tokenizer.
            **kwargs: Forwarded to the underlying template renderer.

        Returns:
            The fully rendered prompt string (not yet tokenized).
        """
        out = self._tok.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            tokenize=False,
            **kwargs,
        )
        return out if isinstance(out, str) else out[0]

    # ------------------------------------------------------------------
    # TokenizerLike — decoding
    # ------------------------------------------------------------------

    def decode(
        self, ids: Sequence[int] | int, *, skip_special_tokens: bool = False
    ) -> str:
        """Decode token IDs back into a string.

        Args:
            ids: One or more token IDs.
            skip_special_tokens: Whether to strip special tokens from the
                output.

        Returns:
            The decoded text.
        """
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Map token IDs to their string representations without merging.

        Unlike :meth:`decode`, this preserves token boundaries, which is
        required for the window-diff detokenizer and for building the decoded
        vocabulary table.

        Args:
            ids: Token IDs to convert.
            skip_special_tokens: Whether to omit special-token strings.

        Returns:
            A list of token strings, one per input ID.  ``None`` entries from
            the HuggingFace API are replaced with empty strings.
        """
        out = self._tok.convert_ids_to_tokens(
            list(ids), skip_special_tokens=skip_special_tokens
        )
        if isinstance(out, str):
            return [out]
        return [t if t is not None else "" for t in out]

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        """Merge a sequence of token strings into readable text.

        Handles byte-fallback tokens (``<0xAB>``) and whitespace normalisation
        as defined by the underlying tokenizer model.

        Args:
            tokens: Token strings as returned by :meth:`convert_ids_to_tokens`.

        Returns:
            The merged text.
        """
        return self._tok.convert_tokens_to_string(list(tokens))

    def new_decode_stream(
        self, *, skip_special_tokens: bool = False
    ) -> DecodeStreamLike | None:
        """Create a fresh incremental decode stream.

        Returns ``None`` when the tokenizer is not Rust-backed (no
        ``backend_tokenizer``) or when the ``tokenizers`` library does not
        expose ``DecodeStream`` (import failure).

        Args:
            skip_special_tokens: Whether the stream should suppress special
                tokens.

        Returns:
            A new :class:`_DecodeStreamAdapter`, or ``None``.
        """
        if self._raw is None or not self._is_fast:
            return None
        try:
            return _DecodeStreamAdapter(self._raw, skip_special_tokens)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # TokenizerLike — vocabulary introspection
    # ------------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the full token-string → token-ID mapping (including added tokens)."""
        return self._vocab

    def get_added_vocab(self) -> dict[str, int]:
        """Return only the tokens added after the base vocabulary was built."""
        return self._added

    def get_decoded_vocab(self) -> list[bytes]:
        """Return the raw byte representation of every token, indexed by ID.

        The list length equals ``max_token_id + 1``.  Entries for undefined
        IDs (gaps between ``vocab_size`` and ``max_token_id``) are empty
        ``bytes``.  Used by guided-decoding engines that operate on byte tries.

        Added-vocabulary tokens are encoded as their literal UTF-8 bytes
        because they are inserted after BPE training and are not subject to
        the byte-fallback mapping.

        Returns:
            A list of ``bytes`` objects, one per token ID slot.
        """
        if self._decoded_vocab is None:
            n = self._max_token_id + 1
            out: list[bytes] = [b""] * n
            # 1. Base vocabulary: convert token string back to human string then UTF-8 encode.
            for tok_str, tid in self._vocab.items():
                if 0 <= tid < n:
                    out[tid] = self._tok.convert_tokens_to_string([tok_str]).encode()
            # 2. Added vocabulary: encode token strings directly as literal UTF-8 bytes.
            for tok_str, tid in self._added.items():
                if 0 <= tid < n:
                    out[tid] = tok_str.encode()
            self._decoded_vocab = out
        return self._decoded_vocab

    # ------------------------------------------------------------------
    # BytesTokenizerLike
    # ------------------------------------------------------------------

    @staticmethod
    def _bytelevel_char_to_byte() -> dict[str, int]:
        """Build the GPT-2 byte-to-unicode character mapping (inverted).

        GPT-2–style byte-level BPE tokenizers map each raw byte to a printable
        Unicode character so the vocabulary contains only printable strings.
        This method inverts that mapping (character → byte value) for use in
        :meth:`token_bytes_table`.

        Returns:
            A dictionary mapping each printable stand-in character to its
            original byte value (0–255).
        """
        # Step 1: List bytes that have natural, printable Latin-1 glyphs.
        # Range 33..126 (ASCII printables, excluding space)
        # Range 161..172 & 174..255
        # (Latin-1 supplement printables, excluding non-breaking space & soft hyphen)
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]  # these bytes map 1:1 to their exact unicode code point
        n = 0
        # Step 2: The remaining 70 non-printable/control bytes are shifted to U+0100 (256) and above
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        # Step 3: Invert into a lookup dictionary: character glyph -> raw byte integer (0..255).
        return {chr(c): b for b, c in zip(bs, cs, strict=True)}

    def token_bytes_table(self) -> list[bytes]:
        """Return the byte-level vocabulary table, indexed by token ID.

        Builds (on first call) a list of ``bytes`` objects where
        ``table[token_id]`` is the raw byte sequence for that token.  The
        table is only meaningful for byte-level BPE tokenizers
        (``supports_chunked_encode is True``); for other tokenizer types an
        empty list is returned.

        Added-vocabulary tokens are encoded as their literal UTF-8 bytes
        rather than going through the GPT-2 byte mapping.

        Returns:
            A list of ``bytes`` of length ``max_token_id + 1``, or an empty
            list when the tokenizer is not byte-level BPE.
        """
        if self._byte_table is not None:
            return self._byte_table
        # If tokenizer is not byte-level BPE (e.g. SentencePiece or word-level),
        # byte table reconstruction is invalid.
        if not self._chunkable:
            self._byte_table = []
            return self._byte_table

        # Inverted mapping: stand-in Unicode character -> raw byte.
        c2b = self._bytelevel_char_to_byte()
        tbl: list[bytes] = [b""] * (self._max_token_id + 1)

        # 1. Base vocabulary: unmap each character in token string back to its raw byte.
        for tok_str, tid in self._vocab.items():
            if not (0 <= tid <= self._max_token_id):
                continue
            try:
                tbl[tid] = bytes(c2b[ch] for ch in tok_str)
            except KeyError:
                # Fallback for special tokens embedded in base vocab: encode literally as UTF-8.
                tbl[tid] = tok_str.encode()

        # 2. Added tokens: always literal UTF-8 strings (e.g. "<|eot_id|>", "<0x0A>").
        for tok_str, tid in self._added.items():
            if 0 <= tid <= self._max_token_id:
                tbl[tid] = tok_str.encode()

        self._byte_table = tbl
        return tbl

    def verify_byte_path(self, samples: Sequence[str] = ()) -> bool:
        """Validate the byte table and update :attr:`byte_path_verified`.

        Delegates to :func:`~ayaka.tokenizers.byte_detok.verify_byte_table`
        with *samples* (or the built-in :data:`_DEFAULT_PROBES` when *samples*
        is empty).  Sets :attr:`byte_path_verified` based on the result so
        that :meth:`~ayaka.tokenizers.detokenizer.IncrementalDetokenizer.create`
        can safely activate the byte backend.

        Args:
            samples: Representative strings to verify.  Falls back to
                :data:`_DEFAULT_PROBES` when empty.

        Returns:
            ``True`` when all probes pass; ``False`` when the table is empty
            or any probe mismatches.
        """
        from ayaka.tokenizers.byte_detok import verify_byte_table

        if not self.token_bytes_table():
            self.byte_path_verified = False
            return False
        self.byte_path_verified = verify_byte_table(
            self, samples or _DEFAULT_PROBES
        )
        return self.byte_path_verified
