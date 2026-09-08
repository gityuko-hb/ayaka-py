"""Structural (Protocol-based) interfaces for tokenizers.

This module defines the contracts that any tokenizer implementation must
satisfy in order to plug into the Ayaka serving pipeline.  No concrete
implementation lives here — conformance is checked structurally at runtime
via ``isinstance`` (every protocol carries ``@runtime_checkable``), so an
adapter or a plain ``transformers.PreTrainedTokenizerFast`` that happens to
expose the right surface will pass the check without inheriting anything.

Dependency direction::

    configs.tokenizer  ←  tokenizers.ports  →  (concrete adapters)
                                                    ↑
                                        TokenizerFactory.from_config

The ``TokenizerConfig`` that ``TokenizerFactory.from_config`` consumes
carries *where* the tokenizer lives and *how* to run it (batching,
chunking, session cache).  The ``ModelSourceConfig.tokenizer_path`` property
in ``configs.model_source`` resolves the *raw path* first; the composition
root is responsible for feeding that path into a ``TokenizerConfig`` and
then into the factory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ayaka.configs.tokenizer import TokenizerConfig


@runtime_checkable
class DecodeStreamLike(Protocol):
    """Incremental byte-level decoder for streaming token output.

    Each call to ``step`` feeds one token ID and returns the text fragment
    that can be flushed to the client, or ``None`` when the decoder needs
    more tokens before it can emit a complete character (e.g. a multi-byte
    UTF-8 sequence split across token boundaries).

    Implementations must be **stateful** — they accumulate partial byte
    sequences between calls and flush only when a valid character boundary
    is reached.
    """

    def step(self, token_id: int) -> str | None:
        """Feed one token and return the decodable text fragment, if any.

        Args:
            token_id: A single integer token ID produced by the model.

        Returns:
            A string fragment ready for output, or ``None`` when the
            accumulated bytes do not yet form a valid character boundary.
        """
        ...


@runtime_checkable
class TokenizerLike(Protocol):
    """The full tokenizer surface that the Ayaka engine requires.

    Any object whose public API structurally matches these signatures can
    serve as the tokenizer for a deployed model.  The interface covers:

    * **Metadata** — ``name_or_path``, ``fingerprint``, special-token IDs,
      ``vocab_size``, ``max_token_id``, ``max_chars_per_token``.
    * **Encoding** — ``encode``, ``encode_batch``, ``apply_chat_template``.
    * **Decoding** — ``decode``, ``convert_ids_to_tokens``,
      ``convert_tokens_to_string``, ``new_decode_stream``.
    * **Vocabulary introspection** — ``get_vocab``, ``get_added_vocab``,
      ``get_decoded_vocab``.

    ``vocab_size`` vs ``max_token_id``:
        ``vocab_size`` is the *logical* vocabulary count reported by the
        model config.  ``max_token_id`` is the highest integer token ID the
        tokenizer can produce — it may be larger than ``vocab_size - 1``
        when the tokenizer has added tokens beyond the base vocabulary.
    """

    @property
    def name_or_path(self) -> str:
        """Hub ID or local directory the tokenizer was loaded from."""
        ...

    def fingerprint(self) -> str:
        """Content-derived hash that changes when the tokenizer data changes.

        Used by caching layers to detect tokenizer updates across restarts.
        """
        ...

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID, or ``None`` if unset."""
        ...
    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token ID, or ``None`` if unset."""
        ...
    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID, or ``None`` if unset."""
        ...
    @property
    def all_special_ids(self) -> frozenset[int]:
        """Frozen set of all registered special-token IDs."""
        ...
    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        """Tuple of all registered special-token strings."""
        ...

    @property
    def vocab_size(self) -> int:
        """Logical vocabulary size from the model config."""
        ...

    @property
    def max_token_id(self) -> int:
        """Highest token ID the tokenizer can produce (≥ vocab_size − 1)."""
        ...

    @property
    def max_chars_per_token(self) -> int:
        """Upper bound on UTF-8 characters a single token may decode to.

        Used to pre-allocate decode buffers and to bound the look-back
        window of stop-string matching in ``DetokenizeParams``.
        """
        ...

    @property
    def is_fast(self) -> bool:
        """Whether the underlying tokenizer is a Rust-backed "fast" tokenizer."""
        ...

    @property
    def supports_chunked_encode(self) -> bool:
        """Whether ``encode`` can safely process text in chunks.

        When ``True``, the engine may split long prompts at character
        boundaries set by ``TokenizerConfig.chunk_chars`` and encode each
        chunk independently, stitching the token lists afterwards.  Byte-level
        BPE tokenizers (GPT-2 family, LLaMA, Qwen) typically support this;
        SentencePiece Unigram tokenizers may not.
        """
        ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode a single text string into token IDs.

        Args:
            text: The input text to tokenize.
            add_special_tokens: Whether to prepend/append BOS/EOS tokens
                as dictated by the model's tokenizer config.

        Returns:
            A list of integer token IDs.
        """
        ...

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode multiple texts in a single call.

        Implementations should exploit the fast tokenizer's batch encoding
        path when available.  The pool-based encoder configured by
        ``TokenizerConfig.encode_pool_workers`` calls this method.

        Args:
            texts: A sequence of input texts.
            add_special_tokens: Whether to prepend/append special tokens.

        Returns:
            A list of token-ID lists, one per input text.
        """
        ...

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
            messages: Chat messages, each a mapping with at least ``role``
                and ``content`` keys.
            tools: Optional tool/function definitions for function-calling
                models.
            add_generation_prompt: Whether to append the assistant turn
                prefix so the model begins generating immediately.
            chat_template: Override the default Jinja2 chat template
                embedded in the tokenizer.
            **kwargs: Forwarded to the underlying template renderer.

        Returns:
            The fully rendered prompt string (not yet tokenized).
        """
        ...

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
        ...

    def convert_ids_to_tokens(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Map token IDs to their string representations without merging.

        Unlike ``decode``, this preserves token boundaries — useful for
        debugging and for building the ``get_decoded_vocab`` table.

        Args:
            ids: Token IDs to convert.
            skip_special_tokens: Whether to omit special-token strings.

        Returns:
            A list of token strings, one per input ID.
        """
        ...

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        """Merge a sequence of token strings into readable text.

        Handles byte-fallback tokens (``<0xAB>``) and whitespace
        normalization as defined by the tokenizer model.

        Args:
            tokens: Token strings as returned by ``convert_ids_to_tokens``.

        Returns:
            The merged text.
        """
        ...

    def new_decode_stream(
        self, *, skip_special_tokens: bool = False
    ) -> DecodeStreamLike | None:
        """Create a fresh incremental decode stream.

        Returns ``None`` when the tokenizer does not support streaming
        decode (e.g. a slow pure-Python tokenizer without byte-level
        tracking).

        Args:
            skip_special_tokens: Whether the stream should silently drop
                special tokens.

        Returns:
            A new ``DecodeStreamLike`` instance, or ``None``.
        """
        ...

    def get_vocab(self) -> dict[str, int]:
        """Return the full token-string → token-ID mapping."""
        ...
    def get_added_vocab(self) -> dict[str, int]:
        """Return only the tokens added after the base vocabulary was built."""
        ...
    def get_decoded_vocab(self) -> list[bytes]:
        """Return the raw byte representation of every token, indexed by ID.

        The list length equals ``max_token_id + 1``.  Entries for undefined
        IDs (gaps between ``vocab_size`` and ``max_token_id``) are empty
        ``bytes``.  Used by guided-decoding engines that operate on byte
        tries.
        """
        ...


@runtime_checkable
class BytesTokenizerLike(Protocol):
    """Optional extension for tokenizers that expose a pre-built byte table.

    A ``TokenizerLike`` that also satisfies ``BytesTokenizerLike`` can
    provide the byte table directly instead of requiring the engine to
    build it via ``get_decoded_vocab``.  This is a performance shortcut
    for guided-decoding backends that need a ``list[bytes]`` keyed by
    token ID.
    """

    def token_bytes_table(self) -> list[bytes]:
        """Return the byte-level vocabulary table, indexed by token ID.

        Equivalent in content to ``get_decoded_vocab`` but may be computed
        more efficiently by implementations that maintain the table
        internally (e.g. tiktoken-based tokenizers).
        """
        ...


@runtime_checkable
class TokenizerFactory(Protocol):
    """Factory protocol for constructing a ``TokenizerLike`` from config.

    The composition root calls ``TokenizerFactory.from_config`` once during
    bootstrap to obtain the tokenizer instance that the engine will use for
    the lifetime of the process.

    The ``TokenizerConfig.tokenizer`` field carries the hub ID or local path
    — typically resolved from ``ModelSourceConfig.tokenizer_path``, which
    defaults to the model path when no separate tokenizer is specified.
    """

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> TokenizerLike:
        """Build a tokenizer from the given configuration.

        Args:
            config: Tokenizer configuration specifying the source path,
                mode (``hf``, ``tiktoken``, ``mistral``, etc.), and
                runtime settings (batching, chunking, session cache).

        Returns:
            A fully initialized tokenizer satisfying ``TokenizerLike``.
        """
        ...
