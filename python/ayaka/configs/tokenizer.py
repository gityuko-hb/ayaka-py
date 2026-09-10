"""Configuration and data-transfer types for the tokenizer subsystem.

This module contains:

* **TokenizerConfig** — immutable settings that control *which* tokenizer to
  load and *how* to run it (batching, chunking, prefix-session cache).
* **DetokenizeParams** — per-request parameters for the incremental
  detokenizer (stop strings, special-token handling).
* **DetokUpdate** — one tick of detokenizer output (delta text, stop match,
  stall indicator).
* **EncodeResult** — the product of a single encode operation, including
  prefix-reuse and chunking metadata.
* **tokenizer_config_from_source** — bridge function that builds a
  ``TokenizerConfig`` from a ``ModelSourceConfig``.

Data flow::

    ModelSourceConfig
        │
        ├─ .tokenizer_path  ──→  tokenizer
        ├─ .revision         ──→  revision   ("" → None)
        ├─ .allows_remote_code → trust_remote_code
        └─ .cache_dir        ──→  download_dir ("" → None)
        │
        ▼  tokenizer_config_from_source(source, **serving_kwargs)
    TokenizerConfig(tokenizer=<path>, ...)
        │
        ▼  TokenizerFactory.from_config(config)
    TokenizerLike
        │
        ▼  encode / decode
    EncodeResult / DetokenizeParams / DetokUpdate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ayaka.configs.base import ConfigMixin

if TYPE_CHECKING:
    from ayaka.configs.model_source import ModelSourceConfig

TokenizerMode = Literal["auto", "hf", "slow", "tiktoken", "mistral"]
"""Backend selection for the tokenizer implementation.

* ``"auto"`` — inspect ``tokenizer_config.json`` and pick the fastest
  compatible backend automatically.
* ``"hf"`` — force the Rust-backed ``tokenizers`` library
  (``PreTrainedTokenizerFast``).
* ``"slow"`` — force the pure-Python ``PreTrainedTokenizer``.  Required
  for models whose tokenizer cannot be represented by the Rust engine
  (rare; some SentencePiece Unigram models).
* ``"tiktoken"`` — use the ``tiktoken`` library directly.  Required for
  OpenAI-family byte-pair encodings (GPT-4, o-series).
* ``"mistral"`` — use Mistral's custom ``mistral-common`` tokenizer.
"""

DetokBackend = Literal["auto", "stream", "window"]
"""Detokenization strategy for incremental decode.

* ``"auto"`` — choose based on ``TokenizerLike.new_decode_stream``
  availability: ``"stream"`` when a ``DecodeStreamLike`` is returned,
  ``"window"`` otherwise.
* ``"stream"`` — feed tokens one-by-one into a ``DecodeStreamLike``.
  Lowest latency but requires the tokenizer to support it.
* ``"window"`` — decode a trailing window of token IDs on every step and
  diff against the previous output.  Universal but slightly higher CPU
  cost per token.
"""


@dataclass(frozen=True, slots=True)
class TokenizerConfig(ConfigMixin):
    """Immutable configuration for loading and operating a tokenizer.

    ``tokenizer`` is the only required field; it names the hub ID or local
    directory that contains the tokenizer files.  All other fields have
    defaults suitable for the common case (HuggingFace fast tokenizer,
    no batching pool, no session cache).

    The remaining fields fall into three groups:

    **Loading** — ``mode``, ``revision``, ``trust_remote_code``,
    ``download_dir``, ``skip_tokenizer_init``.

    **Encoding runtime** — ``truncation_side``, ``encode_pool_workers``,
    ``encode_batch_window_ms``, ``encode_max_batch``, ``long_prompt_chars``,
    ``chunk_chars``.

    **Prefix session cache** — ``enable_session_cache``,
    ``session_cache_capacity``, ``verify_session_cache``.

    Attributes:
        tokenizer: Hub ID (``"meta-llama/Llama-3-8B"``) or local path to
            the directory containing ``tokenizer.json`` /
            ``tokenizer.model``.  Usually populated from
            ``ModelSourceConfig.tokenizer_path``.
        mode: Which tokenizer backend to use.  See ``TokenizerMode``.
        revision: Git revision (branch / tag / SHA) for hub downloads.
            ``None`` means HEAD.
        trust_remote_code: Allow execution of Python code shipped inside
            the tokenizer repo.  Default ``False`` — the engine refuses
            to run checkpoint-supplied code unless explicitly opted in.
        download_dir: Override the default HuggingFace cache directory.
            ``None`` means ``HF_HOME`` / the library default.
        truncation_side: Which end to truncate when a prompt exceeds the
            model's maximum position embeddings.  ``"left"`` (default)
            keeps the most recent context, which is usually correct for
            chat models.
        skip_tokenizer_init: When ``True``, the factory returns a
            lightweight stub that satisfies ``TokenizerLike`` but cannot
            encode or decode.  Used by weight-validation and profiling
            tools that need the engine's config graph but never tokenize.
        detokenize_backend: Incremental detokenization strategy.
            See ``DetokBackend``.
        encode_pool_workers: Number of background threads in the encode
            pool.  ``0`` (default) means encode synchronously in the
            caller's thread.
        encode_batch_window_ms: How long (in milliseconds) to hold an
            incoming encode request before dispatching a batch to the pool.
            Only meaningful when ``encode_pool_workers > 0``.
        encode_max_batch: Maximum number of texts to batch in a single
            ``encode_batch`` call.  Caps memory usage in the tokenizer's
            Rust runtime.
        long_prompt_chars: Character-length threshold above which a prompt
            is considered "long" and may be chunked for encoding.
        chunk_chars: Target character count per chunk when splitting long
            prompts for chunked encoding.  Only used when
            ``TokenizerLike.supports_chunked_encode`` is ``True``.
        enable_session_cache: Enable the prefix-session cache, which
            stores recently encoded prompt prefixes and reuses them on
            subsequent requests that share the same prefix.
        session_cache_capacity: Maximum number of entries in the session
            cache (an LRU ring).
        verify_session_cache: When ``True``, re-encode every cache hit and
            assert equality.  Useful for development and CI; too expensive
            for production.
    """

    tokenizer: str | Path
    mode: TokenizerMode = "auto"
    revision: str | None = None
    trust_remote_code: bool = False
    download_dir: str | None = None
    truncation_side: Literal["left", "right"] = "left"
    skip_tokenizer_init: bool = False
    detokenize_backend: DetokBackend = "auto"
    encode_pool_workers: int = 0
    encode_batch_window_ms: float = 1.0
    encode_max_batch: int = 32
    long_prompt_chars: int = 256 * 1024
    chunk_chars: int = 64 * 1024
    enable_session_cache: bool = False
    session_cache_capacity: int = 4096
    verify_session_cache: bool = False


@dataclass(slots=True)
class DetokenizeParams:
    """Per-request parameters that govern incremental detokenization.

    Passed to the detokenizer at the start of each generation request.
    Mutable (not ``frozen``) because the scheduler may adjust ``stop``
    mid-flight when a request is preempted and resumed.

    Attributes:
        skip_special_tokens: Strip special tokens (BOS, EOS, PAD, etc.)
            from the output text.  Default ``True`` for user-facing output.
        spaces_between_special_tokens: Insert a space before and after
            each special token in the decoded output.
        stop: Tuple of stop strings.  The detokenizer watches the
            accumulated output for any of these substrings and signals a
            match via ``DetokUpdate.stop_matched``.
        include_stop_str_in_output: When ``True``, the matched stop string
            is included in the final ``delta`` rather than being consumed
            silently.
        accumulate_text: When ``True``, the ``delta`` in each
            ``DetokUpdate`` contains the *entire* decoded text so far, not
            just the new fragment.  Used by logprobs endpoints that need
            the full context for alignment.
    """

    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    stop: tuple[str, ...] = ()
    include_stop_str_in_output: bool = False
    accumulate_text: bool = False

    def max_stop_len(self) -> int:
        """Return the character length of the longest stop string.

        Used to size the look-back window in the ``"window"``
        detokenization backend.  Returns ``0`` when no stop strings are
        configured.
        """
        return max((len(s) for s in self.stop), default=0)


@dataclass(slots=True)
class DetokUpdate:
    """One tick of incremental detokenizer output.

    Produced by the detokenizer each time one or more new tokens are
    decoded.  The scheduler inspects ``stop_matched`` to decide whether
    to terminate the request.

    Attributes:
        delta: New text produced in this tick.  Empty when the detokenizer
            is stalled (see ``stalled``).
        stop_matched: The stop string that was matched, or ``None`` if
            generation should continue.
        rewind_chars: Number of characters at the *end* of ``delta`` that
            should be removed from the output buffer.  Non-zero only when
            a stop string is matched partially inside a prior delta and
            must be un-emitted.
        stalled: ``True`` when the detokenizer received tokens but cannot
            emit any text yet — typically because a multi-byte UTF-8
            character is incomplete or a stop-string look-ahead is
            pending.
    """

    delta: str = ""
    stop_matched: str | None = None
    rewind_chars: int = 0
    stalled: bool = False


@dataclass(slots=True)
class EncodeResult:
    """The product of encoding a single prompt.

    Carries the token IDs and metadata about how the encoding was
    performed, so upstream code (the scheduler, the KV-cache manager)
    can make informed decisions about prefix reuse and chunking.

    Attributes:
        token_ids: The encoded token-ID sequence.
        reused_prefix_len: Number of leading tokens that were served from
            the session cache rather than re-encoded.  ``0`` when the
            session cache is disabled or missed.
        truncated: ``True`` when the prompt exceeded
            ``max_position_embeddings`` and was truncated according to
            ``TokenizerConfig.truncation_side``.
        chunks: Number of independent chunks the prompt was split into
            for encoding.  ``1`` when chunked encoding was not used.
        extra: Opaque bag for implementation-specific metadata (e.g.
            offset mappings, attention masks) that the engine does not
            interpret but may forward to downstream consumers.
    """

    token_ids: list[int]
    reused_prefix_len: int = 0
    truncated: bool = False
    chunks: int = 1
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bridge: ModelSourceConfig → TokenizerConfig
# ---------------------------------------------------------------------------

def tokenizer_config_from_source(
    source: ModelSourceConfig,
    *,
    mode: TokenizerMode = "auto",
    truncation_side: Literal["left", "right"] = "left",
    skip_tokenizer_init: bool = False,
    detokenize_backend: DetokBackend = "auto",
    encode_pool_workers: int = 0,
    encode_batch_window_ms: float = 1.0,
    encode_max_batch: int = 32,
    long_prompt_chars: int = 256 * 1024,
    chunk_chars: int = 64 * 1024,
    enable_session_cache: bool = False,
    session_cache_capacity: int = 4096,
    verify_session_cache: bool = False,
) -> TokenizerConfig:
    """Build a ``TokenizerConfig`` from a ``ModelSourceConfig``.

    This is the **single bridge** between the "where does the model live?"
    config and the "how should the tokenizer run?" config.  Four fields are
    derived from *source*; the remaining thirteen are tokenizer-specific and
    exposed as keyword arguments with the same defaults as ``TokenizerConfig``.

    Field mapping::

        ModelSourceConfig              TokenizerConfig
        ─────────────────              ───────────────
        source.tokenizer_path     →    tokenizer        (str)
        source.revision           →    revision         ("" → None)
        source.allows_remote_code →    trust_remote_code (bool)
        source.cache_dir          →    download_dir     ("" → None)

    Args:
        source: The model-source config that identifies the hub ID or local
            path, revision, trust policy, and cache directory.
        mode: Tokenizer backend.  See ``TokenizerMode``.
        truncation_side: Which end to truncate on overflow.
        skip_tokenizer_init: Return a stub instead of a real tokenizer.
        detokenize_backend: Incremental decode strategy.
        encode_pool_workers: Background encode-pool thread count (0 = sync).
        encode_batch_window_ms: Batching window before pool dispatch.
        encode_max_batch: Max texts per ``encode_batch`` call.
        long_prompt_chars: Char threshold for "long prompt" chunking.
        chunk_chars: Target chars per chunk.
        enable_session_cache: Turn on prefix-session caching.
        session_cache_capacity: LRU ring size for the session cache.
        verify_session_cache: Re-encode cache hits and assert equality.

    Returns:
        An immutable ``TokenizerConfig`` ready for
        ``TokenizerFactory.from_config``.

    Example::

        from ayaka.configs.model_source import ModelSourceConfig
        from ayaka.configs.tokenizer import tokenizer_config_from_source

        source = ModelSourceConfig(model="meta-llama/Llama-3-8B")
        tok_cfg = tokenizer_config_from_source(source, mode="hf")
    """
    return TokenizerConfig(
        tokenizer=source.tokenizer_path,
        mode=mode,
        revision=source.revision or None,
        trust_remote_code=source.allows_remote_code,
        download_dir=source.cache_dir or None,
        truncation_side=truncation_side,
        skip_tokenizer_init=skip_tokenizer_init,
        detokenize_backend=detokenize_backend,
        encode_pool_workers=encode_pool_workers,
        encode_batch_window_ms=encode_batch_window_ms,
        encode_max_batch=encode_max_batch,
        long_prompt_chars=long_prompt_chars,
        chunk_chars=chunk_chars,
        enable_session_cache=enable_session_cache,
        session_cache_capacity=session_cache_capacity,
        verify_session_cache=verify_session_cache,
    )

