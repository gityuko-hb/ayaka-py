"""Validated tokenizer settings and immutable encoding/decoding values.

Modes without an adapter are rejected by the factory. Character chunk settings
are reserved for a proven chunk-safe adapter; HfTokenizer encodes whole prompts.
The session cache stores exact whole-prompt encodings, never acquired KV state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.utils.validation import require_frozen, require_int, require_text

if TYPE_CHECKING:
    from ayaka.configs.model_source import ModelSourceConfig

TokenizerMode = Literal["auto", "hf", "slow", "tiktoken", "mistral"]
DetokBackend = Literal["auto", "bytes", "stream", "window"]


def require_bool(value: bool, name: str) -> None:
    """Reject truthy substitutes at public boundaries."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


@dataclass(frozen=True, slots=True)
class TokenizerConfig(ConfigMixin):
    """Loading and bounded encode-service settings.

    workers=0 executes in the submitting thread. A positive worker count enables
    bounded, coalesced background batches. The adapter is serialized because
    slow/custom tokenizers need not be thread-safe. Limits include queued and
    running jobs; bytes count UTF-8 payload or eight bytes per input token ID.
    Chunking is disabled for the HF adapter until exact equivalence is proven.
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
    encode_max_pending: int = 256
    encode_max_pending_bytes: int = 16 << 20
    session_cache_max_bytes: int = 16 << 20

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, (str, Path)):
            raise TypeError("tokenizer must be a string or Path")
        require_text(str(self.tokenizer), "tokenizer")
        for name, allowed in (
            ("mode", ("auto", "hf", "slow", "tiktoken", "mistral")),
            ("truncation_side", ("left", "right")),
            ("detokenize_backend", ("auto", "bytes", "stream", "window")),
        ):
            if getattr(self, name) not in allowed:
                raise ConfigError(f"tokenizer.{name}", "INVALID_CHOICE", str(allowed))
        for name in ("revision", "download_dir"):
            value = getattr(self, name)
            if value is not None:
                require_text(value, name)
        for name in (
            "trust_remote_code",
            "skip_tokenizer_init",
            "enable_session_cache",
            "verify_session_cache",
        ):
            require_bool(getattr(self, name), name)
        require_int(self.encode_pool_workers, "encode_pool_workers")
        for name in (
            "encode_max_batch",
            "long_prompt_chars",
            "chunk_chars",
            "session_cache_capacity",
            "encode_max_pending",
            "encode_max_pending_bytes",
            "session_cache_max_bytes",
        ):
            require_int(getattr(self, name), name, minimum=1)
        window = self.encode_batch_window_ms
        if type(window) not in (float, int) or not math.isfinite(window) or window < 0:
            raise ValueError("encode_batch_window_ms must be finite and non-negative")
        if self.verify_session_cache and not self.enable_session_cache:
            raise ConfigError(
                "tokenizer.verify_session_cache",
                "CACHE_DISABLED",
                "verification requires the session cache",
            )
        if self.skip_tokenizer_init and (
            self.enable_session_cache or self.detokenize_backend != "auto"
        ):
            raise ConfigError(
                "tokenizer.skip_tokenizer_init",
                "TOKENIZER_DISABLED",
                "token-ID-only mode cannot cache text or select a decoder",
            )


@dataclass(frozen=True, slots=True)
class DetokenizeParams:
    """Fixed per-request decode policy; accumulate_text only retains output_text.

    update() always returns a delta. Preemption must retain decoder state rather
    than mutate this policy. Text stop gating for min_tokens belongs to the
    output owner (ResolvedStopPolicy), not to the scheduler.
    """

    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    stop: tuple[str, ...] = ()
    include_stop_str_in_output: bool = False
    accumulate_text: bool = False

    def __post_init__(self) -> None:
        require_frozen(self, "detokenize params")
        for name in (
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "include_stop_str_in_output",
            "accumulate_text",
        ):
            require_bool(getattr(self, name), name)
        if type(self.stop) is not tuple:
            raise TypeError("stop must be a tuple")
        for value in self.stop:
            if type(value) is not str or not value:
                raise ValueError("stop strings must be non-empty strings")

    def max_stop_len(self) -> int:
        return max(map(len, self.stop), default=0)


@dataclass(frozen=True, slots=True)
class DetokUpdate:
    """Append-only text update; consumed_tokens stops at the first text match."""

    delta: str = ""
    stop_matched: str | None = None
    rewind_chars: int = 0
    stalled: bool = False
    consumed_tokens: int = 0

    def __post_init__(self) -> None:
        require_frozen(self, "detokenizer update")
        if type(self.delta) is not str:
            raise TypeError("delta must be text")
        if self.stop_matched is not None and (
            type(self.stop_matched) is not str or not self.stop_matched
        ):
            raise ValueError("stop_matched must be non-empty text")
        require_int(self.rewind_chars, "rewind_chars")
        if self.rewind_chars:
            raise ValueError("append-only output does not support rewind")
        require_int(self.consumed_tokens, "consumed_tokens")
        require_bool(self.stalled, "stalled")


@dataclass(frozen=True, slots=True)
class EncodeResult:
    """Owned IDs; reused_prefix_len is encode-cache reuse, never acquired KV.

    extra is a deeply immutable tuple of key/value pairs, replacing the mutable
    metadata dict. Service callers receive tuple IDs rather than shared lists.
    """

    token_ids: tuple[int, ...]
    reused_prefix_len: int = 0
    truncated: bool = False
    chunks: int = 1
    extra: tuple[tuple[str, str | int | bool], ...] = ()

    def __post_init__(self) -> None:
        require_frozen(self, "encode result")
        if type(self.token_ids) is not tuple or type(self.extra) is not tuple:
            raise TypeError("token_ids and extra must be tuples")
        for token in self.token_ids:
            require_int(token, "token id")
        require_int(self.reused_prefix_len, "reused_prefix_len")
        if self.reused_prefix_len > len(self.token_ids):
            raise ValueError("reused prefix exceeds encoded length")
        require_int(self.chunks, "chunks", minimum=1)
        require_bool(self.truncated, "truncated")
        keys = []
        for pair in self.extra:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("extra entries must be pairs")
            key, value = pair
            require_text(key, "extra key")
            if type(value) not in (str, int, bool):
                raise TypeError("extra values must be string, int or bool")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate extra key")


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
    encode_max_pending: int = 256,
    encode_max_pending_bytes: int = 16 << 20,
    session_cache_max_bytes: int = 16 << 20,
) -> TokenizerConfig:
    """Bridge model-source loading policy and tokenizer-service settings."""
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
        encode_max_pending=encode_max_pending,
        encode_max_pending_bytes=encode_max_pending_bytes,
        session_cache_max_bytes=session_cache_max_bytes,
    )
