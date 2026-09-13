"""Concrete tokenizer loading and capability resolution, without model allocation."""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.configs.base import ConfigError
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.tokenizers.contracts import TokenizerCapabilities
from ayaka.tokenizers.ports import TokenizerLike


@dataclass(frozen=True, slots=True)
class LoadedTokenizer:
    """Resource bundle, not a serializable config or a KV ownership descriptor."""

    tokenizer: TokenizerLike | None
    capabilities: TokenizerCapabilities

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, TokenizerCapabilities):
            raise TypeError("capabilities must be TokenizerCapabilities")
        if (self.tokenizer is None) != (self.capabilities.backend == "disabled"):
            raise ValueError("adapter and capability state disagree")


def validate_loading_config(config: TokenizerConfig) -> None:
    """Reject settings which P1 cannot honor, including legacy chunk tuning."""
    if not isinstance(config, TokenizerConfig):
        raise TypeError("config must be TokenizerConfig")
    if config.mode not in ("auto", "hf", "slow"):
        raise ConfigError("tokenizer.mode", "UNSUPPORTED_BACKEND", f"no adapter for {config.mode}")
    if config.long_prompt_chars != 256 * 1024 or config.chunk_chars != 64 * 1024:
        raise ConfigError(
            "tokenizer.chunk_chars",
            "UNSUPPORTED_CHUNKING",
            "P1 encodes whole prompts; chunk tuning is not supported",
        )


class DefaultTokenizerFactory:
    @classmethod
    def from_config(cls, config: TokenizerConfig) -> LoadedTokenizer:
        """Resolve explicit modes before loading; disabled mode has no text adapter."""
        validate_loading_config(config)
        if config.skip_tokenizer_init:
            return LoadedTokenizer(
                None,
                TokenizerCapabilities(
                    "disabled",
                    False,
                    False,
                    False,
                    False,
                    False,
                ),
            )
        from ayaka.tokenizers.hf_tokenizer import HfTokenizer

        tokenizer = HfTokenizer.from_config(config)
        byte_decode = tokenizer.verify_byte_path()
        stream = tokenizer.new_decode_stream() is not None
        if config.detokenize_backend == "bytes" and not byte_decode:
            raise ConfigError(
                "tokenizer.detokenize_backend",
                "UNSUPPORTED_BACKEND",
                "byte-table verification failed",
            )
        if config.detokenize_backend == "stream" and not stream:
            raise ConfigError(
                "tokenizer.detokenize_backend", "UNSUPPORTED_BACKEND", "DecodeStream is unavailable"
            )
        return LoadedTokenizer(
            tokenizer,
            TokenizerCapabilities(
                "hf" if tokenizer.is_fast else "slow",
                True,
                True,
                True,
                stream,
                byte_decode,
                False,
            ),
        )
