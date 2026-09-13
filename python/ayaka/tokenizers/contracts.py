"""Immutable preprocessing inputs, independent of transport and GPU ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ayaka.configs.tokenizer import require_bool
from ayaka.utils.validation import require_frozen, require_int


@dataclass(frozen=True, slots=True)
class TextInput:
    text: str
    add_special_tokens: bool = True

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be str")
        require_bool(self.add_special_tokens, "add_special_tokens")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("P1 supports system/user/assistant text messages only")
        if type(self.content) is not str:
            raise TypeError("chat content must be text")


@dataclass(frozen=True, slots=True)
class ChatInput:
    messages: tuple[ChatMessage, ...]
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template: str | None = None

    def __post_init__(self) -> None:
        require_frozen(self, "chat")
        if type(self.messages) is not tuple or not self.messages:
            raise ValueError("messages must be a non-empty tuple")
        if any(not isinstance(message, ChatMessage) for message in self.messages):
            raise TypeError("messages must contain ChatMessage values")
        require_bool(self.add_generation_prompt, "add_generation_prompt")
        require_bool(self.continue_final_message, "continue_final_message")
        if self.add_generation_prompt and self.continue_final_message:
            raise ValueError("generation prompt and final-message continuation conflict")
        if self.continue_final_message and self.messages[-1].role != "assistant":
            raise ValueError("continuation requires a final assistant message")
        if self.chat_template is not None and (
            type(self.chat_template) is not str or not self.chat_template
        ):
            raise ValueError("chat_template must be non-empty text")


@dataclass(frozen=True, slots=True)
class TokenIdsInput:
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.token_ids) is not tuple:
            raise TypeError("token_ids must be an immutable tuple")
        for token in self.token_ids:
            require_int(token, "token id")


type EncodeInput = TextInput | ChatInput | TokenIdsInput


@dataclass(frozen=True, slots=True)
class EncodeOptions:
    """Explicit output reservation and opt-in truncation before Request creation."""

    max_output_tokens: int = 128
    max_input_tokens: int | None = None
    truncate: bool = False
    cache_namespace: str = ""

    def __post_init__(self) -> None:
        require_int(self.max_output_tokens, "max_output_tokens", minimum=1)
        if self.max_input_tokens is not None:
            require_int(self.max_input_tokens, "max_input_tokens", minimum=1)
        require_bool(self.truncate, "truncate")
        if type(self.cache_namespace) is not str or len(self.cache_namespace) > 256:
            raise ValueError("cache_namespace must be a string of at most 256 characters")


@dataclass(frozen=True, slots=True)
class TokenizerCapabilities:
    """Actual loaded adapter capabilities; never inferred from a requested mode."""

    backend: Literal["hf", "slow", "disabled"]
    text_encode: bool
    text_decode: bool
    chat: bool
    decode_stream: bool
    byte_decode: bool
    chunked_encode: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("hf", "slow", "disabled"):
            raise ValueError("unknown loaded backend")
        flags = (
            self.text_encode,
            self.text_decode,
            self.chat,
            self.decode_stream,
            self.byte_decode,
            self.chunked_encode,
        )
        for flag in flags:
            require_bool(flag, "capability")
        if self.backend == "disabled" and any(flags):
            raise ValueError("disabled backend cannot advertise text capabilities")
        if (self.decode_stream or self.byte_decode) and not self.text_decode:
            raise ValueError("decoder capability requires text_decode")
