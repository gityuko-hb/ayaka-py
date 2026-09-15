from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from ayaka.configs.tokenizer import TokenizerConfig

# DEFAULT_PAT_STR uses the OpenAI cl100k regex split pattern.
# It acts as the fallback default, which can be overridden by vocab metadata.
DEFAULT_PAT_STR = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=32)
def _get_jinja_template(chat_template: str):
    """Compile and cache Jinja chat templates to avoid repeated AST parsing."""
    from jinja2 import Template

    return Template(chat_template)


class SpecialTokenPolicy(StrEnum):
    """Defines how to handle literal special tokens found in user-provided content."""

    # Default: Raise an exception.
    # Literal special tokens (e.g., <|im_end|>) indicate potential prompt injection.
    REJECT = "reject"

    # Insert U+200B (ZeroWidth Space) to break the token structure while preserving visual fidelity.
    ESCAPE = "escape"

    # Permissive (vLLM style)
    # Retains literal strings, leaving chat templates vulnerable to hijacking.
    ALLOW = "allow"


class SpecialTokenInContent(ValueError):
    """Raised when user or tool content contains literal special tokens under REJECT policy."""

    pass


def load_tiktoken_vocab(
    path: Path,
) -> tuple[dict[bytes, int], dict[str, int], str | None, int | None]:
    """Load vocabulary from a file.

    Returns:
        tuple: (mergeable_ranks, special_tokens, pat_str, explicit_n_vocab)
    """
    raw_head = path.read_bytes()[:7]
    # Strip UTF-8 BOM if present
    if raw_head.startswith(b"\xef\xbb\xbf"):
        raw_head = raw_head[3:]
    head = raw_head.lstrip()

    tried: list[str] = []

    # Case 1: JSON format (xtok or flat base64 mapping)
    if head[:1] == b"{":
        tried.append("json")
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if "regular_tokens" in obj:  # xtok format (e.g., StepFun)
            merge = {bytes(it["bytes"]): it["token"] for it in obj["regular_tokens"]}
            special = {
                bytes(it["bytes"]).decode(): it["token"] for it in obj.get("special_tokens", [])
            }
            return merge, special, obj.get("pat_str"), obj.get("vocab_size")

        # Flat JSON mapping: {token_b64: rank}
        try:
            merge = {base64.b64decode(k): int(v) for k, v in obj.items()}
            return merge, {}, None, None
        except (binascii.Error, ValueError) as e:
            raise ValueError(
                f"{path}: JSON file is neither xtok format nor a flat {{token_b64: rank}} mapping"
            ) from e

    # Case 2: Line-based format '<base64> <rank>'
    tried.append("line-delimited '<base64> <rank>'")
    merge: dict[bytes, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"{path}:{lineno}: Expected '<base64> <rank>', but received {line[:40]!r}. "
                f"Attempted formats: {', '.join(tried)}"
            )
        merge[base64.b64decode(parts[0])] = int(parts[1])

    if not merge:
        raise ValueError(f"{path}: Vocabulary file is empty. Attempted formats: {', '.join(tried)}")

    return merge, {}, None, None


class TiktokenTokenizer:
    __slots__ = (
        "_enc",
        "_name",
        "_fp",
        "_tbl",
        "_special_str2id",
        "_special_ids",
        "_special_re",
        "_policy",
        "_max_token_id",
        "_max_chars",
        "_eos",
        "_bos",
        "_pad",
        "_vocab_str",
        "_trunc_side",
    )

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> TiktokenTokenizer:
        path = Path(config.tokenizer)
        if path.is_dir():
            for name in ("tiktoken.model", "tokenizer.model", "tokenizer.tiktoken"):
                if (path / name).is_file():
                    path = path / name
                    break
            else:
                raise FileNotFoundError(
                    f"Tiktoken vocabulary file not found in {path} "
                    "(searched for tiktoken.model, tokenizer.model, tokenizer.tiktoken)"
                )

        merge, special, pat, n_vocab = load_tiktoken_vocab(path)

        cfg_path = path.parent / "tokenizer_config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            for tid_s, info in (cfg.get("added_tokens_decoder") or {}).items():
                content = (info or {}).get("content")
                if content:
                    special.setdefault(content, int(tid_s))

        return cls(
            mergeable_ranks=merge,
            special_tokens=special,
            pat_str=pat or DEFAULT_PAT_STR,
            explicit_n_vocab=n_vocab,
            name=str(config.tokenizer),
            truncation_side=config.truncation_side,
        )

    def __init__(
        self,
        *,
        mergeable_ranks: Mapping[bytes, int],
        special_tokens: Mapping[str, int],
        pat_str: str = DEFAULT_PAT_STR,
        explicit_n_vocab: int | None = None,
        name: str = "tiktoken",
        truncation_side: str = "left",
        special_token_policy: SpecialTokenPolicy = SpecialTokenPolicy.REJECT,
    ):
        import tiktoken

        kwargs: dict[str, Any] = {
            "name": name,
            "pat_str": pat_str,
            "mergeable_ranks": dict(mergeable_ranks),
            "special_tokens": dict(special_tokens),
        }
        if explicit_n_vocab is not None:
            kwargs["explicit_n_vocab"] = explicit_n_vocab

        self._enc = tiktoken.Encoding(**kwargs)
        self._name = name
        self._policy = special_token_policy
        self._trunc_side = truncation_side

        self._special_str2id = dict(special_tokens)
        self._special_ids = frozenset(special_tokens.values())

        # Sort tokens in descending order by length to avoid partial prefix matching
        lits = sorted(special_tokens, key=len, reverse=True)
        self._special_re = re.compile("|".join(re.escape(x) for x in lits)) if lits else None

        # Build token-to-bytes lookup table
        self._max_token_id = max(
            max(mergeable_ranks.values(), default=-1),
            max(special_tokens.values(), default=-1),
        )
        tbl: list[bytes] = [b""] * (self._max_token_id + 1)
        for b, i in mergeable_ranks.items():
            tbl[i] = b
        for s, i in special_tokens.items():
            tbl[i] = s.encode("utf-8")
        self._tbl = tbl

        # Compute max characters, including special tokens
        max_mergeable = max((len(b) for b in mergeable_ranks), default=1)
        max_special = max((len(s.encode("utf-8")) for s in special_tokens), default=1)
        self._max_chars = max(max_mergeable, max_special)

        # Token fallbacks
        self._eos = (
            special_tokens.get("<|im_end|>")
            or special_tokens.get("<|endoftext|>")
            or special_tokens.get("</s>")
        )
        self._bos = special_tokens.get("<|im_start|>") or special_tokens.get("<s>")
        self._pad = (
            special_tokens.get("<|pad|>")
            or special_tokens.get("<pad>")
            or special_tokens.get("[PAD]")
            or self._eos
        )
        self._vocab_str: dict[str, int] | None = None

        # State fingerprint (includes regex pattern & explicit vocab size
        # to prevent KV cache collisions)
        h = hashlib.blake2b(digest_size=16)
        h.update(pat_str.encode("utf-8") + b"\0")
        if explicit_n_vocab is not None:
            h.update(explicit_n_vocab.to_bytes(4, "little"))
        for b, i in sorted(mergeable_ranks.items(), key=lambda kv: kv[1]):
            h.update(i.to_bytes(4, "little") + b + b"\0")
        for s, i in sorted(special_tokens.items(), key=lambda kv: kv[1]):
            h.update(i.to_bytes(4, "little") + s.encode("utf-8") + b"\0")
        self._fp = h.hexdigest()

    # -- Identifiers
    @property
    def name_or_path(self) -> str:
        return self._name

    def fingerprint(self) -> str:
        return self._fp

    # -- Metadata
    @property
    def eos_token_id(self) -> int | None:
        return self._eos

    @property
    def bos_token_id(self) -> int | None:
        return self._bos

    @property
    def pad_token_id(self) -> int | None:
        return self._pad

    @property
    def all_special_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        return tuple(self._special_str2id)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    @property
    def max_chars_per_token(self) -> int:
        return self._max_chars

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def supports_chunked_encode(self) -> bool:
        return True

    # -- Representations
    def token_bytes_table(self) -> list[bytes]:
        return self._tbl

    # -- Encoding
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode user-provided text.

        Guarantees that no special token IDs are produced; any literal special
        tokens present in the text are treated as plain text bytes.
        """
        del add_special_tokens
        return self._enc.encode(text, allowed_special=set(), disallowed_special=())

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        del add_special_tokens
        return self._enc.encode_batch(list(texts), allowed_special=set(), disallowed_special=())

    def sanitize_content(self, content: Any) -> Any:
        """Sanitize user/tool-provided inputs according to SpecialTokenPolicy.

        Handles strings, nested lists, and dictionaries (e.g., multimodal payloads).
        Must be applied prior to template rendering to ensure subsequent literal
        special token matching in `encode_rendered` is safe.
        """
        if self._special_re is None or self._policy is SpecialTokenPolicy.ALLOW:
            return content

        if isinstance(content, str):
            if self._policy is SpecialTokenPolicy.REJECT:
                m = self._special_re.search(content)
                if m is not None:
                    raise SpecialTokenInContent(
                        f"Content contains literal special token {m.group(0)!r} at index {m.start()}. "  # noqa: E501
                        f"Set special_token_policy=ESCAPE to neutralize it, or "
                        f"ALLOW to follow permissive behavior (vulnerable to template injection)."
                    )
                return content

            # Neutralize any token format by inserting U+200B after the first character
            return self._special_re.sub(
                lambda m: m.group(0)[:1] + "\u200b" + m.group(0)[1:],
                content,
            )

        if isinstance(content, list):
            return [self.sanitize_content(item) for item in content]
        if isinstance(content, dict):
            return {k: self.sanitize_content(v) for k, v in content.items()}

        return content

    def encode_rendered(self, rendered: str) -> list[int]:
        """Encode an already-rendered prompt string using native Rust execution.

        Special tokens mapped in the vocabulary are converted to special token IDs,
        while all other components are parsed as regular BPE tokens.
        """
        return self._enc.encode(rendered, allowed_special="all")

    def truncate(self, ids: list[int], max_length: int | None) -> tuple[list[int], bool]:
        """Truncate token IDs to max_length based on the configured truncation side."""
        if max_length is None or len(ids) <= max_length:
            return ids, False
        if max_length <= 0:
            return [], True
        return (ids[-max_length:] if self._trunc_side == "left" else ids[:max_length]), True

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        if chat_template is None:
            raise ValueError("TiktokenTokenizer requires an explicit `chat_template`.")

        template = _get_jinja_template(chat_template)

        clean = []
        for m in messages:
            msg_dict = dict(m)
            if "content" in msg_dict and msg_dict["content"] is not None:
                msg_dict["content"] = self.sanitize_content(msg_dict["content"])
            clean.append(msg_dict)

        return template.render(
            messages=clean,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    # -- Decoding
    def decode(self, ids: Sequence[int] | int, *, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            seq = [ids]
        else:
            # Handles PyTorch tensors, NumPy arrays, and standard sequences
            seq = [int(i) for i in ids]

        if skip_special_tokens:
            sp = self._special_ids
            seq = [i for i in seq if i not in sp]

        return self._enc.decode(seq)

    def convert_ids_to_tokens(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Convert token IDs to token strings using reversible surrogateescape decoding."""
        out = []
        for item in ids:
            i = int(item)
            if skip_special_tokens and i in self._special_ids:
                continue
            b = self._tbl[i] if 0 <= i < len(self._tbl) else b""
            out.append(b.decode("utf-8", errors="surrogateescape"))
        return out

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        raw = b"".join(t.encode("utf-8", errors="surrogateescape") for t in tokens)
        return raw.decode("utf-8", errors="replace")

    def new_decode_stream(self, *, skip_special_tokens: bool = False):
        return None

    # Vocab Access
    def get_vocab(self) -> dict[str, int]:
        if self._vocab_str is None:
            v = {
                b.decode("utf-8", errors="surrogateescape"): i for i, b in enumerate(self._tbl) if b
            }
            self._vocab_str = v
        return self._vocab_str

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._special_str2id)

    def get_decoded_vocab(self) -> list[bytes]:
        return self._tbl
