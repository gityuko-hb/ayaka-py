"""Verified byte-table decoding; finalization is shared with all backends."""

from __future__ import annotations

import codecs
import random
from collections.abc import Sequence

from ayaka.configs.tokenizer import DetokenizeParams
from ayaka.tokenizers.detokenizer import IncrementalDetokenizer
from ayaka.utils.validation import require_int


def verify_byte_table(tokenizer, samples: Sequence[str], *, n_random: int = 512) -> bool:
    """Check concatenated byte decoding against both skip policies.

    This is a capability probe, not a proof for arbitrary decoder pipelines.
    Only valid vocabulary IDs are sampled, including added/special tokens.
    """
    table = tokenizer.token_bytes_table()
    if not table:
        return False
    valid = tuple(sorted(set(tokenizer.get_vocab().values())))
    if not valid or any(token < 0 or token >= len(table) for token in valid):
        return False

    def ok(ids: Sequence[int]) -> bool:
        for skip in (False, True):
            kept = [token for token in ids if not skip or token not in tokenizer.all_special_ids]
            data = b"".join(table[token] for token in kept)
            if data.decode("utf-8", errors="replace") != tokenizer.decode(
                ids, skip_special_tokens=skip
            ):
                return False
        return True

    for sample in samples:
        if not ok(tokenizer.encode(sample, add_special_tokens=False)):
            return False
    for token in tokenizer.all_special_ids:
        if not ok([token]):
            return False
    rng = random.Random(0)
    return all(ok(rng.choices(valid, k=rng.randint(1, 24))) for _ in range(n_random))


class ByteIncrementalDetokenizer(IncrementalDetokenizer):
    def __init__(
        self, tokenizer, params: DetokenizeParams, *, prompt_token_ids: Sequence[int] = ()
    ) -> None:
        super().__init__(params)
        if not params.skip_special_tokens:
            raise ValueError("byte backend requires skip_special_tokens=True")
        self._tbl = tuple(tokenizer.token_bytes_table())
        if not self._tbl:
            raise ValueError("empty byte table")
        self._valid_ids = frozenset(tokenizer.get_vocab().values())
        self._special = frozenset(tokenizer.all_special_ids)
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        for token in prompt_token_ids:
            require_int(token, "prompt token id")
            self._validate_token(token)
            self._decode_next(token)
        if self._dec.getstate()[0]:
            raise ValueError("prompt ends inside a UTF-8 character")

    def _validate_token(self, token_id: int) -> None:
        if token_id not in self._valid_ids or token_id >= len(self._tbl):
            raise ValueError(f"unknown token id: {token_id}")

    def _decode_next(self, token_id: int) -> str:
        if token_id in self._special:
            return ""
        return self._dec.decode(self._tbl[token_id])

    def _decode_tail(self) -> str:
        return self._dec.decode(b"", final=True)

    @property
    def n_invalid_ids(self) -> int:
        """Invalid IDs now raise rather than being silently discarded."""
        return 0
