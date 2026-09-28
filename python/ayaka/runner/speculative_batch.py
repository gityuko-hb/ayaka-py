"""Linear-chain speculation payload shared by draft/verify execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ayaka.utils.validation import require_int, require_text


class SpeculativeRole(StrEnum):
    DRAFT = "draft"
    VERIFY = "verify"
    DRAFT_EXTEND = "draft_extend"


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionBatch:
    """One linear candidate chain, with explicit validity and logical positions."""

    draft_identity: str
    target_identity: str
    prefix: tuple[int, ...]
    candidates: tuple[int, ...]
    vocab_size: int

    def __post_init__(self) -> None:
        require_text(self.draft_identity, "draft identity")
        require_text(self.target_identity, "target identity")
        require_int(self.vocab_size, "vocab_size", minimum=1)
        if not self.prefix or len(self.candidates) > 16:
            raise ValueError("invalid speculative prefix/width")
        for token in (*self.prefix, *self.candidates):
            require_int(token, "token id")
            if token >= self.vocab_size:
                raise ValueError("candidate token is outside the shared vocabulary")

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(range(len(self.prefix) - 1, len(self.prefix) + len(self.candidates)))

    @property
    def query_offsets(self) -> tuple[int, int]:
        return (0, len(self.candidates) + 1)

    @property
    def valid(self) -> tuple[bool, ...]:
        return (True,) * (len(self.candidates) + 1)
