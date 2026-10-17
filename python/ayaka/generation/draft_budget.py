"""Generated and verified proposal budgets (anchor and bonus excluded)."""

from dataclasses import dataclass
from enum import StrEnum

from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class DraftBudget:
    generated: int
    verified: int

    def __post_init__(self) -> None:
        require_int(self.generated, "generated")
        require_int(self.verified, "verified")
        if not self.verified <= self.generated <= 255:
            raise ValueError("draft budget requires verified <= generated <= 255")

    def validate(self, cap: int) -> None:
        require_int(cap, "proposal cap")
        if self.generated > min(cap, 255) or (self.generated and not self.verified):
            raise ValueError("draft budget is invalid or exceeds the method's proposal width")


class DraftPolicy(StrEnum):
    """Method factories carry the fixed budget separately from this policy."""

    FIXED = "fixed"
    ADAPTIVE = "adaptive"
