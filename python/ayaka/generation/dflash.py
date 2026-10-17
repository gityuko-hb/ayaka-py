"""DFlash/DSpark block-draft method; numerical execution belongs to the backend."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ayaka.generation.contracts import Demand, bounded
from ayaka.generation.draft_budget import DraftBudget
from ayaka.generation.draft_control import DraftLoad
from ayaka.generation.method import MethodCheckpoint, MethodKind, MethodRequirements
from ayaka.generation.mtp import DrafterState


@dataclass(frozen=True, slots=True)
class DFlash:
    artifact: str
    width: int
    budget: DraftBudget | None = None
    load: DraftLoad | None = None

    def __post_init__(self) -> None:
        if not self.artifact.strip():
            raise ValueError("DFlash identity must be nonempty")
        bounded(self.width, "DFlash proposal width", 255, 1)
        if self.budget is not None:
            self.budget.validate(self.width)

    def with_budget(self, budget: DraftBudget) -> DFlash:
        return replace(self, budget=budget)

    def adaptive(self) -> DFlash:
        return replace(self, load=DraftLoad())

    def identity(self) -> str:
        suffix = ":adaptive" if self.load is not None else ""
        if self.budget is not None:
            suffix += f":{self.budget.generated}:{self.budget.verified}"
        return f"dflash:{self.artifact}:{self.width}{suffix}"

    def proposals(self) -> int:
        return self.width

    def priming_requires_successor(self) -> bool:
        return False

    def embeds_prompt_tokens(self) -> bool:
        return False

    def requires(self) -> MethodRequirements:
        return MethodRequirements(Demand.FEATURES, Demand.NONE, Demand.FEATURES, True)

    def create(self, checkpoint: MethodCheckpoint | None = None) -> DrafterState:
        return DrafterState(
            MethodKind.DFLASH,
            self.width,
            self.budget or DraftBudget(self.width, self.width),
            self.load,
            checkpoint,
        )
