"""Engine role configuration: prefill-node / decode-node disaggregation.

A role decides *which phase* this engine node serves so a scheduler per role
can be built on the shared preemption/policy machinery of ``sched/``. ``AUTO``
keeps today's single-node continuous behavior; ``PREFILL`` schedules prefill
chunks only and hands completed prompts to the KV transfer; ``DECODE`` only
schedules requests whose KV already landed (locally cached or transferred).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.utils.validation import require_int

__all__ = ["EngineRole", "RoleConfig", "ResolvedRolePlan"]


class EngineRole(enum.StrEnum):
    """Phase this node owns in a prefill/decode disaggregated deployment."""

    AUTO = "auto"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class RoleConfig(ConfigMixin):
    """Per-node role plus the KV-aware routing hints it consumes."""

    role: EngineRole = EngineRole.AUTO
    node_id: str | None = None
    """Fabric identity of this node; ``None`` derives from node_rank."""
    min_prefix_tokens: int = 0
    """A KV-aware route only matters above this saved-prefix depth."""
    handoff_timeout_s: float = 30.0
    """Grace period for the P→D KV handoff before a retry/abandon."""
    decode_only_requests: bool = False
    """Decode role refuses prompts with local compute left (prefill belongs to P)."""

    def __post_init__(self) -> None:
        if not isinstance(self.role, EngineRole):
            raise ConfigError(
                "role.role",
                "ROLE_INVALID",
                f"role must be one of {tuple(item.value for item in EngineRole)}",
            )
        if self.node_id is not None and not self.node_id.strip():
            raise ConfigError(
                "role.node_id",
                "NODE_ID_BLANK",
                "node_id must be a non-blank string or None",
            )
        require_int(self.min_prefix_tokens, "role.min_prefix_tokens", minimum=0)
        if self.handoff_timeout_s <= 0:
            raise ConfigError(
                "role.handoff_timeout_s",
                "HANDOFF_TIMEOUT_INVALID",
                "handoff timeout must be positive",
            )

    def resolve(self, *, node_rank: int = 0) -> ResolvedRolePlan:
        node_id = self.node_id if self.node_id is not None else f"node-{node_rank}"
        return ResolvedRolePlan(
            role=self.role,
            node_id=node_id,
            min_prefix_tokens=self.min_prefix_tokens,
            handoff_timeout_s=self.handoff_timeout_s,
            decode_only_requests=self.decode_only_requests,
        )


@dataclass(frozen=True, slots=True)
class ResolvedRolePlan(ConfigMixin):
    """Concrete role consumed by the scheduler factory and the fabric."""

    role: EngineRole
    node_id: str
    min_prefix_tokens: int
    handoff_timeout_s: float
    decode_only_requests: bool

    def __post_init__(self) -> None:
        if not isinstance(self.role, EngineRole):
            raise ConfigError(
                "role.role",
                "ROLE_RESOLVED_INVALID",
                "resolved plan must carry a validated EngineRole",
            )
        if not self.node_id:
            raise ConfigError(
                "role.node_id", "NODE_ID_RESOLVED_BLANK", "resolved node_id must not be blank"
            )
