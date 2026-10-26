"""Cost-to-ready decision between restoring a tiered prefix and recomputing it (research R3).

A host- or remote-tier prefix hit is only worth restoring if the copy path is
ready sooner than recomputing the prefill, with an uncertainty margin, and only
if the transfer fits its hard budgets. This module is the decision function and
its inputs; it is shadow-only: nothing in the tiering path calls it yet, so the
existing promotion behaviour is unchanged. Inputs are measured throughputs of
the current topology (bytes/s for the copy path, tokens/s for prefill), never
datasheet numbers.

Partial restores must stop at a boundary every cache group can restore
(``common_resume_boundary``); the longest single-group match is not a valid
boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from ayaka.prefix.service import common_resume_boundary
from ayaka.utils.validation import require_int

__all__ = ["RestoreChoice", "RestoreDecision", "RestoreInputs", "decide_restore"]


class RestoreChoice(StrEnum):
    RESTORE = "restore"
    RECOMPUTE = "recompute"


@dataclass(frozen=True, slots=True)
class RestoreInputs:
    #: Restorable positions per cache group (each group's own checkpoints).
    group_boundaries: tuple[tuple[int, ...], ...]
    #: Longest prefix the request could reuse at all (its prompt minus the query).
    limit: int
    #: KV bytes per restored token across every group.
    bytes_per_token: int
    #: Measured copy-path throughput (host/remote -> device) in bytes per second.
    restore_bytes_per_second: float
    #: Fixed restore latency: metadata verification, lease publication, fetch RTT.
    restore_overhead_seconds: float
    #: Measured prefill throughput for this model on this device, tokens per second.
    prefill_tokens_per_second: float
    #: Relative uncertainty the restore estimate must beat recompute by (e.g. 0.2).
    margin: float = 0.2
    #: Hard budgets: free pinned staging bytes and free transfer slots.
    pinned_bytes_available: int = 0
    transfer_slots_available: int = 0

    def __post_init__(self) -> None:
        require_int(self.limit, "limit")
        require_int(self.bytes_per_token, "bytes_per_token", minimum=1)
        require_int(self.pinned_bytes_available, "pinned_bytes_available")
        require_int(self.transfer_slots_available, "transfer_slots_available")
        for name in (
            "restore_bytes_per_second",
            "prefill_tokens_per_second",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive measured throughput")
        if not math.isfinite(self.restore_overhead_seconds) or self.restore_overhead_seconds < 0:
            raise ValueError("restore_overhead_seconds must be finite and nonnegative")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class RestoreDecision:
    choice: RestoreChoice
    #: Tokens a restore would cover (a boundary every group can restore).
    boundary: int
    restore_seconds: float
    recompute_seconds: float
    reason: str


def decide_restore(inputs: RestoreInputs) -> RestoreDecision:
    """Restore only when it is ready sooner by the margin and fits every budget."""
    boundary = common_resume_boundary(inputs.group_boundaries, inputs.limit)
    if boundary <= 0:
        return RestoreDecision(RestoreChoice.RECOMPUTE, 0, math.inf, 0.0, "no common boundary")
    nbytes = boundary * inputs.bytes_per_token
    restore = inputs.restore_overhead_seconds + nbytes / inputs.restore_bytes_per_second
    recompute = boundary / inputs.prefill_tokens_per_second
    if inputs.transfer_slots_available < 1:
        return RestoreDecision(
            RestoreChoice.RECOMPUTE, boundary, restore, recompute, "no free transfer slot"
        )
    if nbytes > inputs.pinned_bytes_available:
        return RestoreDecision(
            RestoreChoice.RECOMPUTE, boundary, restore, recompute, "pinned staging budget"
        )
    if restore * (1.0 + inputs.margin) < recompute:
        return RestoreDecision(
            RestoreChoice.RESTORE, boundary, restore, recompute, "ready sooner by the margin"
        )
    return RestoreDecision(
        RestoreChoice.RECOMPUTE, boundary, restore, recompute, "not faster by the margin"
    )
