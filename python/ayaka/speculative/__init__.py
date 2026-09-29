"""Scheduler-aware, transactional speculative decoding subsystem.

The dense draft-model path in :mod:`ayaka.runner.speculative_runner` remains the
golden reference until the paged production path passes parity tests.
"""

from ayaka.speculative.config import (
    MAX_DRAFT_TOKENS,
    AcceptanceMethod,
    NGramConfig,
    NGramSelection,
    PolicyConfig,
    PolicyKind,
    SpeculativeDecodingConfig,
)
from ayaka.speculative.mode import (
    Certification,
    ModeCapabilities,
    SpeculativeMode,
    mode_capabilities,
    resolve_mode,
)
from ayaka.speculative.plan import SpecDisableReason, SpeculativeSlicePlan

__all__ = [
    "MAX_DRAFT_TOKENS",
    "AcceptanceMethod",
    "Certification",
    "ModeCapabilities",
    "NGramConfig",
    "NGramSelection",
    "PolicyConfig",
    "PolicyKind",
    "SpecDisableReason",
    "SpeculativeDecodingConfig",
    "SpeculativeMode",
    "SpeculativeSlicePlan",
    "mode_capabilities",
    "resolve_mode",
]
