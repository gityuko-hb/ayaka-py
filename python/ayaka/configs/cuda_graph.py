"""Resolve the existing operator config, without introducing another enable flag."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ayaka.configs.base import validate_buckets
from ayaka.configs.serving import ServingConfig
from ayaka.device.nvidia_profile import (
    NvidiaExecutionProfile,
    ResolvedExecutionProfile,
)
from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class DecodeCudaGraphConfig:
    requested: bool
    requested_buckets: tuple[int, ...] | None
    buckets: tuple[int, ...]
    provenance: str
    backend: str = "full"
    profile: NvidiaExecutionProfile = NvidiaExecutionProfile.CONSTRAINED

    @classmethod
    def resolve(
        cls,
        config: ServingConfig,
        *,
        max_requests: int,
        max_tokens: int,
        profile: NvidiaExecutionProfile = NvidiaExecutionProfile.CONSTRAINED,
        admission: ResolvedExecutionProfile | None = None,
    ) -> DecodeCudaGraphConfig:
        """Resolve the decode graph policy from the operator config.

        This runs before the device is known, so it can only apply capacity
        ceilings. ``admission`` carries the budget-resolved set from
        :mod:`~ayaka.device.nvidia_profile` once the device and
        memory plan exist; passing it is what replaces the old fixed ladder with
        a decision the device and budget actually justify.

        Args:
            config: The operator's serving config.
            max_requests: Scheduler request ceiling.
            max_tokens: Token capacity ceiling for this runner.
            profile: Which candidate ladder to start from.
            admission: A budget-admitted profile. Its ``admitted_buckets``
                replace the capacity default; an empty set means eager.

        Raises:
            ValueError: when an explicit bucket set is malformed or exceeds the
                runner ceilings. A malformed operator request is a startup
                error, never a silent trim.
        """
        require_int(max_requests, "max_requests", minimum=1)
        require_int(max_tokens, "max_tokens", minimum=1)
        if config.decode_graph is not None and type(config.decode_graph) is not bool:
            raise TypeError("decode_graph must be a boolean")
        ceiling = min(max_requests, max_tokens)
        explicit = config.graph_buckets
        if explicit is not None:
            validate_buckets(explicit)
            if config.decode_graph and explicit[-1] > ceiling:
                raise ValueError("explicit graph_buckets exceed runner capacity; lower the set")
        if admission is not None:
            buckets = admission.admitted_buckets
            provenance = f"{admission.provenance}:{admission.profile.value}"
        elif explicit is not None:
            buckets = explicit
            provenance = "explicit"
        else:
            # Pre-bootstrap capacity default. Bring-up replaces this with the
            # budget-admitted set; this keeps the config resolvable before a
            # device exists.
            candidates = _CAPACITY_DEFAULTS[profile]
            buckets = tuple(b for b in candidates if b <= ceiling)
            provenance = f"capacity_default:{profile.value}"
        return cls(
            bool(config.decode_graph),
            explicit,
            buckets if config.decode_graph else (),
            provenance,
            profile=profile,
        )


#: Capacity-only fallback ladders, used before a device is known. The real
#: policy lives in :mod:`~ayaka.device.nvidia_profile`; keeping a
#: copy here would be a second authority, so this only mirrors the same names
#: for the pre-bootstrap window.
_CAPACITY_DEFAULTS: dict[NvidiaExecutionProfile, tuple[int, ...]] = {
    NvidiaExecutionProfile.CONSTRAINED: (1, 2, 4, 8),
    NvidiaExecutionProfile.LATENCY: (1, 2, 4, 8, 16, 32),
    NvidiaExecutionProfile.BALANCED: (1, 2, 4, 8, 16, 32, 64, 128),
    NvidiaExecutionProfile.THROUGHPUT: (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256),
    NvidiaExecutionProfile.EXPLICIT: (),
}


@dataclass(frozen=True, slots=True)
class ExecutionStartupReport:
    requested: bool
    requested_buckets: tuple[int, ...] | None
    resolved_buckets: tuple[int, ...]
    captured_buckets: tuple[int, ...]
    provenance: str
    graph_backend: str
    attention_backend: str
    device: str
    flight_slots: int
    reserve_bytes: int
    capture_bytes: int
    capture_seconds: float
    peak_capture_bytes: int
    state: str
    fallback_reason: str
    requested_profile: str = "constrained"
    #: EP1 bring-up facts. Every field is optional so a report produced by an
    #: older bootstrap path still reads; an absent value means "not resolved",
    #: never a silent zero.
    device_sm: str = ""
    device_uuid: str = ""
    arch_reachability: str = ""
    partition_kind: str = ""
    partition_budget_bytes: int = 0
    driver_card_bytes: int | None = None
    framework_ptx_architecture: int | None = None
    support_tiers: Mapping[str, str] = field(default_factory=dict)
    support_reasons: tuple[str, ...] = ()
    profile_explanation: str = ""
    phases: Mapping[str, object] = field(default_factory=dict)
