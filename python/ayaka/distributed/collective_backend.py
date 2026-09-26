"""Collective policy dispatch and group agreement.

This module is the seam between the runtime-owned process group and the
optional custom collective backends DC4 will implement. It owns exactly three
things:

* the user-facing policy decision (``torch`` / ``auto`` / ``custom_required``)
  and the stable reason codes a decision carries;
* the one-time setup agreement that proves every rank sees the same
  membership, generation, capability and probe evidence *before* any data
  launch;
* the per-call dispatcher that implements :class:`CommunicationBackend` while
  never adding a host rendezvous to the hot path: a call is routed from the
  already-agreed plan plus local tensor metadata only.

Failure semantics are fail-closed (INV-2/INV-3): an unsupported call falls back
to Torch only before the custom backend is touched, a custom error raised with
:class:`CollectivePreLaunchError` is still rollback-safe, and every other error
after the custom backend was entered is sticky — the dispatcher refuses all
further calls and never retries or falls back on already-mutated state.

Importing this module is torch-free and side-effect-free; the verified probe is
imported lazily and only when a custom backend actually asks for evidence.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NoReturn, Protocol, runtime_checkable

from ayaka.configs.base import ConfigMixin
from ayaka.configs.parallel import CollectivePolicy
from ayaka.distributed.device import (
    AsyncHandle,
    CommOpType,
    CommunicationBackend,
    DeviceGroup,
)
from ayaka.types import DType
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.validation import require_text

__all__ = [
    "CUSTOM_ALIGNMENT_BYTES",
    "CUSTOM_BACKEND_NAME",
    "DEFAULT_AGREEMENT_TIMEOUT_S",
    "DEFAULT_SIGNAL_ALGORITHM",
    "DISPATCH_SCHEMA_VERSION",
    "AgreementChannel",
    "CollectiveAgreementError",
    "CollectiveCapability",
    "CollectiveDispatchBackend",
    "CollectiveDispatchError",
    "CollectiveDispatchFailed",
    "CollectivePlan",
    "CollectivePostLaunchError",
    "CollectivePreLaunchError",
    "CollectiveReason",
    "CollectiveRequiredError",
    "CollectiveSetup",
    "CustomCapability",
    "CustomCollective",
    "DispatchDecision",
    "DispatchMetrics",
    "DispatchMetricsSnapshot",
    "group_reason",
    "probe_collective_capability",
    "setup_collective_backend",
]

#: Wire schema for :class:`CollectivePlan`; a different value is rejected.
DISPATCH_SCHEMA_VERSION = 1

#: Default deadline for every setup collective when the caller passes none.
DEFAULT_AGREEMENT_TIMEOUT_S = 30.0

#: Arrow/vector alignment a custom backend must declare.
CUSTOM_ALIGNMENT_BYTES = 16

CUSTOM_BACKEND_NAME = "custom_all_reduce"
DEFAULT_SIGNAL_ALGORITHM = "signal_epoch"

_CUSTOM_CAPABILITY = "distributed.collective_backend"


class CollectiveReason(StrEnum):
    """Stable decision/fallback codes shared with probe and runtime metrics.

    The first block is the minimum vocabulary the spec requires; the
    ``prelaunch_failed``/``postlaunch_failed`` codes are extensions used by
    call metrics so a failure gate can distinguish rollback from quarantine.
    """

    OK = "ok"
    SINGLETON = "singleton"
    POLICY_TORCH = "policy_torch"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    CUSTOM_UNAVAILABLE = "custom_unavailable"
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    UNSUPPORTED_OP = "unsupported_op"
    UNSUPPORTED_LAYOUT = "unsupported_layout"
    UNSUPPORTED_SIZE = "unsupported_size"
    P2P_UNVERIFIED = "p2p_unverified"
    ATOMIC_UNSUPPORTED = "atomic_unsupported"
    WORKSPACE_UNAVAILABLE = "workspace_unavailable"
    GRAPH_UNCERTIFIED = "graph_uncertified"
    GENERATION_MISMATCH = "generation_mismatch"
    RANK_DISAGREEMENT = "rank_disagreement"
    PRELAUNCH_FAILED = "prelaunch_failed"
    POSTLAUNCH_FAILED = "postlaunch_failed"


#: Fixed priority used to collapse per-rank refusal reasons into the one reason
#: every rank records. Keeping the order explicit makes the group-visible
#: decision identical no matter which rank refused first.
_REASON_PRIORITY: tuple[CollectiveReason, ...] = (
    CollectiveReason.GENERATION_MISMATCH,
    CollectiveReason.WORKSPACE_UNAVAILABLE,
    CollectiveReason.ATOMIC_UNSUPPORTED,
    CollectiveReason.P2P_UNVERIFIED,
    CollectiveReason.GRAPH_UNCERTIFIED,
    CollectiveReason.UNSUPPORTED_DTYPE,
    CollectiveReason.UNSUPPORTED_OP,
    CollectiveReason.UNSUPPORTED_LAYOUT,
    CollectiveReason.UNSUPPORTED_SIZE,
    CollectiveReason.UNSUPPORTED_PLATFORM,
    CollectiveReason.CUSTOM_UNAVAILABLE,
    CollectiveReason.RANK_DISAGREEMENT,
    CollectiveReason.POLICY_TORCH,
    CollectiveReason.SINGLETON,
    CollectiveReason.OK,
)
_REASON_RANK = {reason: index for index, reason in enumerate(_REASON_PRIORITY)}


def group_reason(reasons: Sequence[CollectiveReason]) -> CollectiveReason:
    """Collapse per-rank reasons into one deterministic group-visible code."""
    if not reasons:
        raise ValueError("a group reason needs at least one rank reason")
    for reason in reasons:
        if not isinstance(reason, CollectiveReason):
            raise TypeError("group reasons must be CollectiveReason values")
    return min(reasons, key=lambda reason: _REASON_RANK[reason])


class CollectiveDispatchError(RuntimeError):
    """Base class for policy, agreement and dispatch failures."""


class CollectiveAgreementError(CollectiveDispatchError):
    """The group refused a custom backend under ``custom_required``."""

    def __init__(self, reason: CollectiveReason, *, detail: str = "") -> None:
        self.reason = reason
        suffix = f": {detail}" if detail else ""
        super().__init__(f"collective group agreement refused ({reason.value}){suffix}")


class CollectiveRequiredError(CollectiveDispatchError):
    """A call is not eligible for the agreed custom backend under ``custom_required``."""

    def __init__(self, reason: CollectiveReason, *, detail: str = "") -> None:
        self.reason = reason
        suffix = f": {detail}" if detail else ""
        super().__init__(f"custom collective is required but unavailable ({reason.value}){suffix}")


class CollectivePreLaunchError(CollectiveDispatchError):
    """Raised by a custom backend when it failed *before* mutating or launching.

    Only this error is rollback-safe: the dispatcher may delegate the call to
    Torch (``auto``) or refuse it (``custom_required``).
    """


class CollectivePostLaunchError(CollectiveDispatchError):
    """Raised by a custom backend after device work was launched."""


class CollectiveDispatchFailed(CollectiveDispatchError):
    """A dispatcher is quarantined after a post-launch failure."""

    def __init__(self, detail: str, *, reason: CollectiveReason | None = None) -> None:
        self.reason = reason
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Capability records
# ---------------------------------------------------------------------------


def _sorted_enums(values: Sequence[Any], key: str) -> tuple[Any, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{key} must be a non-empty sequence")
    normalized = {value: None for value in values}
    return tuple(
        sorted(normalized, key=lambda item: getattr(item, "value", getattr(item, "label", "")))
    )


def _require_positive_pair(value: Any, label: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(part) is not int or part <= 0 for part in value)
    ):
        raise ValueError(f"{label} must be a pair of positive integers")
    return value


@dataclass(frozen=True, slots=True)
class CustomCapability(ConfigMixin):
    """What one custom collective backend implementation can serve.

    ``supported_ops``/``supported_dtypes`` are explicit allow-lists: a custom
    backend is never assumed to serve an op or dtype it did not declare.
    ``min_bytes``/``max_bytes``/``alignment`` bound the payload it accepts and
    ``workspace_bytes`` is the budget the group agrees on at setup.
    """

    backend: str = CUSTOM_BACKEND_NAME
    version: str = ""
    algorithm: str = DEFAULT_SIGNAL_ALGORITHM
    supported_ops: tuple[CommOpType, ...] = (CommOpType.SUM,)
    supported_dtypes: tuple[DType, ...] = (DType.FP32, DType.FP16, DType.BF16)
    min_bytes: int = 0
    max_bytes: int | None = None
    alignment: int = CUSTOM_ALIGNMENT_BYTES
    workspace_bytes: int = 0
    graph_certified: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("backend", self.backend),
            ("algorithm", self.algorithm),
        ):
            require_text(value, f"custom capability {label}")
        if not isinstance(self.version, str):
            raise TypeError("custom capability version must be a string")
        object.__setattr__(self, "supported_ops", _sorted_enums(self.supported_ops, "ops"))
        object.__setattr__(self, "supported_dtypes", _sorted_enums(self.supported_dtypes, "dtypes"))
        for label, value in (
            ("supported_ops", self.supported_ops),
            ("supported_dtypes", self.supported_dtypes),
        ):
            for entry in value:
                if label == "supported_ops" and not isinstance(entry, CommOpType):
                    raise TypeError("custom capability ops must be CommOpType values")
                if label == "supported_dtypes" and not isinstance(entry, DType):
                    raise TypeError("custom capability dtypes must be DType values")
        for label, value, minimum in (
            ("min_bytes", self.min_bytes, 0),
            ("alignment", self.alignment, 1),
            ("workspace_bytes", self.workspace_bytes, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"custom capability {label} must be an integer >= {minimum}")
        if self.max_bytes is not None:
            if type(self.max_bytes) is not int or self.max_bytes < self.min_bytes:
                raise ValueError("custom capability max_bytes must be >= min_bytes")
        if type(self.graph_certified) is not bool:
            raise TypeError("custom capability graph_certified must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Stable wire form: op values and dtype labels, never tuple payloads."""
        return {
            "backend": self.backend,
            "version": self.version,
            "algorithm": self.algorithm,
            "supported_ops": [op.value for op in self.supported_ops],
            "supported_dtypes": [dtype.label for dtype in self.supported_dtypes],
            "min_bytes": self.min_bytes,
            "max_bytes": self.max_bytes,
            "alignment": self.alignment,
            "workspace_bytes": self.workspace_bytes,
            "graph_certified": self.graph_certified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CustomCapability:
        if not isinstance(payload, Mapping):
            raise TypeError("custom capability payload must be a mapping")
        ops = payload.get("supported_ops", ())
        dtypes = payload.get("supported_dtypes", ())
        if not isinstance(ops, (list, tuple)) or not isinstance(dtypes, (list, tuple)):
            raise ValueError("custom capability ops and dtypes must be sequences")
        try:
            resolved_dtypes = tuple(_DTYPE_BY_LABEL[str(item)] for item in dtypes)
        except KeyError as exc:
            raise ValueError(f"unknown custom capability dtype {exc.args[0]!r}") from exc
        return cls(
            backend=str(payload.get("backend", CUSTOM_BACKEND_NAME)),
            version=str(payload.get("version", "")),
            algorithm=str(payload.get("algorithm", DEFAULT_SIGNAL_ALGORITHM)),
            supported_ops=tuple(CommOpType(str(item)) for item in ops),
            supported_dtypes=resolved_dtypes,
            min_bytes=int(payload.get("min_bytes", 0)),
            max_bytes=(None if payload.get("max_bytes") is None else int(payload["max_bytes"])),
            alignment=int(payload.get("alignment", CUSTOM_ALIGNMENT_BYTES)),
            workspace_bytes=int(payload.get("workspace_bytes", 0)),
            graph_certified=bool(payload.get("graph_certified", False)),
        )


@dataclass(frozen=True, slots=True)
class CollectiveCapability(ConfigMixin):
    """One rank's local view of what the group could agree on.

    ``reason`` is the local eligibility verdict; ``probe_agreement_key`` is the
    digest ranks compare to prove they ran the same verified probe, and
    ``verified_rank_pairs`` names the ordered group-rank pairs that probe
    verified (ordinals never cross ranks).
    """

    group_name: str
    ranks: tuple[int, ...]
    world_size: int
    local_rank: int
    session_generation: tuple[int, int]
    custom: CustomCapability | None = None
    reason: CollectiveReason = CollectiveReason.CUSTOM_UNAVAILABLE
    probe_status: str = ""
    probe_reason: str = ""
    probe_agreement_key: str = ""
    verified_rank_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        require_text(self.group_name, "collective capability group_name")
        if len(self.ranks) != self.world_size or self.world_size < 1:
            raise ValueError("collective capability world_size must match its rank list")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("collective capability ranks must be unique")
        if not 0 <= self.local_rank < self.world_size:
            raise ValueError("collective capability local_rank is out of range")
        _require_positive_pair(self.session_generation, "collective capability session_generation")
        if not isinstance(self.reason, CollectiveReason):
            raise TypeError("collective capability reason must be a CollectiveReason")
        if self.custom is not None and not isinstance(self.custom, CustomCapability):
            raise TypeError("collective capability custom must be a CustomCapability")
        for source, destination in self.verified_rank_pairs:
            if not 0 <= source < self.world_size or not 0 <= destination < self.world_size:
                raise ValueError("verified rank pairs must name ranks inside the group")
            if source == destination:
                raise ValueError("verified rank pairs must be ordered pairs of distinct ranks")

    def to_dict(self) -> dict[str, Any]:
        """Stable wire form used by plan digests and broadcast admission."""
        return {
            "group_name": self.group_name,
            "ranks": list(self.ranks),
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "session_generation": list(self.session_generation),
            "custom": self.custom.to_dict() if self.custom is not None else None,
            "reason": self.reason.value,
            "probe_status": self.probe_status,
            "probe_reason": self.probe_reason,
            "probe_agreement_key": self.probe_agreement_key,
            "verified_rank_pairs": [list(pair) for pair in self.verified_rank_pairs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CollectiveCapability:
        if not isinstance(payload, Mapping):
            raise TypeError("collective capability payload must be a mapping")
        ranks = payload.get("ranks", ())
        generation = payload.get("session_generation", ())
        pairs = payload.get("verified_rank_pairs", ())
        custom = payload.get("custom")
        return cls(
            group_name=str(payload.get("group_name", "")),
            ranks=tuple(int(rank) for rank in ranks),
            world_size=int(payload.get("world_size", 0)),
            local_rank=int(payload.get("local_rank", 0)),
            session_generation=(int(generation[0]), int(generation[1])),
            custom=CustomCapability.from_dict(custom) if isinstance(custom, Mapping) else None,
            reason=CollectiveReason(
                payload.get("reason", CollectiveReason.CUSTOM_UNAVAILABLE.value)
            ),
            probe_status=str(payload.get("probe_status", "")),
            probe_reason=str(payload.get("probe_reason", "")),
            probe_agreement_key=str(payload.get("probe_agreement_key", "")),
            verified_rank_pairs=tuple((int(pair[0]), int(pair[1])) for pair in pairs),
        )

    @property
    def custom_available(self) -> bool:
        """Whether this rank can participate in the custom backend at all."""
        return self.reason is CollectiveReason.OK and self.custom is not None

    def admission_reason(self, proposed: CollectiveCapability) -> CollectiveReason:
        """Whether this rank admits a proposed profile, and why not otherwise.

        The comparison is deliberately limited to fields that must be identical
        for one group generation; ``local_rank`` and ordinal placement are the
        rank's own business and never compared.
        """
        if not isinstance(proposed, CollectiveCapability):
            raise TypeError("admission requires a CollectiveCapability proposal")
        if (
            proposed.group_name != self.group_name
            or proposed.ranks != self.ranks
            or proposed.world_size != self.world_size
        ):
            return CollectiveReason.RANK_DISAGREEMENT
        if proposed.session_generation != self.session_generation:
            return CollectiveReason.GENERATION_MISMATCH
        if self.reason is not CollectiveReason.OK:
            return self.reason
        if proposed.reason is not CollectiveReason.OK:
            return proposed.reason
        if proposed.probe_agreement_key != self.probe_agreement_key:
            return CollectiveReason.P2P_UNVERIFIED
        if proposed.verified_rank_pairs != self.verified_rank_pairs:
            return CollectiveReason.P2P_UNVERIFIED
        if (self.custom is None) != (proposed.custom is None):
            return CollectiveReason.RANK_DISAGREEMENT
        if self.custom is not None and proposed.custom is not None:
            if self.custom.fingerprint != proposed.custom.fingerprint:
                return CollectiveReason.RANK_DISAGREEMENT
        return CollectiveReason.OK


def _custom_reason(probe_reason: Any) -> CollectiveReason:
    """Map a DC2 probe reason onto the collective decision vocabulary."""
    if probe_reason in {
        "unsupported_platform",
        "no_cuda_device",
        "peer_access_absent",
        "single_device",
    }:
        return CollectiveReason.UNSUPPORTED_PLATFORM
    if probe_reason == "atomic_unsupported":
        return CollectiveReason.ATOMIC_UNSUPPORTED
    return CollectiveReason.P2P_UNVERIFIED


def probe_collective_capability(
    *,
    group: DeviceGroup,
    session_generation: tuple[int, int],
    custom: CustomCapability | None,
    probe: Callable[..., Any] | None = None,
    timeout_s: float = 120.0,
    need_ordering: bool = True,
    need_remote_atomic: bool = False,
    device_ordinals: Sequence[int] | None = None,
) -> CollectiveCapability:
    """Build one rank's capability from an optional custom declaration + probe.

    No probe is spawned when ``custom`` is ``None``: a build without a custom
    backend (DC3 has none yet) must stay cheap and host-only. A custom backend
    without verified ordered pairs is reported as ``p2p_unverified`` and can
    never be agreed.
    """
    if group.size < 1:
        raise ValueError("collective capability needs a nonempty group")
    ranks = tuple(group.ranks) if group.ranks else tuple(range(group.size))
    if len(ranks) != group.size:
        raise ValueError("collective capability needs one rank per device")
    _require_positive_pair(session_generation, "session_generation")
    if type(timeout_s) not in (int, float) or timeout_s <= 0:
        raise ValueError("probe timeout must be positive")
    if device_ordinals is None:
        device_ordinals = tuple(device.index for device in group.devices)

    if group.size <= 1:
        return CollectiveCapability(
            group_name=group.name,
            ranks=ranks,
            world_size=group.size,
            local_rank=group.local_rank,
            session_generation=session_generation,
            custom=custom,
            reason=CollectiveReason.SINGLETON,
        )
    if custom is None:
        return CollectiveCapability(
            group_name=group.name,
            ranks=ranks,
            world_size=group.size,
            local_rank=group.local_rank,
            session_generation=session_generation,
            reason=CollectiveReason.CUSTOM_UNAVAILABLE,
        )

    from ayaka.distributed.p2p_probe import probe_verified_topology

    active_probe = probe if probe is not None else probe_verified_topology
    report = active_probe(
        device_ordinals,
        need_ordering=need_ordering,
        need_remote_atomic=need_remote_atomic,
        timeout_s=float(timeout_s),
    )
    verified_pairs: list[tuple[int, int]] = []
    for source in ranks:
        for destination in ranks:
            if source == destination:
                continue
            source_ordinal = device_ordinals[ranks.index(source)]
            destination_ordinal = device_ordinals[ranks.index(destination)]
            if report.verified_for(
                source_ordinal,
                destination_ordinal,
                need_ordering=need_ordering,
                need_remote_atomic=need_remote_atomic,
            ):
                verified_pairs.append((source, destination))
    verified = tuple(verified_pairs)
    reason = CollectiveReason.OK
    if report.status != "verified":
        reason = _custom_reason(report.reason)
    elif len(verified) != group.size * (group.size - 1):
        reason = CollectiveReason.P2P_UNVERIFIED
    return CollectiveCapability(
        group_name=group.name,
        ranks=ranks,
        world_size=group.size,
        local_rank=group.local_rank,
        session_generation=session_generation,
        custom=custom,
        reason=reason,
        probe_status=str(getattr(report.status, "value", report.status)),
        probe_reason=str(getattr(report.reason, "value", report.reason)),
        probe_agreement_key=str(report.agreement_key()),
        verified_rank_pairs=verified,
    )


# ---------------------------------------------------------------------------
# Agreement wire record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CollectivePlan(ConfigMixin):
    """The one decision every rank makes before any data-plane launch.

    ``enabled`` means the custom backend is the agreed route; otherwise the
    plan records the deterministic ``reason`` the group fell back (or refused).
    The digest in :meth:`encode` protects the broadcast body from corruption.
    """

    requested_policy: CollectivePolicy = CollectivePolicy.TORCH
    enabled: bool = False
    reason: CollectiveReason = CollectiveReason.POLICY_TORCH
    capability: CollectiveCapability | None = None
    schema_version: int = DISPATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DISPATCH_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported collective plan schema_version {self.schema_version!r}; "
                f"expected {DISPATCH_SCHEMA_VERSION}"
            )
        if not isinstance(self.requested_policy, CollectivePolicy):
            raise TypeError("collective plan policy must be a CollectivePolicy")
        if type(self.enabled) is not bool:
            raise TypeError("collective plan enabled must be a boolean")
        if not isinstance(self.reason, CollectiveReason):
            raise TypeError("collective plan reason must be a CollectiveReason")
        if self.enabled and (self.reason is not CollectiveReason.OK or self.capability is None):
            raise ValueError("an enabled collective plan needs a capability and the OK reason")

    def to_dict(self) -> dict[str, Any]:
        """Stable plan wire form; delegates to each nested record's canonical form."""
        return {
            "schema_version": self.schema_version,
            "requested_policy": self.requested_policy.value,
            "enabled": self.enabled,
            "reason": self.reason.value,
            "capability": self.capability.to_dict() if self.capability is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CollectivePlan:
        if not isinstance(payload, Mapping):
            raise TypeError("collective plan payload must be a mapping")
        capability = payload.get("capability")
        return cls(
            requested_policy=CollectivePolicy(payload["requested_policy"]),
            enabled=bool(payload.get("enabled", False)),
            reason=CollectiveReason(payload.get("reason", CollectiveReason.POLICY_TORCH.value)),
            capability=(
                CollectiveCapability.from_dict(capability)
                if isinstance(capability, Mapping)
                else None
            ),
            schema_version=int(payload.get("schema_version", 0)),
        )

    def encode(self) -> str:
        """Canonical JSON body plus a sha256 digest for the control channel."""
        body = self.canonical_json()
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return json.dumps(
            {"body": json.loads(body), "sha256": digest},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def decode(cls, text: str) -> CollectivePlan:
        """Decode and verify a plan broadcast by the source rank."""
        if not isinstance(text, str):
            raise TypeError("collective plan wire payload must be text")
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("collective plan wire payload is not valid JSON") from exc
        if not isinstance(envelope, Mapping) or "body" not in envelope or "sha256" not in envelope:
            raise ValueError("collective plan wire payload is missing its envelope fields")
        body = json.dumps(
            envelope["body"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest != envelope["sha256"]:
            raise ValueError("collective plan wire payload failed its digest check")
        return cls.from_dict(envelope["body"])


# ---------------------------------------------------------------------------
# Decision and metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    """The pure routing decision for one call, before any backend is entered."""

    backend: str
    custom: bool
    reason: CollectiveReason
    noop: bool = False


@dataclass(frozen=True, slots=True)
class DispatchMetricsSnapshot:
    """Counters for one dispatcher; never contains handles or pointers."""

    calls: int
    bytes: int
    fallbacks: int
    backends: tuple[tuple[str, int], ...]
    reasons: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class DispatchMetrics:
    """Thread-safe backend/algorithm/fallback accounting (spec section 8)."""

    calls: int = 0
    bytes: int = 0
    fallbacks: int = 0
    backends: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(
        init=False, repr=False, compare=False, default_factory=threading.RLock
    )

    def record(
        self,
        *,
        backend: str,
        reason: CollectiveReason,
        nbytes: int,
        fallback: bool,
    ) -> None:
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("metric byte counts must be non-negative integers")
        with self._lock:
            self.calls += 1
            self.bytes += nbytes
            if fallback:
                self.fallbacks += 1
            self.backends[backend] = self.backends.get(backend, 0) + 1
            self.reasons[reason.value] = self.reasons.get(reason.value, 0) + 1

    def snapshot(self) -> DispatchMetricsSnapshot:
        with self._lock:
            return DispatchMetricsSnapshot(
                calls=self.calls,
                bytes=self.bytes,
                fallbacks=self.fallbacks,
                backends=tuple(sorted(self.backends.items())),
                reasons=tuple(sorted(self.reasons.items())),
            )


# ---------------------------------------------------------------------------
# Custom backend and agreement channel protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CustomCollective(Protocol):
    """A custom collective implementation the dispatcher may route to.

    ``all_reduce`` must honour the caller tensor mutation contract: an
    out-of-place implementation copies the result back into the caller tensor
    and includes that copy in its completion handle. Raising
    :class:`CollectivePreLaunchError` is the only signal that permits a
    fallback; any other exception is treated as post-launch.
    """

    @property
    def name(self) -> str: ...

    def all_reduce(
        self,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None: ...

    def close(self) -> None: ...


@runtime_checkable
class AgreementChannel(Protocol):
    """The minimal control surface the setup agreement needs.

    :class:`~ayaka.distributed.process_group.TorchDistributedKVProcessGroup`
    satisfies it; tests substitute a scripted or loopback channel. Every call
    carries a deadline so a wedged rank becomes a fail-closed error.
    """

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def broadcast_text(
        self,
        text: str | None,
        *,
        source_rank: int,
        timeout_s: float | None = None,
    ) -> str: ...

    def all_grant(self, local_grant: bool, *, timeout_s: float | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class CollectiveSetup:
    """Runtime-owner inputs used by ``build_parallel_runtime`` for setup.

    The owner probes capability (``probe_collective_capability``) and supplies
    the already-initialized control channel and Torch fallback; the assembly
    seam only performs the agreement and returns the dispatcher.
    """

    control: AgreementChannel
    capability: CollectiveCapability
    custom_factory: Callable[[CollectivePlan], CustomCollective] | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capability, CollectiveCapability):
            raise TypeError("collective setup capability must be a CollectiveCapability")
        if self.timeout_s is not None and (
            type(self.timeout_s) not in (int, float) or self.timeout_s <= 0
        ):
            raise ValueError("collective setup timeout must be positive")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TORCH_BACKEND = "torch.distributed"
_DTYPE_BY_TORCH_NAME: dict[str, DType] = {
    dtype.torch_name: dtype for dtype in DType if dtype.torch_name is not None
}
_DTYPE_BY_LABEL: dict[str, DType] = {dtype.label: dtype for dtype in DType}


def _tensor_dtype(tensor: Any) -> DType | None:
    name = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
    return _DTYPE_BY_TORCH_NAME.get(name)


def _tensor_nbytes(tensor: Any) -> int:
    try:
        numel = int(tensor.numel())
        return numel * int(tensor.element_size())
    except (AttributeError, TypeError, ValueError):
        return 0


class CollectiveDispatchBackend:
    """Route collectives by the agreed plan while implementing the public protocol.

    All unsupported operations, unsupported metadata and non-agreed groups go
    to the Torch fallback *before* the custom backend is entered. The dispatcher
    adds no collective, probe, allocation or handle exchange to a call: the
    decision reads the plan plus the tensor's local metadata only.
    """

    def __init__(
        self,
        *,
        policy: CollectivePolicy,
        group: DeviceGroup,
        plan: CollectivePlan,
        fallback: CommunicationBackend | None,
        custom: CustomCollective | None = None,
        metrics: DispatchMetrics | None = None,
    ) -> None:
        if not isinstance(policy, CollectivePolicy):
            raise TypeError("dispatcher policy must be a CollectivePolicy")
        if not isinstance(plan, CollectivePlan):
            raise TypeError("dispatcher plan must be a CollectivePlan")
        if plan.enabled and custom is None:
            raise ValueError("an enabled plan requires a custom backend")
        self._policy = policy
        self._group = group
        self._plan = plan
        self._fallback = fallback
        self._custom = custom
        self._metrics = metrics if metrics is not None else DispatchMetrics()
        self._failed = False
        self._failure: str | None = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        """Stable backend name; mirrors the public protocol's ``name`` field."""
        if self._plan.enabled and self._custom is not None:
            return self._custom.name
        return _TORCH_BACKEND

    @property
    def policy(self) -> CollectivePolicy:
        return self._policy

    @property
    def group(self) -> DeviceGroup:
        return self._group

    @property
    def plan(self) -> CollectivePlan:
        return self._plan

    @property
    def metrics(self) -> DispatchMetrics:
        return self._metrics

    @property
    def custom(self) -> CustomCollective | None:
        return self._custom

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def failure(self) -> str | None:
        """Failure class name after a post-launch failure; never a pointer."""
        return self._failure

    def close(self) -> None:
        """Release the custom backend; only safe after drain/recovery."""
        custom, self._custom = self._custom, None
        if custom is not None:
            custom.close()

    def capture_graph(
        self,
        *,
        dtype: Any,
        numel: int,
        generation_provider: Callable[[], tuple[int, int]] | None = None,
    ) -> Any:
        """Capture a registered collective only for an agreed graph profile.

        The default profile remains eager-only. An operator must opt into graph
        capture on every rank before group agreement; the concrete custom
        communicator then owns the graph, pins and replay tickets.
        """
        capability = self._plan.capability
        custom_capability = capability.custom if capability is not None else None
        if (
            self._failed
            or not self._plan.enabled
            or custom_capability is None
            or not custom_capability.graph_certified
            or self._custom is None
        ):
            raise CollectiveRequiredError(
                CollectiveReason.GRAPH_UNCERTIFIED,
                detail="the agreed collective profile does not permit graph capture",
            )
        capture = getattr(self._custom, "capture_graph", None)
        if not callable(capture):
            raise CollectiveRequiredError(CollectiveReason.GRAPH_UNCERTIFIED)
        try:
            return capture(dtype=dtype, numel=numel, generation_provider=generation_provider)
        except CollectivePostLaunchError as exc:
            self._fail_closed(exc)

    def evaluate(
        self,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
    ) -> DispatchDecision:
        """Pure routing decision for one call (no control traffic, no mutation)."""
        with self._lock:
            if self._failed:
                raise CollectiveDispatchFailed(
                    f"collective dispatcher is quarantined after a post-launch failure "
                    f"({self._failure})",
                    reason=CollectiveReason.POSTLAUNCH_FAILED,
                )
        if not isinstance(group, DeviceGroup):
            raise TypeError("collective dispatch requires an explicit DeviceGroup")
        if group.size <= 1 or group.is_trivial:
            return DispatchDecision(
                backend="none", custom=False, reason=CollectiveReason.SINGLETON, noop=True
            )
        if not self._plan.enabled or self._custom is None:
            return DispatchDecision(backend=_TORCH_BACKEND, custom=False, reason=self._plan.reason)
        capability = self._plan.capability
        if (
            capability is None
            or group.name != capability.group_name
            or tuple(group.ranks) != capability.ranks
        ):
            return DispatchDecision(
                backend=_TORCH_BACKEND, custom=False, reason=CollectiveReason.UNSUPPORTED_OP
            )
        custom = capability.custom
        if custom is None:
            return DispatchDecision(
                backend=_TORCH_BACKEND, custom=False, reason=CollectiveReason.CUSTOM_UNAVAILABLE
            )
        if op not in custom.supported_ops:
            return DispatchDecision(
                backend=_TORCH_BACKEND, custom=False, reason=CollectiveReason.UNSUPPORTED_OP
            )
        dtype = _tensor_dtype(tensor)
        if dtype is None or dtype not in custom.supported_dtypes:
            return DispatchDecision(
                backend=_TORCH_BACKEND, custom=False, reason=CollectiveReason.UNSUPPORTED_DTYPE
            )
        if not bool(tensor.is_contiguous()) or int(tensor.storage_offset()) < 0:
            return DispatchDecision(
                backend=_TORCH_BACKEND, custom=False, reason=CollectiveReason.UNSUPPORTED_LAYOUT
            )
        numel = int(tensor.numel())
        if numel == 0:
            return DispatchDecision(
                backend=self._custom.name, custom=True, reason=CollectiveReason.OK, noop=True
            )
        alignment = custom.alignment
        if alignment > 1 and int(tensor.data_ptr()) % alignment:
            return DispatchDecision(
                backend=_TORCH_BACKEND, custom=False, reason=CollectiveReason.UNSUPPORTED_LAYOUT
            )
        nbytes = numel * int(tensor.element_size())
        if nbytes < custom.min_bytes or (
            custom.max_bytes is not None and nbytes > custom.max_bytes
        ):
            return DispatchDecision(
                backend=_TORCH_BACKEND, custom=False, reason=CollectiveReason.UNSUPPORTED_SIZE
            )
        return DispatchDecision(backend=self._custom.name, custom=True, reason=CollectiveReason.OK)

    def all_reduce(
        self,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None:
        decision = self.evaluate(tensor, group, op)
        nbytes = _tensor_nbytes(tensor)
        if decision.noop:
            self._metrics.record(
                backend=decision.backend,
                reason=decision.reason,
                nbytes=nbytes,
                fallback=False,
            )
            return None
        if not decision.custom:
            if self._policy is CollectivePolicy.CUSTOM_REQUIRED:
                raise CollectiveRequiredError(
                    decision.reason,
                    detail="the call is not eligible for the agreed custom backend",
                )
            return self._torch_all_reduce(tensor, group, op, async_op, decision, nbytes)
        custom = self._custom
        if custom is None:
            raise CollectiveDispatchFailed(
                "the agreed plan lost its custom backend",
                reason=CollectiveReason.CUSTOM_UNAVAILABLE,
            )
        self._metrics.record(
            backend=decision.backend, reason=CollectiveReason.OK, nbytes=nbytes, fallback=False
        )
        try:
            return custom.all_reduce(tensor, group, op, async_op)
        except CollectivePreLaunchError as exc:
            self._metrics.record(
                backend=_TORCH_BACKEND,
                reason=CollectiveReason.PRELAUNCH_FAILED,
                nbytes=nbytes,
                fallback=True,
            )
            if self._policy is CollectivePolicy.CUSTOM_REQUIRED:
                raise CollectiveRequiredError(
                    CollectiveReason.PRELAUNCH_FAILED,
                    detail="the custom backend failed before launch",
                ) from exc
            return self._torch_all_reduce(
                tensor,
                group,
                op,
                async_op,
                DispatchDecision(
                    backend=_TORCH_BACKEND,
                    custom=False,
                    reason=CollectiveReason.PRELAUNCH_FAILED,
                ),
                nbytes,
                record=False,
            )
        except CollectivePostLaunchError as exc:
            self._fail_closed(exc)
        except BaseException as exc:
            self._fail_closed(exc)

    def _torch_all_reduce(
        self,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType,
        async_op: bool,
        decision: DispatchDecision,
        nbytes: int,
        *,
        record: bool = True,
    ) -> AsyncHandle | None:
        fallback = self._fallback
        if fallback is None:
            raise CollectiveDispatchFailed(
                "no Torch fallback is attached to this dispatcher",
                reason=CollectiveReason.UNSUPPORTED_PLATFORM,
            )
        if record:
            self._metrics.record(
                backend=decision.backend,
                reason=decision.reason,
                nbytes=nbytes,
                fallback=True,
            )
        return fallback.all_reduce(tensor, group, op, async_op)

    def _delegate(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._lock:
            if self._failed:
                raise CollectiveDispatchFailed(
                    f"collective dispatcher is quarantined after a post-launch failure "
                    f"({self._failure})",
                    reason=CollectiveReason.POSTLAUNCH_FAILED,
                )
        if self._fallback is None:
            raise CollectiveDispatchFailed(
                f"no Torch fallback is attached for {method}",
                reason=CollectiveReason.UNSUPPORTED_PLATFORM,
            )
        self._metrics.record(
            backend=_TORCH_BACKEND,
            reason=CollectiveReason.UNSUPPORTED_OP,
            nbytes=0,
            fallback=True,
        )
        return getattr(self._fallback, method)(*args, **kwargs)

    def all_gather(
        self, output: Any, tensor: Any, group: DeviceGroup, async_op: bool = False
    ) -> AsyncHandle | None:
        return self._delegate("all_gather", output, tensor, group, async_op)

    def reduce_scatter(
        self,
        output: Any,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None:
        return self._delegate("reduce_scatter", output, tensor, group, op, async_op)

    def broadcast(
        self, tensor: Any, group: DeviceGroup, src_rank: int = 0, async_op: bool = False
    ) -> AsyncHandle | None:
        return self._delegate("broadcast", tensor, group, src_rank, async_op)

    def all_to_all(
        self, output: Any, tensor: Any, group: DeviceGroup, async_op: bool = False
    ) -> AsyncHandle | None:
        return self._delegate("all_to_all", output, tensor, group, async_op)

    def send(self, tensor: Any, group: DeviceGroup, dst_rank: int) -> AsyncHandle | None:
        return self._delegate("send", tensor, group, dst_rank)

    def recv(self, tensor: Any, group: DeviceGroup, src_rank: int) -> AsyncHandle | None:
        return self._delegate("recv", tensor, group, src_rank)

    def barrier(self, group: DeviceGroup) -> None:
        self._delegate("barrier", group)

    def _fail_closed(self, exc: BaseException) -> NoReturn:
        """Quarantine the dispatcher; never fall back or retry after this point."""
        with self._lock:
            self._failed = True
            self._failure = type(exc).__name__
        self._metrics.record(
            backend=type(self._custom).__name__ if self._custom is not None else "custom",
            reason=CollectiveReason.POSTLAUNCH_FAILED,
            nbytes=0,
            fallback=False,
        )
        quarantine = getattr(self._custom, "quarantine", None)
        if callable(quarantine):
            quarantine()
        if isinstance(exc, CollectiveDispatchError):
            raise exc
        raise CollectiveDispatchFailed(
            f"custom collective failed after launch: {type(exc).__name__}",
            reason=CollectiveReason.POSTLAUNCH_FAILED,
        ) from exc


# ---------------------------------------------------------------------------
# Setup agreement
# ---------------------------------------------------------------------------


def _group_matches_capability(group: DeviceGroup, capability: CollectiveCapability) -> bool:
    ranks = tuple(group.ranks) if group.ranks else tuple(range(group.size))
    return (
        capability.group_name == group.name
        and capability.ranks == ranks
        and capability.world_size == group.size
        and capability.local_rank == group.local_rank
    )


def _broadcast_group_reason(
    control: AgreementChannel,
    local_reason: CollectiveReason,
    timeout_s: float | None,
) -> CollectiveReason:
    """Ordered per-rank reason broadcast; every rank computes the same result."""
    reasons: list[CollectiveReason] = []
    for rank in range(control.world_size):
        text = local_reason.value if rank == control.rank else None
        received = control.broadcast_text(text, source_rank=rank, timeout_s=timeout_s)
        reasons.append(CollectiveReason(received))
    return group_reason(reasons)


def setup_collective_backend(
    *,
    policy: CollectivePolicy,
    group: DeviceGroup,
    control: AgreementChannel | None = None,
    fallback: CommunicationBackend | None = None,
    capability: CollectiveCapability | None = None,
    custom_factory: Callable[[CollectivePlan], CustomCollective] | None = None,
    timeout_s: float | None = None,
) -> CollectiveDispatchBackend:
    """Agree one plan for the group and return the dispatching backend.

    A singleton group short-circuits: no control channel, probe, custom
    backend or workspace is created. A multi-rank group broadcasts the source
    rank's profile, admits or refuses it locally, votes exactly once, and only
    installs a custom plan when every rank agreed. ``auto`` falls back to the
    attached Torch backend after a completed refusal; ``custom_required``
    raises :class:`CollectiveAgreementError` instead. Any collective failure or
    timeout raises immediately — fail-closed, never a silent fallback.
    """
    if not isinstance(policy, CollectivePolicy):
        raise TypeError("collective policy must be a CollectivePolicy")
    if group.size < 1:
        raise ValueError("collective setup needs a nonempty group")
    if timeout_s is not None and (type(timeout_s) not in (int, float) or timeout_s <= 0):
        raise ValueError("collective setup timeout must be positive")

    if group.size <= 1:
        plan = CollectivePlan(
            requested_policy=policy,
            enabled=False,
            reason=CollectiveReason.SINGLETON,
            capability=capability,
        )
        return CollectiveDispatchBackend(
            policy=policy, group=group, plan=plan, fallback=fallback, custom=None
        )
    if policy is CollectivePolicy.TORCH:
        plan = CollectivePlan(
            requested_policy=policy,
            enabled=False,
            reason=CollectiveReason.POLICY_TORCH,
            capability=capability,
        )
        return CollectiveDispatchBackend(
            policy=policy, group=group, plan=plan, fallback=fallback, custom=None
        )
    if control is None or fallback is None or capability is None:
        raise CapabilityError(
            _CUSTOM_CAPABILITY,
            detail="custom collective setup needs a control channel, a Torch fallback and "
            "a probed capability",
            remedy="supply these at build_parallel_runtime or leave the policy on torch",
        )
    if control.world_size != group.size or control.rank != group.local_rank:
        raise ValueError("the agreement channel must cover the logical group in local-rank order")
    if not _group_matches_capability(group, capability):
        raise ValueError("capability does not describe the group being agreed")
    deadline = DEFAULT_AGREEMENT_TIMEOUT_S if timeout_s is None else float(timeout_s)

    local_reason = capability.reason
    custom: CustomCollective | None = None
    if local_reason is CollectiveReason.OK and custom_factory is not None:
        proposal = CollectivePlan(
            requested_policy=policy,
            enabled=True,
            reason=CollectiveReason.OK,
            capability=capability,
        )
        try:
            custom = custom_factory(proposal)
            if custom is None:
                raise RuntimeError("the custom factory returned no backend")
        except BaseException:
            custom = None
            local_reason = CollectiveReason.WORKSPACE_UNAVAILABLE
    elif local_reason is CollectiveReason.OK:
        local_reason = CollectiveReason.CUSTOM_UNAVAILABLE

    proposal = CollectivePlan(
        requested_policy=policy,
        enabled=local_reason is CollectiveReason.OK,
        reason=local_reason,
        capability=capability,
    )
    try:
        text = proposal.encode() if control.rank == 0 else None
        received = control.broadcast_text(text, source_rank=0, timeout_s=deadline)
        decoded = CollectivePlan.decode(received)
        if control.rank == 0 and decoded.canonical_json() != proposal.canonical_json():
            raise CollectiveDispatchError("source plan changed during broadcast")
        if decoded.capability is None:
            admitted_reason = CollectiveReason.RANK_DISAGREEMENT
        elif local_reason is not CollectiveReason.OK:
            admitted_reason = local_reason
        else:
            admitted_reason = capability.admission_reason(decoded.capability)
        granted = control.all_grant(admitted_reason is CollectiveReason.OK, timeout_s=deadline)
    except BaseException:
        if custom is not None:
            custom.close()
        raise
    if granted:
        plan = CollectivePlan(
            requested_policy=policy,
            enabled=True,
            reason=CollectiveReason.OK,
            capability=capability,
        )
        return CollectiveDispatchBackend(
            policy=policy, group=group, plan=plan, fallback=fallback, custom=custom
        )

    if custom is not None:
        custom.close()
        custom = None
    refusal = (
        local_reason
        if local_reason is not CollectiveReason.OK
        else (
            admitted_reason
            if admitted_reason is not CollectiveReason.OK
            else CollectiveReason.RANK_DISAGREEMENT
        )
    )
    resolved = _broadcast_group_reason(control, refusal, deadline)
    if policy is CollectivePolicy.CUSTOM_REQUIRED:
        raise CollectiveAgreementError(
            resolved,
            detail="custom backend is required but the group did not agree",
        )
    plan = CollectivePlan(
        requested_policy=policy,
        enabled=False,
        reason=resolved,
        capability=capability,
    )
    return CollectiveDispatchBackend(
        policy=policy, group=group, plan=plan, fallback=fallback, custom=None
    )
