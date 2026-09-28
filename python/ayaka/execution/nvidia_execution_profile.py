"""Bucket, flight-slot and memory-budget policy for an NVIDIA execution lane.

Policy is chosen from what the device can do and what memory is actually
available, never from a device name. A GPU model string is a marketing label
that survives a silent silicon change; compute capability, partition capacity
and an admission check are facts that do not.

The estimate produced here is a precheck, not an acceptance. Only capture
measures the real footprint, and only capture can reject a set. That ordering is
deliberate: a resolver that trusted its own arithmetic would admit a set that
then fails at capture with a half-built registry.

Every decision this module makes is recorded with the input that produced it, so
an operator reading a startup report can tell a 1-2 bucket set on a 4 GiB
laptop from a deliberate small-batch profile on a 24 GiB card.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ayaka.execution.execution_capabilities import (
    ExecutionCapabilityReport,
    SupportTier,
)
from ayaka.execution.shape_key import validate_buckets
from ayaka.utils.validation import require_int

__all__ = [
    "BudgetEvidence",
    "NvidiaExecutionProfile",
    "ProfileDecision",
    "ProfileRejection",
    "ResolvedExecutionProfile",
    "default_min_headroom",
    "resolve_profile",
]

MIB: Final[int] = 1 << 20
GIB: Final[int] = 1 << 30


class NvidiaExecutionProfile(StrEnum):
    """Named starting points for the bucket and budget trade-off.

    These are candidate sets, not measured optima. None of them is correct for
    every device; the point of naming them is that the choice is visible in the
    startup report instead of buried in a default.
    """

    CONSTRAINED = "constrained"
    LATENCY = "latency"
    BALANCED = "balanced"
    THROUGHPUT = "throughput"
    EXPLICIT = "explicit"


#: Candidate decode buckets per profile, before capacity and budget admission.
#: These ascend and are deduplicated by construction; ``validate_buckets``
#: re-checks any operator-supplied set against the same rules.
_CANDIDATES: Final[dict[NvidiaExecutionProfile, tuple[int, ...]]] = {
    NvidiaExecutionProfile.CONSTRAINED: (1, 2, 4, 8),
    NvidiaExecutionProfile.LATENCY: (1, 2, 4, 8, 16, 32),
    NvidiaExecutionProfile.BALANCED: (1, 2, 4, 8, 16, 32, 64, 128),
    NvidiaExecutionProfile.THROUGHPUT: (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256),
    NvidiaExecutionProfile.EXPLICIT: (),
}

#: Absolute floors for a graph reserve to be worth attempting at all. Below the
#: floor the honest answer is eager, not a bucket set too small to be useful.
_GRAPH_RESERVE_FLOOR: Final[int] = 32 * MIB

#: Headroom bounds. The headroom a set must clear is a fraction of what this
#: partition actually has, not a constant: 768 MiB is the figure the plan
#: proposes for a small development card, and applying it to every device would
#: starve a large one while still being too tight for a tiny one.
_HEADROOM_FRACTION: Final[float] = 0.10
_HEADROOM_MIN: Final[int] = 64 * MIB
_HEADROOM_MAX: Final[int] = 768 * MIB

#: Flight slots. One is the correctness baseline; two is what lets a host
#: snapshot be prepared while a device batch is still in flight.
_DEFAULT_FLIGHT_SLOTS: Final[int] = 1


def default_min_headroom(partition_capacity_bytes: int) -> int:
    """Headroom floor for a partition, scaled to its capacity.

    Ten percent, clamped into a band that keeps a 4 GiB development card
    workable while stopping a 24 GiB card from treating a 768 MiB cushion as
    the whole story. A device whose capacity could not be measured falls back
    to the upper anchor, because an unknown budget is the case that most needs
    the conservative end.
    """
    if partition_capacity_bytes <= 0:
        return _HEADROOM_MAX
    scaled = int(partition_capacity_bytes * _HEADROOM_FRACTION)
    return max(_HEADROOM_MIN, min(_HEADROOM_MAX, scaled))


def _floor_price(bucket: int) -> int:
    """Price a graph as if its only state were the reserve floor.

    Used when no real per-bucket price is available. It is deliberately the most
    optimistic price that still funds a capture, so an unpriced lane is admitted
    the smallest possible set rather than a set nobody measured.
    """
    return 0


class ProfileRejection(StrEnum):
    """Why a candidate bucket was not admitted.

    Bounded set, because these become metric labels.
    """

    ABOVE_REQUEST_CEILING = "above_request_ceiling"
    ABOVE_TOKEN_CEILING = "above_token_ceiling"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BELOW_RESERVE_FLOOR = "below_reserve_floor"
    INSUFFICIENT_HEADROOM = "insufficient_headroom"
    UNPRICED = "unpriced"
    GRAPH_UNSUPPORTED = "graph_unsupported"
    NOT_CAPTURABLE = "not_capturable"


@dataclass(frozen=True, slots=True)
class ProfileDecision:
    """One policy choice with the input that forced it."""

    subject: str
    choice: str
    because: str

    def __str__(self) -> str:
        return f"{self.subject}={self.choice} ({self.because})"


@dataclass(frozen=True, slots=True)
class BudgetEvidence:
    """The arithmetic behind an admission, in absolute bytes.

    ``available`` is what the memory plan already left after subtracting
    non-KV allocations once. It is not recomputed here, because subtracting
    ``non_kv_bytes`` a second time is precisely the double count this field
    exists to make visible.
    """

    partition_capacity_bytes: int
    available_bytes: int
    non_kv_bytes: int
    measured: bool
    graph_reserve_bytes: int
    safety_reserve_bytes: int
    headroom_after_graph_bytes: int

    @property
    def admits_graph(self) -> bool:
        """Whether a graph reserve of any size could be funded here."""
        return self.available_bytes > 0 and self.non_kv_bytes >= 0


@dataclass(frozen=True, slots=True)
class ResolvedExecutionProfile:
    """An admitted bucket set plus every number and reason behind it."""

    profile: NvidiaExecutionProfile
    provenance: str
    candidate_buckets: tuple[int, ...]
    admitted_buckets: tuple[int, ...]
    rejected: tuple[tuple[int, ProfileRejection], ...]
    flight_slots: int
    graph_reserve_bytes: int
    safety_reserve_bytes: int
    budget: BudgetEvidence
    decisions: tuple[ProfileDecision, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.admitted_buckets:
            validate_buckets(self.admitted_buckets)
        require_int(self.flight_slots, "flight_slots", minimum=1)

    def explain(self) -> str:
        """One block an operator can read to see why this set was admitted."""
        lines = [
            f"profile={self.profile.value} provenance={self.provenance}",
            f"candidates={self.candidate_buckets} admitted={self.admitted_buckets}",
            f"flight_slots={self.flight_slots} "
            f"graph_reserve={self.graph_reserve_bytes} "
            f"safety_reserve={self.safety_reserve_bytes}",
            f"budget: partition={self.budget.partition_capacity_bytes} "
            f"available={self.budget.available_bytes} non_kv={self.budget.non_kv_bytes} "
            f"measured={self.budget.measured}",
        ]
        for bucket, reason in self.rejected:
            lines.append(f"rejected bucket {bucket}: {reason.value}")
        for note in self.notes:
            lines.append(f"note: {note}")
        for decision in self.decisions:
            lines.append(f"decision: {decision}")
        return "\n".join(lines)


def _ceil_buckets(candidates: Sequence[int], ceiling: int) -> tuple[int, ...]:
    return tuple(bucket for bucket in candidates if bucket <= ceiling)


#: Reason recorded when a bucket's cost cannot be priced separately.
_UNPRICED = ProfileRejection.UNPRICED


def _estimate_graph_bytes(
    admitted: Sequence[int],
    *,
    activation_bytes: int,
    flight_slots: int,
    price_persistent: Callable[[int], int] | None,
) -> int:
    """Precheck a reserve for a candidate set.

    The persistent part is priced by the *largest* admitted bucket, which is how
    the attention builder sizes its graph state, and how the existing Triton
    estimator in the serving runtime already prices it. Price per extra bucket
    is deliberately not modelled here: a scalar model of capture count would be
    invented arithmetic, and capture reconciles the real footprint anyway.

    The floor keeps a tiny model able to admit a cold recapture after a resize,
    instead of discovering that a warmed allocator is a prerequisite.
    """
    largest = max(admitted, default=0)
    if largest < 1:
        return 0
    if price_persistent is None:
        # Without a per-bucket price the set cannot be discriminated on cost.
        return 0
    per_slot = max(_GRAPH_RESERVE_FLOOR, price_persistent(largest) + activation_bytes)
    return per_slot * flight_slots


def resolve_profile(
    report: ExecutionCapabilityReport,
    *,
    profile: NvidiaExecutionProfile = NvidiaExecutionProfile.BALANCED,
    explicit_buckets: tuple[int, ...] | None = None,
    max_requests: int,
    max_tokens: int,
    activation_bytes: int = 0,
    price_persistent: Callable[[int], int] | None = None,
    flight_slots: int = _DEFAULT_FLIGHT_SLOTS,
    available_bytes: int | None = None,
    non_kv_bytes: int = 0,
    measured: bool = False,
    safety_reserve_bytes: int = 512 * MIB,
    min_headroom_bytes: int | None = None,
) -> ResolvedExecutionProfile:
    """Resolve the bucket set, flight slots and graph reserve for one lane.

    Args:
        report: The capability report from EP1.1. Its graph tier decides whether
            capture is even a candidate.
        profile: Which starting point to use when the operator gave no set.
        explicit_buckets: An operator's exact set. It is admitted as given or
            refused with a reason; it is never silently trimmed.
        max_requests: Scheduler request ceiling. Independent of any graph
            ceiling.
        max_tokens: Token capacity ceiling for this runner.
        activation_bytes: Measured bootstrap activation peak.
        price_persistent: Prices the attention builder's persistent graph state
            for a given largest bucket, returning bytes. The Triton estimator
            ``estimate_graph_state_bytes(max_batch_size=...)`` already has this
            shape. When it is ``None`` the cost of one bucket cannot be compared
            against another, so only the smallest set is admitted and the rest
            are recorded as ``UNPRICED`` rather than assumed affordable.
        flight_slots: Overlap capacity. Two is needed to stage a device batch
            while the previous one is still resident.
        available_bytes: Memory left after non-KV allocations, from
            :func:`~ayaka.configs.memory.plan_device_memory`. Pass the plan's
            value; never pass the nominal card capacity.
        non_kv_bytes: Non-KV allocation total, for reporting only. It was
            already subtracted from ``available_bytes`` upstream and is not
            subtracted again.
        measured: Whether the profile numbers came from a real measurement. An
            unmeasured profile is a precheck and says so.
        safety_reserve_bytes: Held back for allocator fragmentation and
            co-tenancy.
        min_headroom_bytes: Headroom a set must clear after funding the reserve.
            ``None`` scales it to the partition; pass a value to override.

    Returns:
        A :class:`ResolvedExecutionProfile` whose ``admitted_buckets`` may be
        empty. Empty means eager, and is a legitimate outcome, not a failure.

    Raises:
        ValueError: when an explicit set is malformed or above the runner
            ceilings. A malformed operator request is a startup error, not a
            silent downgrade.
    """
    require_int(max_requests, "max_requests", minimum=1)
    require_int(max_tokens, "max_tokens", minimum=1)
    require_int(flight_slots, "flight_slots", minimum=1)
    require_int(activation_bytes, "activation_bytes", minimum=0)
    require_int(safety_reserve_bytes, "safety_reserve_bytes", minimum=0)

    decisions: list[ProfileDecision] = []
    notes: list[str] = []

    partition_capacity = report.device.capacity_bytes
    if available_bytes is None:
        available_bytes = 0
        notes.append("no memory plan was supplied; graph budget is not computable")
    if partition_capacity <= 0:
        notes.append("partition capacity is unknown; the budget is a precheck, not an estimate")
    if min_headroom_bytes is None:
        min_headroom_bytes = default_min_headroom(partition_capacity)
        decisions.append(
            ProfileDecision(
                "min_headroom",
                str(min_headroom_bytes),
                f"scaled to {_HEADROOM_FRACTION:.0%} of a "
                f"{partition_capacity}-byte partition, clamped to "
                f"[{_HEADROOM_MIN}, {_HEADROOM_MAX}]",
            )
        )
    else:
        require_int(min_headroom_bytes, "min_headroom_bytes", minimum=0)

    graph_supported = report.verdict(SupportTier.REQUESTED_FEATURES).supported
    if not graph_supported:
        decisions.append(
            ProfileDecision(
                "admitted_buckets",
                "()",
                "the capability report does not support the requested graph features",
            )
        )
        return ResolvedExecutionProfile(
            profile=profile,
            provenance="graph_unsupported",
            candidate_buckets=(),
            admitted_buckets=(),
            rejected=(),
            flight_slots=flight_slots,
            graph_reserve_bytes=0,
            safety_reserve_bytes=safety_reserve_bytes,
            budget=_evidence(
                partition_capacity, available_bytes, non_kv_bytes, measured, 0, safety_reserve_bytes
            ),
            decisions=tuple(decisions),
            notes=tuple(notes),
        )

    ceiling = min(max_requests, max_tokens)
    decisions.append(
        ProfileDecision(
            "ceiling",
            str(ceiling),
            f"min(max_requests={max_requests}, max_tokens={max_tokens})",
        )
    )

    if explicit_buckets is not None:
        validate_buckets(explicit_buckets)
        chosen_profile = NvidiaExecutionProfile.EXPLICIT
        candidates = explicit_buckets
        provenance = "explicit"
        if candidates[-1] > ceiling:
            # Refusing beats trimming: a trimmed explicit set silently changes
            # what the operator asked for and hides the ceiling they missed.
            raise ValueError(
                f"explicit graph_buckets top out at {candidates[-1]} but this runner's "
                f"ceiling is {ceiling}; lower the set or raise the capacity"
            )
        decisions.append(
            ProfileDecision("candidates", str(candidates), "operator supplied this exact set")
        )
    else:
        chosen_profile = profile
        candidates = _CANDIDATES[profile]
        candidates = _ceil_buckets(candidates, ceiling)
        provenance = "auto"
        decisions.append(
            ProfileDecision(
                "candidates",
                str(candidates),
                f"profile {profile.value} capped at ceiling {ceiling}",
            )
        )

    if not candidates:
        decisions.append(
            ProfileDecision("admitted_buckets", "()", "no candidate fits the runner ceiling")
        )
        return ResolvedExecutionProfile(
            profile=chosen_profile,
            provenance=provenance,
            candidate_buckets=(),
            admitted_buckets=(),
            rejected=(),
            flight_slots=flight_slots,
            graph_reserve_bytes=0,
            safety_reserve_bytes=safety_reserve_bytes,
            budget=_evidence(
                partition_capacity, available_bytes, non_kv_bytes, measured, 0, safety_reserve_bytes
            ),
            decisions=tuple(decisions),
            notes=tuple(notes),
        )

    admitted: list[int] = []
    rejected: list[tuple[int, ProfileRejection]] = []
    reserve = 0

    if explicit_buckets is not None:
        # An explicit set is the operator's decision and is admitted verbatim.
        # Budget policy governs which sets this module *proposes*; it does not
        # get to quietly narrow one an operator asked for, because a silent trim
        # reports success for a configuration nobody chose. The real hard limit
        # stays where it can be measured: capture reconciles the actual footprint
        # against the reserve and fails startup if the device cannot hold it.
        admitted = list(explicit_buckets)
        reserve = _estimate_graph_bytes(
            explicit_buckets,
            activation_bytes=activation_bytes,
            flight_slots=flight_slots,
            price_persistent=price_persistent or _floor_price,
        )
        if available_bytes > 0:
            shortfall = available_bytes - safety_reserve_bytes - reserve
            if shortfall < 0:
                decisions.append(
                    ProfileDecision(
                        "graph_reserve",
                        str(reserve),
                        f"the explicit set leaves {shortfall} bytes against the budget; "
                        "capture will refuse it if the device cannot hold the reserve",
                    )
                )
        else:
            notes.append(
                "the explicit set was admitted without a budget check; no memory "
                "snapshot was available at bootstrap"
            )
        return ResolvedExecutionProfile(
            profile=chosen_profile,
            provenance=provenance,
            candidate_buckets=tuple(candidates),
            admitted_buckets=tuple(admitted),
            rejected=(),
            flight_slots=flight_slots,
            graph_reserve_bytes=reserve,
            safety_reserve_bytes=safety_reserve_bytes,
            budget=_evidence(
                partition_capacity,
                available_bytes,
                non_kv_bytes,
                measured,
                reserve,
                safety_reserve_bytes,
            ),
            decisions=tuple(decisions),
            notes=tuple(notes),
        )

    if price_persistent is None:
        # Without a per-bucket price the cost of one bucket cannot be compared
        # against another. Admitting the whole ladder would be asserting a
        # comparison nobody made, so only the cheapest set is admitted and the
        # rest are recorded as unpriced rather than assumed affordable.
        smallest = min(candidates)
        admitted.append(smallest)
        rejected.extend((bucket, _UNPRICED) for bucket in candidates if bucket != smallest)
        reserve = _estimate_graph_bytes(
            (smallest,),
            activation_bytes=activation_bytes,
            flight_slots=flight_slots,
            price_persistent=_floor_price,
        )
        decisions.append(
            ProfileDecision(
                "admitted_buckets",
                str((smallest,)),
                "no per-bucket price was supplied, so the larger buckets were not "
                "shown to be affordable and were not admitted",
            )
        )
        admitted.sort()
        rejected.sort()
        return ResolvedExecutionProfile(
            profile=chosen_profile,
            provenance=provenance,
            candidate_buckets=tuple(candidates),
            admitted_buckets=tuple(admitted),
            rejected=tuple(rejected),
            flight_slots=flight_slots,
            graph_reserve_bytes=reserve,
            safety_reserve_bytes=safety_reserve_bytes,
            budget=_evidence(
                partition_capacity,
                available_bytes,
                non_kv_bytes,
                measured,
                reserve,
                safety_reserve_bytes,
            ),
            decisions=tuple(decisions),
            notes=tuple(notes),
        )

    # Descending order makes the incremental cost of each admission visible, and
    # matches the capture order the graph pool already uses.
    working = tuple(sorted(candidates, reverse=True))

    for bucket in working:
        trial = _estimate_graph_bytes(
            (bucket, *admitted),
            activation_bytes=activation_bytes,
            flight_slots=flight_slots,
            price_persistent=price_persistent,
        )
        if trial < _GRAPH_RESERVE_FLOOR:
            rejected.append((bucket, ProfileRejection.BELOW_RESERVE_FLOOR))
            continue
        headroom = available_bytes - safety_reserve_bytes - trial
        if headroom < min_headroom_bytes:
            rejected.append(
                (bucket, ProfileRejection.BUDGET_EXHAUSTED)
                if available_bytes <= 0
                else (bucket, ProfileRejection.INSUFFICIENT_HEADROOM)
            )
            continue
        admitted.append(bucket)
        reserve = trial

    if not admitted and candidates:
        # One bounded retry on the smallest set, with the reserve dropped to its
        # floor. It relaxes the reserve size only: the headroom floor still
        # holds, because a budget that cannot fund the safety margin is a
        # reason to run eager, not a reason to under-reserve.
        smallest = min(candidates)
        retry_reserve = _estimate_graph_bytes(
            (smallest,),
            activation_bytes=activation_bytes,
            flight_slots=flight_slots,
            price_persistent=price_persistent or _floor_price,
        )
        retry_headroom = available_bytes - safety_reserve_bytes - retry_reserve
        if retry_reserve > 0 and retry_headroom >= min_headroom_bytes:
            admitted.append(smallest)
            reserve = retry_reserve
            decisions.append(
                ProfileDecision(
                    "graph_reserve",
                    str(retry_reserve),
                    f"reduced to the {smallest}-bucket floor after the sized set did not fit",
                )
            )
        else:
            decisions.append(
                ProfileDecision(
                    "admitted_buckets",
                    "()",
                    f"even the {smallest}-bucket floor leaves "
                    f"{retry_headroom} bytes against a {min_headroom_bytes} minimum",
                )
            )

    if not admitted:
        decisions.append(
            ProfileDecision(
                "admitted_buckets",
                "()",
                "no candidate fit the budget; eager is the honest outcome",
            )
        )

    admitted.sort()
    if admitted and tuple(admitted) != tuple(candidates):
        decisions.append(
            ProfileDecision(
                "admitted_buckets",
                str(tuple(admitted)),
                f"trimmed from {tuple(candidates)} under the available budget",
            )
        )
    if not measured:
        notes.append("budget arithmetic is a precheck; capture reconciles the measured footprint")

    return ResolvedExecutionProfile(
        profile=chosen_profile,
        provenance=provenance,
        candidate_buckets=tuple(candidates),
        admitted_buckets=tuple(admitted),
        rejected=tuple(rejected),
        flight_slots=flight_slots,
        graph_reserve_bytes=reserve,
        safety_reserve_bytes=safety_reserve_bytes,
        budget=_evidence(
            partition_capacity,
            available_bytes,
            non_kv_bytes,
            measured,
            reserve,
            safety_reserve_bytes,
        ),
        decisions=tuple(decisions),
        notes=tuple(notes),
    )


def _evidence(
    partition_capacity: int,
    available: int,
    non_kv: int,
    measured: bool,
    reserve: int,
    safety: int,
) -> BudgetEvidence:
    return BudgetEvidence(
        partition_capacity_bytes=partition_capacity,
        available_bytes=available,
        non_kv_bytes=non_kv,
        measured=measured,
        graph_reserve_bytes=reserve,
        safety_reserve_bytes=safety,
        headroom_after_graph_bytes=available - reserve - safety,
    )
