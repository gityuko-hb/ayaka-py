"""Host memory policy — the safety ceiling on pinned memory.

Pinned memory is not "host memory that happens to be faster". It is memory the
kernel may not swap or reclaim, so every pinned byte is a byte the OS has
permanently lost. Over-pin and the failure is not a slow engine — it is the
host OOM killer choosing a victim, and on a container it is usually us.

So the effective limit is the minimum over two independent sources:

  operator settings     ``host_tier_bytes`` and ``host_pinned_max_bytes``
  system ceiling        what the machine can survive

The system ceiling is itself a minimum over several probes — container limit,
``MemAvailable``, PSI pressure, per-node free memory — each of which catches a
case the others miss.  Every probe lives in :mod:`ayaka.utils.host_info` and
degrades to "unknown" rather than raising: a policy that throws because
``/sys/fs/cgroup`` has an unexpected layout would make the engine unstartable on
a host it could have served fine.

:meth:`HostMemoryPolicy.can_pin` then gates each runtime request against that
ceiling, because "this configuration could never be safe" and "this *moment* is
not safe" are different failures with different remediations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from ayaka.types import MemoryTier
from ayaka.utils.host_info import (
    cgroup_memory,
    host_available_bytes,
    host_ram_bytes,
    memory_pressure,
    numa_free_bytes,
)

__all__ = [
    "HostMemoryFacts",
    "HostMemoryPolicy",
    "HostLimitsSettings",
    "MirrorDecision",
    "PinDecision",
    "PinRefusal",
    "probe_host_memory",
]


class HostLimitsSettings(Protocol):
    """The four operator ceilings this policy reads.

    A Protocol rather than importing the config schema: ``ayaka.memory`` has no
    business depending on it, and structural typing still checks the shape at
    every call site. The values come from ``MemoryConfig.host``
    (``HostMemoryLimits``); the requested tier size comes from
    ``CacheConfig.tiering.host_bytes``.
    """

    @property
    def pinned_max_bytes(self) -> int | None: ...

    @property
    def pinned_max_ratio(self) -> float: ...

    @property
    def min_available_bytes(self) -> int: ...

    @property
    def min_available_ratio(self) -> float: ...

    @property
    def allow_pageable_fallback(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class HostMemoryFacts:
    """What the machine says about itself.

    ``None`` means a probe could not determine a value. Zero is a real, known
    observation -- most importantly, zero available bytes must never be treated
    as permission to pin more memory.
    """

    mem_total_bytes: int | None = None
    mem_available_bytes: int | None = None
    cgroup_limit_bytes: int | None = None
    cgroup_current_bytes: int | None = None
    numa_nodes: int = 1
    numa_free_bytes: tuple[int, ...] = ()
    #: Percentage of the last 10 s some task stalled on memory. Above 10% the
    #: host is already reclaiming hard, and pinning more is how a slow machine
    #: becomes a dead one.
    psi_some_avg10: float = 0.0
    psi_full_avg10: float = 0.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        byte_facts = (
            self.mem_total_bytes,
            self.mem_available_bytes,
            self.cgroup_limit_bytes,
            self.cgroup_current_bytes,
        )
        if any(value is not None and value < 0 for value in byte_facts):
            raise ValueError("host memory byte facts must be non-negative")
        if self.numa_nodes <= 0:
            raise ValueError("numa_nodes must be positive")
        if any(value < 0 for value in self.numa_free_bytes):
            raise ValueError("NUMA free-byte facts must be non-negative")
        if self.numa_free_bytes and len(self.numa_free_bytes) != self.numa_nodes:
            raise ValueError("numa_free_bytes must have one value per NUMA node")
        if self.psi_some_avg10 < 0 or self.psi_full_avg10 < 0:
            raise ValueError("PSI averages must be non-negative")

    @property
    def scope_total_bytes(self) -> int:
        """The size of the box we are actually in.

        A container with a 16 GiB cgroup limit on a 512 GiB host must size
        against 16, not 512 — and ``MemTotal`` inside that container still
        reports 512, which is exactly how a percentage-of-MemTotal rule kills a
        pod.
        """
        if self.cgroup_limit_bytes is not None:
            if self.mem_total_bytes is not None:
                return min(self.cgroup_limit_bytes, self.mem_total_bytes)
            return self.cgroup_limit_bytes
        return self.mem_total_bytes if self.mem_total_bytes is not None else 0

    @property
    def scope_available_bytes(self) -> int | None:
        if self.cgroup_limit_bytes is not None and self.cgroup_current_bytes is not None:
            headroom = max(self.cgroup_limit_bytes - self.cgroup_current_bytes, 0)
            if self.mem_available_bytes is not None:
                return min(headroom, self.mem_available_bytes)
            return headroom
        return self.mem_available_bytes

    @property
    def under_pressure(self) -> bool:
        """``full`` PSI means *every* runnable task stalled — the host is already
        thrashing, not merely busy."""
        return self.psi_full_avg10 >= 1.0 or self.psi_some_avg10 >= 10.0

    @property
    def known(self) -> bool:
        return self.scope_total_bytes > 0


class PinRefusal(str):
    """A reason ``can_pin`` said no. A plain string subclass so it logs cleanly
    and still compares by value."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class PinDecision:
    allowed: bool
    reason: PinRefusal | None = None
    requested_bytes: int = 0
    headroom_bytes: int = 0

    def __bool__(self) -> bool:
        return self.allowed


@dataclass(frozen=True, slots=True)
class MirrorDecision:
    """Where one host KV mirror should actually be allocated.

    ``tier`` is ``None`` only when the request is refused. A fallback decision
    is still an allowed decision: the caller must build the mirror at
    ``tier`` and charge the ledger there, while logging ``reason``.
    """

    allowed: bool
    tier: MemoryTier | None = None
    fallback: bool = False
    reason: PinRefusal | None = None
    requested_bytes: int = 0
    headroom_bytes: int = 0

    def __bool__(self) -> bool:
        return self.allowed

    def __post_init__(self) -> None:
        if self.allowed and self.tier is None:
            raise ValueError("an allowed mirror decision must name a tier")
        if not self.allowed and self.tier is not None:
            raise ValueError("a refused mirror decision cannot name a tier")
        if self.fallback and not self.allowed:
            raise ValueError("a refused mirror decision cannot be a fallback")
        if self.tier is not None and self.tier not in (
            MemoryTier.HOST_PINNED,
            MemoryTier.HOST_PAGEABLE,
        ):
            raise ValueError(f"{self.tier} is not a host tier")


@dataclass(frozen=True, slots=True)
class HostMemoryPolicy:
    """Resolves config plus facts into one number, and gates runtime pinning.

    Split from ``CacheConfig`` because the config is a *request* and this is a
    *decision*: the same config produces different limits on a 512 GiB host and
    in a 16 GiB pod, and only one of those is knowable at config time.
    """

    host_tier_bytes: int
    host_pinned_max_bytes: int | None = None
    host_pinned_max_ratio: float = 0.25
    host_min_available_bytes: int = 4 << 30
    host_min_available_ratio: float = 0.10
    #: Whether a refused pinned mirror may degrade to pageable memory. A
    #: pageable mirror keeps the tier functional at reduced DMA throughput;
    #: ``False`` makes the pin ceiling a hard startup refusal.
    allow_pageable_fallback: bool = True
    facts: HostMemoryFacts = HostMemoryFacts()

    def __post_init__(self) -> None:
        if self.host_tier_bytes < 0:
            raise ValueError("host_tier_bytes must be non-negative")
        if not 0.0 < self.host_pinned_max_ratio <= 1.0:
            raise ValueError(
                f"host_pinned_max_ratio={self.host_pinned_max_ratio} must be in (0, 1]"
            )
        if not 0.0 <= self.host_min_available_ratio < 1.0:
            raise ValueError(
                f"host_min_available_ratio={self.host_min_available_ratio} must be in [0, 1)"
            )
        if self.host_pinned_max_bytes is not None and self.host_pinned_max_bytes < 0:
            raise ValueError("host_pinned_max_bytes must be non-negative")
        if not isinstance(self.allow_pageable_fallback, bool):
            raise TypeError("allow_pageable_fallback must be a bool")

    @classmethod
    def from_limits(
        cls,
        host_tier_bytes: int,
        limits: HostLimitsSettings,
        facts: HostMemoryFacts | None = None,
        *,
        allow_pageable_fallback: bool | None = None,
    ) -> HostMemoryPolicy:
        """Combine the requested tier size with the operator ceilings.

        ``limits`` is ``MemoryConfig.host`` (a ``HostMemoryLimits``); the
        machine facts are probed once here and refreshed only when a caller
        explicitly re-probes through :meth:`can_pin`.
        """
        if allow_pageable_fallback is None:
            allow_pageable_fallback = limits.allow_pageable_fallback
        return cls(
            host_tier_bytes=host_tier_bytes,
            host_pinned_max_bytes=limits.pinned_max_bytes,
            host_pinned_max_ratio=limits.pinned_max_ratio,
            host_min_available_bytes=limits.min_available_bytes,
            host_min_available_ratio=limits.min_available_ratio,
            allow_pageable_fallback=allow_pageable_fallback,
            facts=facts if facts is not None else probe_host_memory(),
        )

    # ── the ceiling ──────────────────────────────────────────────────────────

    @property
    def available_floor_bytes(self) -> int:
        """How much the engine must leave available to everything else."""
        scope = self.facts.scope_total_bytes
        return max(
            self.host_min_available_bytes,
            int(scope * self.host_min_available_ratio),
        )

    @property
    def physical_headroom_bytes(self) -> int:
        """What could be pinned right now without breaching the floor.

        Uses *available*, not total: a host with 512 GiB installed and 500 GiB in
        page cache and anonymous pages has no room to pin, and a
        percentage-of-total rule cannot see that.
        """
        available = self.facts.scope_available_bytes
        if available is None:
            return 0
        return max(available - self.available_floor_bytes, 0)

    def effective_pinned_limit(self) -> int:
        """The total pinned-memory ceiling: the minimum of every real constraint.

        ``host_tier_bytes`` is the *requested mirror size*, not a ceiling: a
        resize may briefly hold the replacement mirror while the old one is
        still pinned, and a SWAP must be allowed when the machine ceiling has
        room for both. The request itself is bounded by :meth:`can_pin`, which
        compares the actual bytes against this limit.

        When the machine is unknown (no ``/proc``, no cgroup) no system ceiling
        exists and the operator's requested tier size stands — refusing to run
        because a probe failed would be worse than trusting the number a human
        typed.
        """
        limits: list[int] = []
        if self.host_pinned_max_bytes is not None:
            limits.append(self.host_pinned_max_bytes)
        scope = self.facts.scope_total_bytes
        if scope:
            limits.append(int(scope * self.host_pinned_max_ratio))
        if self.facts.scope_available_bytes is not None:
            limits.append(self.physical_headroom_bytes)
        if not limits:
            return max(self.host_tier_bytes, 0)
        return max(min(limits), 0)

    def explain(self) -> str:
        """Why the limit is what it is. The single most useful log line when an
        operator asks for 64 GiB of host tier and gets 3."""
        scope = self.facts.scope_total_bytes
        parts = [f"requested={self.host_tier_bytes >> 20}MiB"]
        if self.host_pinned_max_bytes is not None:
            parts.append(f"max_bytes={self.host_pinned_max_bytes >> 20}MiB")
        if scope:
            parts.append(
                f"ratio={self.host_pinned_max_ratio:.0%}"
                f"×{scope >> 20}MiB={int(scope * self.host_pinned_max_ratio) >> 20}MiB"
            )
            parts.append(f"floor={self.available_floor_bytes >> 20}MiB")
        if self.facts.scope_available_bytes is not None:
            parts.append(f"headroom={self.physical_headroom_bytes >> 20}MiB")
        return (
            f"effective_pinned_limit={self.effective_pinned_limit() >> 20}MiB "
            f"= min({', '.join(parts)}) [{self.facts.source}]"
        )

    # ── runtime admission ────────────────────────────────────────────────────

    def can_pin(
        self,
        nbytes: int,
        *,
        already_pinned_bytes: int,
        facts: HostMemoryFacts | None = None,
    ) -> PinDecision:
        """Gate one pinning request.

        The ceiling is computed once at bootstrap; this runs per request,
        because the two failure modes are different. The ceiling stops a
        configuration that could never be safe; this stops a *moment* that is
        not — another process ballooned, the page cache grew, PSI spiked.

        Pass fresh ``facts`` to re-probe the *system*; the budget is always
        measured against the bootstrap ceiling, because that is the engine's
        stable allotment and it must not shrink under a transient.

        System conditions are checked before the budget on purpose. When a
        starved host drives the headroom to zero, "budget exhausted: 0 of 0" is
        true and useless; "would leave 1 GiB available, below the 6 GiB floor"
        names the thing an operator can act on.
        """
        live = facts if facts is not None else self.facts
        limit = self.effective_pinned_limit()
        if already_pinned_bytes < 0:
            raise ValueError("already_pinned_bytes must be non-negative")
        headroom = max(limit - already_pinned_bytes, 0)

        if nbytes <= 0:
            return PinDecision(False, PinRefusal("request must be positive"), nbytes, headroom)

        available = live.scope_available_bytes
        if available is not None:
            floor = max(
                self.host_min_available_bytes,
                int(live.scope_total_bytes * self.host_min_available_ratio),
            )
            remaining = available - nbytes
            if remaining < floor:
                return PinDecision(
                    False,
                    PinRefusal(
                        f"would leave {remaining >> 20}MiB available, below the "
                        f"{floor >> 20}MiB floor"
                    ),
                    nbytes,
                    headroom,
                )
        if (
            live.cgroup_limit_bytes is not None
            and live.cgroup_limit_bytes > 0
            and live.cgroup_current_bytes is not None
        ):
            used = live.cgroup_current_bytes / live.cgroup_limit_bytes
            if used > 0.90:
                return PinDecision(
                    False, PinRefusal(f"cgroup at {used:.0%} of its limit"), nbytes, headroom
                )
        if live.under_pressure:
            return PinDecision(
                False,
                PinRefusal(
                    f"memory pressure (PSI some={live.psi_some_avg10:.1f} "
                    f"full={live.psi_full_avg10:.1f}); the host is already reclaiming"
                ),
                nbytes,
                headroom,
            )
        if live.numa_free_bytes and max(live.numa_free_bytes) < nbytes:
            # Pinned pages cannot migrate between nodes, so a pin that does not
            # fit on one node either fails or lands split — and a split pinned
            # buffer means every DMA from the far half crosses the interconnect
            # for the life of the engine.
            return PinDecision(
                False,
                PinRefusal(
                    f"no NUMA node has {nbytes >> 20}MiB free "
                    f"(best {max(live.numa_free_bytes) >> 20}MiB); a split pinned "
                    "buffer pays cross-socket DMA forever"
                ),
                nbytes,
                headroom,
            )

        if nbytes > headroom:
            return PinDecision(
                False,
                PinRefusal(
                    f"pinned budget exhausted: {already_pinned_bytes >> 20}MiB of a "
                    f"{limit >> 20}MiB limit is already pinned"
                ),
                nbytes,
                headroom,
            )
        return PinDecision(True, None, nbytes, headroom)

    def decide_mirror(
        self,
        nbytes: int,
        *,
        requested_tier: MemoryTier,
        already_pinned_bytes: int,
        facts: HostMemoryFacts | None = None,
    ) -> MirrorDecision:
        """Resolve the tier a host KV mirror is actually allowed to use.

        A pageable request never consults the pin ceiling: pageable memory is
        reclaimable and carries no host-OOM risk. A pinned request that the
        ceiling refuses degrades to pageable only when the caller's policy
        allows it; otherwise the mirror is refused and the caller decides
        between a startup error and a retryable rejection.
        """
        if requested_tier is MemoryTier.HOST_PAGEABLE:
            return MirrorDecision(
                True,
                tier=MemoryTier.HOST_PAGEABLE,
                fallback=False,
                requested_bytes=nbytes,
                headroom_bytes=self.effective_pinned_limit(),
            )
        if requested_tier is not MemoryTier.HOST_PINNED:
            raise ValueError(f"{requested_tier} is not a host tier")
        decision = self.can_pin(
            nbytes,
            already_pinned_bytes=already_pinned_bytes,
            facts=facts,
        )
        if decision:
            return MirrorDecision(
                True,
                tier=MemoryTier.HOST_PINNED,
                fallback=False,
                reason=None,
                requested_bytes=nbytes,
                headroom_bytes=decision.headroom_bytes,
            )
        if self.allow_pageable_fallback:
            return MirrorDecision(
                True,
                tier=MemoryTier.HOST_PAGEABLE,
                fallback=True,
                reason=decision.reason,
                requested_bytes=nbytes,
                headroom_bytes=decision.headroom_bytes,
            )
        return MirrorDecision(
            False,
            tier=None,
            fallback=False,
            reason=decision.reason,
            requested_bytes=nbytes,
            headroom_bytes=decision.headroom_bytes,
        )


# ────────────────────────────── probes ───────────────────────────────────────


def probe_host_memory() -> HostMemoryFacts:
    """Read the machine. Never raises, never imports torch.

    Every ``/proc`` and ``/sys`` read lives in :mod:`ayaka.utils.host_info`; this
    function only decides which facts matter to the policy.
    ``AYAKA_DISABLE_HOST_PROBE=1`` returns empty facts, which is how the fast
    lane stays deterministic on a runner whose cgroup layout is not ours.
    """
    if os.environ.get("AYAKA_DISABLE_HOST_PROBE") not in (None, "", "0"):
        return HostMemoryFacts(source="probe disabled")
    cgroup = cgroup_memory()
    some, full = memory_pressure()
    numa = numa_free_bytes()
    return HostMemoryFacts(
        mem_total_bytes=host_ram_bytes(),
        mem_available_bytes=host_available_bytes(),
        cgroup_limit_bytes=cgroup.limit_bytes,
        cgroup_current_bytes=cgroup.current_bytes,
        numa_nodes=max(len(numa), 1),
        numa_free_bytes=numa,
        psi_some_avg10=some,
        psi_full_avg10=full,
        source=f"meminfo+{cgroup.flavour}",
    )
