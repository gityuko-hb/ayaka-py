"""Host memory probes — one reader for every ``/proc`` and ``/sys`` fact.

Two callers need the same machine facts and previously each grew its own
parser: ``configs/hardware.py`` wants total RAM, NUMA geometry and a device's
NUMA node; ``ayaka.memory.host.host_policy`` wants the container limit,
available memory, per-node free bytes and PSI pressure.  Two parsers of the
same kernel interfaces drift — one learns about cgroup v1 sentinels, the other
does not — and the failure is silent: a ceiling computed from the wrong number.

Every probe here degrades to ``None`` / ``0`` / ``()`` rather than raising.  A
machine with an unexpected cgroup layout or no ``/proc`` at all is a machine the
engine can still run on; refusing to start because a probe failed would be the
worse answer.

None of these functions import torch or initialise a CUDA context, so they are
safe to call before the fork boundary.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CgroupMemory",
    "cgroup_memory",
    "host_available_bytes",
    "host_ram_bytes",
    "host_total_ram_bytes",
    "memory_pressure",
    "numa_free_bytes",
    "numa_node_count",
    "pci_numa_node",
]

_MEMINFO = Path("/proc/meminfo")
_PRESSURE_MEMORY = Path("/proc/pressure/memory")
_CGROUP_V2 = Path("/sys/fs/cgroup")
_CGROUP_V1_MEM = Path("/sys/fs/cgroup/memory")
_NUMA_BASE = Path("/sys/devices/system/node")
_PCI_DEVICES = Path("/sys/bus/pci/devices")

#: cgroup v1 reports "unlimited" as a sentinel near ``2**63`` rather than a
#: word, so a raw ``int`` read of that file is not usable without this filter.
_CGROUP_UNLIMITED = 1 << 62


@dataclass(frozen=True, slots=True)
class CgroupMemory:
    """The container's memory limit, when one applies.

    ``None`` means unknown or unlimited; zero is a real observation.  The
    ``flavour`` names which hierarchy answered, because an operator debugging a
    limit needs to know whether ``memory.max`` or ``memory.limit_in_bytes`` was
    the file that produced it.
    """

    limit_bytes: int | None = None
    current_bytes: int | None = None
    flavour: str = "no-cgroup"


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(_MEMINFO, encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if not parts:
                    continue
                try:
                    values[key] = int(parts[0]) * 1024
                except ValueError:
                    continue
    except OSError:
        pass
    return values


def host_ram_bytes() -> int | None:
    """``MemTotal``: the machine's RAM, which inside a container is the host's."""
    return _meminfo().get("MemTotal")


def host_available_bytes() -> int | None:
    """``MemAvailable``: what could be used without reclaiming.

    A host can have 512 GiB installed and none of it available; a rule based on
    ``MemTotal`` cannot see that.
    """
    return _meminfo().get("MemAvailable")


def cgroup_memory() -> CgroupMemory:
    """``(limit, current, flavour)`` from cgroup v2, falling back to v1."""
    limit = _read_int(_CGROUP_V2 / "memory.max")
    current = _read_int(_CGROUP_V2 / "memory.current")
    if limit is not None or current is not None:
        return CgroupMemory(limit, current, "cgroup-v2")
    limit = _read_int(_CGROUP_V1_MEM / "memory.limit_in_bytes")
    current = _read_int(_CGROUP_V1_MEM / "memory.usage_in_bytes")
    if limit is not None and limit >= _CGROUP_UNLIMITED:
        limit = None
    if limit is not None or current is not None:
        return CgroupMemory(limit, current, "cgroup-v1")
    return CgroupMemory()


def host_total_ram_bytes() -> int | None:
    """Total host RAM, honouring a cgroup limit when one applies.

    A container's ``MemTotal`` is the *host's*, not the container's, so a
    ceiling computed from ``/proc/meminfo`` alone will happily exceed the cgroup
    limit and get the process OOM-killed by the kernel rather than refused by an
    allocator.  The answer is therefore the minimum over every limit the kernel
    exposes.
    """
    limits: list[int] = []
    total = host_ram_bytes()
    if total is not None:
        limits.append(total)
    for path in (_CGROUP_V2 / "memory.max", _CGROUP_V1_MEM / "memory.limit_in_bytes"):
        value = _read_int(path)
        if value is not None and 0 < value < _CGROUP_UNLIMITED:
            limits.append(value)
    return min(limits) if limits else None


def numa_node_count() -> int:
    """Number of NUMA nodes; ``1`` when the topology is unknown."""
    try:
        entries = os.listdir(_NUMA_BASE)
    except OSError:
        return 1
    nodes = sum(1 for entry in entries if re.fullmatch(r"node\d+", entry))
    return max(1, nodes)


def numa_free_bytes() -> tuple[int, ...]:
    """Free bytes per NUMA node, in node order; empty when unknown.

    Pinned pages cannot migrate between nodes, so a pin that does not fit on
    one node either fails or lands split — and a split pinned buffer pays
    cross-socket DMA for the life of the engine.
    """
    try:
        nodes = sorted(entry for entry in os.listdir(_NUMA_BASE) if re.fullmatch(r"node\d+", entry))
    except OSError:
        return ()
    free: list[int] = []
    for node in nodes:
        try:
            for line in (_NUMA_BASE / node / "meminfo").read_text().splitlines():
                if "MemFree:" in line:
                    free.append(int(line.split()[-2]) * 1024)
                    break
            else:
                free.append(0)
        except (OSError, ValueError, IndexError):
            free.append(0)
    return tuple(free)


def pci_numa_node(bus_id: str) -> int:
    """NUMA node of a PCI device, or ``-1`` when it cannot be determined."""
    if not bus_id:
        return -1
    clean = bus_id.lower().strip()
    if not clean.startswith("0000:"):
        clean = f"0000:{clean}"
    try:
        value = int((_PCI_DEVICES / clean / "numa_node").read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return -1
    return value if value >= 0 else -1


def memory_pressure() -> tuple[float, float]:
    """``(some.avg10, full.avg10)`` from ``/proc/pressure/memory``.

    PSI is the only probe that answers "is the host *struggling*" rather than
    "how much is free".  A machine at 99% used and idle is fine; a machine at
    70% used and reclaiming constantly is not, and only PSI distinguishes them.
    """
    try:
        text = _PRESSURE_MEMORY.read_text(encoding="ascii")
    except OSError:
        return 0.0, 0.0
    some = full = 0.0
    for line in text.splitlines():
        match = re.match(r"(some|full)\s+avg10=([\d.]+)", line)
        if not match:
            continue
        if match.group(1) == "some":
            some = float(match.group(2))
        else:
            full = float(match.group(2))
    return some, full
