"""Verified peer topology probe and cache.

The driver-level ``PeerTopology`` matrix answers "does CUDA report peer
access". This module answers the question a custom backend actually needs:
did an isolated producer/consumer pair really export, map, write, read and
signal through peer memory, and what exactly failed when it did not.

Report vocabulary:

``verified``     the real data path passed for the pair;
``unsupported``  a driver/platform fact says the pair cannot work;
``unknown``      the probe could not prove it (timeout, crash, missing
                 primitive) -- never silently reported as a pass.

Importing this module is torch-free and side-effect-free. The real probe
spawns processes (never forks a CUDA context), gives every child a deadline
and reaps stragglers, and the cache degrades to "probe again" on any error
rather than taking the engine down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ayaka.distributed.topology import (
    PhysicalDeviceIdentity,
    _optional_torch,
    physical_device_identity,
    visible_device_identities,
)
from ayaka.utils.import_utils import has_module

__all__ = [
    "PROBE_SCHEMA_VERSION",
    "PROBE_VERSION",
    "CacheLookup",
    "DeviceProbeBackend",
    "PairProbeReport",
    "ProbeBackend",
    "ProbeEnvironment",
    "ProbeReason",
    "ProbeStatus",
    "VerifiedTopologyReport",
    "cache_disabled",
    "capture_environment",
    "default_cache_path",
    "driver_version",
    "load_cached_report",
    "probe_verified_topology",
    "validate_report",
    "write_cached_report",
]

logger = logging.getLogger(__name__)

#: Cache/report schema. A different value is rejected, never interpreted.
PROBE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
#: Probe semantics version; bump when what "verified" proves changes.
PROBE_VERSION = "dc2-v1"

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_EPOCHS = 1000
DEFAULT_SPIN_BUDGET = 5_000_000
DEFAULT_ATOMIC_TIMES = 64
#: Bytes one writer owns in every peer's workspace for the read/write gate.
DEFAULT_SLOT_BYTES = 4096
#: Signal pads are separated by at least one cache line.
SIGNAL_PAD_BYTES = 128
SIGNAL_PAD_WORDS = SIGNAL_PAD_BYTES // 4
SESSION_GENERATION = (1, 1)

CACHE_ENV = "AYAKA_P2P_PROBE_CACHE"
DISABLE_CACHE_ENV = "AYAKA_P2P_PROBE_DISABLE_CACHE"
FORCE_ENV = "AYAKA_P2P_PROBE_FORCE"
TEST_CRASH_ENV = "AYAKA_P2P_PROBE_TEST_CRASH"
TEST_HANG_ENV = "AYAKA_P2P_PROBE_TEST_HANG"
TEST_MALFORMED_ENV = "AYAKA_P2P_PROBE_TEST_MALFORMED"

_TRUTHY = {"1", "true", "yes", "on"}
_DRIVER_VERSION_RE = re.compile(r"Kernel Module\s+([0-9][^\s]*)")


class ProbeStatus(StrEnum):
    """Tri-state capability outcome; ``SKIP`` does not exist on purpose."""

    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    VERIFIED = "verified"


class ProbeReason(StrEnum):
    """Stable reason codes shared with DC3's dispatch policy."""

    OK = "ok"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    NO_CUDA_DEVICE = "no_cuda_device"
    SINGLE_DEVICE = "single_device"
    PEER_ACCESS_ABSENT = "peer_access_absent"
    IPC_UNAVAILABLE = "ipc_unavailable"
    P2P_UNVERIFIED = "p2p_unverified"
    ATOMIC_UNSUPPORTED = "atomic_unsupported"
    IDENTITY_MISMATCH = "identity_mismatch"
    ORDINAL_ONLY_IDENTITY = "ordinal_only_identity"
    TIMEOUT = "timeout"
    CHILD_CRASH = "child_crash"
    MALFORMED_OUTPUT = "malformed_output"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_DISABLED = "cache_disabled"
    CACHE_STALE = "cache_stale"
    CACHE_CORRUPT = "cache_corrupt"
    CACHE_UNWRITABLE = "cache_unwritable"
    CACHE_FORCED = "cache_forced"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def driver_version() -> str:
    """NVIDIA driver version from ``/proc`` on Linux, else ``""``.

    Deliberately ``/proc``-only: no subprocess, no NVML requirement, so the
    cache fingerprint is still driver-bound on a host without either.
    """
    try:
        text = Path("/proc/driver/nvidia/version").read_text(encoding="ascii", errors="replace")
    except OSError:
        return ""
    match = _DRIVER_VERSION_RE.search(text)
    return match.group(1) if match else ""


def _load_nvml() -> Any | None:
    """Import and initialize NVML, or return None.

    NVML is optional by contract: missing NVML is missing topology detail,
    not a missing P2P path (spec section 7).
    """
    if not has_module("pynvml"):
        return None
    try:
        from importlib import import_module

        nvml = import_module("pynvml")
        nvml.nvmlInit()
    except Exception:
        logger.debug("NVML init failed; fabric detail stays unknown", exc_info=True)
        return None
    return nvml


def _nvml_available() -> bool:
    """Whether NVML can be initialized here; always shuts the handle down."""
    nvml = _load_nvml()
    if nvml is None:
        return False
    try:
        return True
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def nvml_link_kind(source_uuid: str, destination_uuid: str) -> str:
    """``"nvlink"``, ``"pcie"`` or ``"unknown"`` for one ordered pair.

    Returns ``"unknown"`` whenever NVML is unavailable or ambiguous; this
    field is informational and never decides ``verified``.
    """
    if not source_uuid or not destination_uuid:
        return "unknown"
    nvml = _load_nvml()
    if nvml is None:
        return "unknown"
    try:
        source = nvml.nvmlDeviceGetHandleByUUID(source_uuid)
        destination = nvml.nvmlDeviceGetHandleByUUID(destination_uuid)
        status = nvml.nvmlDeviceGetP2PStatus(
            source,
            destination,
            nvml.NVML_P2P_CAPS_INDEX_NVLINK,
        )
        return "nvlink" if status == nvml.NVML_P2P_STATUS_OK else "pcie"
    except Exception:
        return "unknown"
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class ProbeEnvironment:
    """Everything a cached probe.

    ``identities`` is the ordered local-ordinal -> physical-GPU mapping. Cache
    validation compares both this mapping and the flat fingerprint, so a
    ``CUDA_VISIBLE_DEVICES`` permutation invalidates a positive cache even
    though the set of physical GPUs is unchanged.
    """

    platform: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_runtime: str = ""
    triton_version: str = ""
    driver_version: str = ""
    device_count: int = 0
    cuda_visible_devices: str = ""
    nvidia_visible_devices: str = ""
    nvml_available: bool = False
    identities: tuple[PhysicalDeviceIdentity, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_runtime": self.cuda_runtime,
            "triton_version": self.triton_version,
            "driver_version": self.driver_version,
            "device_count": self.device_count,
            "cuda_visible_devices": self.cuda_visible_devices,
            "nvidia_visible_devices": self.nvidia_visible_devices,
            "nvml_available": self.nvml_available,
            "identities": [identity.to_dict() for identity in self.identities],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeEnvironment:
        if not isinstance(payload, Mapping):
            raise TypeError("probe environment payload must be a mapping")
        raw_identities = payload.get("identities", ())
        if not isinstance(raw_identities, (list, tuple)):
            raise ValueError("probe environment identities must be a sequence")
        return cls(
            platform=payload.get("platform", ""),
            python_version=payload.get("python_version", ""),
            torch_version=payload.get("torch_version", ""),
            cuda_runtime=payload.get("cuda_runtime", ""),
            triton_version=payload.get("triton_version", ""),
            driver_version=payload.get("driver_version", ""),
            device_count=payload.get("device_count", 0),
            cuda_visible_devices=payload.get("cuda_visible_devices", ""),
            nvidia_visible_devices=payload.get("nvidia_visible_devices", ""),
            nvml_available=bool(payload.get("nvml_available", False)),
            identities=tuple(PhysicalDeviceIdentity.from_dict(item) for item in raw_identities),
        )

    def fingerprint(self) -> str:
        """Stable digest of this environment, identities included."""
        return _sha256_json(self.to_dict())


def capture_environment(
    *,
    identities: tuple[PhysicalDeviceIdentity, ...] | None = None,
) -> ProbeEnvironment:
    """Read the environment a probe result is bound to. Never raises."""
    torch = _optional_torch()
    torch_version = cuda_runtime = ""
    if torch is not None:
        torch_version = str(getattr(torch, "__version__", ""))
        cuda_runtime = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    triton_version = ""
    if has_module("triton"):
        try:
            import triton

            triton_version = str(getattr(triton, "__version__", ""))
        except Exception:  # pragma: no cover - broken triton install
            triton_version = ""
    resolved = visible_device_identities() if identities is None else tuple(identities)
    return ProbeEnvironment(
        platform=f"{sys.platform}-{platform.machine()}",
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_runtime=cuda_runtime,
        triton_version=triton_version,
        driver_version=driver_version(),
        device_count=len(resolved),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "") or "",
        nvidia_visible_devices=os.environ.get("NVIDIA_VISIBLE_DEVICES", "") or "",
        nvml_available=_nvml_available(),
        identities=resolved,
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairProbeReport:
    """What the probe proved for one ordered (source -> destination) pair.

    ``same_device`` pairs exercise IPC between two processes on one physical
    GPU; they are valid same-device evidence and are never counted as
    cross-device P2P proof (G-DC2.5).
    """

    source_rank: int
    destination_rank: int
    source_device: int
    destination_device: int
    status: ProbeStatus
    reason: ProbeReason
    source_identity: PhysicalDeviceIdentity | None = None
    destination_identity: PhysicalDeviceIdentity | None = None
    same_device: bool = False
    peer_access: bool | None = None
    ipc_verified: bool = False
    write_verified: bool = False
    read_verified: bool = False
    ordering: bool | None = None
    atomic: bool | None = None
    fabric: str = "unknown"
    bytes_tested: int = 0
    epochs: int = 0
    elapsed_s: float = 0.0
    error: str = ""

    def __post_init__(self) -> None:
        for label, value in (
            ("source_rank", self.source_rank),
            ("destination_rank", self.destination_rank),
            ("source_device", self.source_device),
            ("destination_device", self.destination_device),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"pair {label} must be a non-negative integer")
        if not isinstance(self.status, ProbeStatus):
            raise TypeError("pair status must be a ProbeStatus")
        if not isinstance(self.reason, ProbeReason):
            raise TypeError("pair reason must be a ProbeReason")
        for label, value in (("bytes_tested", self.bytes_tested), ("epochs", self.epochs)):
            if type(value) is not int or value < 0:
                raise ValueError(f"pair {label} must be a non-negative integer")

    @property
    def cross_device(self) -> bool:
        """Whether this pair can count as cross-device P2P evidence."""
        return not self.same_device

    @property
    def verified(self) -> bool:
        return self.status is ProbeStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rank": self.source_rank,
            "destination_rank": self.destination_rank,
            "source_device": self.source_device,
            "destination_device": self.destination_device,
            "source_identity": (
                self.source_identity.to_dict() if self.source_identity is not None else None
            ),
            "destination_identity": (
                self.destination_identity.to_dict()
                if self.destination_identity is not None
                else None
            ),
            "status": self.status.value,
            "reason": self.reason.value,
            "same_device": self.same_device,
            "peer_access": self.peer_access,
            "ipc_verified": self.ipc_verified,
            "write_verified": self.write_verified,
            "read_verified": self.read_verified,
            "ordering": self.ordering,
            "atomic": self.atomic,
            "fabric": self.fabric,
            "bytes_tested": self.bytes_tested,
            "epochs": self.epochs,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PairProbeReport:
        if not isinstance(payload, Mapping):
            raise TypeError("pair probe payload must be a mapping")
        source_identity = payload.get("source_identity")
        destination_identity = payload.get("destination_identity")
        return cls(
            source_rank=payload["source_rank"],
            destination_rank=payload["destination_rank"],
            source_device=payload["source_device"],
            destination_device=payload["destination_device"],
            status=ProbeStatus(payload["status"]),
            reason=ProbeReason(payload["reason"]),
            source_identity=(
                PhysicalDeviceIdentity.from_dict(source_identity)
                if isinstance(source_identity, Mapping)
                else None
            ),
            destination_identity=(
                PhysicalDeviceIdentity.from_dict(destination_identity)
                if isinstance(destination_identity, Mapping)
                else None
            ),
            same_device=bool(payload.get("same_device", False)),
            peer_access=payload.get("peer_access"),
            ipc_verified=bool(payload.get("ipc_verified", False)),
            write_verified=bool(payload.get("write_verified", False)),
            read_verified=bool(payload.get("read_verified", False)),
            ordering=payload.get("ordering"),
            atomic=payload.get("atomic"),
            fabric=str(payload.get("fabric", "unknown")),
            bytes_tested=int(payload.get("bytes_tested", 0)),
            epochs=int(payload.get("epochs", 0)),
            elapsed_s=float(payload.get("elapsed_s", 0.0)),
            error=str(payload.get("error", "")),
        )


@dataclass(frozen=True, slots=True)
class VerifiedTopologyReport:
    """Aggregate probe result for one probe run (fresh or cached)."""

    status: ProbeStatus
    reason: ProbeReason
    environment: ProbeEnvironment
    pairs: tuple[PairProbeReport, ...] = ()
    probe_version: str = PROBE_VERSION
    schema_version: int = PROBE_SCHEMA_VERSION
    cache_reason: ProbeReason = ProbeReason.CACHE_MISS
    created_unix: float = 0.0
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProbeStatus):
            raise TypeError("report status must be a ProbeStatus")
        if not isinstance(self.reason, ProbeReason):
            raise TypeError("report reason must be a ProbeReason")
        if self.schema_version != PROBE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported probe schema_version {self.schema_version!r}; "
                f"expected {PROBE_SCHEMA_VERSION}"
            )

    def pair(self, source_device: int, destination_device: int) -> PairProbeReport | None:
        """The report for one ordered pair of local device ordinals."""
        for entry in self.pairs:
            if (
                entry.source_device == source_device
                and entry.destination_device == destination_device
            ):
                return entry
        return None

    def verified_for(
        self,
        source_device: int,
        destination_device: int,
        *,
        need_ordering: bool = True,
        need_remote_atomic: bool = False,
    ) -> bool:
        """Whether a pair is verified for the primitive set a backend needs."""
        entry = self.pair(source_device, destination_device)
        if entry is None or not entry.verified:
            return False
        if need_ordering and entry.ordering is not True:
            return False
        if need_remote_atomic and entry.atomic is not True:
            return False
        return True

    def agreement_key(self) -> str:
        """Digest ranks compare to prove they agree on the profile.

        Volatile fields (timestamps, durations, cache state, per-pair error
        strings) are excluded; statuses, reasons, identities and primitive
        outcomes are included.
        """
        payload = {
            "probe_version": self.probe_version,
            "status": self.status.value,
            "reason": self.reason.value,
            "environment": self.environment.to_dict(),
            "pairs": [
                {
                    key: value
                    for key, value in entry.to_dict().items()
                    if key not in {"elapsed_s", "error"}
                }
                for entry in self.pairs
            ],
        }
        return _sha256_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "probe_version": self.probe_version,
            "status": self.status.value,
            "reason": self.reason.value,
            "environment": self.environment.to_dict(),
            "pairs": [entry.to_dict() for entry in self.pairs],
            "cache_reason": self.cache_reason.value,
            "created_unix": self.created_unix,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerifiedTopologyReport:
        if not isinstance(payload, Mapping):
            raise TypeError("verified topology payload must be a mapping")
        raw_pairs = payload.get("pairs", ())
        if not isinstance(raw_pairs, (list, tuple)):
            raise ValueError("verified topology pairs must be a sequence")
        return cls(
            status=ProbeStatus(payload["status"]),
            reason=ProbeReason(payload["reason"]),
            environment=ProbeEnvironment.from_dict(payload["environment"]),
            pairs=tuple(PairProbeReport.from_dict(item) for item in raw_pairs),
            probe_version=str(payload.get("probe_version", "")),
            schema_version=int(payload.get("schema_version", 0)),
            cache_reason=ProbeReason(payload.get("cache_reason", ProbeReason.CACHE_MISS.value)),
            created_unix=float(payload.get("created_unix", 0.0)),
            duration_s=float(payload.get("duration_s", 0.0)),
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheLookup:
    """Outcome of a cache read: the hit or the reason there is none."""

    report: VerifiedTopologyReport | None
    reason: ProbeReason


def default_cache_path() -> Path | None:
    """Cache file path, honouring ``AYAKA_P2P_PROBE_CACHE``.

    An explicitly empty override disables caching; otherwise the XDG cache
    directory is used. ``None`` means "no cache in this process".
    """
    override = os.environ.get(CACHE_ENV)
    if override is not None:
        override = override.strip()
        return Path(override).expanduser() if override else None
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "ayaka" / "p2p_probe.json"


def cache_disabled() -> bool:
    """Whether the kill-switch environment variable disables the cache."""
    return os.environ.get(DISABLE_CACHE_ENV, "").strip().lower() in _TRUTHY


def _live_peer_access(source: int, destination: int) -> bool:
    torch = _optional_torch()
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.can_device_access_peer(source, destination))
    except Exception:
        return False


def validate_report(
    report: VerifiedTopologyReport,
    *,
    environment: ProbeEnvironment | None = None,
) -> tuple[bool, ProbeReason]:
    """Re-check a (possibly cached) report against the live host.

    Cached identity is never enough on its own: identities are re-read and
    every cross pair that claimed driver peer access is re-queried before the
    report is trusted at setup (spec section 7).
    """
    try:
        identities = visible_device_identities()
    except Exception:
        return False, ProbeReason.CACHE_STALE
    expected = report.environment.identities
    if len(identities) != len(expected):
        return False, ProbeReason.CACHE_STALE
    for live, stored in zip(identities, expected, strict=True):
        if not live.same_physical_device(stored):
            return False, ProbeReason.CACHE_STALE
    if environment is not None and environment.fingerprint() != report.environment.fingerprint():
        return False, ProbeReason.CACHE_STALE
    for entry in report.pairs:
        if entry.same_device or entry.peer_access is False:
            continue
        if not _live_peer_access(entry.source_device, entry.destination_device):
            return False, ProbeReason.CACHE_STALE
    return True, ProbeReason.OK


def load_cached_report(
    environment: ProbeEnvironment,
    *,
    path: str | Path | None = None,
    live: bool = True,
) -> CacheLookup:
    """Load a cached report if it is valid for ``environment`` right now.

    Only ``verified`` reports are served. Corruption, schema drift, a
    fingerprint mismatch or failed live revalidation all mean "probe again",
    never "trust a stale positive" (G-DC2.3).
    """
    if cache_disabled():
        return CacheLookup(None, ProbeReason.CACHE_DISABLED)
    resolved = Path(path) if path is not None else default_cache_path()
    if resolved is None:
        return CacheLookup(None, ProbeReason.CACHE_DISABLED)
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return CacheLookup(None, ProbeReason.CACHE_MISS)
    except OSError:
        return CacheLookup(None, ProbeReason.CACHE_CORRUPT)
    try:
        report = VerifiedTopologyReport.from_dict(json.loads(text))
    except Exception:
        return CacheLookup(None, ProbeReason.CACHE_CORRUPT)
    if report.probe_version != PROBE_VERSION:
        return CacheLookup(None, ProbeReason.CACHE_CORRUPT)
    if report.status is not ProbeStatus.VERIFIED:
        return CacheLookup(None, ProbeReason.CACHE_MISS)
    if report.environment.fingerprint() != environment.fingerprint():
        return CacheLookup(None, ProbeReason.CACHE_STALE)
    if live:
        ok, reason = validate_report(report, environment=environment)
        if not ok:
            return CacheLookup(None, reason)
    return CacheLookup(report, ProbeReason.CACHE_HIT)


def write_cached_report(
    report: VerifiedTopologyReport,
    *,
    path: str | Path | None = None,
) -> ProbeReason:
    """Persist a verified report with an atomic, concurrent-safe write.

    Returns the reason code instead of raising: an unwritable cache is a
    degradation, not a probe failure. Non-verified reports are not cached.
    """
    if cache_disabled():
        return ProbeReason.CACHE_DISABLED
    if report.status is not ProbeStatus.VERIFIED:
        return ProbeReason.CACHE_MISS
    resolved = Path(path) if path is not None else default_cache_path()
    if resolved is None:
        return ProbeReason.CACHE_DISABLED
    payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
    temp_name = ""
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{resolved.name}.", suffix=".tmp", dir=str(resolved.parent)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, resolved)
    except OSError:
        logger.debug("probe cache write failed", exc_info=True)
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        return ProbeReason.CACHE_UNWRITABLE
    return ProbeReason.OK


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProbeBackend(Protocol):
    """Injectable probe source; the default spawns real worker processes."""

    def identities(self, devices: tuple[int, ...]) -> tuple[PhysicalDeviceIdentity, ...]: ...

    def probe_pairs(
        self,
        devices: tuple[int, ...],
        identities: tuple[PhysicalDeviceIdentity, ...],
        *,
        need_ordering: bool,
        need_remote_atomic: bool,
        timeout_s: float,
        epochs: int,
        spin_budget: int,
    ) -> tuple[PairProbeReport, ...]: ...


# ---------------------------------------------------------------------------
# Layout shared by parent and workers
# ---------------------------------------------------------------------------


def _align_words(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _layout(world: int, epochs: int, slot_bytes: int = DEFAULT_SLOT_BYTES) -> dict[str, int]:
    """Int32-word offsets of every region inside one rank's workspace.

    Layout per workspace: ``world`` payload slots, then ``world`` order
    regions (epoch payloads), then per-source write/order signal pads, one
    done pad and one atomic counter pad. Pads are 128 bytes apart.
    """
    if slot_bytes % 4 != 0:
        raise ValueError("slot_bytes must be a multiple of 4")
    order_words = _align_words(epochs + 1, SIGNAL_PAD_WORDS)
    slots_words = (world * slot_bytes) // 4
    order_base = slots_words
    signal_base = order_base + world * order_words
    write_sig = signal_base
    order_sig = write_sig + world * SIGNAL_PAD_WORDS
    done_sig = order_sig + world * SIGNAL_PAD_WORDS
    counter = done_sig + SIGNAL_PAD_WORDS
    workspace_words = counter + SIGNAL_PAD_WORDS
    return {
        "slot_bytes": slot_bytes,
        "order_words": order_words,
        "workspace_words": workspace_words,
        "order_base": order_base,
        "write_sig": write_sig,
        "order_sig": order_sig,
        "done_sig": done_sig,
        "counter": counter,
    }


# ---------------------------------------------------------------------------
# Process harness
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _wait_for_json(path: Path, deadline: float, poll_s: float = 0.02) -> dict[str, Any]:
    while True:
        if path.is_file():
            try:
                return _read_json(path)
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(poll_s)


def _pattern_tensor(slot_bytes: int, writer: int, device: Any) -> Any:
    import torch

    index = torch.arange(slot_bytes, dtype=torch.int64, device=device)
    return ((index * 31 + writer * 37 + 11) % 251).to(torch.uint8)


def _maybe_test_hooks(rank: int, work: Path) -> None:
    """Deterministic failure injection, read only when the env var is set."""
    crash = os.environ.get(TEST_CRASH_ENV, "").strip()
    if crash and crash in {"1", str(rank)}:
        raise RuntimeError(f"{TEST_CRASH_ENV} forced a crash on rank {rank}")
    hang = os.environ.get(TEST_HANG_ENV, "").strip()
    if hang and hang in {"1", str(rank)}:
        time.sleep(3600.0)
    malformed = os.environ.get(TEST_MALFORMED_ENV, "").strip()
    if malformed and malformed in {"1", str(rank)}:
        results = work / "results"
        results.mkdir(parents=True, exist_ok=True)
        (results / f"rank{rank}.json").write_text("{not-json", encoding="utf-8")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _probe_rank_worker(rank: int, workdir: str) -> None:
    """One probe participant; runs in a spawned process, never forked."""
    work = Path(workdir)
    try:
        job = _read_json(work / "job.json")
        _maybe_test_hooks(rank, work)
        result = _run_rank_protocol(rank, job, work)
        _write_json(work / "results" / f"rank{rank}.json", result)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 - reported through the result file
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "rank": rank,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json(work / "results" / f"rank{rank}.json", payload)
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _run_rank_protocol(rank: int, job: Mapping[str, Any], work: Path) -> dict[str, Any]:
    """Export, import every peer, prove write/read/ordering/atomic, close."""
    import torch

    from ayaka.kernel.triton import p2p_signal
    from ayaka.runtime.ipc_engine import IpcRegionDescriptor, IpcRegionRegistry

    devices = [int(value) for value in job["devices"]]
    world = int(job["world_size"])
    need_ordering = bool(job["need_ordering"])
    need_remote_atomic = bool(job["need_remote_atomic"])
    epochs = int(job["epochs"])
    spin_budget = int(job["spin_budget"])
    atomic_times = int(job["atomic_times"])
    layout = {key: int(value) for key, value in dict(job["layout"]).items()}
    device = devices[rank]
    torch.cuda.set_device(device)
    own_identity = physical_device_identity(device)
    expected_identity = PhysicalDeviceIdentity.from_dict(job["identities"][rank])
    if own_identity.uuid and expected_identity.uuid and own_identity.uuid != expected_identity.uuid:
        raise RuntimeError(
            f"device identity changed under rank {rank}: expected "
            f"{expected_identity.uuid}, process sees {own_identity.uuid}"
        )

    peers = [other for other in range(world) if other != rank]
    workspace = torch.zeros(layout["workspace_words"], dtype=torch.int32, device=f"cuda:{device}")
    workspace_bytes = workspace.view(torch.uint8)
    registry = IpcRegionRegistry(device=device, session_generation=SESSION_GENERATION)
    descriptor = registry.export(workspace, close_acks=peers)
    wire_descriptor = replace(
        descriptor,
        physical_device_uuid=own_identity.uuid,
        exporter_rank=rank,
        device=device,
    )
    _write_json(work / "rendezvous" / f"identity{rank}.json", own_identity.to_dict())
    _write_json(work / "rendezvous" / f"descriptor{rank}.json", wire_descriptor.to_dict())

    deadline = time.monotonic() + max(5.0, float(job["timeout_s"]) - 5.0)
    imports: dict[int, tuple[Any, IpcRegionRegistry, IpcRegionDescriptor, Any]] = {}
    try:
        for other in peers:
            peer_identity = PhysicalDeviceIdentity.from_dict(
                _wait_for_json(work / "rendezvous" / f"identity{other}.json", deadline)
            )
            peer_descriptor = IpcRegionDescriptor.from_dict(
                _wait_for_json(work / "rendezvous" / f"descriptor{other}.json", deadline)
            )
            importer = IpcRegionRegistry(
                device=device,
                session_generation=SESSION_GENERATION,
                physical_device_uuid=peer_identity.uuid or None,
            )
            view = importer.open(peer_descriptor)
            imports[other] = (view, importer, peer_descriptor, view.tensor.view(torch.int32))

        own_words = workspace
        observed_patterns: dict[str, bool] = {}
        observed_ordering: dict[str, bool] = {}
        ordering_epochs = epochs if need_ordering else 0

        # Phase 1: publish everything this rank owes every peer.
        for other in peers:
            _, _, _, peer_words = imports[other]
            peer_bytes = peer_words.view(torch.uint8)
            slot_off = rank * layout["slot_bytes"]
            peer_bytes[slot_off : slot_off + layout["slot_bytes"]].copy_(
                _pattern_tensor(layout["slot_bytes"], rank, f"cuda:{device}")
            )
            order_off = layout["order_base"] + rank * layout["order_words"]
            p2p_signal.publish_epochs(
                peer_words[order_off : order_off + layout["order_words"]],
                peer_words[layout["write_sig"] + rank : layout["write_sig"] + rank + 1],
                epochs=1,
            )
            if need_ordering:
                p2p_signal.publish_epochs(
                    peer_words[order_off : order_off + layout["order_words"]],
                    peer_words[layout["order_sig"] + rank : layout["order_sig"] + rank + 1],
                    epochs=epochs,
                )
            if need_remote_atomic:
                p2p_signal.remote_atomic_add(
                    peer_words[layout["counter"] : layout["counter"] + 1], atomic_times
                )
        if need_remote_atomic:
            p2p_signal.release_signal(own_words[layout["done_sig"] : layout["done_sig"] + 1])

        # Phase 2: consume what every peer owes this rank, then verify.
        out = torch.zeros(epochs + 1, dtype=torch.int32, device=f"cuda:{device}")
        for other in peers:
            _, _, _, peer_words = imports[other]
            other_order = layout["order_base"] + other * layout["order_words"]
            p2p_signal.consume_epochs(
                own_words[other_order : other_order + layout["order_words"]],
                own_words[layout["write_sig"] + other : layout["write_sig"] + other + 1],
                out,
                epochs=1,
            )
            expected = _pattern_tensor(layout["slot_bytes"], other, f"cuda:{device}")
            slot_off = other * layout["slot_bytes"]
            observed_patterns[str(other)] = bool(
                torch.equal(
                    workspace_bytes[slot_off : slot_off + layout["slot_bytes"]],
                    expected,
                )
            )
            if need_ordering:
                p2p_signal.consume_epochs(
                    own_words[other_order : other_order + layout["order_words"]],
                    own_words[layout["order_sig"] + other : layout["order_sig"] + other + 1],
                    out,
                    epochs=epochs,
                    spin_budget=spin_budget,
                )
                reference = torch.arange(1, epochs + 1, dtype=torch.int32, device=f"cuda:{device}")
                observed_ordering[str(other)] = bool(torch.equal(out[1 : epochs + 1], reference))
        atomic_observed = 0
        atomic_expected = len(peers) * atomic_times
        if need_remote_atomic:
            for other in peers:
                _, _, _, peer_words = imports[other]
                other_order = layout["order_base"]
                p2p_signal.consume_epochs(
                    peer_words[other_order : other_order + layout["order_words"]],
                    peer_words[layout["done_sig"] : layout["done_sig"] + 1],
                    out,
                    epochs=1,
                )
            torch.cuda.synchronize()
            atomic_observed = int(own_words[layout["counter"]].item())

        # Cleanup: importer closes first and ACKs; exporter waits for ACKs.
        for other, (view, importer, peer_descriptor, _) in imports.items():
            view.close()
            importer.close(peer_descriptor)
            _write_json(work / "rendezvous" / f"ack{other}_{rank}.json", {"ok": True})
        mappings_closed = registry.opens == 0 and all(
            importer.opens == 0 for _, importer, _, _ in imports.values()
        )
        for other in peers:
            _wait_for_json(work / "rendezvous" / f"ack{rank}_{other}.json", deadline)
            registry.confirm_close(descriptor, other)
        registry.release(descriptor)
        registry.close_all(force=True)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "rank": rank,
            "device": device,
            "identity": own_identity.to_dict(),
            "observed_patterns": observed_patterns,
            "observed_ordering": observed_ordering,
            "ordering_epochs": ordering_epochs,
            "atomic_observed": atomic_observed,
            "atomic_expected": atomic_expected,
            "atomic_ok": atomic_observed == atomic_expected,
            "mappings_closed": mappings_closed,
            "error": None,
        }
    finally:
        for view, importer, _, _ in imports.values():
            try:
                view.close()
            except Exception:
                pass
            try:
                importer.close_all(force=True)
            except Exception:
                pass
        try:
            registry.close_all(force=True)
        except Exception:
            pass


@dataclass(slots=True)
class _RankOutcome:
    result: dict[str, Any] | None
    reason: ProbeReason
    error: str = ""


def _validate_rank_result(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "rank",
        "device",
        "identity",
        "observed_patterns",
        "observed_ordering",
        "ordering_epochs",
        "atomic_observed",
        "atomic_expected",
        "atomic_ok",
        "mappings_closed",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"rank result is missing {sorted(missing)}")
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported rank result schema {payload['schema_version']!r}")
    if not isinstance(payload["observed_patterns"], Mapping):
        raise ValueError("observed_patterns must be a mapping")
    if not isinstance(payload["observed_ordering"], Mapping):
        raise ValueError("observed_ordering must be a mapping")
    for key in ("rank", "device", "ordering_epochs", "atomic_observed", "atomic_expected"):
        if type(payload[key]) is not int:
            raise ValueError(f"rank result {key} must be an integer")
    if not isinstance(payload["atomic_ok"], bool):
        raise ValueError("atomic_ok must be a boolean")
    if not isinstance(payload["mappings_closed"], bool):
        raise ValueError("mappings_closed must be a boolean")
    PhysicalDeviceIdentity.from_dict(payload["identity"])


def _read_outcome(
    process: Any,
    result_path: Path,
    expected_identity: PhysicalDeviceIdentity,
    timed_out: bool,
) -> _RankOutcome:
    if timed_out or process.is_alive():
        return _RankOutcome(None, ProbeReason.TIMEOUT, "worker exceeded the probe deadline")
    if process.exitcode != 0:
        detail = f"worker exited with code {process.exitcode}"
        try:
            payload = _read_json(result_path)
        except (json.JSONDecodeError, OSError, ValueError):
            payload = {}
        if isinstance(payload.get("error"), str):
            detail = f"{detail}: {payload['error']}"
        return _RankOutcome(None, ProbeReason.CHILD_CRASH, detail)
    try:
        payload = _read_json(result_path)
    except FileNotFoundError:
        return _RankOutcome(None, ProbeReason.MALFORMED_OUTPUT, "worker produced no result file")
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return _RankOutcome(None, ProbeReason.MALFORMED_OUTPUT, f"unreadable result: {exc}")
    try:
        _validate_rank_result(payload)
    except (ValueError, TypeError, KeyError) as exc:
        return _RankOutcome(None, ProbeReason.MALFORMED_OUTPUT, f"invalid result: {exc}")
    try:
        child_identity = PhysicalDeviceIdentity.from_dict(payload["identity"])
    except (ValueError, TypeError):
        return _RankOutcome(None, ProbeReason.MALFORMED_OUTPUT, "invalid identity in result")
    if (
        expected_identity.uuid
        and child_identity.uuid
        and child_identity.uuid != expected_identity.uuid
    ):
        return _RankOutcome(
            None,
            ProbeReason.IDENTITY_MISMATCH,
            f"worker sees {child_identity.uuid}, parent expects {expected_identity.uuid}",
        )
    return _RankOutcome(dict(payload), ProbeReason.OK)


def _run_rank_job(
    devices: tuple[int, ...],
    identities: tuple[PhysicalDeviceIdentity, ...],
    *,
    need_ordering: bool,
    need_remote_atomic: bool,
    timeout_s: float,
    epochs: int,
    spin_budget: int,
) -> dict[int, _RankOutcome]:
    """Spawn one worker per rank and collect their results under a deadline."""
    if type(timeout_s) not in (int, float) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive number")
    if len(devices) != len(identities):
        raise ValueError("devices and identities must have the same length")
    workdir = Path(tempfile.mkdtemp(prefix="ayaka-p2p-probe-"))
    try:
        rendezvous = workdir / "rendezvous"
        results = workdir / "results"
        rendezvous.mkdir()
        results.mkdir()
        job = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "world_size": len(devices),
            "devices": [int(device) for device in devices],
            "identities": [identity.to_dict() for identity in identities],
            "need_ordering": bool(need_ordering),
            "need_remote_atomic": bool(need_remote_atomic),
            "epochs": int(epochs),
            "spin_budget": int(spin_budget),
            "atomic_times": DEFAULT_ATOMIC_TIMES,
            "timeout_s": float(timeout_s),
            "layout": _layout(len(devices), epochs),
        }
        _write_json(workdir / "job.json", job)

        context = mp.get_context("spawn")
        processes = [
            context.Process(
                target=_probe_rank_worker,
                args=(rank, str(workdir)),
                name=f"p2p-probe-rank{rank}",
            )
            for rank in range(len(devices))
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + float(timeout_s)
        try:
            for process in processes:
                process.join(max(0.0, deadline - time.monotonic()))
            alive_ranks = {rank for rank, process in enumerate(processes) if process.is_alive()}
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(2.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(2.0)
        return {
            rank: _read_outcome(
                process,
                results / f"rank{rank}.json",
                identities[rank],
                rank in alive_ranks,
            )
            for rank, process in enumerate(processes)
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pair assembly
# ---------------------------------------------------------------------------


def _build_cross_pair(
    source: int,
    destination: int,
    devices: tuple[int, ...],
    identities: tuple[PhysicalDeviceIdentity, ...],
    outcomes: Mapping[int, _RankOutcome],
    *,
    need_ordering: bool,
    need_remote_atomic: bool,
    epochs: int,
    slot_bytes: int,
) -> PairProbeReport:
    source_identity = identities[source]
    destination_identity = identities[destination]
    same_device = source_identity.same_physical_device(destination_identity) or (
        devices[source] == devices[destination]
    )
    source_outcome = outcomes.get(source, _RankOutcome(None, ProbeReason.CHILD_CRASH, ""))
    destination_outcome = outcomes.get(destination, _RankOutcome(None, ProbeReason.CHILD_CRASH, ""))
    peer_access: bool | None
    status = ProbeStatus.UNKNOWN
    reason = ProbeReason.P2P_UNVERIFIED
    error = ""
    ipc_verified = write_verified = read_verified = False
    ordering: bool | None = None
    atomic: bool | None = None

    if not source_identity.trustworthy or not destination_identity.trustworthy:
        reason = ProbeReason.ORDINAL_ONLY_IDENTITY
        error = "at least one rank exposed no UUID/PCI identity"
        peer_access = None
    elif not same_device and _live_peer_access(devices[source], devices[destination]) is False:
        status = ProbeStatus.UNSUPPORTED
        reason = ProbeReason.PEER_ACCESS_ABSENT
        peer_access = False
    elif source_outcome.result is None or destination_outcome.result is None:
        failed = source_outcome if source_outcome.result is None else destination_outcome
        reason = failed.reason
        error = failed.error
        peer_access = True
    else:
        peer_access = True
        source_result = source_outcome.result
        destination_result = destination_outcome.result
        ipc_verified = bool(source_result["mappings_closed"]) and bool(
            destination_result["mappings_closed"]
        )
        write_verified = bool(destination_result["observed_patterns"].get(str(source), False))
        read_verified = bool(source_result["observed_patterns"].get(str(destination), False))
        if need_ordering:
            ordering = bool(destination_result["observed_ordering"].get(str(source), False))
        if need_remote_atomic:
            atomic = bool(destination_result["atomic_ok"])
        primitive_ok = (not need_ordering or ordering is True) and (
            not need_remote_atomic or atomic is True
        )
        if ipc_verified and write_verified and read_verified and primitive_ok:
            status = ProbeStatus.VERIFIED
            reason = ProbeReason.OK
        else:
            reason = (
                ProbeReason.ATOMIC_UNSUPPORTED
                if need_remote_atomic and atomic is False
                else ProbeReason.P2P_UNVERIFIED
            )
            failures: list[str] = []
            if not ipc_verified:
                failures.append("IPC mappings were not closed cleanly")
            if not write_verified:
                failures.append(f"rank {source} write not observed by rank {destination}")
            if not read_verified:
                failures.append(f"rank {destination} write not observed by rank {source}")
            if need_ordering and ordering is not True:
                failures.append("release/acquire ordering did not verify")
            if need_remote_atomic and atomic is not True:
                failures.append("remote atomic RMW did not verify")
            error = "; ".join(failures) or "peer pair did not verify"
    return PairProbeReport(
        source_rank=source,
        destination_rank=destination,
        source_device=devices[source],
        destination_device=devices[destination],
        source_identity=source_identity,
        destination_identity=destination_identity,
        status=status,
        reason=reason,
        same_device=same_device,
        peer_access=peer_access,
        ipc_verified=ipc_verified,
        write_verified=write_verified,
        read_verified=read_verified,
        ordering=ordering,
        atomic=atomic,
        fabric=nvml_link_kind(source_identity.uuid, destination_identity.uuid),
        bytes_tested=slot_bytes if (write_verified or read_verified) else 0,
        epochs=epochs if ordering is not None else 0,
        error=error,
    )


def _probe_same_device(
    device: int,
    identity: PhysicalDeviceIdentity,
    *,
    timeout_s: float,
    epochs: int,
    spin_budget: int,
) -> PairProbeReport:
    """Two spawned processes on one physical GPU: same-device IPC evidence."""
    outcomes = _run_rank_job(
        (device, device),
        (identity, identity),
        need_ordering=False,
        need_remote_atomic=False,
        timeout_s=timeout_s,
        epochs=epochs,
        spin_budget=spin_budget,
    )
    exporter = outcomes.get(0, _RankOutcome(None, ProbeReason.CHILD_CRASH, ""))
    importer = outcomes.get(1, _RankOutcome(None, ProbeReason.CHILD_CRASH, ""))

    status = ProbeStatus.UNKNOWN
    reason = ProbeReason.P2P_UNVERIFIED
    error = ""
    ipc_verified = write_verified = read_verified = False
    if not identity.trustworthy:
        reason = ProbeReason.ORDINAL_ONLY_IDENTITY
        error = "same-device IPC cannot be attributed to a physical GPU without UUID/PCI"
    elif exporter.result is None or importer.result is None:
        failed = exporter if exporter.result is None else importer
        reason = failed.reason
        error = failed.error
    else:
        write_verified = bool(importer.result["observed_patterns"].get("0", False))
        read_verified = bool(exporter.result["observed_patterns"].get("1", False))
        ipc_verified = bool(exporter.result["mappings_closed"]) and bool(
            importer.result["mappings_closed"]
        )
        if ipc_verified and write_verified and read_verified:
            status = ProbeStatus.VERIFIED
            reason = ProbeReason.OK
        else:
            error = "same-device IPC read/write did not verify"
    return PairProbeReport(
        source_rank=0,
        destination_rank=1,
        source_device=device,
        destination_device=device,
        source_identity=identity,
        destination_identity=identity,
        status=status,
        reason=reason,
        same_device=True,
        peer_access=True,
        ipc_verified=ipc_verified,
        write_verified=write_verified,
        read_verified=read_verified,
        ordering=None,
        atomic=None,
        fabric="self",
        bytes_tested=DEFAULT_SLOT_BYTES if status is ProbeStatus.VERIFIED else 0,
        epochs=0,
        error=error,
    )


class DeviceProbeBackend:
    """Default backend: driver probe plus real spawned producer/consumer work."""

    def identities(self, devices: tuple[int, ...]) -> tuple[PhysicalDeviceIdentity, ...]:
        return tuple(physical_device_identity(device) for device in devices)

    def probe_pairs(
        self,
        devices: tuple[int, ...],
        identities: tuple[PhysicalDeviceIdentity, ...],
        *,
        need_ordering: bool,
        need_remote_atomic: bool,
        timeout_s: float,
        epochs: int,
        spin_budget: int,
    ) -> tuple[PairProbeReport, ...]:
        reports: list[PairProbeReport] = []
        world = len(devices)
        has_cross_work = world >= 2 and any(
            _live_peer_access(devices[source], devices[destination])
            for source in range(world)
            for destination in range(world)
            if source != destination
        )
        outcomes: dict[int, _RankOutcome] = {}
        if has_cross_work:
            outcomes = _run_rank_job(
                devices,
                identities,
                need_ordering=need_ordering,
                need_remote_atomic=need_remote_atomic,
                timeout_s=timeout_s,
                epochs=epochs,
                spin_budget=spin_budget,
            )
        for source in range(world):
            for destination in range(world):
                if source == destination:
                    continue
                if not has_cross_work:
                    reports.append(
                        PairProbeReport(
                            source_rank=source,
                            destination_rank=destination,
                            source_device=devices[source],
                            destination_device=devices[destination],
                            source_identity=identities[source],
                            destination_identity=identities[destination],
                            status=ProbeStatus.UNSUPPORTED,
                            reason=ProbeReason.PEER_ACCESS_ABSENT,
                            peer_access=False,
                            fabric=nvml_link_kind(
                                identities[source].uuid, identities[destination].uuid
                            ),
                        )
                    )
                    continue
                reports.append(
                    _build_cross_pair(
                        source,
                        destination,
                        devices,
                        identities,
                        outcomes,
                        need_ordering=need_ordering,
                        need_remote_atomic=need_remote_atomic,
                        epochs=epochs,
                        slot_bytes=DEFAULT_SLOT_BYTES,
                    )
                )
        for device, identity in zip(devices, identities, strict=True):
            reports.append(
                _probe_same_device(
                    device,
                    identity,
                    timeout_s=timeout_s,
                    epochs=epochs,
                    spin_budget=spin_budget,
                )
            )
        return tuple(reports)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _normalize_devices(devices: Sequence[int | str] | None) -> tuple[int, ...]:
    if devices is None:
        torch = _optional_torch()
        if torch is None or not torch.cuda.is_available():
            return ()
        return tuple(range(torch.cuda.device_count()))
    resolved: list[int] = []
    for entry in devices:
        if isinstance(entry, bool) or not isinstance(entry, (int, str)):
            raise TypeError("devices entries must be int ordinals or 'cuda:N' strings")
        if isinstance(entry, int):
            ordinal = entry
        else:
            text = entry.strip()
            if not text.startswith("cuda"):
                raise ValueError(f"device {entry!r} is not a CUDA device")
            _, _, index = text.partition(":")
            if not index:
                raise ValueError(f"device {entry!r} must name an ordinal, e.g. 'cuda:0'")
            ordinal = int(index)
        if ordinal < 0:
            raise ValueError("device ordinals must be non-negative")
        if ordinal not in resolved:
            resolved.append(ordinal)
    return tuple(resolved)


def _aggregate(
    pairs: tuple[PairProbeReport, ...],
    *,
    single_device: bool,
) -> tuple[ProbeStatus, ProbeReason]:
    if single_device:
        return ProbeStatus.UNSUPPORTED, ProbeReason.SINGLE_DEVICE
    cross = [entry for entry in pairs if entry.cross_device]
    if not cross:
        return ProbeStatus.UNSUPPORTED, ProbeReason.SINGLE_DEVICE
    statuses = {entry.status for entry in cross}
    if statuses == {ProbeStatus.VERIFIED}:
        return ProbeStatus.VERIFIED, ProbeReason.OK
    if ProbeStatus.UNKNOWN in statuses:
        for entry in cross:
            if entry.status is ProbeStatus.UNKNOWN:
                return ProbeStatus.UNKNOWN, entry.reason
    for entry in cross:
        if entry.status is ProbeStatus.UNSUPPORTED:
            return ProbeStatus.UNSUPPORTED, entry.reason
    return ProbeStatus.UNKNOWN, ProbeReason.P2P_UNVERIFIED


def _empty_report(
    environment: ProbeEnvironment,
    status: ProbeStatus,
    reason: ProbeReason,
    cache_reason: ProbeReason,
) -> VerifiedTopologyReport:
    return VerifiedTopologyReport(
        status=status,
        reason=reason,
        environment=environment,
        pairs=(),
        cache_reason=cache_reason,
        created_unix=time.time(),
    )


def probe_verified_topology(
    devices: Sequence[int | str] | None = None,
    *,
    need_ordering: bool = True,
    need_remote_atomic: bool = False,
    force: bool | None = None,
    use_cache: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    epochs: int = DEFAULT_EPOCHS,
    spin_budget: int = DEFAULT_SPIN_BUDGET,
    cache_path: str | Path | None = None,
    backend: ProbeBackend | None = None,
) -> VerifiedTopologyReport:
    """Probe (or load a verified cache of) the ordered-pair peer topology.

    Args:
        devices: local CUDA ordinals, e.g. ``(0, 1)`` or ``("cuda:0",)``;
            defaults to every device visible to this process.
        need_ordering: require the release/acquire signal primitive for a pair
            to count as ``verified`` (the DC0 protocol needs it).
        need_remote_atomic: additionally require the remote atomic RMW probe.
        force: ignore a valid cache and reprobe; defaults to
            ``AYAKA_P2P_PROBE_FORCE``.
        use_cache: read/write the cache at all.
        timeout_s: per-job wall-clock deadline for spawned workers.
        epochs: release/acquire epochs published per ordered pair.
        spin_budget: finite consumer spin budget; exhaustion is a failure.
        cache_path: explicit cache file, overriding the environment default.
        backend: injectable probe source; defaults to spawned real workers.

    Returns:
        A :class:`VerifiedTopologyReport`. Unsupported/failed probes are
        reported, not raised, so callers can follow the Torch fallback path.
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(spin_budget) is not int or spin_budget < 1:
        raise ValueError("spin_budget must be a positive integer")
    if type(timeout_s) not in (int, float) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive number")

    ordinals = _normalize_devices(devices)
    active_backend: ProbeBackend = backend if backend is not None else DeviceProbeBackend()
    identities = tuple(active_backend.identities(ordinals))
    environment = capture_environment(identities=identities)
    if force is None:
        force = os.environ.get(FORCE_ENV, "").strip().lower() in _TRUTHY

    if not sys.platform.startswith("linux"):
        return _empty_report(
            environment,
            ProbeStatus.UNSUPPORTED,
            ProbeReason.UNSUPPORTED_PLATFORM,
            ProbeReason.CACHE_DISABLED if not use_cache else ProbeReason.CACHE_MISS,
        )
    if not ordinals or not identities:
        return _empty_report(
            environment,
            ProbeStatus.UNSUPPORTED,
            ProbeReason.NO_CUDA_DEVICE,
            ProbeReason.CACHE_DISABLED if not use_cache else ProbeReason.CACHE_MISS,
        )

    cache_reason = ProbeReason.CACHE_MISS if use_cache else ProbeReason.CACHE_DISABLED
    if use_cache and not force:
        lookup = load_cached_report(environment, path=cache_path)
        if lookup.report is not None:
            return replace(lookup.report, cache_reason=ProbeReason.CACHE_HIT)
        cache_reason = lookup.reason
    elif force:
        cache_reason = ProbeReason.CACHE_FORCED

    started = time.monotonic()
    pairs = tuple(
        active_backend.probe_pairs(
            ordinals,
            identities,
            need_ordering=need_ordering,
            need_remote_atomic=need_remote_atomic,
            timeout_s=float(timeout_s),
            epochs=epochs,
            spin_budget=spin_budget,
        )
    )
    duration_s = time.monotonic() - started
    status, reason = _aggregate(pairs, single_device=len(ordinals) < 2)
    report = VerifiedTopologyReport(
        status=status,
        reason=reason,
        environment=environment,
        pairs=pairs,
        cache_reason=cache_reason,
        created_unix=time.time(),
        duration_s=duration_s,
    )
    if use_cache and status is ProbeStatus.VERIFIED:
        write_reason = write_cached_report(report, path=cache_path)
        if write_reason is ProbeReason.CACHE_UNWRITABLE:
            report = replace(report, cache_reason=ProbeReason.CACHE_UNWRITABLE)
    return report
