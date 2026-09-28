"""Capability report composed from the existing device owners.

This module owns no hardware probe. Every fact here is read from an authority
that already existed: :class:`~ayaka.utils.torch_utils.DeviceProfile` for the
device itself, :func:`~ayaka.device_comm.topology.physical_device_identity` for
cross-process identity, ``CC_LIMITS``/``_L2_BYTES`` for per-architecture
constants, :func:`~ayaka.device.platform.arch_reachability` for what this build
can actually launch, and the already-bound attention backend instances for
kernel-level declarations. A sixth, independent detection path is exactly the
duplication this package exists to prevent.

Support is reported in three tiers that never imply one another
(``AYAKA_EXECUTION_MASTER_PLAN.md`` 10.1): the package runs, the model runs
eager, and the requested graph features run. A device can pass the first and
fail the second; the report says which and why.

Importing this module never initializes a CUDA context. Facts that would need
one are resolved through the already-cached owners, so a CPU-only import path
and an import during graph capture both stay safe.
"""

from __future__ import annotations

import json
import platform as host_platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ayaka.device.platform import (
    ArchReachability,
    arch_reachability,
    current_platform,
    framework_architectures,
    framework_ptx,
)
from ayaka.device_comm.topology import PhysicalDeviceIdentity, physical_device_identity
from ayaka.types import DType
from ayaka.utils.import_utils import CapabilityError, has_module
from ayaka.utils.validation import require_int, require_text

#: The declared NVIDIA support matrix, shipped with the package.
_MATRIX_FILE = Path(__file__).resolve().parent / "nvidia_support_matrix.json"

_SUPPORT_MATRIX: tuple[SupportRow, ...] | None = None

__all__ = [
    "BackendFacts",
    "CapabilityReason",
    "DeviceIdentity",
    "ExecutionCapabilityReport",
    "GraphState",
    "PackageFacts",
    "RuntimeFacts",
    "SupportRow",
    "SupportState",
    "SupportTier",
    "SupportVerdict",
    "arch_reachability",
    "compute_dtype",
    "resolve_capabilities",
    "support_matrix",
]


def compute_dtype(torch_dtype: object) -> DType:
    """Map a framework dtype onto the runtime's own :class:`DType`.

    The capability report reasons about dtypes, not about ``torch.dtype``. An
    unmapped value is a real answer too: it is a dtype the runtime has no
    capability rule for, so it falls back to BF16 rather than being assumed
    supported.
    """
    for candidate in DType:
        if candidate.torch_dtype is not None and candidate.torch_dtype is torch_dtype:
            return candidate
    return DType.BF16


class CapabilityReason(StrEnum):
    """Why a tier is not satisfied.

    The set is closed on purpose: these become metric labels, so an open-ended
    string reason would grow cardinality without bound. Add a member when a new
    refusal is real, never a formatted message.
    """

    NO_CUDA_DEVICE = "no_cuda_device"
    ARCH_NOT_BUILT = "arch_not_built"
    ARCH_PTX_ONLY = "arch_ptx_only"
    FRAMEWORK_TARGETS_UNKNOWN = "framework_targets_unknown"
    DEVICE_IDENTITY_UNKNOWN = "device_identity_unknown"
    PARTITION_CAPACITY_UNKNOWN = "partition_capacity_unknown"
    PACKAGE_MISSING = "package_missing"
    PACKAGE_VERSION_UNSUPPORTED = "package_version_unsupported"
    DTYPE_NOT_NATIVE = "dtype_not_native"
    DTYPE_EMULATED = "dtype_emulated"
    ATTENTION_BACKEND_UNSUPPORTED = "attention_backend_unsupported"
    ATTENTION_GRAPH_UNSAFE = "attention_graph_unsafe"
    ATTENTION_REQUIRES_CUDA = "attention_requires_cuda"
    ATTENTION_HOST_LENS = "attention_host_lens"
    ATTENTION_RAGGED_UNSUPPORTED = "attention_ragged_unsupported"
    KV_LAYOUT_UNSUPPORTED = "kv_layout_unsupported"
    MODEL_SHAPE_EXCEEDS_PARTITION = "model_shape_exceeds_partition"
    GRAPH_NOT_REQUESTED = "graph_not_requested"
    GRAPH_MULTI_GROUP = "graph_multi_group"
    GRAPH_LM_HEAD_GATHER = "graph_lm_head_gather"
    DISTRIBUTED_UNSUPPORTED = "distributed_unsupported"


class SupportTier(StrEnum):
    """The three questions a support report answers, weakest first.

    Ordered, but deliberately not usable as a numeric threshold: a device may
    satisfy ``MODEL_EAGER`` while failing ``REQUESTED_FEATURES``, and the caller
    must read the verdict it actually needs.
    """

    PACKAGE_RUNTIME = "package_runtime"
    MODEL_EAGER = "model_eager"
    REQUESTED_FEATURES = "requested_features"


@dataclass(frozen=True, slots=True)
class SupportVerdict:
    """One tier's outcome.

    ``blocking`` is what withheld support and ``advisory`` is what an operator
    should still see: a JIT-only device, an emulated dtype, an unidentified GPU
    all work while costing something. Keeping them apart is what lets the report
    say "supported, but expect a first-launch JIT" instead of either hiding the
    cost or refusing outright.
    """

    supported: bool
    blocking: tuple[CapabilityReason, ...] = ()
    advisory: tuple[CapabilityReason, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        blocking = tuple(dict.fromkeys(self.blocking))
        advisory = tuple(dict.fromkeys(self.advisory))
        overlap = set(blocking) & set(advisory)
        if overlap:
            raise ValueError(f"reasons cannot be both blocking and advisory: {sorted(overlap)}")
        if self.supported != (not blocking):
            raise ValueError("supported must be true exactly when no blocking reason is present")
        object.__setattr__(self, "blocking", blocking)
        object.__setattr__(self, "advisory", advisory)
        object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def reasons(self) -> tuple[CapabilityReason, ...]:
        """Every reason, blocking first, so metrics can label both alike."""
        return self.blocking + self.advisory


def _verdict(
    blocking: Sequence[CapabilityReason] = (),
    advisory: Sequence[CapabilityReason] = (),
    evidence: Mapping[str, object] | None = None,
) -> SupportVerdict:
    """Build a verdict from blocking and advisory reasons."""
    return SupportVerdict(
        supported=not tuple(blocking),
        blocking=tuple(blocking),
        advisory=tuple(advisory),
        evidence=evidence or {},
    )


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """Who the device is, and how much of that can be proven.

    ``uuid``/``pci_bus_id`` are the cross-process comparable fields; an empty
    ``identity_key`` means nothing can be proven about which physical GPU this
    is, and the report says so rather than guessing.
    """

    index: int
    name: str
    compute_capability: tuple[int, int] | None
    sm: int
    #: Partition capacity in bytes as CUDA presents it. On a MIG instance this is
    #: the instance, never the parent card: CUDA enumerates a compute instance as
    #: its own device, so this is already partition-correct.
    capacity_bytes: int
    multiprocessor_count: int
    uuid: str = ""
    pci_bus_id: str = ""
    reachability: ArchReachability = ArchReachability.UNREACHABLE
    #: Set when a driver-level snapshot disagreed with what torch reported.
    cross_check: str = ""

    def __post_init__(self) -> None:
        require_int(self.index, "device index", minimum=0)
        require_int(self.sm, "device sm", minimum=20)
        require_int(self.capacity_bytes, "device capacity_bytes", minimum=0)

    @property
    def identity_key(self) -> str:
        """Cross-process key, or an empty string when nothing is provable."""
        return self.uuid or self.pci_bus_id

    @property
    def trustworthy(self) -> bool:
        """Whether this identity can name a physical GPU across processes."""
        return bool(self.identity_key)

    @property
    def is_cuda(self) -> bool:
        return self.compute_capability is not None

    def describe(self) -> str:
        if not self.is_cuda:
            return "cpu"
        return f"cuda:{self.index} {self.name} sm_{self.sm} {self.reachability.value}"

    def supports_dtype(self, dtype: DType) -> bool:
        """Whether ``dtype`` has native acceleration on this compute capability.

        The thresholds are the named floors already owned by
        :mod:`ayaka.utils.torch_utils`; the compute capability is read from this
        record rather than re-queried by ordinal, so a report stays consistent
        with the device it describes even when a caller injects one.
        """
        from ayaka.utils.torch_utils import SM_ADA, SM_AMPERE, SM_HOPPER

        if self.compute_capability is None:
            return False
        cc = self.compute_capability
        if dtype in (DType.FP8_E4M3, DType.FP8_E5M2, DType.FP4_E2M1):
            # FP8 needs Ada for arithmetic; FP4 needs Blackwell. Storage of these
            # dtypes works wherever the type exists -- that gap is the KV cache
            # validator's concern, not an execution-support question.
            floor = SM_HOPPER if dtype is DType.FP4_E2M1 else SM_ADA
            return cc >= floor
        if dtype is DType.BF16:
            return cc >= SM_AMPERE
        return True


@dataclass(frozen=True, slots=True)
class RuntimeFacts:
    """Software stack, and the build targets it was compiled for."""

    torch_version: str
    cuda_runtime_version: str | None
    driver_version: str
    os_name: str
    os_release: str
    machine: str
    python_version: str
    cuda_initialized: bool
    #: Compute capabilities the installed framework ships cubins for, and the
    #: PTX it embeds. Read from the framework, never from a project file.
    framework_architectures: tuple[int, ...]
    framework_ptx_architecture: int | None


@dataclass(frozen=True, slots=True)
class PackageFacts:
    """Optional dependency availability, resolved by import spec only.

    Presence is not version compatibility and neither is correctness; a
    quant or attention lane stays unsupported until its own evidence exists.
    """

    available: Mapping[str, bool]

    def has(self, name: str) -> bool:
        return bool(self.available.get(name, False))

    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, present in self.available.items() if not present))


@dataclass(frozen=True, slots=True)
class BackendFacts:
    """Graph-safety read from the attention instances the runner already bound.

    Every field mirrors a declaration the backend or its metadata builder
    already carries. Nothing here probes a kernel and nothing here selects a
    backend; the runner made that choice before this report exists.
    """

    backend_names: tuple[str, ...]
    group_count: int
    #: Lowest declared support across bound groups, as its integer rank.
    cudagraph_support: int
    cudagraph_support_name: str
    reads_host_lens: bool
    supports_ragged_mixed: bool
    requires_cuda: bool
    kv_layouts: tuple[str, ...]
    kv_dtypes: tuple[str, ...]
    head_dims_qk: tuple[int, ...]
    head_dims_vo: tuple[int, ...]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionCapabilityReport:
    """Immutable bring-up result: who the device is and what it can run."""

    device: DeviceIdentity
    runtime: RuntimeFacts
    packages: PackageFacts
    backends: BackendFacts
    tiers: Mapping[SupportTier, SupportVerdict]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiers", dict(self.tiers))

    def verdict(self, tier: SupportTier) -> SupportVerdict:
        """The verdict for ``tier``; every tier is always present."""
        return self.tiers[tier]

    @property
    def graph_supported(self) -> bool:
        """Whether the requested graph features are supported, as read."""
        return self.tiers[SupportTier.REQUESTED_FEATURES].supported

    @property
    def reasons(self) -> tuple[CapabilityReason, ...]:
        """Every reason across all tiers, deduplicated, tier order preserved."""
        seen: dict[CapabilityReason, None] = {}
        for tier in SupportTier:
            for reason in self.tiers[tier].reasons:
                seen.setdefault(reason, None)
        return tuple(seen)


def _resolve_identity(index: int) -> PhysicalDeviceIdentity | None:
    """Read cross-process identity without ever making it mandatory.

    A device that reports no UUID and no PCI address is still usable for eager
    execution in one process. It is simply not identifiable across processes,
    and the report records that instead of refusing to run.
    """
    try:
        return physical_device_identity(index)
    except (CapabilityError, ValueError):
        return None


def _resolve_device(index: int, hardware: object | None) -> DeviceIdentity:
    """Compose the device record from torch, the arch authority and the identity.

    ``hardware`` is an optional driver-level cross-check. It never becomes a
    third source of truth: it can only add a disagreement note, never replace a
    value torch measured, and never contributes memory capacity. NVML reports
    the parent card, so trusting it for capacity would size a MIG partition from
    memory it was not given.
    """
    from ayaka.utils.torch_utils import cuda_available, device_count, device_profile

    if not cuda_available() or index >= device_count():
        return DeviceIdentity(
            index=index,
            name="cpu",
            compute_capability=None,
            sm=20,
            capacity_bytes=0,
            multiprocessor_count=0,
        )

    profile = device_profile(index)
    sm = profile.compute_capability
    if sm is None:
        # A visible CUDA device whose capability cannot be read is not a device
        # we can reason about; treat it as CPU rather than inventing a CC.
        return DeviceIdentity(
            index=index,
            name=profile.name,
            compute_capability=None,
            sm=20,
            capacity_bytes=0,
            multiprocessor_count=profile.multiprocessor_count,
        )
    sm_int = sm[0] * 10 + sm[1]
    try:
        reach = arch_reachability(sm_int)
    except CapabilityError:
        # Targets unresolvable is reported through RuntimeFacts and the
        # PACKAGE_RUNTIME verdict, not by pretending the device is unreachable.
        reach = ArchReachability.UNREACHABLE

    identity = _resolve_identity(index)
    note = _cross_check_note(hardware, sm, identity)
    return DeviceIdentity(
        index=index,
        name=profile.name,
        compute_capability=sm,
        sm=sm_int,
        capacity_bytes=profile.total_memory_bytes,
        multiprocessor_count=profile.multiprocessor_count,
        uuid=identity.uuid if identity else "",
        pci_bus_id=identity.pci_bus_id if identity else "",
        reachability=reach,
        cross_check=note,
    )


def _cross_check_note(
    hardware: object | None, sm: tuple[int, int], identity: PhysicalDeviceIdentity | None
) -> str:
    """Compare a driver snapshot against torch; report only disagreement."""
    if hardware is None:
        return ""
    capabilities = getattr(hardware, "capabilities", ())
    if not capabilities:
        return ""
    notes: list[str] = []
    driver = capabilities[0]
    driver_cc = (getattr(driver, "sm_major", None), getattr(driver, "sm_minor", None))
    if driver_cc != (None, None) and driver_cc != sm:
        notes.append(f"driver compute capability {driver_cc[0]}.{driver_cc[1]} != torch {sm}")
    devices = getattr(hardware, "devices", ())
    if devices and identity is not None:
        driver_uuid = getattr(devices[0], "uuid", "")
        if driver_uuid and identity.uuid and driver_uuid != identity.uuid:
            notes.append("driver UUID disagrees with torch device UUID")
    return "; ".join(notes)


def _resolve_runtime() -> RuntimeFacts:
    from ayaka.utils.torch_utils import torch_version

    cuda_runtime: str | None = None
    try:
        from ayaka.utils.torch_utils import torch

        version = getattr(torch.version, "cuda", None)
        cuda_runtime = str(version) if version else None
    except CapabilityError:
        cuda_runtime = None

    # Both accessors absorb a framework that cannot be asked and return an empty
    # set and a ``None`` respectively, which makes every device read as
    # UNREACHABLE. That is the safe direction to fail in, so no guard is needed
    # here and none of this can raise.
    return RuntimeFacts(
        torch_version=".".join(str(part) for part in torch_version()),
        cuda_runtime_version=cuda_runtime,
        driver_version=_driver_version(),
        os_name=host_platform.system(),
        os_release=host_platform.release(),
        machine=host_platform.machine(),
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        cuda_initialized=current_platform().is_cuda and _cuda_initialized(),
        framework_architectures=tuple(framework_architectures()),
        framework_ptx_architecture=framework_ptx(),
    )


def _cuda_initialized() -> bool:
    try:
        from ayaka.utils.torch_utils import torch

        return bool(torch.cuda.is_initialized())
    except (CapabilityError, AttributeError):
        return False


def _driver_version() -> str:
    """Driver version via NVML when present; empty is a valid answer.

    The driver is a partition-invariant fact, so NVML is the right source for
    it. Missing NVML is not a refusal, and inventing a version is not an option.
    """
    from ayaka.utils.nvml_utils import get_driver_version, nvml_session

    try:
        with nvml_session() as pynvml:
            if pynvml is None:
                return ""
            return get_driver_version(pynvml)
    except Exception:  # NVML is advisory here; never fail bring-up over it
        return ""


def _resolve_packages() -> PackageFacts:
    from ayaka.utils.torch_utils import has_dtype

    return PackageFacts(
        available={
            "torch": has_module("torch"),
            "triton": has_module("triton"),
            "flash_attn": has_module("flash_attn"),
            "flashinfer": has_module("flashinfer"),
            "pynvml": has_module("pynvml"),
            "cuda": has_dtype("float8_e4m3fn"),
        }
    )


def _resolve_backends(
    backends: Mapping[str, object] | None,
    builders: Mapping[str, object] | None = None,
) -> BackendFacts:
    """Read graph-safety declarations off the already-bound attention instances.

    A missing backend mapping means the report was requested before attention
    binding. That is not a failure of the device, so the graph tier stays
    undecided rather than guessing.

    The metadata builders are a separate mapping from the backends: the runner
    owns ``backends`` for kernel dispatch and ``builders`` for planning, and a
    backend does not carry its own builder. Graph safety is declared by the
    builder, so both are needed.
    """
    if not backends:
        return BackendFacts(
            backend_names=(),
            group_count=0,
            cudagraph_support=0,
            cudagraph_support_name="UNBOUND",
            reads_host_lens=False,
            supports_ragged_mixed=False,
            requires_cuda=False,
            kv_layouts=(),
            kv_dtypes=(),
            head_dims_qk=(),
            head_dims_vo=(),
            detail="no attention backend bound",
        )

    from ayaka.types import AttentionCudaGraphSupport

    names = tuple(sorted(backends))
    supports: list[int] = []
    host_lens: list[bool] = []
    ragged: list[bool] = []
    requires_cuda: list[bool] = []
    layouts: list[str] = []
    dtypes: list[str] = []
    qk: set[int] = set()
    vo: set[int] = set()

    for name, backend in backends.items():
        device = getattr(backend, "device", None)
        requires_cuda.append(getattr(device, "type", "cpu") == "cuda")
        ragged.append(bool(getattr(backend, "supports_ragged_mixed", False)))
        spec = getattr(getattr(backend, "spec", None), "head_dim_qk", None)
        if isinstance(spec, int):
            qk.add(spec)
        vo_dim = getattr(getattr(backend, "spec", None), "head_dim_vo", None)
        if isinstance(vo_dim, int):
            vo.add(vo_dim)
        cache = getattr(backend, "kv_cache", None)
        dtype = getattr(cache, "dtype", None)
        if dtype is not None:
            dtypes.append(str(dtype))
        layout = getattr(cache, "layout", None)
        if layout is not None:
            layouts.append(str(layout))

        builder = (builders or {}).get(name)
        if builder is not None:
            support = getattr(builder, "cudagraph_support", None)
            if support is not None:
                supports.append(int(support))
            host_lens.append(bool(getattr(builder, "reads_host_lens", False)))

    rank = min(supports) if supports else int(AttentionCudaGraphSupport.NEVER)
    try:
        support_name = AttentionCudaGraphSupport(rank).name
    except ValueError:
        support_name = f"UNKNOWN({rank})"
    return BackendFacts(
        backend_names=names,
        group_count=len(backends),
        cudagraph_support=rank,
        cudagraph_support_name=support_name if supports else "UNDECLARED",
        reads_host_lens=any(host_lens) if host_lens else False,
        supports_ragged_mixed=bool(ragged) and all(ragged),
        requires_cuda=any(requires_cuda),
        kv_layouts=tuple(sorted(set(layouts))),
        kv_dtypes=tuple(sorted(set(dtypes))),
        head_dims_qk=tuple(sorted(qk)),
        head_dims_vo=tuple(sorted(vo)),
        detail="" if supports else "no metadata builder bound for graph safety",
    )


def _package_tier(device: DeviceIdentity, runtime: RuntimeFacts) -> SupportVerdict:
    """Can this interpreter and build talk to the device at all?"""
    evidence: dict[str, object] = {
        "torch": runtime.torch_version,
        "cuda_runtime": runtime.cuda_runtime_version,
        "driver": runtime.driver_version,
        "framework_architectures": runtime.framework_architectures,
        "framework_ptx_architecture": runtime.framework_ptx_architecture,
        "device": device.describe(),
    }
    if not device.is_cuda:
        return _verdict([CapabilityReason.NO_CUDA_DEVICE], evidence=evidence)
    if not runtime.framework_architectures:
        # The framework reports no cubin targets at all: a CPU-only torch
        # build, or a build whose target list cannot be read. Every device
        # then reads UNREACHABLE, which is the safe direction to fail.
        return _verdict([CapabilityReason.FRAMEWORK_TARGETS_UNKNOWN], evidence=evidence)
    blocking: list[CapabilityReason] = []
    advisory: list[CapabilityReason] = []
    if device.reachability is ArchReachability.UNREACHABLE:
        blocking.append(CapabilityReason.ARCH_NOT_BUILT)
    elif device.reachability is ArchReachability.PTX_JIT_ONLY:
        # Reachable, but a first-launch JIT cost exists and is not a claim of
        # qualification. Advisory, so the tier still passes.
        advisory.append(CapabilityReason.ARCH_PTX_ONLY)
    if not device.trustworthy:
        advisory.append(CapabilityReason.DEVICE_IDENTITY_UNKNOWN)
    if device.capacity_bytes <= 0:
        # Budget arithmetic is refused later, where it is actually needed. Not
        # knowing the capacity does not stop a device from running eager.
        advisory.append(CapabilityReason.PARTITION_CAPACITY_UNKNOWN)
    return _verdict(blocking, advisory, evidence)


def _eager_tier(
    device: DeviceIdentity,
    runtime: RuntimeFacts,
    packages: PackageFacts,
    backends: BackendFacts,
    *,
    dtype: DType,
) -> SupportVerdict:
    """Can the model run eagerly, without any capture involved?"""
    evidence: dict[str, object] = {
        "dtype": dtype.label,
        "attention_backends": backends.backend_names,
        "kv_layouts": backends.kv_layouts,
        "kv_dtypes": backends.kv_dtypes,
        "triton": packages.has("triton"),
    }
    blocking: list[CapabilityReason] = []
    advisory: list[CapabilityReason] = []

    if not device.is_cuda:
        # A CPU reference lane is a legitimate eager target; the graph tier is
        # what fails, not eager execution.
        return _verdict(advisory=[], evidence=evidence)

    blocking: list[CapabilityReason] = []
    advisory: list[CapabilityReason] = []
    if device.reachability is ArchReachability.UNREACHABLE:
        blocking.append(CapabilityReason.ARCH_NOT_BUILT)

    # Native versus emulated is a fact about the build, not a guess: a dtype
    # whose tensor type merely exists is not an accelerated one. Both answers
    # come from the record and the package facts, never a fresh device probe.
    if dtype in (DType.FP8_E4M3, DType.FP8_E5M2, DType.FP4_E2M1):
        if not device.supports_dtype(dtype):
            if packages.has("triton"):
                advisory.append(CapabilityReason.DTYPE_EMULATED)
            else:
                blocking.append(CapabilityReason.DTYPE_NOT_NATIVE)
    elif not device.supports_dtype(dtype):
        blocking.append(CapabilityReason.DTYPE_NOT_NATIVE)

    if backends.requires_cuda and not device.is_cuda:
        blocking.append(CapabilityReason.ATTENTION_REQUIRES_CUDA)
    if not backends.backend_names:
        evidence["attention"] = backends.detail
    return _verdict(blocking, advisory, evidence)


def _feature_tier(
    device: DeviceIdentity,
    backends: BackendFacts,
    *,
    graph_requested: bool,
    graph_backend: str,
    single_group_required: bool = True,
    lm_head_gathers: bool = False,
) -> SupportVerdict:
    """Can the requested capture features run on this device and binding?"""
    from ayaka.types import AttentionCudaGraphSupport

    evidence: dict[str, object] = {
        "graph_backend": graph_backend,
        "graph_requested": graph_requested,
        "attention_support": backends.cudagraph_support_name,
        "groups": backends.group_count,
        "reads_host_lens": backends.reads_host_lens,
    }
    if not graph_requested:
        # Nothing was asked for, so nothing failed. Recording this as a reason
        # would put "the operator declined" in the same bucket as "this device
        # cannot", which is the distinction the three tiers exist to preserve.
        return SupportVerdict(supported=True, evidence=evidence)

    blocking: list[CapabilityReason] = []
    advisory: list[CapabilityReason] = []
    if not device.is_cuda:
        blocking.append(CapabilityReason.NO_CUDA_DEVICE)
    if device.reachability is ArchReachability.UNREACHABLE:
        blocking.append(CapabilityReason.ARCH_NOT_BUILT)
    elif device.reachability is ArchReachability.PTX_JIT_ONLY:
        advisory.append(CapabilityReason.ARCH_PTX_ONLY)
    if backends.cudagraph_support < AttentionCudaGraphSupport.PURE_DECODE:
        blocking.append(CapabilityReason.ATTENTION_GRAPH_UNSAFE)
    if backends.reads_host_lens:
        # Advisory, not blocking. It says ``build`` reads host sequence lengths,
        # which constrains a future pinned-staging overlap; it does not say the
        # backend is uncapturable, because ``build_for_replay`` is copy-only by
        # contract and takes every bound from device tensors. Conflating the two
        # would refuse a graph path that demonstrably works.
        advisory.append(CapabilityReason.ATTENTION_HOST_LENS)
    if single_group_required and backends.group_count > 1:
        blocking.append(CapabilityReason.GRAPH_MULTI_GROUP)
    if lm_head_gathers:
        blocking.append(CapabilityReason.GRAPH_LM_HEAD_GATHER)
    if not backends.backend_names:
        blocking.append(CapabilityReason.ATTENTION_BACKEND_UNSUPPORTED)
    return _verdict(blocking, advisory, evidence)


def resolve_capabilities(
    *,
    device_index: int = 0,
    device: DeviceIdentity | None = None,
    dtype: DType = DType.BF16,
    backends: Mapping[str, object] | None = None,
    builders: Mapping[str, object] | None = None,
    graph_requested: bool = False,
    graph_backend: str = "full",
    lm_head_gathers: bool = False,
    single_group_required: bool = True,
    hardware: object | None = None,
) -> ExecutionCapabilityReport:
    """Resolve the bring-up capability report without probing new hardware.

    Args:
        device_index: Local CUDA ordinal this worker owns.
        device: A pre-resolved device record to report on instead of detecting
            one. Bring-up passes the record it already built; tests pass a
            synthetic one to exercise the full matrix without hardware. It is
            the same code path either way, which is what keeps a fake-matrix
            result meaningful.
        dtype: Compute dtype the model will run in; drives the native/emulated
            distinction.
        backends: The attention instances the runner already bound, keyed by
            group name. Their declarations are read, never re-selected.
        builders: The metadata builders the runner already bound, keyed by
            the same group names. Graph safety is declared here, because a
            backend does not carry its own builder.
        graph_requested: Whether the operator asked for capture. A false value
            makes the feature tier pass with the request recorded as evidence,
            rather than a hardware failure, which keeps the two separable.
        graph_backend: Requested graph backend, recorded as evidence.
        lm_head_gathers: Whether the LM head requires a vocabulary gather.
        single_group_required: Whether the requested backend qualifies only with
            one attention group.
        hardware: Optional driver-level snapshot used purely as a cross-check.
            It can add a disagreement note and nothing else; in particular it
            never supplies memory capacity.

    Returns:
        An immutable :class:`ExecutionCapabilityReport`. It records what the
        device can do; it never mutates configuration and never chooses a
        backend.
    """
    require_int(device_index, "device_index", minimum=0)
    resolved = device if device is not None else _resolve_device(device_index, hardware)
    runtime = _resolve_runtime()
    packages = _resolve_packages()
    backend_facts = _resolve_backends(backends, builders)
    return ExecutionCapabilityReport(
        device=resolved,
        runtime=runtime,
        packages=packages,
        backends=backend_facts,
        tiers={
            SupportTier.PACKAGE_RUNTIME: _package_tier(resolved, runtime),
            SupportTier.MODEL_EAGER: _eager_tier(
                resolved, runtime, packages, backend_facts, dtype=dtype
            ),
            SupportTier.REQUESTED_FEATURES: _feature_tier(
                resolved,
                backend_facts,
                graph_requested=graph_requested,
                graph_backend=graph_backend,
                single_group_required=single_group_required,
                lm_head_gathers=lm_head_gathers,
            ),
        },
    )


class SupportState(StrEnum):
    """Product support for a hardware row.

    Independent of work-package state and of a test result. A row can be
    ``SUPPORTED`` for eager while a graph gate for it is still ``NOT_RUN``; and a
    work package can be ``ACCEPTED`` while every row it touched is
    ``UNSUPPORTED``. Collapsing these axes is how a release ends up claiming
    coverage it never measured.
    """

    SUPPORTED = "SUPPORTED"
    REFERENCE = "REFERENCE"
    EXPERIMENTAL = "EXPERIMENTAL"
    PARTIAL = "PARTIAL"
    NOT_WIRED = "NOT_WIRED"
    UNSUPPORTED = "UNSUPPORTED"


class GraphState(StrEnum):
    """Whether decode-graph capture is qualified on a hardware row.

    This is a support claim, not a description, so it obeys the same evidence
    rule as :class:`SupportState`. Keeping it in the matrix at all is only
    defensible because of that: a graph claim nobody checks is how a release
    ends up advertising a feature that was never run.
    """

    QUALIFIED = "qualified"
    PLANNED = "planned"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SupportRow:
    """One declared hardware row and the evidence behind its state."""

    id: str
    group: str
    support: SupportState
    test_result: str
    compute_capabilities: tuple[int, ...]
    evidence: str | None
    graph: GraphState = GraphState.PLANNED
    reason: str = ""
    remedy: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        require_text(self.id, "support row id")
        require_text(self.group, f"{self.id} group")
        if self.support is SupportState.SUPPORTED:
            # A SUPPORTED row is a claim about hardware that was run. Without
            # evidence it is a guess wearing a passing grade, which is exactly
            # what a fake or unrun row would otherwise become.
            if not self.evidence:
                raise ValueError(
                    f"support row {self.id!r} is SUPPORTED without evidence; "
                    "a row with no hardware run must not be SUPPORTED"
                )
            if self.test_result != "PASS":
                raise ValueError(
                    f"support row {self.id!r} is SUPPORTED with test_result "
                    f"{self.test_result!r}; only PASS qualifies"
                )
        if self.support is SupportState.UNSUPPORTED and not self.reason:
            raise ValueError(f"support row {self.id!r} is UNSUPPORTED without a reason")
        # The graph claim is a support claim, so it is held to the same rule and
        # kept consistent with the row it belongs to.
        if self.graph is GraphState.QUALIFIED and self.support is not SupportState.SUPPORTED:
            raise ValueError(
                f"support row {self.id!r} claims qualified graph capture while support is "
                f"{self.support.value}; a graph claim is a support claim"
            )
        if self.graph is GraphState.UNAVAILABLE and self.support is not SupportState.UNSUPPORTED:
            raise ValueError(
                f"support row {self.id!r} marks graph capture unavailable while support is "
                f"{self.support.value}; 'unavailable' is for rows nothing can run on"
            )


def support_matrix() -> tuple[SupportRow, ...]:
    """Load the declared NVIDIA support rows.

    Data, not code, so the matrix can be reviewed and diffed without a code
    change. Loading validates the invariants above, which is the mechanism that
    stops an unrun row from being published as supported.
    """
    global _SUPPORT_MATRIX
    if _SUPPORT_MATRIX is not None:
        return _SUPPORT_MATRIX
    try:
        data = json.loads(_MATRIX_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CapabilityError(
            "support_matrix",
            detail=f"support matrix not found at {_MATRIX_FILE}",
            remedy="restore the packaged nvidia_support_matrix.json",
        ) from exc
    rows = tuple(
        SupportRow(
            id=entry["id"],
            group=entry["group"],
            support=SupportState(entry["support"]),
            test_result=entry["test_result"],
            compute_capabilities=tuple(entry.get("compute_capabilities", ())),
            evidence=entry.get("evidence"),
            graph=GraphState(entry.get("graph", "planned")),
            reason=entry.get("reason", ""),
            remedy=entry.get("remedy", ""),
            notes=entry.get("notes", ""),
        )
        for entry in data["rows"]
    )
    _SUPPORT_MATRIX = rows
    return rows
