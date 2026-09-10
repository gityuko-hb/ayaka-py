from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ayaka.types import DeviceKind, DType


@dataclass(frozen=True, slots=True)
class DeviceRef:
    kind: DeviceKind = DeviceKind.CUDA
    index: int = 0
    uuid: str = ""

    def __str__(self) -> str:
        return f"{self.kind}:{self.index}"

    @property
    def is_cuda(self) -> bool:
        return self.kind is DeviceKind.CUDA

@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """What the planner is allowed to assume about a GPU.

    Every field is something a scheduling or kernel decision actually reads:

      * ``sm_major/minor`` gates dtypes — bf16 needs ≥ 8.0, fp8 needs ≥ 8.9,
        and on 8.6 (the 3050) a bf16 tensor-core GEMM runs at half the fp16 rate.
      * ``shared_mem_per_block`` is the FlashAttention tile ceiling: 48 KB
        default, 99 KB opt-in on 8.0/9.0, 100 KB on 8.6 — a tile that needs more
        simply cannot be launched.
      * ``l2_bytes`` decides whether a KV block resident set fits in L2 between
        layers, which is the difference between an HBM-bound and an L2-bound
        decode step.
      * ``registers_per_sm`` bounds occupancy; a kernel at 128 regs/thread caps
        at 512 threads/SM on 8.6.
    """

    name: str = ""
    sm_major: int = 0
    sm_minor: int = 0
    num_sms: int = 0
    hbm_bytes: int = 0
    l2_bytes: int = 0
    shared_mem_per_block: int = 0
    shared_mem_per_sm: int = 0
    registers_per_sm: int = 0
    max_threads_per_sm: int = 0
    warp_size: int = 32
    memory_bandwidth_bps: int = 0
    supports_nvlink: bool = False
    pci_bus_id: str = ""
    numa_node: int = -1

    @property
    def compute_capability(self) -> tuple[int, int]:
        return (self.sm_major, self.sm_minor)

    def supports(self, dtype: DType) -> bool:
        cc = self.compute_capability
        if dtype in (DType.FP8_E4M3, DType.FP8_E5M2):
            return cc >= (8, 9)
        if dtype is DType.BF16:
            return cc >= (8, 0)
        if dtype is DType.FP4_E2M1:
            return cc >= (10, 0)
        return True

class LinkKind(enum.StrEnum):
    """Physical hardware interconnect path between two execution or storage devices.

    Used by topology discovery engines and communication cost models to calculate
    transfer latencies, validate P2P DMA accessibility, and select optimal collective
    communication algorithms.
    """

    SELF = "self"
    """Intra-device memory access (local GPU-to-GPU memory transfer on the same device)."""

    NVLINK = "nvlink"
    """Direct point-to-point NVLink bridge
    between two adjacent GPUs without an intervening switch."""

    NVSWITCH = "nvswitch"
    """Multi-GPU switched NVLink fabric (e.g., DGX/HGX baseboards with full-bisection bandwidth)."""

    PCIE = "pcie"
    """Intra-node Host PCIe bus or dedicated PCIe packet switch within the same CPU NUMA socket."""

    QPI = "qpi"
    """Cross-socket host interconnect
    (Intel UPI/QPI, AMD Infinity Fabric) traversing NUMA domains."""

    NETWORK = "network"
    """Inter-node network interface (InfiniBand, RoCE v2, standard Ethernet via RDMA/TCP)."""

    @property
    def is_intra_node(self) -> bool:
        """Whether the communication path resides entirely within a single physical machine node."""
        return self is not LinkKind.NETWORK

    @property
    def is_nvlink_fabric(self) -> bool:
        """Whether the link utilizes high-speed NVLink hardware interconnects."""
        return self in (LinkKind.NVLINK, LinkKind.NVSWITCH)

    @property
    def is_numa_crossing(self) -> bool:
        """Whether the link traverses the host CPU cross-socket interconnect (NUMA penalty)."""
        return self is LinkKind.QPI

    @property
    def supports_direct_p2p_dma(self) -> bool:
        """Whether the link supports zero-copy peer-to-peer CUDA IPC without host staging."""
        return self in (LinkKind.SELF, LinkKind.NVLINK, LinkKind.NVSWITCH, LinkKind.PCIE)

class CommOpType(enum.StrEnum):
    """Reduction operators for distributed collective communication primitives.

    Specifies the mathematical reduction applied across ranks during
    ``AllReduce``, ``ReduceScatter``, and ``Reduce`` operations in distributed
    runtime fabrics (NCCL, RCCL, Gloo, or MPI).
    """

    SUM = "sum"
    """Element-wise summation. Standard for Tensor Parallel linear projections
    and MoE activation combination."""

    MAX = "max"
    """Element-wise maximum. Used for distributed logit normalization,
    stable softmax scaling across ranks, and distributed top-k sampling."""

    MIN = "min"
    """Element-wise minimum. Used for global bounding, latency synchronization,
    and distributed sequence length clipping."""

    AVG = "avg"
    """Element-wise arithmetic mean. Supported natively via hardware/NCCL average
    reduction without requiring post-reduction rank-count division."""

    PROD = "prod"
    """Element-wise product. Used for multi-dimensional shape reductions,
    cumulative probabilistic scaling, and geometric tensor operations."""

    @property
    def is_idempotent(self) -> bool:
        """Whether applying this operator multiple times to identical inputs preserves values
        (e.g., max(x, x) == x)."""
        return self in (CommOpType.MAX, CommOpType.MIN)

    @property
    def is_linear(self) -> bool:
        """Whether the reduction operator exhibits mathematical linearity over vectors."""
        return self in (CommOpType.SUM, CommOpType.AVG)

@dataclass(frozen=True, slots=True)
class DeviceGroup:
    """A set of devices addressed as one collective domain.

    ``name`` is what a :class:`~ayaka.plan.CommOp` refers to ("tp",
    "pp", "ep"), so the plan stays free of concrete communicator objects.
    """

    name: str
    devices: tuple[DeviceRef, ...]
    ranks: tuple[int, ...] = ()
    local_rank: int = 0

    def __post_init__(self) -> None:
        if self.ranks and len(self.ranks) != len(self.devices):
            raise ValueError(f"group {self.name}: ranks/devices length mismatch")
        if not 0 <= self.local_rank < max(len(self.devices), 1):
            raise ValueError(f"group {self.name}: local_rank {self.local_rank} out of range")

    @property
    def size(self) -> int:
        return len(self.devices)

    @property
    def is_trivial(self) -> bool:
        """Single-rank group — every collective is identity.  The P0 path."""
        return self.size <= 1

class AsyncHandle(Protocol):
    """Returned by non-blocking collectives.  Deliberately minimal: the executor
    only ever needs to know whether it can proceed, and to force a join."""

    def wait(self) -> None: ...

    def is_completed(self) -> bool: ...

@runtime_checkable
class CommunicationBackend(Protocol):
    """NCCL / gloo / RDMA / MPI are interchangeable behind this.

    Tensors are ``Any`` cannot name a torch type.  Implementations live in
    ``ayaka.distributed.communication``.

    ``async_op=True`` returns an :class:`AsyncHandle` and the caller is
    responsible for the join.  Every collective takes an explicit ``group`` —
    there is no ambient "current group", because an ambient group is exactly how
    TP and PP collectives get crossed under micro_batching.
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

    def all_gather(
        self, output: Any, tensor: Any, group: DeviceGroup, async_op: bool = False
    ) -> AsyncHandle | None: ...

    def reduce_scatter(
        self,
        output: Any,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None: ...

    def broadcast(
        self, tensor: Any, group: DeviceGroup, src_rank: int = 0, async_op: bool = False
    ) -> AsyncHandle | None: ...

    def all_to_all(
        self, output: Any, tensor: Any, group: DeviceGroup, async_op: bool = False
    ) -> AsyncHandle | None: ...

    def send(self, tensor: Any, group: DeviceGroup, dst_rank: int) -> AsyncHandle | None: ...

    def recv(self, tensor: Any, group: DeviceGroup, src_rank: int) -> AsyncHandle | None: ...

    def barrier(self, group: DeviceGroup) -> None: ...
