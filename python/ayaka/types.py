from __future__ import annotations

import enum
from enum import IntEnum, StrEnum
from typing import Final


class DType(enum.Enum):
    """Internal data type enumeration for the execution runtime.

    Deliberately decoupled from ``torch.dtype``: the protocol layer must remain
    importable on environments without PyTorch or CUDA drivers installed.
    Conversion adapters are located in ``ayaka.backends.torch``.

    Attributes:
        label (str): Canonical lowercase string identifier (e.g., "fp16", "int4").
        bits (int): Bit width per element.
        sub_byte (bool): Flag indicating whether the type occupies less than 1 byte
            and requires packing.
    """

    # Floating Point Types
    FP64 = ("fp64", 64, False)
    FP32 = ("fp32", 32, False)
    FP16 = ("fp16", 16, False)
    BF16 = ("bf16", 16, False)
    FP8_E4M3 = ("fp8_e4m3", 8, False)
    FP8_E5M2 = ("fp8_e5m2", 8, False)
    FP4_E2M1 = ("fp4_e2m1", 4, True)

    # Integer & Quantized Types
    INT64 = ("int64", 64, False)
    INT32 = ("int32", 32, False)
    INT8 = ("int8", 8, False)
    UINT8 = ("uint8", 8, False)
    INT4 = ("int4", 4, True)

    # Boolean Type
    BOOL = ("bool", 8, False)

    def __init__(self, label: str, bits: int, sub_byte: bool) -> None:
        self.label: Final[str] = label
        self.bits: Final[int] = bits
        self.sub_byte: Final[bool] = sub_byte

    @classmethod
    def from_str(cls, name: str) -> DType:
        """Lookup a DType instance by its string identifier (case-insensitive).

        Args:
            name: The dtype name (e.g., "fp16", "FP16", "int4").

        Returns:
            The matching DType enum member.

        Raises:
            ValueError: If the identifier does not match any registered DType.
        """
        normalized = name.strip().lower()
        for member in cls:
            if member.label == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(
            f"Unknown dtype: '{name}'. Valid options: {[m.label for m in cls]}"
        )

    @property
    def itemsize(self) -> float:
        """Bytes per element.

        Note:
            Fractional for sub-byte types (e.g., 0.5 for 4-bit types). Callers
            requiring exact byte allocation sizes must use :meth:`nbytes`.
        """
        return self.bits / 8.0

    @property
    def is_floating_point(self) -> bool:
        """Whether this dtype represents a standard or micro-scaled floating-point type."""
        return self in (
            DType.FP32,
            DType.FP16,
            DType.BF16,
            DType.FP8_E4M3,
            DType.FP8_E5M2,
            DType.FP4_E2M1,
        )

    @property
    def is_integer(self) -> bool:
        """Whether this dtype represents a signed or unsigned integer type."""
        return self in (DType.INT64, DType.INT32, DType.INT8, DType.UINT8, DType.INT4)

    def nbytes(self, numel: int) -> int:
        """Calculate the exact byte storage required for a given number of elements.

        Sub-byte types are packed 2-per-byte along the contiguous dimension.
        An odd element count is rounded up to a whole byte boundary.

        Args:
            numel: Total number of logical elements in the tensor.

        Returns:
            Total bytes required for physical storage.

        Raises:
            ValueError: If ``numel`` is negative.
        """
        if numel < 0:
            raise ValueError(f"numel must be non-negative, got {numel}")
        total_bits = numel * self.bits
        return (total_bits + 7) // 8

    def __repr__(self) -> str:
        return f"DType.{self.name}"

class DeviceKind(enum.StrEnum):
    """Hardware target categories for storage and kernel execution.

    Attributes:
        CPU: Host RAM and CPU execution.
        CUDA: GPU accelerator memory (VRAM/HBM) and CUDA cores.
        DISK: Local persistent block storage (NVMe/SSD) for eviction and swap.
        REMOTE: Disaggregated network storage targets (RDMA/TCP).
    """

    CPU = "cpu"
    CUDA = "cuda"
    DISK = "disk"
    REMOTE = "remote"

class MemoryTier(enum.IntEnum):
    """Hardware memory hierarchy, ordered strictly from fastest to slowest.

    The hierarchy is split into two categories:
    1. **Non-allocatable Tiers (REGISTER, SHARED, L2):**
       Used by kernel cost models, latency estimation, and occupancy calculators.
    2. **Allocatable Residency Tiers (DEVICE -> DISK):**
       Tracked in ledger accounts for physical memory allocation and eviction.
    3. **Target Endpoint (REMOTE):**
       Names external offload targets outside the local memory ledger.

    Being an ``IntEnum``, ordering is load-bearing:
    - Faster tiers have smaller integer values (e.g., ``DEVICE < HOST_PINNED``).
    - Promotion corresponds to ``tier - 1``.
    - Demotion corresponds to ``tier + 1``.
    """

    REGISTER = 0
    SHARED = 1
    L2 = 2
    DEVICE = 3
    HOST_PINNED = 4
    HOST_PAGEABLE = 5
    DISK = 6
    REMOTE = 7

    @property
    def allocatable(self) -> bool:
        """Whether this tier is manageable by the local memory allocator."""
        return MemoryTier.DEVICE <= self <= MemoryTier.DISK

    @property
    def is_host(self) -> bool:
        """Whether this tier resides on host system memory."""
        return self in (MemoryTier.HOST_PINNED, MemoryTier.HOST_PAGEABLE)

    def promoted(self) -> MemoryTier:
        """Return the next faster residency tier, clamped at ``DEVICE``.

        Returns:
            The upgraded MemoryTier.

        Raises:
            ValueError: If called on a non-residency kernel tier (REGISTER, SHARED, L2).
        """
        if self < MemoryTier.DEVICE:
            raise ValueError(
                f"{self.name} is a kernel cost model tier, not a residency tier"
            )
        return MemoryTier(max(int(MemoryTier.DEVICE), int(self) - 1))

    def demoted(self) -> MemoryTier:
        """Return the next slower residency tier, clamped at ``REMOTE``.

        Returns:
            The downgraded MemoryTier.

        Raises:
            ValueError: If called on a non-residency kernel tier (REGISTER, SHARED, L2).
        """
        if self < MemoryTier.DEVICE:
            raise ValueError(
                f"{self.name} is a kernel cost model tier, not a residency tier"
            )
        return MemoryTier(min(int(MemoryTier.REMOTE), int(self) + 1))

class MemoryOwner(enum.StrEnum):
    """Explicit ownership tag for memory allocation and ledger accounting.

    Enforces deterministic memory attribution: every allocated buffer on any
    managed :class:`MemoryTier` must declare exactly one owner. The memory
    allocator rejects untagged allocations.

    Attributes:
        WEIGHT: Static model parameters and persistent quantized weight tensors.
        ACTIVATION: Intermediate activations generated during forward passes.
        KV: Paged Key-Value cache blocks managed by the sequence memory manager.
        COMM: Communication staging and collective buffers (NCCL / P2P / AllReduce).
        COMPILE: Persistent artifacts from graph captures, CUDA Graphs, and JIT compilation.
        WORKSPACE: Ephemeral scratchpad buffers reserved by math libraries (cuBLAS, cuDNN).
    """

    WEIGHT = "weight"
    ACTIVATION = "activation"
    KV = "kv"
    COMM = "comm"
    COMPILE = "compile"
    WORKSPACE = "workspace"

    @property
    def is_dynamic(self) -> bool:
        """Whether this allocation varies dynamically based on concurrent request load.

        Used by the scheduler's admission controller to compute real-time headroom.
        """
        return self in (MemoryOwner.ACTIVATION, MemoryOwner.KV)

    @property
    def is_static(self) -> bool:
        """Whether this allocation represents static baseline engine overhead."""
        return not self.is_dynamic

class Layout(enum.StrEnum):
    """Physical memory layout hints for memory planning and kernel dispatching.

    Layout serves as a performance hint and does not alter the mathematical
    semantics of tensor data.

    Attributes:
        ROW_MAJOR: Standard C-contiguous row-major layout.
        COL_MAJOR: Fortran-contiguous column-major layout.
        PACKED_SUB_BYTE: Packed layout for sub-byte quantized types.
        BLOCKED: Block/Tiled layout optimized for Tensor Cores or block-quantization.
    """

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    PACKED_SUB_BYTE = "packed_sub_byte"
    BLOCKED = "blocked"


class KVLayoutKind(enum.StrEnum):
    """Physical layout arrangement of a single Key-Value cache block.

    This is an engine-global configuration. Modifying this layout at runtime
    invalidates all active block tables.

    Attributes:
        NHD: Shape [block_size, num_kv_heads, head_dim] (FlashInfer default).
        HND: Shape [num_kv_heads, block_size, head_dim] (vLLM / FlashAttention default).
        NLD: Shape [(page, slot, latent_dim)] MLA latent and RoPE planes
    """

    NHD = "nhd"
    HND = "hnd"
    NLD = "NLD"


class StreamRole(enum.StrEnum):
    """CUDA stream roles for explicit concurrency and pipeline overlapping.

    The runtime maintains one dedicated stream per role per device to eliminate
    ad-hoc stream creation and reduce synchronization overhead.

    Attributes:
        COMPUTE: Main stream for forward passes and attention kernels.
        H2D: Asynchronous Host-to-Device memory copy stream.
        D2H: Asynchronous Device-to-Host memory copy stream.
        P2P: Peer-to-Peer direct GPU-to-GPU transfer stream (NVLink/PCIe).
        COMM: Inter-node collective communications stream (NCCL / AllReduce).
        KV: Dedicated stream for KV-cache block transfers and page management.
    """

    COMPUTE = "compute"
    H2D = "h2d"
    D2H = "d2h"
    P2P = "p2p"
    COMM = "comm"
    KV = "kv"

class AttentionType(StrEnum):
    """One value per KV-pool family.

    A group declares one; a backend declares the set it serves
    (``BackendInfo.supported_types``). Pool layout follows the type, so this is also what
    ``KVCacheGeometry`` keys on -- one taxonomy, not two.
    """

    FULL = "full"  # uniform causal MHA/GQA over a paged K/V pool
    SWA = "swa"  # sliding-window (optionally with sinks) over a paged K/V pool
    MLA = "mla"  # latent-KV MLA: one latent row per token, absorbed kv_b
    LINEAR = "linear"  # GDN / Mamba recurrent-state layers

    @property
    def backend_driven(self) -> bool:
        """Whether this group constrains attention-backend selection.

        LINEAR layers reach their kernels through the linear-state runtime, so a model
        mixing GDN with GQA must not have its GQA backend narrowed by the GDN group.
        """
        return self is not AttentionType.LINEAR

    @property
    def is_paged_kv(self) -> bool:
        return self in (AttentionType.FULL, AttentionType.SWA, AttentionType.MLA)

class ForwardMode(IntEnum):
    """What a forward is doing, as data rather than as ``max(query_len) == 1``.

    A bool is not enough, concretely: a prompt that hits the prefix cache completely arrives
    as a ONE-TOKEN prefill. It looks exactly like decode by query length, but the scheduler
    has not staged decode-only addressing for it and it must not take the replay path.
    """

    IDLE = 0  # padding-only batch (graph warmup, rank sync)
    PREFILL = 1  # fresh sequences, no cached prefix
    EXTEND = 2  # partial prefix hit, or chunked prefill continuation
    DECODE = 3  # one token per request
    TARGET_VERIFY = 4  # uniform k-token query per request, tree or linear mask
    DRAFT_EXTEND = 5  # draft model catching up after an accepted block

    @property
    def is_decode_like(self) -> bool:
        """Uniform query length per request -> eligible for a decode-shaped kernel."""
        return self in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY)

    @property
    def is_prefill_like(self) -> bool:
        return self in (ForwardMode.PREFILL, ForwardMode.EXTEND, ForwardMode.DRAFT_EXTEND)

    @property
    def has_cached_prefix(self) -> bool:
        """Whether ``computed_lens`` can be non-zero (i.e. the K/V loop must page)."""
        return self is not ForwardMode.PREFILL

class AttentionCudaGraphSupport(IntEnum):
    """How much of a batch a backend can serve from inside a captured graph.

    The graph runner reads this to decide replay eligibility; it does NOT probe the backend
    or catch exceptions. Ordered, so ``support >= required`` is a valid check.
    """

    NEVER = 0  # e.g. a backend whose plan() allocates per step
    PURE_DECODE = 1  # DECODE only (one query per request)
    UNIFORM_QUERY = 2  # DECODE + TARGET_VERIFY (same query len for every request)
    ALWAYS = 3  # any mode, including ragged prefill

class KVCacheDtype(StrEnum):
    """Storage dtype of the KV pool, independent of the model's compute dtype.

    Deliberately NOT gated on compute capability. FP8 *storage* is bytes and works anywhere
    the dtype exists; only FP8 *arithmetic* needs sm_89 (``torch_utils.supports_fp8`` says
    exactly this). An FP8 KV cache on Ampere is legal and slower, not illegal -- the kernel
    dequantize on load. Refusing it on arch would deny the memory saving to precisely the
    cards that need it most.

    Scales are per-layer scalars living on the pool and read as DEVICE tensors, so a
    re-calibrated scale takes effect without recapturing the graph.
    """

    AUTO = "auto"  # same as the model dtype; no scale, no dequant
    FP8_E4M3 = "fp8_e4m3"  # 1-4-3, no inf; the KV default (more mantissa)
    FP8_E5M2 = "fp8_e5m2"  # 1-5-2, wider range, coarser

    @property
    def is_quantized(self) -> bool:
        return self is not KVCacheDtype.AUTO

    @property
    def torch_dtype_name(self) -> str | None:
        """Canonical name for ``torch_utils.torch_dtype``; None for AUTO."""
        return {
            KVCacheDtype.FP8_E4M3: "float8_e4m3fn",
            KVCacheDtype.FP8_E5M2: "float8_e5m2",
        }.get(self)

    @property
    def element_bytes(self) -> int | None:
        """Bytes per element, or None for AUTO (the model dtype decides).

        This is the number ``KVBudgetSpec.bytes_per_token_per_rank`` multiplies, which is
        why the quantization choice must reach capacity planning and not just the kernel.
        """
        return 1 if self.is_quantized else None

class MaskKind(StrEnum):
    CAUSAL = "causal"
    FULL = "full"  # bidirectional (encoder / embedding / reranker models)
    SLIDING = "sliding"  # causal within a window
