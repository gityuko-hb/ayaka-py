from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ayaka.types import (
    DType,
    KVLayoutKind,
    MemoryOwner,
    MemoryTier,
    StreamRole,
)

#! Batch plan
class TokenKind(enum.StrEnum):
    PREFILL = "prefill"  # first chunk of a new sequence
    EXTEND = "extend"  # continuation chunk (chunked prefill / prefix hit)
    DECODE = "decode"  # exactly one token per sequence
    SPEC_VERIFY = "spec_verify"  # k draft tokens verified in one pass
    
dataclass(frozen=True, slots=True)
class BatchEntry:
    """One sequence's slice of this step.

    ``token_offset`` is the position of the first scheduled token inside the
    *logical* sequence (prompt + generated), i.e. the RoPE base offset.  It is
    NOT an index into any buffer — buffer placement is ``slot_offset``.
    """
    
    request_id: str
    kind: TokenKind
    token_offset: int
    num_tokens: int
    num_cached_tokens: int  # prefix-cache hit length, already in KV
    slot_offset: int  # start index in the flattened token buffer
    seq_len_after: int  # sequence length once this step lands

    def __post_init__(self) -> None:
        if self.num_tokens < 1:
            raise ValueError(f"{self.request_id}: num_tokens must be >= 1")
        if self.kind is TokenKind.DECODE and self.num_tokens != 1:
            raise ValueError(f"{self.request_id}: decode entry must carry exactly 1 token")

@dataclass(frozen=True, slots=True)
class BatchPlan:
    """The shape of the step.  ``padded_num_tokens`` is what the kernels and the
    CUDA graph actually see; the gap is masked, never garbage-collected mid-step."""

    entries: tuple[BatchEntry, ...]
    num_tokens: int
    padded_num_tokens: int
    num_seqs: int
    max_seq_len: int
    is_pure_decode: bool  # gates the CUDA-graph path

    def __post_init__(self) -> None:
        if self.padded_num_tokens < self.num_tokens:
            raise ValueError("padded_num_tokens < num_tokens")
        if len(self.entries) != self.num_seqs:
            raise ValueError("num_seqs disagrees with len(entries)")
        
#! ComputePlan
@dataclass(frozen=True, slots=True)
class ComputePlan:
    """Numerics and layer span.  ``layer_range`` is half-open and is the PP
    stage's slice — a non-zero start means this rank is not the first stage and
    must receive hidden states before layer ``start``."""

    dtype: DType
    kv_dtype: DType
    layer_range: tuple[int, int]
    num_micro_batches: int = 1
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    
#! MemoryPlan
@dataclass(frozen=True, slots=True)
class WorkspaceRequest:
    """A scratch reservation for one step.

    The executor asks the workspace manager for exactly this
    and nothing else.  ``alignment`` of 256 is the floor for any buffer a
    tensor-core GEMM touches; 128 is enough for vectorized ``ld.global.v4``.
    """

    name: str
    nbytes: int
    owner: MemoryOwner
    tier: MemoryTier = MemoryTier.DEVICE
    alignment: int = 256

    def __post_init__(self) -> None:
        if self.alignment & (self.alignment - 1):
            raise ValueError(f"{self.name}: alignment {self.alignment} is not a power of two")


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Aggregate the workspace size and estimate the activation bytes required for the run."""
    workspaces: tuple[WorkspaceRequest, ...] = ()
    activation_bytes: int = 0
    peak_bytes_estimate: int = 0

    @property
    def workspace_bytes(self) -> int:
        return sum(w.nbytes for w in self.workspaces)
    
class MappingOpKind(enum.StrEnum):
    """Primitive memory operations on physical KV-cache block tables.

    Defines the discrete transactional operations supported by the virtual memory
    manager and block allocator across hierarchical :class:`MemoryTier` levels.
    """
    
    APPEND = "append"  # bind fresh blocks to the tail of a sequence
    """Bind fresh, uninitialized physical memory blocks to the sequence tail."""
    
    RELEASE = "release"  # drop refcount; eviction is a side effect of alloc
    """Decrement reference count on mapped blocks. Blocks with zero references
    become immediately reusable or eligible for cache eviction."""
    SHARE = "share"  # bump refcount on a prefix hit (full pages only)
    """Increment reference count on existing sealed prefix blocks for zero-copy
    multi-sequence KV cache reuse."""
    
    COPY = "copy"  # materialism a private tail out of a sealed block
    """Perform Copy-on-Write (CoW) to materialize a private mutable block from
    a shared read-only block."""
    
    PROMOTE = "promote"  # tier n → tier n-1
    """Migrate block data to a faster memory tier (e.g., Host RAM to GPU VRAM)."""
    
    DEMOTE = "demote"  # tier n → tier n+1
    """Evict or offload block data to a slower memory tier (e.g., GPU VRAM to Host RAM/Disk)."""
    
    @property
    def is_tier_migration(self) -> bool:
        """Whether this operation moves data across hierarchical storage tiers."""
        return self in (MappingOpKind.PROMOTE, MappingOpKind.DEMOTE)

    @property
    def modifies_refcount(self) -> bool:
        """Whether this operation directly modifies block reference counters."""
        return self in (MappingOpKind.SHARE, MappingOpKind.RELEASE, MappingOpKind.COPY)

    @property
    def allocates_physical_memory(self) -> bool:
        """Whether this operation requires allocating a new physical block."""
        return self in (MappingOpKind.APPEND, MappingOpKind.COPY, MappingOpKind.PROMOTE)
    
@dataclass(frozen=True, slots=True)
class MappingOp:
    kind: MappingOpKind
    request_id: str
    block_ids: tuple[int, ...]
    src_tier: MemoryTier = MemoryTier.DEVICE
    dst_tier: MemoryTier = MemoryTier.DEVICE
    stream: StreamRole = StreamRole.KV
    
@dataclass(frozen=True, slots=True)
class KVPlan:
    """Block tables plus the mapping deltas to apply at the batch boundary.

    ``block_tables`` is dense per sequence, row-padded to ``max_blocks_per_seq``
    so it copies H2D as one contiguous int32 rectangle — a ragged table would
    cost one memcpy per sequence and dominate the step at 128 seqs.

    ``slot_mapping`` is per scheduled token: the flat index of the KV slot the
    token's K/V is written to.  Length must equal ``BatchPlan.num_tokens``.
    """

    block_tables: tuple[tuple[int, ...], ...]
    slot_mapping: tuple[int, ...]
    mapping_ops: tuple[MappingOp, ...] = ()
    block_size: int = 16
    max_blocks_per_seq: int = 0
    layout: KVLayoutKind = KVLayoutKind.NHD

#! WeightResidencyPlan
@dataclass(frozen=True, slots=True)
class WeightResidency:
    """Where one weight shard must be before the step runs.  A non-HBM
    ``current_tier`` with HBM ``required_tier`` is a prefetch the executor must
    issue on the H2D stream and wait on before the owning layer."""

    weight_name: str
    required_tier: MemoryTier
    current_tier: MemoryTier
    nbytes: int
    stream: StreamRole = StreamRole.H2D

    @property
    def needs_transfer(self) -> bool:
        return self.required_tier != self.current_tier
    
@dataclass(frozen=True, slots=True)
class WeightResidencyPlan:
    """Per-step weight residency."""

    residency: tuple[WeightResidency, ...] = ()
    adapter_ids: tuple[str, ...] = ()

    @property
    def transfer_bytes(self) -> int:
        return sum(r.nbytes for r in self.residency if r.needs_transfer)
    
#! ParallelPlan
@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """TP/PP/EP are execution strategies, never model architecture.
    Model code reads *nothing* from here — only ``ayaka.distributed`` does."""
    
    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    sp_enabled: bool = False
    
    def __post_init__(self) -> None:
        for size, rank, name in (
            (self.tp_size, self.tp_rank, "tp"),
            (self.pp_size, self.pp_rank, "pp"),
            (self.dp_size, self.dp_rank, "dp"),
            (self.ep_size, self.ep_rank, "ep"),
            (self.cp_size, self.cp_rank, "cp"),
        ):
            if size < 1:
                raise ValueError(f"{name}_size must be >= 1")
            if not 0 <= rank < size:
                raise ValueError(f"{name}_rank {rank} out of range for size {size}")

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.dp_size * self.cp_size

    @property
    def is_single_process(self) -> bool:
        """When true every collective in ``ayaka.distributed`` degrades to identity"""
        return self.world_size == 1 and self.ep_size == 1
    
#! CommunicationPlan
class CommOpKind(enum.StrEnum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    ALL_TO_ALL = "all_to_all"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"
    

@dataclass(frozen=True, slots=True)
class CommOp:
    kind: CommOpKind
    group: str  # named process group: "tp" | "pp" | "ep" | ...
    nbytes: int
    dtype: DType
    stream: StreamRole = StreamRole.COMM
    peer_rank: int | None = None  # SEND/RECV only
    layer_index: int | None = None  # anchor for overlap scheduling
    
@dataclass(frozen=True, slots=True)
class CommunicationPlan:
    ops: tuple[CommOp, ...] = ()
    overlap_with_compute: bool = True

    @property
    def total_bytes(self) -> int:
        return sum(op.nbytes for op in self.ops)
    
#! KernelPlan
@dataclass(frozen=True, slots=True)
class KernelChoice:
    """Model layer names an *operation*; the plan names the
    *implementation*.  ``op`` is one of the ComputeBackend verbs."""

    op: str  # "gemm" | "attention" | "rmsnorm" | "rope" | "moe" | "sample"
    backend: str  # "torch" | "cuda" | "triton" | "flashinfer" | ...
    variant: str = "default"
    tile: tuple[int, ...] = ()  # backend-specific tile/stage hint


@dataclass(frozen=True, slots=True)
class KernelPlan:
    choices: tuple[KernelChoice, ...] = ()
    attention_backend: str = "torch"  # engine-global mode, never per-request

    def choice_for(self, op: str) -> KernelChoice | None:
        for c in self.choices:
            if c.op == op:
                return c
        return None

#! GraphPlan
class GraphMode(enum.StrEnum):
    EAGER = "eager"
    CAPTURE = "capture"
    REPLAY = "replay"
    
@dataclass(frozen=True, slots=True)
class GraphPlan:
    """CUDA-graph decision.  ``bucket`` is the padded batch size the graph was
    captured for; a replay with a different bucket is a correctness bug, not a
    slow path, because captured kernels bake in their launch dims."""

    mode: GraphMode = GraphMode.EAGER
    bucket: int = 0
    graph_key: str = ""

#! SamplingPlan
@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """The *shape* of this step's sampling, decided before the forward pass.

    What is here and what is deliberately not: this branch carries counts and
    flags, never producers.  A grammar matcher is a live object with mutable
    state; putting one in a frozen plan would make the plan unhashable, make it
    unshippable across a process boundary, and quietly give the executor a
    handle it could advance.  The producers stay in ``ayaka.sampling`` and are
    reached through the mask pipeline both sides already hold.

    ``all_greedy`` is computed on host staging in ``ayaka.sampling.metadata``,
    so it costs no sync.  It is what lets the executor take the argmax fast
    path — no softmax, no RNG, no second read of the logits — and what fixes
    the CUDA-graph bucket.

    ``num_mask_rows`` is not ``num_rows``: under speculative verification one
    sequence contributes ``propose_step + 1`` mask rows, so the bitmask arena
    is taller than the batch.  Conflating the two is how a spec-decode path
    ends up writing past the end of a mask buffer sized for decode.
    """

    num_rows: int = 0
    num_mask_rows: int = 0
    all_greedy: bool = True
    any_penalty: bool = False
    custom_ops: tuple[str, ...] = ()
    
    def __post_init__(self) -> None:
        if self.num_rows < 0 or self.num_mask_rows < 0:
            raise ValueError("sampling row counts must be non-negative")
        if self.num_mask_rows and self.num_mask_rows < self.num_rows:
            raise ValueError(
                f"num_mask_rows {self.num_mask_rows} < num_rows {self.num_rows}: "
                "every constrained row needs at least one mask row"
            )

    @property
    def any_mask(self) -> bool:
        return self.num_mask_rows > 0

    @property
    def graph_capturable(self) -> bool:
        """Tier-2 custom ops run arbitrary user code and cannot be captured.

        Stated as a property of the plan rather than discovered at capture
        time: TensorRT-LLM discovers it at capture time and the failure is a
        cryptic driver error inside a context manager.
        """
        return not self.custom_ops
    
#! ExecutionPlan
@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """The whole step, decided.  Immutable and self-contained: an executor given
    only this object and the model weights can run the step."""
    
    plan_id: str
    step_id: int
    batch: BatchPlan
    compute: ComputePlan
    memory: MemoryPlan
    kv: KVPlan
    weight_residency: WeightResidencyPlan = field(default_factory=WeightResidencyPlan)
    parallel: ParallelPlan = field(default_factory=ParallelPlan)
    communication: CommunicationPlan = field(default_factory=CommunicationPlan)
    kernel: KernelPlan = field(default_factory=KernelPlan)
    graph: GraphPlan = field(default_factory=GraphPlan)
    sampling: SamplingPlan = field(default_factory=SamplingPlan)

    # the plan is the unit of tracing.
    trace_ids: tuple[str, ...] = ()
    created_ns: int = 0

    def __post_init__(self) -> None:
        if len(self.kv.slot_mapping) != self.batch.num_tokens:
            raise ValueError(
                f"slot_mapping has {len(self.kv.slot_mapping)} entries, "
                f"batch has {self.batch.num_tokens} tokens"
            )
        if len(self.kv.block_tables) != self.batch.num_seqs:
            raise ValueError(
                f"block_tables has {len(self.kv.block_tables)} rows, "
                f"batch has {self.batch.num_seqs} sequences"
            )
        if self.graph.mode is GraphMode.REPLAY and not self.batch.is_pure_decode:
            raise ValueError("graph replay requested for a non pure-decode batch")
        if self.sampling.num_rows > self.batch.num_seqs:
            raise ValueError(
                f"sampling plans {self.sampling.num_rows} rows but the batch has "
                f"{self.batch.num_seqs} sequences"
            )
        if self.graph.mode is GraphMode.REPLAY and not self.sampling.graph_capturable:
            raise ValueError(
                f"graph replay requested with custom ops {self.sampling.custom_ops}; "
                "tier-2 ops are the slow path by design"
            )