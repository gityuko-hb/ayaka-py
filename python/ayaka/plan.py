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