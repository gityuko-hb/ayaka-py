from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

from ayaka.types import DType, Layout, MemoryOwner, MemoryTier


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Pure description of a buffer."""

    shape: tuple[int, ...]
    dtype: DType
    layout: Layout = Layout.ROW_MAJOR
    tier: MemoryTier = MemoryTier.DEVICE
    owner: MemoryOwner = MemoryOwner.ACTIVATION
    alignment: int = 256
    name: str = ""

    def __post_init__(self) -> None:
        if any(d < 0 for d in self.shape):
            raise ValueError(f"{self.name}: negative dim in {self.shape}")
        if self.dtype.sub_byte and self.layout is not Layout.PACKED_SUB_BYTE:
            raise ValueError(f"{self.name}: {self.dtype.label} must use Layout.PACKED_SUB_BYTE")

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.dtype.nbytes(self.numel)

    @property
    def padded_nbytes(self) -> int:
        a = self.alignment
        return (self.nbytes + a - 1) // a * a

    def contiguous_strides(self) -> tuple[int, ...]:
        """Element strides for the declared layout.  Sub-byte dtypes are packed
        along the last dim, so the last stride is still 1 *in elements* — a
        caller converting to bytes must go through ``dtype.nbytes``."""
        if self.layout is Layout.COL_MAJOR:
            strides: list[int] = []
            acc = 1
            for d in self.shape:
                strides.append(acc)
                acc *= d
            return tuple(strides)
        strides = [1] * len(self.shape)
        acc = 1
        for i in range(len(self.shape) - 1, -1, -1):
            strides[i] = acc
            acc *= self.shape[i]
        return tuple(strides)

    def with_shape(self, shape: tuple[int, ...]) -> TensorSpec:
        return TensorSpec(
            shape=shape,
            dtype=self.dtype,
            layout=self.layout,
            tier=self.tier,
            owner=self.owner,
            alignment=self.alignment,
            name=self.name,
        )


@dataclass(frozen=True, slots=True)
class TensorHandle:
    """A live buffer.

    ``data`` is deliberately ``Any``: the protocol layer must not import torch,
    so the torch tensor (or the raw ``cudaMalloc`` pointer, or the DLPack
    capsule) travels opaquely.  Only ``ayaka.backends.*`` may unwrap it.

    ``device_ptr`` is populated whenever the buffer is device-resident, so
    kernels launched through ``csrc`` never need to touch ``data`` at all.
    """

    spec: TensorSpec
    data: Any = None
    device_ptr: int = 0
    device_index: int = -1
    offset: int = 0  # byte offset into the owning MemoryRegion

    @property
    def is_device(self) -> bool:
        return self.spec.tier <= MemoryTier.DEVICE and self.device_index >= 0

    def is_aligned(self) -> bool:
        return self.device_ptr % self.spec.alignment == 0


class ShardKind(enum.StrEnum):
    """How one logical weight is split across ranks.

    COLUMN → split output dim; the result needs an all-gather (or is consumed by
             a ROW shard, which is the fused case).
    ROW    → split input dim; the result needs an all-reduce.
    EXPERT → split the expert dim (EP); no collective, an all-to-all instead.
    """

    REPLICATED = "replicated"
    COLUMN = "column"
    ROW = "row"
    EXPERT = "expert"
    HEAD = "head"  # query heads: split along heads, keeps head_dim intact
    # Distinct from HEAD so KV-ness is *declared* by the family table
    # and never inferred.  The old shape heuristic — "dim 0 equals
    # num_kv_heads x head_dim" — has no answer under MLA, where the cached
    # tensor is a single latent and `kv_a_proj_with_mqa` must be replicated on
    # every rank; it silently sharded it instead.
    KV_HEAD = "kv_head"  # K/V heads: partitioned, or replicated when tp > kv_heads


@dataclass(frozen=True, slots=True)
class ShardSpec:
    kind: ShardKind = ShardKind.REPLICATED
    dim: int = -1
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid shard rank {self.rank}/{self.world_size}")
        if self.kind is not ShardKind.REPLICATED and self.dim < 0:
            raise ValueError(f"{self.kind} shard needs an explicit dim")

    def shard_shape(self, full: tuple[int, ...]) -> tuple[int, ...]:
        if self.kind is ShardKind.REPLICATED or self.world_size == 1:
            return full
        if full[self.dim] % self.world_size:
            raise ValueError(
                f"dim {self.dim}={full[self.dim]} not divisible by world_size {self.world_size}"
            )
        out = list(full)
        out[self.dim] //= self.world_size
        return tuple(out)

    def slice_bounds(self, full: tuple[int, ...]) -> tuple[int, int]:
        """[start, end) along ``dim`` for this rank — what the loader reads out
        of the safetensors mmap without materializing the full tensor."""
        if self.kind is ShardKind.REPLICATED or self.world_size == 1:
            return 0, full[self.dim] if self.dim >= 0 else 0
        n = full[self.dim] // self.world_size
        return self.rank * n, (self.rank + 1) * n


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """Quantization of one weight.  ``group_size=-1`` is per-channel,
    ``0`` per-tensor.  ``scale_dtype`` is separate because an fp8 weight with
    fp32 scales and an fp8 weight with bf16 scales are different layouts."""

    method: str = "none"  # "none" | "awq" | "gptq" | "fp8" | "int8" | "nvfp4"
    weight_dtype: DType = DType.BF16
    scale_dtype: DType = DType.FP32
    group_size: int = 0
    symmetric: bool = True
    has_zero_point: bool = False

    @property
    def is_quantized(self) -> bool:
        return self.method != "none"


@dataclass(frozen=True, slots=True)
class WeightSource:
    """Where the bytes come from.  ``byte_offset``/``nbytes`` let the loader
    ``pread`` straight into a pinned staging buffer with no intermediate copy."""

    uri: str  # file path or object-store key
    tensor_key: str  # key inside the checkpoint
    byte_offset: int = 0
    nbytes: int = 0
    checksum: str = ""


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """One weight, fully described before a single byte is read.

    The load pipeline (Rule 4) is: discovery → plan → I/O → transform → shard →
    quantize → place → verify.  Every stage reads this object; only the last
    produces a :class:`TensorHandle`.
    """

    name: str
    full_shape: tuple[int, ...]
    spec: TensorSpec
    shard: ShardSpec = field(default_factory=ShardSpec)
    quant: QuantSpec = field(default_factory=QuantSpec)
    source: WeightSource | None = None
    layer_index: int | None = None
    transpose: bool = False

    # ``tied_alias`` names the weight this one *is*, not the one it copies.  A
    # tied LM head shares storage with the embedding: it has no checkpoint entry
    # and must not be charged to the ledger twice, which is why "tied" has to be
    # expressible in the expected schema rather than patched up after binding.
    tied_alias: str = ""

    # A weight the architecture permits but does not require — Qwen2's qkv bias
    # is present, its o_proj bias is not.  Without this flag, "absent" and
    # "missing" are the same observation and the loader cannot tell a correct
    # checkpoint from a truncated one.
    optional: bool = False

    def __post_init__(self) -> None:
        expected = self.shard.shard_shape(self.full_shape)
        if self.spec.shape != expected:
            raise ValueError(
                f"{self.name}: spec.shape {self.spec.shape} != sharded shape {expected}"
            )
        if self.tied_alias:
            if self.tied_alias == self.name:
                raise ValueError(f"{self.name}: weight cannot be tied to itself")
            if self.source is not None:
                raise ValueError(
                    f"{self.name}: tied to {self.tied_alias!r} but also declares a "
                    "checkpoint source — a tied weight has no bytes of its own"
                )

    @property
    def nbytes(self) -> int:
        return self.spec.nbytes

    @property
    def is_tied(self) -> bool:
        return bool(self.tied_alias)
