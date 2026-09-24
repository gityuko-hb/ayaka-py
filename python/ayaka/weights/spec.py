from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass, field
from typing import Any

from ayaka.types import DType, Layout, MemoryOwner, MemoryTier
from ayaka.utils.math_utils import align_up


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
        return align_up(self.nbytes, self.alignment)

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


class FP8Recipe(enum.StrEnum):
    """Supported FP8 inference recipe identifiers.

    These names describe contracts, not proof that a checkpoint adapter or an
    execution backend is implemented for the recipe.
    """

    W8A8_STATIC_PER_TENSOR = "w8a8_static_per_tensor"
    W8A8_DYNAMIC_PER_TENSOR = "w8a8_dynamic_per_tensor"
    W8A8_ROWWISE = "w8a8_rowwise"
    W8A8_BLOCK128 = "w8a8_block128"
    MXFP8 = "mxfp8"
    W8A16 = "w8a16"
    KV_CACHE = "kv_cache"


class FP8ValueFormat(enum.StrEnum):
    """FP8 value encoding; E5M2 is reserved for KV cache recipes."""

    E4M3 = "e4m3"
    E5M2 = "e5m2"


class FP8ActivationScheme(enum.StrEnum):
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"
    CALIBRATED = "calibrated"


class FP8ScaleFormat(enum.StrEnum):
    FP32 = "float32"
    UE8M0 = "ue8m0"


class FP8ScaleDirection(enum.StrEnum):
    """Direction of checkpoint scale values before loader normalization."""

    DEQUANT = "dequant"
    INVERSE = "inverse"


class FP8ZeroPolicy(enum.StrEnum):
    """How zero-valued groups and supplied checkpoint scales are handled."""

    AMAX_FLOOR = "amax_floor"
    CHECKPOINT_SCALE = "checkpoint_scale"


@dataclass(frozen=True, slots=True)
class FP8RecipeSpec:
    """Immutable, explicit description of one FP8 inference recipe.

    ``activation_block`` and ``weight_block`` are logical ``(rows, K)`` block
    sizes. ``None`` means per-tensor/per-row activation scaling or
    per-output-channel weight scaling, as defined by ``recipe``. Layout fields
    are checkpoint metadata identifiers and may describe both A and W scales;
    adapters must validate them against actual tensor shapes before planning or
    allocating layer storage.
    """

    recipe: FP8Recipe
    value_format: FP8ValueFormat
    activation_scheme: FP8ActivationScheme
    activation_block: tuple[int, int] | None
    weight_block: tuple[int, int] | None
    scale_format: FP8ScaleFormat
    scale_direction: FP8ScaleDirection
    logical_scale_layout: str
    physical_scale_layout: str
    zero_policy: FP8ZeroPolicy
    amax_floor: float | None = None
    calibration_source: str | None = None
    excluded_layers: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("recipe", FP8Recipe),
            ("value_format", FP8ValueFormat),
            ("activation_scheme", FP8ActivationScheme),
            ("scale_format", FP8ScaleFormat),
            ("scale_direction", FP8ScaleDirection),
            ("zero_policy", FP8ZeroPolicy),
        ):
            try:
                object.__setattr__(self, name, enum_type(getattr(self, name)))
            except ValueError as error:
                allowed = ", ".join(member.value for member in enum_type)
                raise ValueError(f"{name} must be one of: {allowed}") from error

        for name in ("activation_block", "weight_block"):
            block = getattr(self, name)
            if block is None:
                continue
            if not isinstance(block, tuple) or len(block) != 2:
                raise TypeError(f"{name} must be a (rows, K) tuple or None")
            if any(isinstance(size, bool) or not isinstance(size, int) for size in block):
                raise TypeError(f"{name} entries must be integers")
            if any(size < 1 for size in block):
                raise ValueError(f"{name} entries must be positive")

        for name in ("logical_scale_layout", "physical_scale_layout"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty metadata identifier")
            object.__setattr__(self, name, value.strip())

        if self.zero_policy is FP8ZeroPolicy.AMAX_FLOOR:
            if isinstance(self.amax_floor, bool) or not isinstance(
                self.amax_floor, (int, float)
            ):
                raise TypeError("amax_floor must be a finite positive number for dynamic recipes")
            try:
                floor = float(self.amax_floor)
            except OverflowError as error:
                raise ValueError("amax_floor must be finite and positive") from error
            if not math.isfinite(floor) or floor <= 0.0:
                raise ValueError("amax_floor must be finite and positive")
            object.__setattr__(self, "amax_floor", floor)
        elif self.amax_floor is not None:
            raise ValueError("amax_floor is only valid when zero_policy='amax_floor'")

        if self.calibration_source is not None:
            if not isinstance(self.calibration_source, str) or not self.calibration_source.strip():
                raise ValueError("calibration_source must be None or a non-empty string")
            object.__setattr__(self, "calibration_source", self.calibration_source.strip())
        if not isinstance(self.excluded_layers, tuple):
            raise TypeError("excluded_layers must be a tuple of layer names")
        if any(not isinstance(layer, str) or not layer.strip() for layer in self.excluded_layers):
            raise ValueError("excluded_layers entries must be non-empty strings")
        object.__setattr__(
            self, "excluded_layers", tuple(layer.strip() for layer in self.excluded_layers)
        )
        if len(set(self.excluded_layers)) != len(self.excluded_layers):
            raise ValueError("excluded_layers must not contain duplicates")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an integer")
        if self.version < 1:
            raise ValueError("version must be >= 1")

        uses_checkpoint_calibration = self.activation_scheme in (
            FP8ActivationScheme.STATIC,
            FP8ActivationScheme.CALIBRATED,
        ) or self.recipe is FP8Recipe.W8A16
        if uses_checkpoint_calibration and self.calibration_source is None:
            raise ValueError("static, calibrated, and W8A16 recipes require calibration_source")

        self._validate_recipe_contract()

    def _validate_recipe_contract(self) -> None:
        recipe = self.recipe
        if recipe is FP8Recipe.KV_CACHE:
            if self.activation_scheme is not FP8ActivationScheme.CALIBRATED:
                raise ValueError("KV cache FP8 requires activation_scheme='calibrated'")
            if self.activation_block is not None or self.weight_block is not None:
                raise ValueError("KV cache FP8 uses per-layer/per-plane scales, not GEMM blocks")
            if self.scale_format is not FP8ScaleFormat.FP32:
                raise ValueError("KV cache FP8 scales must use float32")
            if self.zero_policy is not FP8ZeroPolicy.CHECKPOINT_SCALE:
                raise ValueError("KV cache FP8 requires a calibrated checkpoint scale policy")
            return

        if self.value_format is not FP8ValueFormat.E4M3:
            raise ValueError("E5M2 is supported only by the KV cache recipe")

        expected: dict[
            FP8Recipe,
            tuple[
                FP8ActivationScheme,
                tuple[int, int] | None,
                tuple[int, int] | None,
                FP8ScaleFormat,
                FP8ZeroPolicy,
            ],
        ] = {
            FP8Recipe.W8A8_STATIC_PER_TENSOR: (
                FP8ActivationScheme.STATIC, None, None, FP8ScaleFormat.FP32,
                FP8ZeroPolicy.CHECKPOINT_SCALE,
            ),
            FP8Recipe.W8A8_DYNAMIC_PER_TENSOR: (
                FP8ActivationScheme.DYNAMIC, None, None, FP8ScaleFormat.FP32,
                FP8ZeroPolicy.AMAX_FLOOR,
            ),
            FP8Recipe.W8A8_ROWWISE: (
                FP8ActivationScheme.DYNAMIC, None, None, FP8ScaleFormat.FP32,
                FP8ZeroPolicy.AMAX_FLOOR,
            ),
            FP8Recipe.W8A8_BLOCK128: (
                FP8ActivationScheme.DYNAMIC, (1, 128), (128, 128), FP8ScaleFormat.FP32,
                FP8ZeroPolicy.AMAX_FLOOR,
            ),
            FP8Recipe.MXFP8: (
                FP8ActivationScheme.DYNAMIC, (1, 32), (1, 32), FP8ScaleFormat.UE8M0,
                FP8ZeroPolicy.AMAX_FLOOR,
            ),
        }
        if recipe is FP8Recipe.W8A16:
            if self.activation_scheme is not FP8ActivationScheme.NONE:
                raise ValueError("W8A16 requires activation_scheme='none'")
            if self.activation_block is not None:
                raise ValueError("W8A16 does not quantize activations")
            if self.zero_policy is not FP8ZeroPolicy.CHECKPOINT_SCALE:
                raise ValueError("W8A16 requires checkpoint weight scales")
            return

        activation_scheme, activation_block, weight_block, scale_format, zero_policy = expected[
            recipe
        ]
        actual = (
            self.activation_scheme,
            self.activation_block,
            self.weight_block,
            self.scale_format,
            self.zero_policy,
        )
        required = (
            activation_scheme,
            activation_block,
            weight_block,
            scale_format,
            zero_policy,
        )
        if actual != required:
            raise ValueError(
                f"recipe {recipe.value!r} requires activation scheme/block, weight block, "
                f"scale format, and zero policy {required}; got {actual}"
            )

    @property
    def cache_key(self) -> str:
        """Stable recipe metadata included in weight-plan digests."""
        payload = {
            "activation_block": self.activation_block,
            "activation_scheme": self.activation_scheme.value,
            "amax_floor": self.amax_floor,
            "calibration_source": self.calibration_source,
            "excluded_layers": self.excluded_layers,
            "logical_scale_layout": self.logical_scale_layout,
            "physical_scale_layout": self.physical_scale_layout,
            "recipe": self.recipe.value,
            "scale_direction": self.scale_direction.value,
            "scale_format": self.scale_format.value,
            "version": self.version,
            "value_format": self.value_format.value,
            "weight_block": self.weight_block,
            "zero_policy": self.zero_policy.value,
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """Quantization of one weight.  ``group_size=-1`` is per-channel,
    ``0`` per-tensor.  ``scale_dtype`` is separate because an fp8 weight with
    fp32 scales and an fp8 weight with bf16 scales are different layouts.

    FP8 weights require an explicit :class:`FP8RecipeSpec`; the generic group
    size cannot distinguish recipes with separate activation and weight blocks.
    """

    method: str = "none"  # "none" | "awq" | "gptq" | "fp8" | "int8" | "nvfp4"
    weight_dtype: DType = DType.BF16
    scale_dtype: DType = DType.FP32
    group_size: int = 0
    symmetric: bool = True
    has_zero_point: bool = False
    fp8_recipe: FP8RecipeSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")
        if not isinstance(self.weight_dtype, DType) or not isinstance(self.scale_dtype, DType):
            raise TypeError("weight_dtype and scale_dtype must be DType values")
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int):
            raise TypeError("group_size must be an integer")
        if self.group_size < -1:
            raise ValueError("group_size must be -1, 0, or a positive integer")
        if not isinstance(self.symmetric, bool) or not isinstance(self.has_zero_point, bool):
            raise TypeError("symmetric and has_zero_point must be bool values")
        if self.method == "fp8" and self.fp8_recipe is None:
            raise ValueError("FP8 QuantSpec requires an explicit fp8_recipe")
        if self.fp8_recipe is None:
            return
        if self.method != "fp8":
            raise ValueError("fp8_recipe may only be set when method='fp8'")
        if not isinstance(self.fp8_recipe, FP8RecipeSpec):
            raise TypeError("fp8_recipe must be an FP8RecipeSpec")
        if self.fp8_recipe.recipe is FP8Recipe.KV_CACHE:
            raise ValueError("KV cache quantization is owned by KVQuantization, not WeightSpec")
        if self.fp8_recipe.value_format is not FP8ValueFormat.E4M3:
            raise ValueError("FP8 weight recipes in QuantSpec require E4M3 storage")
        if self.weight_dtype is not DType.FP8_E4M3:
            raise ValueError("FP8 weight recipes require weight_dtype=FP8_E4M3")
        expected_scale_dtype = (
            DType.UINT8
            if self.fp8_recipe.scale_format is FP8ScaleFormat.UE8M0
            else DType.FP32
        )
        if self.scale_dtype is not expected_scale_dtype:
            raise ValueError(
                f"{self.fp8_recipe.scale_format.value} scales require "
                f"scale_dtype={expected_scale_dtype.label}"
            )
        if not self.symmetric or self.has_zero_point:
            raise ValueError("FP8 inference recipes require symmetric values without a zero point")
        if self.group_size != 0:
            raise ValueError(
                "FP8RecipeSpec owns FP8 scale granularity; leave generic group_size at 0"
            )

    @property
    def cache_key(self) -> str:
        """Canonical quantization identity for deterministic loader plan digests."""
        recipe = self.fp8_recipe.cache_key if self.fp8_recipe is not None else None
        payload = {
            "fp8_recipe": recipe,
            "group_size": self.group_size,
            "has_zero_point": self.has_zero_point,
            "method": self.method,
            "scale_dtype": self.scale_dtype.label,
            "symmetric": self.symmetric,
            "weight_dtype": self.weight_dtype.label,
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

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
        if (
            self.quant.fp8_recipe is not None
            and self.spec.dtype is not self.quant.weight_dtype
        ):
            raise ValueError(
                f"{self.name}: tensor storage dtype {self.spec.dtype.label} does not match "
                f"FP8 recipe dtype {self.quant.weight_dtype.label}"
            )
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
