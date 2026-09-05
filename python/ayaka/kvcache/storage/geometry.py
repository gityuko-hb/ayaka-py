from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from ayaka.kvcache.storage.dtypes import normalize_storage_dtype, storage_dtype_bytes
from ayaka.kvcache.storage.layout import (
    DEFAULT_KV_ALIGNMENT_BYTES,
    KVStorageKind,
    PlaneLayout,
    PlaneSpec,
    align_up,
)
from ayaka.kvcache.storage.quantization import KVQuantization
from ayaka.types import KVLayoutKind


class BaseKVStorageSpec(ABC):
    """Base geometry contract consumed by storage and capacity planners.

    Subclasses supply the family-specific primitives -- the plane layout, the
    kind, and the compatibility key -- and inherit every derived quantity. That
    split is what keeps the planner's byte math and the backend's allocation
    from drifting: both read :meth:`plane_layout`.
    """

    __slots__ = ()

    layout: KVLayoutKind
    """Element ordering inside a page."""
    dtype: str
    """Canonical storage dtype name, already normalized."""
    quantization: KVQuantization | None
    """Scale policy; ``QuantizationScheme.NONE`` for unquantized caches."""
    num_layers: int
    """Number of layers sharing this geometry."""
    page_size: int
    """Tokens per page: the allocation unit and the attention unit."""
    capacity_pages: int
    """Pages this storage instance can hold."""

    @property
    @abstractmethod
    def kind(self) -> KVStorageKind:
        """Physical cache-content family."""

    @abstractmethod
    def plane_layout(self) -> PlaneLayout:
        """Per-token buffers of one layer, and whether they share an allocation.

        The single description consumed by both :meth:`aligned_total_bytes`
        here and the allocator in :mod:`.backend`.
        """

    @property
    def bytes_per_element(self) -> int:
        """Size of one stored element."""
        return storage_dtype_bytes(self.dtype)

    @property
    def _quantization_policy(self) -> KVQuantization:
        """Return the policy resolved by the concrete spec's ``__post_init__``."""
        if self.quantization is None:
            name = type(self).__name__
            raise RuntimeError(f"{name} did not initialize its quantization policy")
        return self.quantization

    @property
    def elements_per_token_per_layer(self) -> int:
        """Elements one token occupies in one layer, across all planes."""
        return self.plane_layout().elements_per_token

    @property
    def bytes_per_token_per_layer(self) -> int:
        """Bytes one token occupies in one layer.

        The unit every capacity computation is built from.
        """
        return self.elements_per_token_per_layer * self.bytes_per_element

    @property
    def bytes_per_token(self) -> int:
        """Bytes one token occupies across every layer."""
        return self.num_layers * self.bytes_per_token_per_layer

    @property
    def bytes_per_page(self) -> int:
        """Payload bytes of one page across every layer."""
        return self.page_size * self.bytes_per_token

    @property
    def total_bytes(self) -> int:
        """Payload bytes of the whole cache, ignoring alignment padding.

        Use :meth:`aligned_total_bytes` for anything that has to match what is
        actually reserved.
        """
        return self.capacity_pages * self.bytes_per_page

    @property
    def slot_capacity(self) -> int:
        """Number of addressable token slots.

        Slot addressing is flat: ``slot = page_id * page_size + offset``. Every
        bounds check in the backend is against this number.
        """
        return self.capacity_pages * self.page_size

    @property
    def scale_bytes(self) -> int:
        """Device bytes the quantization scales occupy.

        Zero for the per-tensor scheme, whose scales are host floats. See
        :meth:`~ayaka.kvcache.storage.quantization.KVQuantization.scale_bytes`.
        """
        return self._quantization_policy.scale_bytes(
            num_layers=self.num_layers,
            num_planes=len(self.plane_layout()),
        )

    def plane_bytes(self, plane: PlaneSpec) -> int:
        """Payload bytes of one plane, for one layer, across the whole capacity."""
        return (
            self.capacity_pages * self.page_size * plane.elements_per_token * self.bytes_per_element
        )

    def aligned_total_bytes(self, alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES) -> int:
        """Bytes actually reserved, padding included.

        Mirrors the allocation strategy exactly:

        * ``stacked`` -- one allocation per layer holding every plane, so one
            alignment boundary per layer;
        * otherwise -- one allocation per (layer, plane), so one boundary each.

        Scale tensors are added as a single aligned region at the end.

        Raises:
            ValueError: for a non-positive alignment.
        """
        if alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be positive")
        layout = self.plane_layout()
        if layout.stacked:
            per_layer = sum(self.plane_bytes(plane) for plane in layout.planes)
            payload = self.num_layers * align_up(per_layer, alignment_bytes)
        else:
            payload = self.num_layers * sum(
                align_up(self.plane_bytes(plane), alignment_bytes) for plane in layout.planes
            )
        scales = self.scale_bytes
        return payload + (align_up(scales, alignment_bytes) if scales else 0)

    def describe(self) -> str:
        """One-line human summary, for logs and error messages."""
        planes = ", ".join(
            f"{plane.name}{tuple(plane.tail)}" for plane in self.plane_layout().planes
        )
        return (
            f"{self.kind.value}/{self.layout.value} {self.dtype} "
            f"layers={self.num_layers} page_size={self.page_size} "
            f"pages={self.capacity_pages} planes=[{planes}] "
            f"{self.bytes_per_token_per_layer} B/token/layer"
        )


def _check_positive_inits(**values: int) -> None:
    """Reject non-integers, booleans and non-positive extents.

    ``bool`` is excluded explicitly because it subclasses ``int``: without the
    guard, ``MHAStorageSpec(num_layers=True, ...)`` constructs a one-layer
    cache and the mistake surfaces as a shape error much later.
    """
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _normalize_layout(layout: object) -> KVLayoutKind:
    """Normalize an enum member or a case-insensitive layout name."""
    if isinstance(layout, KVLayoutKind):
        return layout
    name = str(layout).strip().upper()
    try:
        return KVLayoutKind[name]
    except KeyError:
        supported = ", ".join(member.name for member in KVLayoutKind)
        raise ValueError(f"unsupported KV layout {layout!r}; supported: {supported}") from None


def _resolve_quantization(dtype: str, quantization: KVQuantization | None) -> KVQuantization:
    """Resolve the dtype default and validate an explicitly supplied policy."""
    policy = KVQuantization.for_dtype(dtype) if quantization is None else quantization
    if not isinstance(policy, KVQuantization):
        raise TypeError(
            f"quantization must be a KVQuantization or None, got {type(policy).__name__}"
        )
    policy.validate_for_dtype(dtype)
    return policy


@dataclass(frozen=True, slots=True)
class MHAStorageSpec(BaseKVStorageSpec):
    """Page-major K/V geometry. Covers MHA, MQA and GQA.

    ``bytes_per_token_per_layer = 2 * num_kv_heads_local * head_dim * dtype_bytes``
    where the factor 2 is K plus V. One logical page maps to the same physical
    page index in every layer.

    Worked example -- Qwen2-0.5B (24 layers, 2 KV heads, head_dim 64, fp16,
    page_size 16)::

        bytes_per_token_per_layer = 2 * 2 * 64 * 2 =     512 B
        bytes_per_token           = 24 * 512      =  12 KiB
        bytes_per_page            = 16 * 12288    = 192 KiB
        one plane, one layer, whole capacity      =   4 KiB * capacity_pages

    4 KiB is a multiple of 256, so this configuration takes no alignment
    padding at all.
    """

    num_layers: int
    """Layers sharing this geometry; one stacked K/V allocation each."""
    num_kv_heads_local: int
    """KV heads **after** tensor-parallel sharding or replication."""
    head_dim: int
    """Per-head K/V dimension."""
    page_size: int
    """Tokens per page."""
    capacity_pages: int
    """Pages this storage instance can hold."""
    dtype: str = "float16"
    layout: KVLayoutKind = KVLayoutKind.NHD
    quantization: KVQuantization | None = None
    """Scale policy; ``None`` derives the safe default from ``dtype``."""

    def __post_init__(self) -> None:
        _check_positive_inits(
            num_layers=self.num_layers,
            num_kv_heads_local=self.num_kv_heads_local,
            head_dim=self.head_dim,
            page_size=self.page_size,
            capacity_pages=self.capacity_pages,
        )
        # frozen=True routes attribute assignment through a raising
        # __setattr__, so normalization has to bypass it.
        object.__setattr__(self, "dtype", normalize_storage_dtype(self.dtype))
        object.__setattr__(self, "layout", _normalize_layout(self.layout))
        if self.layout is KVLayoutKind.NLD:
            raise ValueError("NLD is an MLA layout; MHA storage uses NHD or HND")
        object.__setattr__(
            self,
            "quantization",
            _resolve_quantization(self.dtype, self.quantization),
        )

    @property
    def kind(self) -> KVStorageKind:
        return KVStorageKind.MHA

    def plane_layout(self) -> PlaneLayout:
        """Two equally shaped planes in one allocation.

        Stacked because ``ComputeBackend.attention_decode`` takes a single
        ``kv_cache`` tensor, and because it makes a page's K and V one
        contiguous reservation.
        """

        tail = (self.num_kv_heads_local, self.head_dim)
        return PlaneLayout(
            planes=(PlaneSpec("key", tail), PlaneSpec("value", tail)),
            stacked=True,
        )

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.num_kv_heads_local,
            self.head_dim,
            self.page_size,
            self.dtype,
            self.layout,
            self._quantization_policy.scheme,
        )

    def with_capacity_pages(self, capacity_pages: int) -> MHAStorageSpec:
        return replace(self, capacity_pages=capacity_pages)

    def with_num_layers(self, num_layers: int) -> MHAStorageSpec:
        return replace(self, num_layers=num_layers)


@dataclass(frozen=True, slots=True)
class MLAStorageSpec(BaseKVStorageSpec):
    """Compressed latent / decoupled RoPE geometry for Multi-head Latent Attention.

    ``bytes_per_token_per_layer = (latent_dim + rope_dim) * dtype_bytes`` --
    note the absence of a factor 2. MLA has no separate K and V: after weight
    absorption the compressed latent serves as both (``W_UK`` folded into the
    query projection, ``W_UV`` into the output projection). Modelling MLA as
    synthetic K/V heads gets both the byte count and the layout wrong.

    Worked example -- DeepSeek-V3 (61 layers, latent 512, rope 64)::
        Deepseek-V3 Papers:
        fp16: (512 + 64) * 2 = 1152 B/layer/token -> 68.6 KiB/token
        fp8:  (512 + 64) * 1 =  576 B/layer/token -> 34.3 KiB/token
    """

    latent_dim: int
    """Compressed latent dimension per token."""
    rope_dim: int
    """Decoupled RoPE dimension per token."""
    page_size: int
    """Tokens per page."""
    capacity_pages: int
    """Pages this storage instance can hold."""
    num_layers: int = 1
    dtype: str = "float16"
    layout: KVLayoutKind = KVLayoutKind.NLD
    quantization: KVQuantization | None = None

    def __post_init__(self) -> None:
        _check_positive_inits(
            latent_dim=self.latent_dim,
            rope_dim=self.rope_dim,
            page_size=self.page_size,
            capacity_pages=self.capacity_pages,
            num_layers=self.num_layers,
        )
        object.__setattr__(self, "dtype", normalize_storage_dtype(self.dtype))
        object.__setattr__(self, "layout", _normalize_layout(self.layout))
        if self.layout is not KVLayoutKind.NLD:
            raise ValueError(f"MLA storage materializes NLD only, got {self.layout.value}")
        object.__setattr__(
            self,
            "quantization",
            _resolve_quantization(self.dtype, self.quantization),
        )

    @property
    def kind(self) -> KVStorageKind:
        return KVStorageKind.MLA

    def plane_layout(self) -> PlaneLayout:
        """Two planes of different width, allocated separately.

        Not stacked: stacking requires equal tails, and padding the 64-wide
        RoPE plane up to a 512-wide latent would waste 78% of it.
        """
        return PlaneLayout(
            planes=(
                PlaneSpec("latent", (self.latent_dim,)),
                PlaneSpec("rope", (self.rope_dim,)),
            ),
            stacked=False,
        )

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.latent_dim,
            self.rope_dim,
            self.page_size,
            self.dtype,
            self.layout,
            self._quantization_policy.scheme,
        )

    def with_capacity_pages(self, capacity_pages: int) -> MLAStorageSpec:
        return replace(self, capacity_pages=capacity_pages)

    def with_num_layers(self, num_layers: int) -> MLAStorageSpec:
        return replace(self, num_layers=num_layers)
