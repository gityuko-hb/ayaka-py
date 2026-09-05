from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

#: Alignment shared by planner and materializer, in bytes.
#:
#: 256 satisfies every consumer at once:
#:
#: * two 128 B L1 cache lines, so a page boundary never splits a line;
#: * eight 32 B L2 sectors;
#: * the 16 B operand alignment of ``ld.global.v4.b32`` / ``st.global.v4.b32``,
#:   so attention kernels can issue vectorised loads with no scalar prologue;
#: * the 128 B alignment ``cp.async.bulk`` requires on Hopper.
#:
#: Every independently allocated buffer starts on at least this boundary.
DEFAULT_KV_ALIGNMENT_BYTES: Final[int] = 256

class KVStorageKind(StrEnum):
    """Physical cache-content families.

    ``StrEnum`` (not ``Enum``) because members land inside
    ``compatibility_key`` tuples and in structured logs, and a plain string
    serialises without a custom JSON encoder.
    """

    MHA = "mha"
    """Conventional page-major K/V tensors. Covers MHA, MQA and GQA alike --
    they differ only in ``num_kv_heads_local``, not in physical shape."""

    MLA = "mla"
    """Compressed latent plus decoupled RoPE buffers (DeepSeek-style)."""

    RECURRENT = "recurrent"
    """Recurrent model state.

    Declared but **not constructible**: no spec class produces this kind and
    :func:`~ayaka.cache.kv.storage.validation.validate_kv_storage_support`
    rejects it with ``KIND_UNSUPPORTED``. It stays in the enum so that a model
    config naming a recurrent architecture is classified and *then* refused
    with a sentence, instead of falling through a ``match`` into whichever
    branch happens to be last.
    """

    @property
    def is_implemented(self) -> bool:
        """Whether a storage implementation exists for this kind."""
        return self is not KVStorageKind.RECURRENT

@dataclass(frozen=True, slots=True)
class PlaneSpec:
    """One per-token buffer family within a layer.

    Attributes:
        name: Stable identifier used by accessors and error messages
            (``"key"``, ``"value"``, ``"latent"``, ``"rope"``). Must be unique
            within a :class:`PlaneLayout`.
        tail: Per-token shape, i.e. everything after ``(page, slot)``. For MHA
            this is ``(num_kv_heads_local, head_dim)``; for MLA latent it is
            ``(latent_dim,)``.
    """

    name: str
    tail: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"plane name must be a valid identifier, got {self.name!r}")
        if not self.tail:
            raise ValueError(f"plane {self.name!r} must have a non-empty tail shape")
        for extent in self.tail:
            # bool is a subclass of int, so `True` would otherwise pass as 1.
            if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
                raise ValueError(f"plane {self.name!r} tail extents must be positive integers")

    @property
    def elements_per_token(self) -> int:
        """Number of elements this plane stores for one token."""
        return math.prod(self.tail)


@dataclass(frozen=True, slots=True)
class PlaneLayout:
    """How a layer's planes map onto physical allocations.

    Attributes:
        planes: The planes, in allocation order.
        stacked: Whether all planes share **one** allocation of shape
            ``(num_planes, pages, page_size, *tail)``.

    ``stacked`` is not a style choice; it is forced by a frozen contract.
    ``ComputeBackend.attention_decode`` has taken a single ``kv_cache``
    argument since A0, so an MHA store must be able to hand out one
    ``[2, ...]`` tensor. Two separately allocated tensors cannot be viewed as
    one without copying the whole slab every step.

    Stacking also buys a real invariant: a page's K and V live in one
    contiguous reservation, so the allocator cannot hand out half a page.

    Stacking requires identical tails, which is exactly why MLA cannot stack --
    ``latent_dim`` (e.g. 512) and ``rope_dim`` (e.g. 64) differ, and a stacked
    tensor would have to pad RoPE up to the latent width, wasting
    ``(512 - 64) / 576 = 78%`` of the RoPE plane.
    """

    planes: tuple[PlaneSpec, ...]
    stacked: bool = False

    def __post_init__(self) -> None:
        if not self.planes:
            raise ValueError("a plane layout must declare at least one plane")
        names = [plane.name for plane in self.planes]
        if len(set(names)) != len(names):
            raise ValueError(f"plane names must be unique, got {names}")
        if self.stacked:
            tails = {plane.tail for plane in self.planes}
            if len(tails) != 1:
                raise ValueError(
                    "stacked planes must share one tail shape; "
                    f"got {sorted(tails)} -- allocate them separately instead"
                )

    def __len__(self) -> int:
        return len(self.planes)

    @property
    def names(self) -> tuple[str, ...]:
        """Plane names in allocation order."""
        return tuple(plane.name for plane in self.planes)

    @property
    def elements_per_token(self) -> int:
        """Total elements stored for one token across every plane."""
        return sum(plane.elements_per_token for plane in self.planes)

    def index_of(self, name: str) -> int:
        """Return the position of a plane by name.

        Raises:
            KeyError: if no plane carries that name.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown plane {name!r}; layout has {self.names}") from None


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``.

    Kept general (``alignment`` need not be a power of two) because it only
    runs in planning code, never per step. The bitmask form
    ``(v + a - 1) & ~(a - 1)`` would be faster but would silently produce
    garbage for a non-power-of-two alignment.

    Raises:
        ValueError: for a non-positive alignment or a negative value.
    """
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if value < 0:
        raise ValueError("value must not be negative")
    return ((value + alignment - 1) // alignment) * alignment
