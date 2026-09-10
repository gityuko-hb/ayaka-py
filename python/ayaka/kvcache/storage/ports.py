from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from ayaka.kvcache.storage.geometry import BaseKVStorageSpec


@runtime_checkable
class StackedKVLayout(Protocol):
    """A store that can hand out one ``[num_planes, ...]`` tensor per layer."""

    def layer_cache(self, layer_index: int, /) -> Any:
        """Return ``[2, pages, page_size, num_kv_heads, head_dim]`` for one layer.

        Index 0 is K and index 1 is V. The frozen ``attention_decode`` signature
        implies that convention without ever stating it, so it is stated here.
        """
        ...

@runtime_checkable
class KVStorage(Protocol):
    """Physical tensor allocation associated with one storage specification.

    Implementations preallocate page-major tensors once and expose slot
    scatter / gather primitives. They never allocate or free pages -- that is
    the allocator's job -- and never see request identity.
    """
    @property
    def spec(self) -> BaseKVStorageSpec:
        """Geometry this store was built from. Read-only."""
        ...

    @property
    def capacity_pages(self) -> int: ...

    @property
    def page_size(self) -> int: ...

    @property
    def slot_capacity(self) -> int:
        """``capacity_pages * page_size``; the bound every address is checked against."""
        ...

    @property
    def plane_names(self) -> tuple[str, ...]:
        """Plane names in the order :meth:`write` and :meth:`read` use them."""
        ...

    @property
    def materialized_bytes(self) -> int:
        """Bytes actually held by this store's tensors, measured from the tensors."""
        ...

    def zero_pages(self, physical_pages: Iterable[int], /) -> None:
        """Zero whole pages across every layer and plane."""
        ...

    def write(self, layer_index: int, slots: Any, /, *values: Any) -> None:
        """Scatter one value tensor per plane, in :attr:`plane_names` order."""
        ...

    def write_planes(self, layer_index: int, slots: Any, /, **values: Any) -> None:
        """Scatter planes addressed by name rather than by position."""
        ...

    def read(
        self, layer_index: int, slots: Any, /, *, dtype: Any | None = None, raw: bool = False
    ) -> tuple[Any, ...]:
        """Gather slots, requiring every address to be unique."""
        ...

    def gather(
        self, layer_index: int, slots: Any, /, *, dtype: Any | None = None, raw: bool = False
    ) -> tuple[Any, ...]:
        """Gather slots, allowing repeated padding-slot addresses."""
        ...

    def buffers(self) -> tuple[tuple[Any, ...], ...]:
        """Return per-plane tensor families, in :attr:`plane_names` order."""
        ...

    def close(self) -> None:
        """Release tensor references after callers have drained GPU work."""
        ...
