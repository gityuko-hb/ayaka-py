from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ayaka.kvcache.storage.backend import PagedKVStorage
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec, MHAStorageSpec, MLAStorageSpec
from ayaka.kvcache.storage.layout import KVStorageKind


class _StackedKVMixin:
    """Adds :meth:`layer_cache` to stores whose planes share one allocation."""

    _allocations: tuple[tuple[Any, ...], ...]

    if TYPE_CHECKING:

        def _check_open(self) -> None: ...

        def _check_layer(self, layer_index: int) -> None: ...

    def layer_cache(self, layer_index: int, /) -> Any:
        """Return the single stacked tensor for one layer.

        Shape ``[num_planes, pages, page_size, *tail]``; for MHA index 0 is K
        and index 1 is V. This is what ``ComputeBackend.attention_decode`` and
        ``store_kv`` take.
        """
        self._check_open()
        self._check_layer(layer_index)
        return self._allocations[layer_index][0]


class MHAStorage(_StackedKVMixin, PagedKVStorage):
    """Page-major K/V storage for MHA, MQA and GQA.

    Planes are ``("key", "value")`` and share one allocation per layer, so this
    class satisfies :class:`~ayaka.cache.kv.storage.ports.StackedKVLayout`.
    """

    def __init__(
        self,
        spec: MHAStorageSpec,
        *,
        device: str = "cuda",
        zero_initialize: bool = False,
    ) -> None:
        if not isinstance(spec, MHAStorageSpec):
            raise TypeError(f"spec must be MHAStorageSpec, got {type(spec).__name__}")
        super().__init__(spec, device=device, zero_initialize=zero_initialize)

    @property
    def key_buffers(self) -> tuple[Any, ...]:
        """Per-layer K tensors, each a zero-copy view into the stacked block."""
        return self.plane_buffers("key")

    @property
    def value_buffers(self) -> tuple[Any, ...]:
        """Per-layer V tensors, each a zero-copy view into the stacked block."""
        return self.plane_buffers("value")


class MLAStorage(PagedKVStorage):
    """Compressed latent / decoupled RoPE storage for Multi-head Latent Attention.

    Planes are ``("latent", "rope")`` with different widths, so they are
    allocated separately and this class deliberately does **not** provide
    ``layer_cache``: ``isinstance(store, StackedKVLayout)`` is False, which is
    the honest answer for a family that has no K and no V.
    """

    def __init__(
        self,
        spec: MLAStorageSpec,
        *,
        device: str = "cuda",
        zero_initialize: bool = False,
    ) -> None:
        if not isinstance(spec, MLAStorageSpec):
            raise TypeError(f"spec must be MLAStorageSpec, got {type(spec).__name__}")
        super().__init__(spec, device=device, zero_initialize=zero_initialize)

    @property
    def latent_buffers(self) -> tuple[Any, ...]:
        """Per-layer compressed latent tensors."""
        return self.plane_buffers("latent")

    @property
    def rope_buffers(self) -> tuple[Any, ...]:
        """Per-layer decoupled RoPE tensors."""
        return self.plane_buffers("rope")


def build_kv_storage(
    spec: BaseKVStorageSpec,
    *,
    device: str = "cuda",
    zero_initialize: bool = False,
) -> PagedKVStorage:
    """Build the storage class matching a spec's family.

    Raises:
        NotImplementedError: for a kind with no implementation.
    """
    if isinstance(spec, MHAStorageSpec):
        return MHAStorage(spec, device=device, zero_initialize=zero_initialize)
    if isinstance(spec, MLAStorageSpec):
        return MLAStorage(spec, device=device, zero_initialize=zero_initialize)
    if spec.kind is KVStorageKind.RECURRENT:
        raise NotImplementedError(
            "recurrent-state storage is not implemented: Ayaka has no validated "
            "recurrent model/backend contract"
        )
    raise TypeError(f"no storage implementation for {type(spec).__name__}")
