from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any

from ayaka.exceptions import StorageUnavailableError
from ayaka.kvcache.storage.errors import KVQuantizationError, StorageClosedError
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec
from ayaka.kvcache.storage.index import (
    SlotIndex,
    index_copy_storage,
    index_fill_storage,
    index_select_storage,
    page_index,
    slot_index,
    validated_page_ids,
)
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_utils import require_torch, torch_dtype


def _load_torch_dtype(
    dtype_name: str,
    *,
    consumer: str,
) -> tuple[ModuleType, Any]:
    try:
        module = require_torch(capability=consumer)
        dtype = torch_dtype(dtype_name, capability=consumer)
    except CapabilityError as exc:
        raise StorageUnavailableError(
            f"{consumer} cannot initialize PyTorch storage: {exc}"
        ) from exc
    return module, dtype


class PagedKVStorage:
    """Preallocated page-major storage for an arbitrary set of planes.

    Not abstract -- it is fully usable -- but normally constructed through
    :func:`~ayaka.cache.kv.storage.families.create_kv_storage` or one of the
    family classes in :mod:`.families`, which add the named accessors and,
    where the layout allows it, the
    :class:`~ayaka.cache.kv.storage.ports.StackedKVLayout` capability.
    """

    def __init__(
        self,
        spec: BaseKVStorageSpec,
        *,
        device: str = "cuda",
        zero_initialize: bool = False,
    ) -> None:
        """Allocate every tensor up front.

        Args:
            spec: Geometry. Also fixes the dtype and the quantization policy.
            device: Torch device string.
            zero_initialize: Fill with zeros instead of leaving the allocation
                uninitialized. Off by default because zeroing a multi-gigabyte
                cache costs real bandwidth and every slot is written before it
                is read; turn it on when chasing a correctness bug, where
                reading uninitialized memory would look like a plausible value.
        """
        if not isinstance(spec, BaseKVStorageSpec):
            raise TypeError(f"spec must be a BaseKVStorageSpec, got {type(spec).__name__}")
        if not spec.kind.is_implemented:
            raise NotImplementedError(f"{spec.kind.value} storage has no implementation")
        quantization = spec.quantization
        if quantization is None:
            raise KVQuantizationError(
                f"{type(spec).__name__} did not initialize its quantization policy"
            )

        torch, resolved_dtype = _load_torch_dtype(spec.dtype, consumer=type(self).__name__)
        layout = spec.plane_layout()

        self._spec = spec
        self._layout = layout
        self._quantization = quantization
        self._device = torch.device(device)
        self._torch_dtype = resolved_dtype

        # Held on the instance so the hot paths do not repeat the lazy import.
        # After the first call that is only a `sys.modules` lookup, but it sits
        # inside every write and gather and costs nothing to remove.
        self._torch = torch
        self._closed = False

        factory = torch.zeros if zero_initialize else torch.empty
        page_shape = (spec.capacity_pages, spec.page_size)

        allocations: list[tuple[Any, ...]] = []
        planes: list[tuple[Any, ...]] = []
        if layout.stacked:
            # One allocation per layer holding every plane. Forced by the
            # single-`kv_cache` argument of the frozen attention contract, and
            # good on its own terms: a page's planes become one contiguous
            # reservation, so the allocator cannot hand out half a page.
            shape = (len(layout), *page_shape, *layout.planes[0].tail)
            for _ in range(spec.num_layers):
                block = factory(shape, dtype=resolved_dtype, device=self._device)
                allocations.append((block,))
                planes.append(tuple(block[i] for i in range(len(layout))))
            self._page_dim = 1  # (plane, page, slot, *tail)
        else:
            for _ in range(spec.num_layers):
                per_layer = tuple(
                    factory(
                        (*page_shape, *plane.tail),
                        dtype=resolved_dtype,
                        device=self._device,
                    )
                    for plane in layout.planes
                )
                allocations.append(per_layer)
                planes.append(per_layer)
            self._page_dim = 0  # (page, slot, *tail)

        self._allocations: tuple[tuple[Any, ...], ...] = tuple(allocations)
        self._planes: tuple[tuple[Any, ...], ...] = tuple(planes)

        # Quantization state. Scales are kept as Python floats because they are
        # applied as a scalar multiplier; keeping them device-side would force a
        # device read on every write for no benefit.
        self._scales: dict[tuple[int, int], float] = {}
        self._inverse_scales: dict[tuple[int, int], float] = {}
        self._written: set[tuple[int, int]] = set()

    @property
    def spec(self) -> BaseKVStorageSpec:
        """Geometry this store was built from.

        A read-only property, not a public attribute. Rebinding ``spec`` on a
        live store would leave the geometry describing something the tensors
        are not, and every bounds check derives from the geometry.
        """
        return self._spec

    @property
    def device(self) -> Any:
        """Torch device the tensors live on."""
        return self._device

    @property
    def torch_dtype(self) -> Any:
        """The resolved ``torch.dtype`` of the stored elements.

        Exposed so a caller can build a matching value tensor without
        re-deriving it from ``spec.dtype`` through ``getattr(torch, ...)``.
        """
        return self._torch_dtype

    @property
    def capacity_pages(self) -> int:
        return self._spec.capacity_pages

    @property
    def page_size(self) -> int:
        return self._spec.page_size

    @property
    def slot_capacity(self) -> int:
        return self._spec.slot_capacity

    @property
    def num_layers(self) -> int:
        return self._spec.num_layers

    @property
    def plane_names(self) -> tuple[str, ...]:
        """Plane names in the order :meth:`write` and :meth:`read` use them."""
        return self._layout.names

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def materialized_bytes(self) -> int:
        """Bytes held by this store's tensors, measured from the tensors.

        Measured rather than recomputed from the spec, so it cannot disagree
        with reality. The previous version returned ``spec.total_bytes``, which
        omitted alignment padding while the planner reserved
        ``aligned_total_bytes`` -- a gap that only surfaced when the memory
        ledger tried to reconcile per-owner totals against the driver.
        """
        if self._closed:
            return 0
        return sum(
            tensor.numel() * tensor.element_size()
            for allocation in self._allocations
            for tensor in allocation
        )

    @property
    def reserved_bytes(self) -> int:
        """Bytes the planner reserved for this store, padding included."""
        return self._spec.aligned_total_bytes()

    def __enter__(self) -> PagedKVStorage:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{self.materialized_bytes / 2**20:.1f} MiB"
        return f"<{type(self).__name__} {self._spec.describe()} [{state}]>"

    # buffer access
    def _check_open(self) -> None:
        if self._closed:
            raise StorageClosedError(
                f"{type(self).__name__} was closed; its tensors have been released"
            )

    def _check_layer(self, layer_index: int) -> None:
        """Validate a layer index before any buffer lookup.

        Applied on **every** entry point including :meth:`layer_cache`, which is
        the hottest one. The previous version skipped the check there, so
        ``layer_cache(-1)`` returned the *last* layer instead of raising. A
        pipeline-parallel rank computing a local layer index off by one then
        read the wrong layer, produced plausible logits, and raised nothing.
        """
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError(f"layer_index must be an integer, got {type(layer_index).__name__}")
        if not 0 <= layer_index < self._spec.num_layers:
            raise IndexError(
                f"layer_index {layer_index} is outside storage "
                f"(0 <= index < {self._spec.num_layers})"
            )

    def plane(self, layer_index: int, name: str) -> Any:
        """Return one layer's tensor for a named plane."""
        self._check_open()
        self._check_layer(layer_index)
        return self._planes[layer_index][self._layout.index_of(name)]

    def plane_buffers(self, name: str) -> tuple[Any, ...]:
        """Return the per-layer tensor family for a named plane."""
        self._check_open()
        position = self._layout.index_of(name)
        return tuple(layer[position] for layer in self._planes)

    def buffers(self) -> tuple[tuple[Any, ...], ...]:
        """Per-plane tensor families, in :attr:`plane_names` order.

        For MHA this is ``(key_buffers, value_buffers)`` and for MLA
        ``(latent_buffers, rope_buffers)`` -- same shape as the two-tuple the
        previous API returned, generalized to N planes.
        """
        self._check_open()
        return tuple(self.plane_buffers(name) for name in self.plane_names)

    def close(self) -> None:
        """Drop every tensor reference.

        Only drops references: the caching allocator reclaims the blocks, but
        nothing is returned to the driver (no ``empty_cache()``), and any view a
        caller still holds keeps the memory alive. After this, every method
        raises :class:`~ayaka.cache.kv.storage.errors.StorageClosedError`
        rather than an ``IndexError`` from an emptied tuple.
        """
        self._allocations = ()
        self._planes = ()
        self._scales.clear()
        self._inverse_scales.clear()
        self._written.clear()
        self._closed = True

    def _plane_position(self, name_or_index: str | int) -> int:
        if isinstance(name_or_index, str):
            return self._layout.index_of(name_or_index)
        if not isinstance(name_or_index, int) or isinstance(name_or_index, bool):
            raise TypeError("plane must be a name or an integer position")
        if not 0 <= name_or_index < len(self._layout):
            raise KeyError(f"plane position {name_or_index} is outside {self.plane_names}")
        return name_or_index

    def set_scale(
        self,
        layer_index: int,
        plane: str | int,
        scale: float,
        *,
        force: bool = False,
    ) -> None:
        """Install the dequantization scale for one (layer, plane).

        Stored values relate to real ones by ``real = stored * scale``, matching
        the ``k_scale`` / ``v_scale`` convention of FP8 checkpoints.

        Args:
            layer_index: Target layer.
            plane: Plane name or position.
            scale: A positive, finite multiplier.
            force: Permit changing a scale after data has been written under
                the previous one. Doing so silently reinterprets every token
                already in the cache, so it is refused by default.

        Raises:
            KVQuantizationError: on an unquantized store, a non-positive or
                non-finite scale, or an unforced change after writes.
        """
        self._check_open()
        self._check_layer(layer_index)
        quantization = self._quantization
        if not quantization.is_quantized:
            raise KVQuantizationError(
                f"{self._spec.dtype} storage carries no scale; setting one would be "
                "applied by some code paths and not others"
            )
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            raise KVQuantizationError("scale must be a real number")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise KVQuantizationError(f"scale must be finite and positive, got {scale}")

        key = (layer_index, self._plane_position(plane))
        if key in self._written and not force and self._scales.get(key) != scale:
            raise KVQuantizationError(
                f"layer {key[0]} plane {self.plane_names[key[1]]} already holds data written "
                f"under scale {self._scales.get(key)}. "
                + quantization.explain_static_requirement()
                + " Pass force=True only after zeroing the affected pages."
            )
        self._scales[key] = scale
        self._inverse_scales[key] = 1.0 / scale

    def load_scales(
        self,
        scales: Mapping[tuple[int, str | int], float],
        *,
        force: bool = False,
    ) -> None:
        """Install many scales at once, keyed by ``(layer_index, plane)``."""
        for (layer_index, plane), value in scales.items():
            self.set_scale(layer_index, plane, value, force=force)

    def scale(self, layer_index: int, plane: str | int) -> float:
        """Return the installed scale, or 1.0 for an unquantized store."""
        self._check_layer(layer_index)
        if not self._quantization.is_quantized:
            return 1.0
        return self._scales.get((layer_index, self._plane_position(plane)), 1.0)

    def _require_scale(self, layer_index: int, position: int) -> tuple[float, float]:
        """Return ``(scale, 1/scale)``, enforcing the calibration policy."""
        key = (layer_index, position)
        scale = self._scales.get(key)
        if scale is None:
            if self._quantization.require_calibration:
                raise KVQuantizationError(
                    f"layer {layer_index} plane {self.plane_names[position]} has no calibrated "
                    f"scale. {self._spec.dtype} saturates at "
                    f"±{self._quantization.storage_dtype_max}; writing unscaled activations "
                    "clamps or produces NaN with no diagnostic. Call set_scale() first, or build "
                    "the spec with KVQuantization.per_tensor(require_calibration=False)."
                )
            return 1.0, 1.0
        return scale, self._inverse_scales[key]

    # bulk operations
    def zero_pages(self, physical_pages: Iterable[int], /) -> None:
        """Zero whole pages in every layer and plane.

        Duplicate page IDs are collapsed. Used to sanitize a page between
        ownership lifetimes.

        One kernel launch per allocation -- ``num_layers`` for a stacked layout,
        ``num_layers * num_planes`` otherwise -- rather than one per (layer,
        plane, page).
        """
        self._check_open()
        page_ids = validated_page_ids(physical_pages, capacity_pages=self.capacity_pages)
        if not page_ids:
            return
        torch = self._torch
        indices = page_index(torch, page_ids, device=self._device)
        for allocation in self._allocations:
            for tensor in allocation:
                index_fill_storage(torch, tensor, self._page_dim, indices)

    # scatter
    def _flat(self, tensor: Any, tail: tuple[int, ...]) -> Any:
        """Zero-copy view collapsing ``(pages, page_size)`` into a slot axis."""
        return tensor.reshape(-1, *tail)

    def write(self, layer_index: int, slots: Any, /, *values: Any) -> None:
        """Scatter one value tensor per plane, in :attr:`plane_names` order.

        Args:
            layer_index: Target layer.
            slots: Unique slot addresses -- a Python sequence, a 1-D integer
                tensor, or a prebuilt
                :class:`~ayaka.cache.kv.storage._indexing.SlotIndex`. Pass a
                ``SlotIndex`` to validate once and reuse across every layer of
                a step.
            *values: One tensor per plane, shaped ``(num_tokens, *plane.tail)``.

        Raises:
            SlotAddressError: for a malformed, out-of-range or duplicated address.
            KVQuantizationError: for a quantized plane with no calibrated scale.
            ValueError: on a geometry mismatch.
        """
        self._check_open()
        self._check_layer(layer_index)
        if len(values) != len(self._layout):
            raise ValueError(
                f"expected {len(self._layout)} value tensors {self.plane_names}, got {len(values)}"
            )
        torch = self._torch

        token_count: int | None = None
        for plane, tensor in zip(self._layout.planes, values, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"value for plane {plane.name!r} must be a tensor")
            expected_ndim = 1 + len(plane.tail)
            if tensor.ndim != expected_ndim or tuple(tensor.shape[1:]) != plane.tail:
                tail = ", ".join(map(str, plane.tail))
                raise ValueError(
                    f"plane {plane.name!r} expects (num_tokens, {tail}), got {tuple(tensor.shape)}"
                )
            if token_count is None:
                token_count = int(tensor.shape[0])
            elif int(tensor.shape[0]) != token_count:
                raise ValueError("every plane must carry the same number of tokens")

        index = slot_index(
            torch,
            slots,
            device=self._device,
            capacity_slots=self.slot_capacity,
            require_unique=True,
            expected_count=token_count,
        )

        quantization = self._quantization
        for position, (plane, tensor) in enumerate(zip(self._layout.planes, values, strict=True)):
            payload = tensor
            if quantization.is_quantized:
                _, inverse = self._require_scale(layer_index, position)
                # Scale, then clamp to the representable range, then let
                # index_copy_storage cast. Without the clamp an out-of-range
                # value saturates inside the cast with no diagnostic.
                limit = quantization.storage_dtype_max
                payload = tensor.float().mul(inverse).clamp_(-limit, limit)
            destination = self._flat(self._planes[layer_index][position], plane.tail)
            index_copy_storage(torch, destination, index.tensor, payload)
            self._written.add((layer_index, position))

    def write_planes(self, layer_index: int, slots: Any, /, **values: Any) -> None:
        """Scatter planes addressed by name rather than by position.

        ``store.write_planes(0, slots, key=k, value=v)`` is order-independent
        and self-documenting. :meth:`write` remains the positional form used by
        the hot path.
        """
        missing = set(self.plane_names) - set(values)
        unknown = set(values) - set(self.plane_names)
        if missing or unknown:
            raise ValueError(
                f"planes must be exactly {self.plane_names}; "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        self.write(layer_index, slots, *(values[name] for name in self.plane_names))

    def _gather(
        self,
        layer_index: int,
        slots: Any,
        *,
        dtype: Any | None,
        raw: bool,
        require_unique: bool,
    ) -> tuple[Any, ...]:
        self._check_open()
        self._check_layer(layer_index)
        torch = self._torch
        quantization = self._quantization
        if quantization.is_quantized and not raw and dtype is None:
            raise KVQuantizationError(
                f"{self._spec.dtype} storage holds scaled values; reading without a "
                "target dtype would return raw codes that look like activations. Pass "
                "dtype=... to dequantize, or raw=True if you really want the codes."
            )

        index = slot_index(
            torch,
            slots,
            device=self._device,
            capacity_slots=self.slot_capacity,
            require_unique=require_unique,
        )

        outputs: list[Any] = []
        for position, plane in enumerate(self._layout.planes):
            source = self._flat(self._planes[layer_index][position], plane.tail)
            gathered = index_select_storage(torch, source, index.tensor)
            if quantization.is_quantized and not raw:
                # Dequantize in fp32 first, then cast. Casting first would run
                # the multiply in the target dtype, where a large scale can
                # overflow fp16 even though the real value fits.
                scale, _ = self._require_scale(layer_index, position)
                gathered = gathered.float().mul(scale)
            if dtype is not None:
                gathered = gathered.to(dtype=dtype)
            outputs.append(gathered)
        return tuple(outputs)

    def read(
        self,
        layer_index: int,
        slots: Any,
        /,
        *,
        dtype: Any | None = None,
        raw: bool = False,
    ) -> tuple[Any, ...]:
        """Gather slots, requiring every address to be unique.

        The strict counterpart of :meth:`gather`. Use it wherever a duplicate
        address would indicate a scheduling bug rather than intentional padding.
        """
        return self._gather(layer_index, slots, dtype=dtype, raw=raw, require_unique=True)

    def gather(
        self,
        layer_index: int,
        slots: Any,
        /,
        *,
        dtype: Any | None = None,
        raw: bool = False,
    ) -> tuple[Any, ...]:
        """Gather slots, allowing repeated addresses.

        Retention-aware reference execution gathers the full logical token range
        and resolves evicted leading blocks to a shared padding page, so many
        logical positions legitimately resolve to the same slot.
        """
        return self._gather(layer_index, slots, dtype=dtype, raw=raw, require_unique=False)

    def prepare_slots(
        self,
        slots: Any,
        *,
        require_unique: bool = True,
        validate: bool = True,
    ) -> SlotIndex:
        """Build a reusable, pre-validated slot index.

        Validate once per step and pass the result to every layer's
        :meth:`write`, instead of re-validating the same addresses
        ``num_layers`` times.

        ``validate=False`` issues no ``.item()`` but does not establish write
        uniqueness. Such an index may be gathered, not passed to ``write``.
        For graph writes use a host-validated prepared store such as the paged
        decode adapter; do not fabricate a uniqueness flag for mutable indices.
        """
        self._check_open()
        return slot_index(
            self._torch,
            slots,
            device=self._device,
            capacity_slots=self.slot_capacity,
            require_unique=require_unique,
            validate=validate,
        )
