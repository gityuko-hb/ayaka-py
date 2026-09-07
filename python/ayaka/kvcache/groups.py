from __future__ import annotations

from dataclasses import dataclass

from ayaka.kvcache.retention.policy import HybridRetention, RetentionPolicy
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec


@dataclass(frozen=True, slots=True)
class KVLayerConfig:
    """Storage and retention declaration for one model layer.

    Exactly one layer per config (``storage_spec.num_layers == 1``) with a
    *resolved* retention policy: ``HybridRetention`` is only valid while
    building groups and is rejected here.
    """

    layer_id: int
    storage_spec: BaseKVStorageSpec
    retention: RetentionPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, int) or isinstance(self.layer_id, bool):
            raise TypeError("layer_id must be an integer")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if not isinstance(self.storage_spec, BaseKVStorageSpec):
            raise TypeError("storage_spec must implement BaseKVStorageSpec")
        if self.storage_spec.num_layers != 1:
            raise ValueError("KVLayerConfig storage specs must describe exactly one layer")
        if not isinstance(self.retention, RetentionPolicy):
            raise TypeError("retention must implement RetentionPolicy")
        if isinstance(self.retention, HybridRetention):
            raise ValueError("KVLayerConfig requires the resolved layer retention policy")

@dataclass(frozen=True, slots=True)
class KVCacheGroup:
    """Layers sharing one storage geometry, retention policy, and allocator.

    All members share the same ``compatibility_key``, page capacity, and
    retention, so one ``PageAllocator`` and one physical-page namespace can
    serve the whole group.
    """

    name: str
    layer_ids: tuple[int, ...]
    storage_spec: BaseKVStorageSpec
    retention: RetentionPolicy

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("cache-group name must not be empty")
        layer_ids = tuple(self.layer_ids)
        if not layer_ids:
            raise ValueError("a cache group must contain at least one layer")
        if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in layer_ids):
            raise TypeError("cache-group layer IDs must be integers")
        if any(layer < 0 for layer in layer_ids):
            raise ValueError("cache-group layer IDs must be non-negative")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("cache-group layer IDs must be unique")
        # Sorted layer IDs give deterministic attention ranges and diagnostics.
        if tuple(sorted(layer_ids)) != layer_ids:
            raise ValueError("cache-group layer IDs must be sorted")
        if not isinstance(self.storage_spec, BaseKVStorageSpec):
            raise TypeError("storage_spec must implement KVStorageSpec")
        if self.storage_spec.num_layers != len(layer_ids):
            raise ValueError("storage spec layer count must match cache-group layers")
        if not isinstance(self.retention, RetentionPolicy):
            raise TypeError("retention must implement RetentionPolicy")
        if isinstance(self.retention, HybridRetention):
            raise ValueError("cache groups require one resolved retention policy")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "layer_ids", layer_ids)

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        """Identity used for coalescing: geometry + capacity + retention."""
        return (
            self.storage_spec.compatibility_key,
            self.storage_spec.capacity_pages,
            self.retention.compatibility_key,
        )


def build_kv_cache_groups(
    layers: tuple[KVLayerConfig, ...],
    *,
    name_prefix: str = "kv_group",
) -> tuple[KVCacheGroup, ...]:
    """Deterministically coalesce compatible layer declarations.

    Layers are grouped by (storage compatibility key, capacity, retention
    compatibility key) in ascending layer-id order, and groups are named
    ``{prefix}_{index}`` in first-appearance order, so identical inputs always
    produce identical groups.

    Args:
        layers: One config per model layer; layer IDs must be unique.
        name_prefix: Stable prefix for generated group names.

    Returns:
        Immutable, ordered tuple of coalesced cache groups.

    Raises:
        ValueError: if no layers are given, layer IDs repeat, or the name
            prefix is empty.
        TypeError: if any element is not a ``KVLayerConfig``.
    """

    normalized = tuple(layers)
    if not normalized:
        raise ValueError("at least one KV layer is required")
    if any(not isinstance(layer, KVLayerConfig) for layer in normalized):
        raise TypeError("layers must contain KVLayerConfig values")
    ordered = tuple(sorted(normalized, key=lambda item: item.layer_id))
    if len({layer.layer_id for layer in ordered}) != len(ordered):
        raise ValueError("layer IDs must be globally unique")
    prefix = name_prefix.strip()
    if not prefix:
        raise ValueError("name_prefix must not be empty")

    grouped: dict[tuple[object, ...], list[KVLayerConfig]] = {}
    group_order: list[tuple[object, ...]] = []
    for layer in ordered:
        key = (
            layer.storage_spec.compatibility_key,
            layer.storage_spec.capacity_pages,
            layer.retention.compatibility_key,
        )
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(layer)

    result = []
    for index, group_key in enumerate(group_order):
        members = grouped[group_key]
        first = members[0]
        layer_ids = tuple(member.layer_id for member in members)
        result.append(
            KVCacheGroup(
                name=f"{prefix}_{index}",
                layer_ids=layer_ids,
                # Merged spec: one storage with the combined layer count.
                storage_spec=first.storage_spec.with_num_layers(len(layer_ids)),
                retention=first.retention,
            )
        )
    return tuple(result)
