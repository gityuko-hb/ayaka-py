from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ayaka.types import KVCacheDtype, KVLayoutKind

if TYPE_CHECKING:
    import torch


@runtime_checkable
class PagedKVCache(Protocol):
    """A paged K/V pool for ONE attention group.

    Per-group, not per-model: a hybrid model hands each backend its own pool, so a backend
    never indexes another group's layer space. ``layer_id`` is therefore GROUP-LOCAL --
    ``AttnGroupSpec.layer_ids[i]`` maps to local id ``i``, and the mapping is done once by
    Execution Runtime Layer, not re-derived inside kernels.
    """

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def page_size(self) -> int: ...

    @property
    def num_layers(self) -> int: ...

    @property
    def kv_layout(self) -> KVLayoutKind: ...

    @property
    def kv_cache_dtype(self) -> KVCacheDtype:
        """Storage dtype of the pool. ``dtype`` above stays the COMPUTE dtype.

        Two properties rather than one because they genuinely differ under quantization: a
        bf16 model with an FP8 pool has ``dtype == bfloat16`` and
        ``kv_cache_dtype == FP8_E4M3``, and a kernel needs both -- the first to type its
        accumulator and output, the second to know it must dequantize on load.
        """
        ...

    def key_cache(self, layer_id: int) -> torch.Tensor:
        """``[num_pages, page_size, num_kv_heads, head_dim]`` for NHD.

        For an MLA group this is the single latent slab, ``num_kv_heads == 1`` and
        ``head_dim == kv_lora_rank + qk_rope_head_dim``; ``value_cache`` returns the SAME
        tensor, and V is its leading ``kv_lora_rank`` columns.
        """
        ...

    def value_cache(self, layer_id: int) -> torch.Tensor: ...

    def k_scale(self, layer_id: int) -> torch.Tensor | None:
        """Per-layer dequant scale as a DEVICE scalar tensor, or None when not quantized.

        A device tensor, not a Python float, for one reason: a float baked into a kernel
        launch is baked into the capture too, so re-calibrating scales would need a full
        recapture. Read from device memory it is just content, and content is the one thing
        a replay may change.
        """
        ...

    def v_scale(self, layer_id: int) -> torch.Tensor | None: ...

    def store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Scatter this forward's K/V to flat slots. Must be graph-capturable (no host sync,
        no allocation, no ``.item()``)."""
        ...

@runtime_checkable
class PageTableSource(Protocol):
    """Read-only access to the request page table.

    ``gather_rows`` is the ONLY read, and it must return a fresh tensor -- never a view.
    This is not a convenience: the block table a decode step reads must be a SNAPSHOT,
    because the scheduler's next allocation mutates the live table while this step's graph
    is still replaying. Every sparse backend in FreeToken rediscovers this rule
    independently; here it is a property of the port.

    ``column_stride`` exists so a caller that only needs per-PAGE base entries pays for
    ``[R, C/stride]`` and not ``[R, C]``. FreeToken's DSV4 snapshots the full width
    (``[B, stage_width]`` int64 = 4 MiB per request at a 512K context) and its later
    backends had to fix that class of bug by hand with ``page_table[:, ::block_size]``.
    Strided reads are non-coalesced, but the volume drops by ``stride``, which wins by a
    wide margin at any stride that matters.
    """
    @property
    def num_columns(self) -> int:
        """Columns in the live table == max positions a request can address."""
        ...

    @property
    def device(self) -> torch.device: ...

    def gather_rows(
        self,
        table_idx: torch.Tensor,
        *,
        column_stride: int = 1,
        num_columns: int | None = None,
    ) -> torch.Tensor:
        """``[len(table_idx), ceil(num_columns / column_stride)]`` int32 snapshot.

        ``table_idx`` is an int64 device tensor (the scheduler stages it; no host loop).
        """
        ...

@runtime_checkable
class LinearStateCache(Protocol):
    """Recurrent + conv state slots for ``AttnType.LINEAR`` groups.

    Declared here so hybrid models have ONE metadata channel rather than two. FreeToken
    routes GDN layers around the attention backend entirely (``AttnType.LINEAR`` is
    ``backend_driven == False``, with a separate ``FLAMetadata`` builder), which means a
    hybrid forward carries two unrelated metadata objects and the graph runner has to know
    about both. Ayaka keeps the group taxonomy uniform: a LINEAR group still gets a
    ``group_id``, still lands in the bundle, still has a builder -- it just resolves to the
    linear-state backend instead of an attention backend.
    """

    @property
    def device(self) -> torch.device: ...

    def recurrent_state(self, layer_id: int) -> torch.Tensor: ...

    def conv_state(self, layer_id: int) -> torch.Tensor: ...
