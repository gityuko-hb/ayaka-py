from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from ayaka.attention.metadata import BaseAttentionMetadata, BaseAttentionMetadataBuilder
from ayaka.attention.spec import AttentionGroupSpec
from ayaka.types import AttentionType, KVLayoutKind

if TYPE_CHECKING:
    import torch

    from .ports import PagedKVCache


class BaseAttentionBackend(ABC):
    """Kernel execution for one attention group.

    Holds no per-forward state. Everything a kernel needs arrives through ``metadata``, so
    two forwards can be in flight on two streams without the backend being the shared
    mutable thing between them.
    """

    #: Registry name; set by @register_backend.
    name: ClassVar[str] = ""

    def __init__(
        self,
        group: AttentionGroupSpec,
        kv_cache: PagedKVCache,
        device: torch.device,
    ) -> None:
        self.group = group
        self.spec = group.spec
        self.kv_cache = kv_cache
        self.device = device

    @abstractmethod
    def build_metadata_builder(self) -> BaseAttentionMetadataBuilder:
        """The builder paired with this backend. One per group."""

    @abstractmethod
    def forward(
        self,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: BaseAttentionMetadata,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense attention over the paged pool.

        ``layer_id`` is GROUP-LOCAL. ``query`` is ``[num_tokens, num_qo_heads, head_dim_qk]``
        and ``key``/``value`` are ``[num_tokens, num_kv_heads, head_dim]`` -- flat and
        ragged, never padded to ``[batch, seq, ...]``. Padding a ragged batch to rectangular
        costs both the memset and the wasted MACs, and the indptr form is what every modern
        kernel takes anyway.

        The backend is responsible for storing K/V into the pool before attending; the
        caller does not do it, because a backend that fuses the store into its attention
        kernel must be free to.

        ``output``, when given, is written in place -- this is what lets a captured graph
        keep one output address across replays.
        """

    def get_kv_cache_shape(self, num_pages: int) -> tuple[int, ...]:
        """Physical pool shape this backend addresses, in the group's ``kv_layout``."""
        s = self.spec
        if self.group.attn_type is AttentionType.MLA:
            # One latent row per token, one head. Not K and V.
            return (num_pages, self.group.page_size, 1, s.head_dim_qk)
        if self.group.kv_layout is KVLayoutKind.HND:
            return (num_pages, s.num_kv_heads, self.group.page_size, s.head_dim_qk)
        return (num_pages, self.group.page_size, s.num_kv_heads, s.head_dim_qk)

    def kv_bytes_per_token(self, model_dtype: Any) -> int:
        """Bytes of KV pool one token of this group consumes on THIS rank.

        The number ``KVBudgetSpec.bytes_per_token_per_rank`` is built from, and the reason
        backend selection has to run BEFORE capacity planning: the physical shape is a
        backend property (does it store K and V separately, or one latent row?), and the
        element size is a quantization property. A planner that guesses either sizes the
        pool wrong, and sizing a KV pool wrong is not a rounding error -- it is either an
        OOM at the first long request or permanently stranded HBM.
        """
        from ayaka.utils.torch_utils import dtype_bytes

        s = self.spec
        element = self.group.kv_cache_dtype.element_bytes
        if element is None:
            element = dtype_bytes(model_dtype)
        if self.group.attn_type is AttentionType.MLA:
            return s.head_dim_qk * element  # one latent row, one head, no separate V
        return (s.head_dim_qk + s.head_dim_vo) * s.num_kv_heads * element


@runtime_checkable
class MLAAttentionBackend(Protocol):
    """Latent-KV MLA entry point.

    Separate from ``AttentionBackend.forward`` because the shapes genuinely differ: the model
    absorbs ``kv_b`` into Q and onto the output, so there is no V tensor to pass and the pool
    holds one ``kv_lora_rank + qk_rope_head_dim`` latent row per token rather than K and V.

    Unserved (Llama/Qwen2 are GQA). Declared now so that adding DeepSeek-class models
    is a new backend, not a change to the contract every existing backend implements.
    """

    def forward_mla(
        self,
        layer_id: int,
        q_nope: torch.Tensor,  # [T, H, kv_lora_rank] (kv_b absorbed)
        q_pe: torch.Tensor,  # [T, H, qk_rope_head_dim]
        latent_kv: torch.Tensor,  # [T, kv_lora_rank]
        k_pe: torch.Tensor,  # [T, qk_rope_head_dim]
        metadata: BaseAttentionMetadata,
    ) -> torch.Tensor: ...
