
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.types import AttentionType, KVCacheDtype, KVLayoutKind, MaskKind

if TYPE_CHECKING:
    pass

@dataclass(frozen=True)
class AttentionSpec:
    """Static, per-group attention geometry and behavior.

    Frozen and resolved once at engine init -- NOT a per-call argument. FreeToken threads an
    optional per-call spec through every ``forward`` and then makes four backends raise when
    it is not None; that signature lies about what those backends accept. Here a backend is
    CONSTRUCTED against a spec and ``BackendInfo.check`` refuses at init, so no forward has
    to.

    ``head_dim_qk`` and ``head_dim_vo`` differ for MLA (576 / 512), hence two fields.
    """

    num_qo_heads: int # TP-local
    num_kv_heads: int # TP-local (>= 1; replicated when num_kv_heads < tp_size)
    head_dim_qk: int
    head_dim_vo: int
    sm_scale: float
    mask: MaskKind = MaskKind.CAUSAL
    sliding_window: int | None = None # left context, inclusive of the query token
    logits_soft_cap: float = 0.0  # 0.0 == disabled (Gemma-2 style tanh cap)
    has_sinks: bool = False  # per-head learned logit in the softmax denominator

    def __post_init__(self) -> None:
        if self.num_kv_heads < 1 or self.num_qo_heads % self.num_kv_heads:
            raise ValueError(
                f"num_qo_heads ({self.num_qo_heads}) must be a positive multiple of "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if (self.sliding_window is not None) != (self.mask is MaskKind.SLIDING):
            raise ValueError(
                "sliding_window and MaskKind.SLIDING must be set together "
                f"(window={self.sliding_window}, mask={self.mask})"
            )
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError(f"sliding_window must be >= 1, got {self.sliding_window}")

    @property
    def gqa_group_size(self) -> int:
        return self.num_qo_heads // self.num_kv_heads

    @property
    def is_causal(self) -> bool:
        """Whether a query may attend only positions at or before its own.

        SLIDING is causal *and* bounded on the left; FULL is neither. Backends read this
        rather than comparing against ``MaskKind``, so adding a mask kind does not mean
        editing every kernel's mask branch.
        """
        return self.mask is not MaskKind.FULL

@dataclass(frozen=True)
class AttentionGroupSpec:
    """One KV-cache group: the layers that share a pool, and how it is addressed.

    Hybrid models are the default case, not the exception (Qwen3-Next: 36 GDN + 12 full;
    Gemma-2: alternating SWA/full), so metadata, backend selection and graph buffers are all
    keyed by ``group_id`` from the start.
    """

    group_id: int
    layer_ids: tuple[int, ...]
    attn_type: AttentionType
    spec: AttentionSpec
    page_size: int
    kv_layout: KVLayoutKind = KVLayoutKind.NHD
    kv_cache_dtype: KVCacheDtype = KVCacheDtype.AUTO
    mla: MLAExtras | None = None

    def __post_init__(self) -> None:
        if not self.layer_ids:
            raise ValueError(f"attention group {self.group_id} has no layers")
        if self.page_size < 1 or (self.page_size & (self.page_size - 1)):
            raise ValueError(
                f"page_size must be a power of two, got {self.page_size} "
                f"(group {self.group_id})"
            )
        if self.attn_type is AttentionType.MLA:
            if self.mla is None:
                raise ValueError(f"group {self.group_id} is MLA but carries no MLAExtras")
            if self.spec.num_kv_heads != 1:
                raise ValueError(
                    "MLA stores one latent row per token, so num_kv_heads must be 1, got "
                    f"{self.spec.num_kv_heads} (group {self.group_id})"
                )
            if self.spec.head_dim_qk != self.mla.latent_dim:
                raise ValueError(
                    "MLA head_dim_qk must equal kv_lora_rank + qk_rope_head_dim "
                    f"({self.mla.latent_dim}), got {self.spec.head_dim_qk}"
                )
            if self.spec.head_dim_vo != self.mla.kv_lora_rank:
                raise ValueError(
                    f"MLA head_dim_vo must equal kv_lora_rank ({self.mla.kv_lora_rank}), "
                    f"got {self.spec.head_dim_vo}"
                )
        elif self.mla is not None:
            raise ValueError(
                f"group {self.group_id} carries MLAExtras but is {self.attn_type.value}"
            )

@dataclass(frozen=True)
class MLAExtras:
    """Latent-KV geometry, present exactly when ``AttnType.MLA``.

    The pool holds ONE latent row per token, ``kv_lora_rank + qk_rope_head_dim`` wide, with a
    single KV head. The model absorbs ``W_UK`` into Q and ``W_UV`` onto the output, so at
    decode there is no separate V tensor: V is the leading ``kv_lora_rank`` columns of the
    same row. That aliasing is why MLA decode is cheap, and it is a property of the LAYOUT,
    so it belongs in the protocol rather than inside one backend.
    """

    kv_lora_rank: int  # 512 for DeepSeek-V3
    qk_rope_head_dim: int  # 64 for DeepSeek-V3

    @property
    def latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim
