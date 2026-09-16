"""Model architecture specifications and Rotary Position Embedding (RoPE) scaling.

This module defines:

* :class:`RopeScalingKind` — supported frequency scaling algorithms for RoPE embeddings
  (linear downscaling, dynamic NTK, YaRN, and Llama-3 frequency bands).
* :class:`RopeScaling` — immutable configuration for sequence length extension.
* :class:`ArchitectureConfig` — canonical configuration record defining transformer,
  Mixture-of-Experts (MoE), and state-space model (Mamba) backbones.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.configs.cache import CacheGroupGeometry, CacheLayerKind


class RopeScalingKind(enum.StrEnum):
    """Supported frequency scaling algorithms for Rotary Position Embeddings (RoPE)."""

    NONE = "none"
    """Standard unscaled RoPE embeddings."""

    LINEAR = "linear"
    """Uniform linear downscaling of rotational frequencies by ``1 / factor``."""

    DYNAMIC = "dynamic"
    """Dynamic Neural Tangent Kernel (NTK-aware) scaling computed at runtime."""

    YARN = "yarn"
    """Yet another RoPE extensioN (YaRN) with ramp-based frequency interpolation."""

    LLAMA3 = "llama3"
    """Meta Llama-3 frequency-banded scaling with low- and high-frequency boundaries."""


@dataclass(frozen=True, slots=True)
class RopeScaling(ConfigMixin):
    """Immutable RoPE frequency scaling parameters for extended sequence lengths.

    Controls position embedding scaling when inferencing contexts beyond the model's
    original pre-training limit.

    Attributes:
        kind: RoPE scaling algorithm to apply. See :class:`RopeScalingKind`.
        factor: Context expansion multiplier. Must be 1.0 when scaling is disabled,
            and strictly greater than 1.0 when scaling is enabled.
        original_max_position: Base training context length prior to extension
            (e.g., 8192 for Llama 3). When set, must be positive.
        low_freq_factor: Low-frequency boundary factor for Llama-3 scaling.
            Must satisfy ``0 < low_freq_factor < high_freq_factor``.
        high_freq_factor: High-frequency boundary factor for Llama-3 scaling.
        beta_fast: Fast wavelength cutoff parameter for YaRN scaling.
        beta_slow: Slow wavelength cutoff parameter for YaRN scaling.
            Must satisfy ``beta_slow < beta_fast``.
    """

    kind: RopeScalingKind = RopeScalingKind.NONE
    factor: float = 1.0
    original_max_position: int | None = None
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    def __post_init__(self) -> None:
        """Validate scaling factor and frequency boundaries for the selected algorithm."""
        if self.kind is RopeScalingKind.NONE:
            if self.factor != 1.0:
                raise ConfigError(
                    "model.rope_scaling.factor",
                    "ROPE_FACTOR_WITHOUT_SCALING",
                    "factor must be 1 when rope scaling is disabled",
                )
            return
        if self.factor <= 1.0:
            raise ConfigError(
                "model.rope_scaling.factor",
                "ROPE_FACTOR_INVALID",
                "enabled rope scaling requires factor > 1",
            )
        if self.original_max_position is not None and self.original_max_position < 1:
            raise ConfigError(
                "model.rope_scaling.original_max_position",
                "ROPE_ORIGINAL_CONTEXT_INVALID",
                "original context length must be positive",
            )
        if self.kind is RopeScalingKind.LLAMA3:
            if not 0 < self.low_freq_factor < self.high_freq_factor:
                raise ConfigError(
                    "model.rope_scaling",
                    "LLAMA3_FREQUENCY_RANGE_INVALID",
                    "llama3 scaling requires 0 < low_freq_factor < high_freq_factor",
                )
        if self.kind is RopeScalingKind.YARN and self.beta_slow >= self.beta_fast:
            raise ConfigError(
                "model.rope_scaling",
                "YARN_BETA_RANGE_INVALID",
                "YaRN requires beta_slow < beta_fast",
            )

    @property
    def enabled(self) -> bool:
        """Whether RoPE position scaling is currently active."""
        return self.kind is not RopeScalingKind.NONE

    def effective_context(self, configured_max_position: int) -> int:
        """Calculate effective context length, avoiding duplicate multiplication.

        Args:
            configured_max_position: Max positions configured in the model checkpoint.

        Returns:
            The effective maximum context length after considering RoPE scaling.
        """
        if not self.enabled or self.original_max_position is None:
            return configured_max_position
        scaled = int(self.original_max_position * self.factor)
        return max(configured_max_position, scaled)


@dataclass(frozen=True, slots=True)
class ArchitectureConfig(ConfigMixin):
    """Immutable architectural specification of an LLM backbone.

    Captures layer counts, projection dimensions, normalization constants,
    attention geometry (MHA, GQA, MQA, MLA), Mixture-of-Experts (MoE) topology,
    and hybrid state-space model (Mamba/SSD) parameters.

    The fields fall into several architectural functional groups:

    **Core Transformer Geometry** — ``hidden_size``, ``num_layers``,
    ``num_attention_heads``, ``num_kv_heads``, ``vocab_size``, ``head_dim``,
    ``max_position_embeddings``.

    **Positional Embeddings & Attention** — ``rope_theta``, ``rope_scaling``,
    ``sliding_window``, ``attention_bias``, ``qkv_bias``.

    **Normalization & Feed-Forward** — ``rms_norm_eps``, ``hidden_act``,
    ``tie_word_embeddings``.

    **Multi-Head Latent Attention (MLA)** — ``kv_lora_rank``, ``q_lora_rank``,
    ``qk_rope_head_dim``, ``qk_nope_head_dim``, ``v_head_dim``.

    **Mixture-of-Experts (MoE)** — ``num_experts``, ``num_experts_per_token``,
    ``moe_intermediate_size``, ``shared_expert_intermediate_size``.

    **Hybrid State-Space Models (Mamba)** — ``layer_types``,
    ``mamba_state_elements_per_layer``, ``mamba_conv_state_elements_per_layer``.

    Attributes:
        hidden_size: Dimensionality of the residual hidden states.
        num_layers: Total number of hidden layers in the model backbone.
        num_attention_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads. Equal to
            ``num_attention_heads`` for MHA; smaller for GQA / MQA.
        vocab_size: Vocabulary size of the embedding and LM head layers.
        head_dim: Dimensionality of each individual attention head.
            Defaults to ``hidden_size // num_attention_heads`` when ``None``.
        max_position_embeddings: Maximum sequence position supported by the
            base architecture. Default is ``4096``.
        rms_norm_eps: Numerical epsilon added to variance in RMSNorm / LayerNorm.
        rope_theta: Base frequency period for Rotary Position Embeddings.
        rope_scaling: Configuration for RoPE sequence length extension.
        sliding_window: Optional sliding window attention window size.
        hidden_act: Non-linear activation function in MLP / FFN layers.
        attention_bias: Whether linear projection layers include additive biases.
        qkv_bias: Whether Q, K, V projection layers include additive biases.
        tie_word_embeddings: Whether input and output embedding weights are tied.
        kv_lora_rank: Low-rank projection dimension for KV compression in MLA.
        q_lora_rank: Low-rank projection dimension for Query compression in MLA.
        qk_rope_head_dim: Decoupled rotary head dimension for Q/K in MLA.
        qk_nope_head_dim: Non-rotary head dimension for Q/K in MLA.
        v_head_dim: Value head dimension in MLA.
        num_experts: Total count of routed experts in MoE layers.
        num_experts_per_token: Number of experts routed to each token.
        moe_intermediate_size: Hidden dimension of routed MoE expert MLPs.
        shared_expert_intermediate_size: Hidden dimension of shared expert MLPs.
        layer_types: Layer-by-layer architectural designations (e.g. attention,
            mamba). Empty tuple implies all layers are standard attention.
        mamba_state_elements_per_layer: Recurrent state element count per Mamba layer.
        mamba_conv_state_elements_per_layer: 1D convolution buffer size per Mamba layer.
        architecture: High-level model architecture family (default ``"toy"``).
    """

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int | None = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: RopeScaling = field(default_factory=RopeScaling)
    sliding_window: int | None = None
    hidden_act: str = "silu"
    attention_bias: bool = False
    qkv_bias: bool = False
    tie_word_embeddings: bool = True
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    num_experts: int | None = None
    num_experts_per_token: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    layer_types: tuple[str, ...] = ()
    mamba_state_elements_per_layer: int | None = None
    mamba_conv_state_elements_per_layer: int | None = None
    explicit_cache_groups: tuple[CacheGroupGeometry, ...] = ()
    architecture: str = "toy"

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_layers",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"model.arch.{name}", "MODEL_DIM_INVALID", f"{name} must be >= 1")
        pure_recurrent = bool(self.layer_types) and all(
            item.lower() in ("mamba", "recurrent") for item in self.layer_types
        )
        if not pure_recurrent and (self.num_attention_heads < 1 or self.num_kv_heads < 1):
            raise ConfigError(
                "model.arch.num_attention_heads",
                "ATTENTION_GEOMETRY_MISSING",
                "models with attention layers require positive attention and KV head counts",
            )
        if pure_recurrent and (self.num_attention_heads < 0 or self.num_kv_heads < 0):
            raise ConfigError(
                "model.arch.num_attention_heads",
                "ATTENTION_HEADS_NEGATIVE",
                "attention head counts must be non-negative",
            )
        if self.num_kv_heads and self.num_attention_heads % self.num_kv_heads:
            raise ConfigError(
                "model.arch.num_kv_heads",
                "GQA_GROUP_INVALID",
                "num_attention_heads must be divisible by num_kv_heads",
            )
        if self.head_dim is not None and self.head_dim < 1:
            raise ConfigError(
                "model.arch.head_dim", "HEAD_DIM_INVALID", "head_dim must be positive"
            )
        if (
            not pure_recurrent
            and self.head_dim is None
            and self.hidden_size % self.num_attention_heads
        ):
            raise ConfigError(
                "model.arch.head_dim",
                "HEAD_DIM_NOT_DERIVABLE",
                "pass head_dim explicitly when hidden size is not divisible by attention heads",
            )
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ConfigError(
                "model.arch",
                "MODEL_NUMERIC_PARAMETER_INVALID",
                "norm epsilon and rope theta must be positive",
            )
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ConfigError(
                "model.arch.sliding_window",
                "SWA_WINDOW_INVALID",
                "sliding_window must be None or >= 1",
            )
        self._validate_mla()
        self._validate_moe()
        if self.layer_types and len(self.layer_types) != self.num_layers:
            raise ConfigError(
                "model.arch.layer_types",
                "LAYER_TYPE_COUNT_MISMATCH",
                "layer_types must contain exactly one entry per layer",
            )
        self._validate_cache_group_coverage(self.cache_groups)

    def _validate_mla(self) -> None:
        values = (
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )
        present = [value is not None and value > 0 for value in values]
        if any(present) and not all(present):
            raise ConfigError(
                "model.arch.mla",
                "MLA_GEOMETRY_PARTIAL",
                "MLA requires kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim and v_head_dim",
            )
        for name in (
            "kv_lora_rank",
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "v_head_dim",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ConfigError(
                    f"model.arch.{name}", "MLA_DIM_NEGATIVE", f"{name} must not be negative"
                )

    def _validate_moe(self) -> None:
        experts = self.num_experts or 0
        if experts:
            if not self.num_experts_per_token or not 1 <= self.num_experts_per_token <= experts:
                raise ConfigError(
                    "model.arch.num_experts_per_token",
                    "MOE_TOPK_INVALID",
                    "experts per token must be in [1, num_experts]",
                )
            if not self.moe_intermediate_size or self.moe_intermediate_size < 1:
                raise ConfigError(
                    "model.arch.moe_intermediate_size",
                    "MOE_INTERMEDIATE_INVALID",
                    "MoE models require a positive expert intermediate size",
                )
        elif any(
            value not in (None, 0)
            for value in (
                self.num_experts_per_token,
                self.moe_intermediate_size,
                self.shared_expert_intermediate_size,
            )
        ):
            raise ConfigError(
                "model.arch.moe",
                "MOE_FIELDS_WITHOUT_EXPERTS",
                "MoE-only fields require num_experts",
            )

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.num_attention_heads < 1:
            raise ConfigError(
                "model.arch.head_dim",
                "HEAD_DIM_NOT_APPLICABLE",
                "pure recurrent models do not have an attention head dimension",
            )
        return self.hidden_size // self.num_attention_heads

    @property
    def is_gqa(self) -> bool:
        return bool(self.num_kv_heads) and self.num_kv_heads < self.num_attention_heads

    @property
    def is_mqa(self) -> bool:
        return self.num_kv_heads == 1

    @property
    def gqa_group_size(self) -> int:
        if self.num_kv_heads < 1:
            raise ConfigError(
                "model.arch.num_kv_heads",
                "GQA_NOT_APPLICABLE",
                "pure recurrent models do not have GQA groups",
            )
        return self.num_attention_heads // self.num_kv_heads

    @property
    def is_moe(self) -> bool:
        return bool(self.num_experts)

    @property
    def uses_mla(self) -> bool:
        return bool(self.kv_lora_rank)

    @property
    def max_context_len(self) -> int:
        return self.rope_scaling.effective_context(self.max_position_embeddings)

    @property
    def cache_groups(self) -> tuple[CacheGroupGeometry, ...]:
        if self.explicit_cache_groups:
            return self.explicit_cache_groups
        kinds = self._normalized_layer_kinds()
        groups: list[CacheGroupGeometry] = []
        for kind in CacheLayerKind:
            indices = tuple(index for index, layer_kind in enumerate(kinds) if layer_kind is kind)
            if not indices:
                continue
            if kind is CacheLayerKind.MAMBA:
                groups.append(
                    CacheGroupGeometry(
                        group_id=kind.value,
                        kind=kind,
                        layer_indices=indices,
                        state_elements_per_layer=self.mamba_state_elements_per_layer or 0,
                        conv_state_elements_per_layer=self.mamba_conv_state_elements_per_layer or 0,
                    )
                )
                continue
            heads = 1 if kind is CacheLayerKind.MLA else self.num_kv_heads
            dim = (
                (self.kv_lora_rank or 0) + (self.qk_rope_head_dim or 0)
                if kind is CacheLayerKind.MLA
                else self.effective_head_dim
            )
            groups.append(
                CacheGroupGeometry(
                    group_id=kind.value,
                    kind=kind,
                    layer_indices=indices,
                    num_kv_heads=heads,
                    head_dim=dim,
                    window_size=self.sliding_window
                    if kind is CacheLayerKind.SLIDING_WINDOW
                    else None,
                )
            )
        return tuple(groups)

    def _normalized_layer_kinds(self) -> tuple[CacheLayerKind, ...]:
        if not self.layer_types:
            if self.uses_mla:
                kind = CacheLayerKind.MLA
            elif self.sliding_window is not None:
                kind = CacheLayerKind.SLIDING_WINDOW
            else:
                kind = CacheLayerKind.FULL_ATTENTION
            return (kind,) * self.num_layers
        aliases = {
            "full": CacheLayerKind.FULL_ATTENTION,
            "full_attention": CacheLayerKind.FULL_ATTENTION,
            "attention": CacheLayerKind.FULL_ATTENTION,
            "sliding": CacheLayerKind.SLIDING_WINDOW,
            "sliding_attention": CacheLayerKind.SLIDING_WINDOW,
            "sliding_window": CacheLayerKind.SLIDING_WINDOW,
            "swa": CacheLayerKind.SLIDING_WINDOW,
            "mla": CacheLayerKind.MLA,
            "mamba": CacheLayerKind.MAMBA,
            "recurrent": CacheLayerKind.MAMBA,
        }
        result: list[CacheLayerKind] = []
        for index, raw in enumerate(self.layer_types):
            try:
                kind = aliases[raw.lower()]
            except KeyError as exc:
                raise ConfigError(
                    f"model.arch.layer_types.{index}",
                    "LAYER_TYPE_UNSUPPORTED",
                    f"unsupported layer type {raw!r}; register a family resolver",
                ) from exc
            if kind is CacheLayerKind.FULL_ATTENTION and self.uses_mla:
                kind = CacheLayerKind.MLA
            result.append(kind)
        return tuple(result)

    def _validate_cache_group_coverage(self, groups: tuple[CacheGroupGeometry, ...]) -> None:
        owners: dict[int, str] = {}
        for group in groups:
            for layer in group.layer_indices:
                if layer >= self.num_layers:
                    raise ConfigError(
                        f"model.cache_groups.{group.group_id}",
                        "CACHE_GROUP_LAYER_OUT_OF_RANGE",
                        f"layer {layer} is outside [0, {self.num_layers})",
                    )
                if layer in owners:
                    raise ConfigError(
                        f"model.cache_groups.{group.group_id}",
                        "CACHE_GROUP_LAYER_DUPLICATE",
                        f"layer {layer} is already owned by group {owners[layer]!r}",
                    )
                owners[layer] = group.group_id
        missing = sorted(set(range(self.num_layers)) - owners.keys())
        if missing:
            raise ConfigError(
                "model.cache_groups",
                "CACHE_GROUP_COVERAGE_INCOMPLETE",
                f"layers without cache/state ownership: {missing}",
            )
