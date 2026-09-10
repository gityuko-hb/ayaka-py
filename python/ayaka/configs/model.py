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
    architecture: str = "toy"
