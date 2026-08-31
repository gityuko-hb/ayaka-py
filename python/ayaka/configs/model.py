from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ayaka.configs.base import ConfigError, ConfigMixin
from ayaka.types import DType
from sympy import true

class RopeScalingKind(enum.StrEnum):
    NONE = "none"
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"
    
@dataclass(frozen=True, slots=True)
class RopeScaling(ConfigMixin):
    kind: RopeScalingKind = RopeScalingKind.NONE
    factor: float = 1.0
    original_max_position: int | None = None
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    
    def __post_init__(self) -> None:
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
        return self.kind is not RopeScalingKind.NONE

    def effective_context(self, configured_max_position: int) -> int:
        """Avoid multiplying an already-extended HF max position a second time."""

        if not self.enabled or self.original_max_position is None:
            return configured_max_position
        scaled = int(self.original_max_position * self.factor)
        return max(configured_max_position, scaled)


@dataclass(frozen=True, slots=True)
class ArchitectureConfig(ConfigMixin):
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