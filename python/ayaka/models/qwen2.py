"""Inference-only Qwen2/Qwen2.5 compatible with HuggingFace ``Qwen2ForCausalLM``.

The decoder tree is the LLaMA-style ``model.*`` layout, so the model reuses the
llama decoder modules; Qwen2's own semantics — q/k/v bias without an o bias,
no dynamic NTK/logn, full-rotation RoPE at the checkpoint theta — are carried
by :class:`Qwen2Config`. Sliding-window attention and rope-scaling variants
stay rejected: the certified resident lane is full retention (Qwen2.5
checkpoints ship ``use_sliding_window: false`` and no ``rope_scaling``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from ayaka.layers.linear.methods import LinearMethodBase
from ayaka.layers.quantization.base import BaseQuantization
from ayaka.model_loader.mapping import ExternMapping
from ayaka.model_loader.module import WeightBinding
from ayaka.models._common import (
    ROTARY_CACHE_KEYS,
    DecoderModule,
    load_checkpoint,
    weight_spec,
    write_checkpoint,
)
from ayaka.models.llama import LlamaConfig, LlamaForCausalLM
from ayaka.types import DType
from ayaka.weights.spec import TensorSpec, WeightSpec

__all__ = [
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "load_qwen2_weights",
    "qwen2_expected_weights",
    "qwen2_weight_bindings",
    "qwen2_weight_mapping",
    "write_qwen2_checkpoint",
]

QuantConfig = BaseQuantization | LinearMethodBase | str | None


@dataclass(frozen=True, slots=True)
class Qwen2Config(LlamaConfig):
    """HF ``Qwen2Config`` fields on the shared llama decoder geometry.

    ``from_dict`` keeps every inference-irrelevant field the checkpoint ships
    (``max_window_layers``, ``sliding_window``, dropout) so ``config.json``
    round-trips through :meth:`arch_dict`.
    """

    model_type: ClassVar[str] = "qwen2"

    vocab_size: int = 151936
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    qkv_bias: bool = True
    use_sliding_window: bool = False

    def __post_init__(self) -> None:
        LlamaConfig.__post_init__(self)
        if self.use_sliding_window:
            raise ValueError(
                "qwen2 sliding-window attention is outside the certified resident lane; "
                "checkpoints with use_sliding_window=true are rejected"
            )
        if self.rope_scaling is not None:
            raise ValueError(
                "qwen2 rope_scaling variants are not supported; "
                "the certified lane runs the checkpoint theta directly"
            )

    @property
    def qkv_bias_enabled(self) -> bool:
        """Qwen2 biases q/k/v unconditionally."""
        return self.qkv_bias or self.attention_bias_enabled

    @property
    def o_bias_enabled(self) -> bool:
        """Qwen2 has no o-projection bias."""
        return self.attention_bias_enabled

    def resolved_dtype(self) -> DType:
        """Qwen2 checkpoints are bf16 unless the checkpoint states otherwise."""
        if self.torch_dtype:
            return DType.from_str(self.torch_dtype)
        return DType.BF16


def qwen2_expected_weights(
    config: Qwen2Config,
    tensor_dtype: DType = DType.BF16,
) -> tuple[WeightSpec, ...]:
    """Every tensor an HF Qwen2 checkpoint must provide, in HF naming order."""

    hidden = config.hidden_size
    q_width = config.num_attention_heads * config.resolved_head_dim
    kv_width = config.num_kv_heads * config.resolved_head_dim
    inner = config.intermediate_size
    vocab = config.vocab_size

    def spec(name: str, shape: tuple[int, ...]) -> WeightSpec:
        return weight_spec(name, shape, tensor_dtype)

    specs: list[WeightSpec] = [spec("model.embed_tokens.weight", (vocab, hidden))]
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        specs += [
            spec(f"{prefix}.input_layernorm.weight", (hidden,)),
            spec(f"{prefix}.self_attn.q_proj.weight", (q_width, hidden)),
            spec(f"{prefix}.self_attn.k_proj.weight", (kv_width, hidden)),
            spec(f"{prefix}.self_attn.v_proj.weight", (kv_width, hidden)),
            spec(f"{prefix}.self_attn.o_proj.weight", (hidden, q_width)),
            spec(f"{prefix}.post_attention_layernorm.weight", (hidden,)),
            spec(f"{prefix}.mlp.gate_proj.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.up_proj.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.down_proj.weight", (hidden, inner)),
        ]
        if config.qkv_bias_enabled:
            specs += [
                spec(f"{prefix}.self_attn.q_proj.bias", (q_width,)),
                spec(f"{prefix}.self_attn.k_proj.bias", (kv_width,)),
                spec(f"{prefix}.self_attn.v_proj.bias", (kv_width,)),
            ]
        if config.o_bias_enabled:
            specs.append(spec(f"{prefix}.self_attn.o_proj.bias", (hidden,)))
    specs.append(spec("model.norm.weight", (hidden,)))
    specs.append(
        WeightSpec(
            name="lm_head.weight",
            full_shape=(vocab, hidden),
            spec=TensorSpec(shape=(vocab, hidden), dtype=tensor_dtype, name="lm_head.weight"),
            tied_alias="model.embed_tokens.weight" if config.tie_word_embeddings else "",
        )
    )
    return tuple(specs)


def qwen2_weight_bindings(config: Qwen2Config) -> dict[str, WeightBinding]:
    """Checkpoint key -> module tensor, with q/k/v and gate/up logical shards."""

    bindings: dict[str, WeightBinding] = {
        "model.embed_tokens.weight": WeightBinding("model.embed_tokens.weight"),
        "model.norm.weight": WeightBinding("model.norm.weight"),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        bindings.update(
            {
                f"{prefix}.input_layernorm.weight": WeightBinding(
                    f"{prefix}.input_layernorm.weight"
                ),
                f"{prefix}.self_attn.q_proj.weight": WeightBinding(
                    f"{prefix}.self_attn.qkv_proj.weight", "q"
                ),
                f"{prefix}.self_attn.k_proj.weight": WeightBinding(
                    f"{prefix}.self_attn.qkv_proj.weight", "k"
                ),
                f"{prefix}.self_attn.v_proj.weight": WeightBinding(
                    f"{prefix}.self_attn.qkv_proj.weight", "v"
                ),
                f"{prefix}.self_attn.o_proj.weight": WeightBinding(
                    f"{prefix}.self_attn.o_proj.weight"
                ),
                f"{prefix}.post_attention_layernorm.weight": WeightBinding(
                    f"{prefix}.post_attention_layernorm.weight"
                ),
                f"{prefix}.mlp.gate_proj.weight": WeightBinding(
                    f"{prefix}.mlp.gate_up_proj.weight", "gate"
                ),
                f"{prefix}.mlp.up_proj.weight": WeightBinding(
                    f"{prefix}.mlp.gate_up_proj.weight", "up"
                ),
                f"{prefix}.mlp.down_proj.weight": WeightBinding(f"{prefix}.mlp.down_proj.weight"),
            }
        )
        if config.qkv_bias_enabled:
            bindings.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": WeightBinding(
                        f"{prefix}.self_attn.qkv_proj.bias", "q"
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightBinding(
                        f"{prefix}.self_attn.qkv_proj.bias", "k"
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightBinding(
                        f"{prefix}.self_attn.qkv_proj.bias", "v"
                    ),
                }
            )
        if config.o_bias_enabled:
            bindings[f"{prefix}.self_attn.o_proj.bias"] = WeightBinding(
                f"{prefix}.self_attn.o_proj.bias"
            )
    if not config.tie_word_embeddings:
        bindings["lm_head.weight"] = WeightBinding("lm_head.weight")
    return bindings


def qwen2_weight_mapping(config: Qwen2Config) -> ExternMapping:
    """Declarative ExternMapping: fuse q/k/v and gate/up, whitelist rotary caches."""

    mapping = ExternMapping()
    mapping.add_mapping("model.embed_tokens.weight", "model.embed_tokens.weight")
    mapping.add_mapping("model.norm.weight", "model.norm.weight")
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        mapping.add_mapping(f"{prefix}.input_layernorm.weight", f"{prefix}.input_layernorm.weight")
        mapping.add_mapping(
            f"{prefix}.self_attn.qkv_proj.weight",
            [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ],
            func=lambda q, k, v: torch.cat([q, k, v], dim=0),
        )
        mapping.add_mapping(
            f"{prefix}.self_attn.o_proj.weight", f"{prefix}.self_attn.o_proj.weight"
        )
        mapping.add_mapping(
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        )
        mapping.add_mapping(
            f"{prefix}.mlp.gate_up_proj.weight",
            [f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"],
            func=lambda gate, up: torch.cat([gate, up], dim=0),
        )
        mapping.add_mapping(f"{prefix}.mlp.down_proj.weight", f"{prefix}.mlp.down_proj.weight")
        if config.qkv_bias_enabled:
            mapping.add_mapping(
                f"{prefix}.self_attn.qkv_proj.bias",
                [
                    f"{prefix}.self_attn.q_proj.bias",
                    f"{prefix}.self_attn.k_proj.bias",
                    f"{prefix}.self_attn.v_proj.bias",
                ],
                func=lambda q, k, v: torch.cat([q, k, v], dim=0),
            )
        if config.o_bias_enabled:
            mapping.add_mapping(
                f"{prefix}.self_attn.o_proj.bias", f"{prefix}.self_attn.o_proj.bias"
            )
        for suffix in ROTARY_CACHE_KEYS.suffixes:
            mapping.add_unused(f"{prefix}.self_attn.{suffix}")
    if not config.tie_word_embeddings:
        mapping.add_mapping("lm_head.weight", "lm_head.weight")
    return mapping


def load_qwen2_weights(
    model: Qwen2ForCausalLM,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    validate: bool = True,
    use_mapping: bool = False,
) -> frozenset[str]:
    """Load an HF Qwen2/Qwen2.5 safetensors checkpoint with the bounded reader."""

    config = model.config
    assert isinstance(config, Qwen2Config)
    bindings: dict[str, WeightBinding] | ExternMapping = (
        qwen2_weight_mapping(config) if use_mapping else qwen2_weight_bindings(config)
    )
    return load_checkpoint(
        model,
        checkpoint_dir,
        expected=lambda dtype: qwen2_expected_weights(config, dtype),
        bindings=bindings,
        device=device,
        validate=validate,
    )


def write_qwen2_checkpoint(
    root: Path,
    config: Qwen2Config,
    *,
    seed: int = 1234,
) -> Path:
    """Write a deterministic checkpoint in HF Qwen2 naming order."""

    return write_checkpoint(
        root,
        architectures="Qwen2ForCausalLM",
        config=config,
        specs=qwen2_expected_weights(config, config.resolved_dtype()),
        unit_weight_suffixes=("norm.weight",),
        seed=seed,
    )


class Qwen2ForCausalLM(LlamaForCausalLM):
    """Dense Qwen2 decoder over the shared llama module tree.

    Attention stays a runtime responsibility through the per-forward callback.
    """

    def __init__(
        self,
        config: Qwen2Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: str = "triton",
        quant_config: QuantConfig = None,
    ) -> None:
        super().__init__(
            config,
            device=device,
            dtype=dtype,
            backend=backend,  # type: ignore[arg-type]
            quant_config=quant_config,
        )

    @property
    def transformer(self) -> DecoderModule:
        """Alias for runtimes that reach the decoder as ``transformer``."""
        return self._decoder()
