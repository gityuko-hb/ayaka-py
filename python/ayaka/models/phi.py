"""Inference-only Phi-1.5/Phi-2 compatible with HF ``PhiForCausalLM``.

Reference layout credited to vLLM (Apache-2.0). Phi's decoder layer is
*parallel*: attention and the feed-forward network both consume the same
``input_layernorm`` output and are added back to the residual, unlike
Llama/Qwen's serial ordering. Rotary embedding is applied to the first
``partial_rotary_factor`` slice of every head in NeoX (split-half) style, and
LayerNorm carries both weight and bias. Tensor parallelism, checkpoint schema
and the causal-LM outer contract come from ``ayaka.layers`` and
``ayaka.models._common``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from ayaka.layers._common import LayerBackend
from ayaka.layers.activation import get_act_fn
from ayaka.layers.linear.attention import QKVParallelLinear
from ayaka.layers.linear.core import ColumnParallelLinear, RowParallelLinear
from ayaka.layers.norm import LayerNorm
from ayaka.layers.rotary_embedding import get_rope
from ayaka.model_loader.mapping import ExternMapping
from ayaka.model_loader.module import WeightBinding
from ayaka.models._common import (
    ROTARY_CACHE_KEYS,
    AttentionCallback,
    CausalLM,
    DecoderModule,
    ModelConfigMixin,
    load_checkpoint,
    resolve_layer_range,
    weight_spec,
    write_checkpoint,
)
from ayaka.types import DType
from ayaka.utils.validation import require_int
from ayaka.weights.spec import TensorSpec, WeightSpec

__all__ = [
    "AttentionCallback",
    "PhiConfig",
    "PhiForCausalLM",
    "load_phi_weights",
    "phi_expected_weights",
    "phi_weight_bindings",
    "phi_weight_mapping",
    "write_phi_checkpoint",
]

_SUPPORTED_ACTIVATIONS = frozenset(
    {"gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh", "gelu_quick", "quick_gelu"}
)

_CONFIG_ALIASES: dict[str, str] = {
    "n_positions": "max_position_embeddings",
    "n_inner": "intermediate_size",
}


@dataclass(frozen=True, slots=True)
class PhiConfig(ModelConfigMixin):
    """HF ``PhiConfig`` fields.

    Inference-irrelevant dropout fields are preserved so a checkpoint's
    ``config.json`` round-trips through the inherited :meth:`from_dict` /
    :meth:`arch_dict`, which also accepts the legacy ``n_positions`` /
    ``n_inner`` aliases and the modern bundled ``rope_parameters`` mapping.
    """

    model_type: ClassVar[str] = "phi"
    _CONFIG_ALIASES: ClassVar[Mapping[str, str]] = _CONFIG_ALIASES

    vocab_size: int = 51200
    hidden_size: int = 2048
    intermediate_size: int | None = None
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "gelu_new"
    max_position_embeddings: int = 2048
    partial_rotary_factor: float = 0.5
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Mapping[str, Any] | None = None
    qk_layernorm: bool = False
    initializer_range: float = 0.02
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attention_dropout: float = 0.0
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
        ):
            require_int(getattr(self, name), name, minimum=1)
        if self.intermediate_size is not None:
            require_int(self.intermediate_size, "intermediate_size", minimum=1)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_key_value_heads is not None:
            require_int(self.num_key_value_heads, "num_key_value_heads", minimum=1)
            if self.num_attention_heads % self.num_key_value_heads:
                raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_act not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported hidden_act {self.hidden_act!r}; "
                f"expected one of {sorted(_SUPPORTED_ACTIVATIONS)}"
            )
        if self.qk_layernorm:
            raise ValueError("qk_layernorm is not supported")
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be finite and positive")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.partial_rotary_factor) or not 0 < self.partial_rotary_factor <= 1:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        if self.rotary_dim % 2 or not 2 <= self.rotary_dim <= self.head_dim:
            raise ValueError("rotary_dim must be even and within [2, head_dim]")
        if abs(self.head_dim * self.partial_rotary_factor - self.rotary_dim) > 1e-9:
            raise ValueError("head_dim * partial_rotary_factor must be a whole number of dims")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        if self.num_key_value_heads is not None:
            return self.num_key_value_heads
        return self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def mlp_inner_size(self) -> int:
        """HF ``intermediate_size``, falling back to the canonical ``4 * hidden``."""
        if self.intermediate_size is not None:
            return self.intermediate_size
        return 4 * self.hidden_size

    @property
    def scaling(self) -> float:
        return self.head_dim**-0.5


def phi_expected_weights(
    config: PhiConfig,
    tensor_dtype: DType = DType.FP32,
) -> tuple[WeightSpec, ...]:
    """Every tensor an HF Phi checkpoint must provide, in stable naming order.

    Shapes are global; ``WeightSpec.shard`` stays at its default because the
    expected schema validates names/shapes/dtypes only. ``q_proj``/``k_proj``/
    ``v_proj`` stay separate here — :func:`phi_weight_bindings` packs them into
    ``qkv_proj`` and :func:`phi_weight_mapping` concatenates them.
    """
    hidden = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_kv_heads * config.head_dim
    inner = config.mlp_inner_size
    vocab = config.vocab_size

    def spec(name: str, shape: tuple[int, ...]) -> WeightSpec:
        return weight_spec(name, shape, tensor_dtype)

    specs: list[WeightSpec] = [spec("model.embed_tokens.weight", (vocab, hidden))]
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        specs += [
            spec(f"{prefix}.input_layernorm.weight", (hidden,)),
            spec(f"{prefix}.input_layernorm.bias", (hidden,)),
            spec(f"{prefix}.self_attn.q_proj.weight", (q_width, hidden)),
            spec(f"{prefix}.self_attn.q_proj.bias", (q_width,)),
            spec(f"{prefix}.self_attn.k_proj.weight", (kv_width, hidden)),
            spec(f"{prefix}.self_attn.k_proj.bias", (kv_width,)),
            spec(f"{prefix}.self_attn.v_proj.weight", (kv_width, hidden)),
            spec(f"{prefix}.self_attn.v_proj.bias", (kv_width,)),
            spec(f"{prefix}.self_attn.dense.weight", (hidden, q_width)),
            spec(f"{prefix}.self_attn.dense.bias", (hidden,)),
            spec(f"{prefix}.mlp.fc1.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.fc1.bias", (inner,)),
            spec(f"{prefix}.mlp.fc2.weight", (hidden, inner)),
            spec(f"{prefix}.mlp.fc2.bias", (hidden,)),
        ]
    specs.append(spec("model.final_layernorm.weight", (hidden,)))
    specs.append(spec("model.final_layernorm.bias", (hidden,)))
    specs.append(
        WeightSpec(
            name="lm_head.weight",
            full_shape=(vocab, hidden),
            spec=TensorSpec(shape=(vocab, hidden), dtype=tensor_dtype, name="lm_head.weight"),
            tied_alias="model.embed_tokens.weight" if config.tie_word_embeddings else "",
        )
    )
    specs.append(spec("lm_head.bias", (vocab,)))
    return tuple(specs)


def phi_weight_bindings(config: PhiConfig) -> dict[str, WeightBinding]:
    """Checkpoint key -> module tensor, with logical shard ids for packed QKV.

    The checkpoint stores ``q_proj``/``k_proj``/``v_proj`` (weight and bias)
    separately; each binds to the matching slice of the packed ``qkv_proj``
    parameter, mirroring vLLM's ``stacked_params_mapping``.
    """
    bindings: dict[str, WeightBinding] = {
        "model.embed_tokens.weight": WeightBinding("model.embed_tokens.weight"),
        "model.final_layernorm.weight": WeightBinding("model.final_layernorm.weight"),
        "model.final_layernorm.bias": WeightBinding("model.final_layernorm.bias"),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        bindings[f"{prefix}.input_layernorm.weight"] = WeightBinding(
            f"{prefix}.input_layernorm.weight"
        )
        bindings[f"{prefix}.input_layernorm.bias"] = WeightBinding(f"{prefix}.input_layernorm.bias")
        for shard in ("q", "k", "v"):
            bindings[f"{prefix}.self_attn.{shard}_proj.weight"] = WeightBinding(
                f"{prefix}.self_attn.qkv_proj.weight", shard
            )
            bindings[f"{prefix}.self_attn.{shard}_proj.bias"] = WeightBinding(
                f"{prefix}.self_attn.qkv_proj.bias", shard
            )
        bindings[f"{prefix}.self_attn.dense.weight"] = WeightBinding(
            f"{prefix}.self_attn.dense.weight"
        )
        bindings[f"{prefix}.self_attn.dense.bias"] = WeightBinding(f"{prefix}.self_attn.dense.bias")
        bindings[f"{prefix}.mlp.fc1.weight"] = WeightBinding(f"{prefix}.mlp.fc1.weight")
        bindings[f"{prefix}.mlp.fc1.bias"] = WeightBinding(f"{prefix}.mlp.fc1.bias")
        bindings[f"{prefix}.mlp.fc2.weight"] = WeightBinding(f"{prefix}.mlp.fc2.weight")
        bindings[f"{prefix}.mlp.fc2.bias"] = WeightBinding(f"{prefix}.mlp.fc2.bias")
    if not config.tie_word_embeddings:
        bindings["lm_head.weight"] = WeightBinding("lm_head.weight")
    bindings["lm_head.bias"] = WeightBinding("lm_head.bias")
    return bindings


def phi_weight_mapping(config: PhiConfig) -> ExternMapping:
    """Declarative ExternMapping for Phi checkpoints.

    Maps 1-1 tensors, concatenates q/k/v into the packed ``qkv_proj`` weight and
    bias, and whitelists rotary embedding frequency tensors.
    """
    mapping = ExternMapping()
    mapping.add_mapping("model.embed_tokens.weight", "model.embed_tokens.weight")
    mapping.add_mapping("model.final_layernorm.weight", "model.final_layernorm.weight")
    mapping.add_mapping("model.final_layernorm.bias", "model.final_layernorm.bias")
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        mapping.add_mapping(f"{prefix}.input_layernorm.weight", f"{prefix}.input_layernorm.weight")
        mapping.add_mapping(f"{prefix}.input_layernorm.bias", f"{prefix}.input_layernorm.bias")
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
            f"{prefix}.self_attn.qkv_proj.bias",
            [
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ],
            func=lambda q, k, v: torch.cat([q, k, v], dim=0),
        )
        mapping.add_mapping(f"{prefix}.self_attn.dense.weight", f"{prefix}.self_attn.dense.weight")
        mapping.add_mapping(f"{prefix}.self_attn.dense.bias", f"{prefix}.self_attn.dense.bias")
        mapping.add_mapping(f"{prefix}.mlp.fc1.weight", f"{prefix}.mlp.fc1.weight")
        mapping.add_mapping(f"{prefix}.mlp.fc1.bias", f"{prefix}.mlp.fc1.bias")
        mapping.add_mapping(f"{prefix}.mlp.fc2.weight", f"{prefix}.mlp.fc2.weight")
        mapping.add_mapping(f"{prefix}.mlp.fc2.bias", f"{prefix}.mlp.fc2.bias")
        for suffix in ROTARY_CACHE_KEYS.suffixes:
            mapping.add_unused(f"{prefix}.self_attn.{suffix}")

    if not config.tie_word_embeddings:
        mapping.add_mapping("lm_head.weight", "lm_head.weight")
    mapping.add_mapping("lm_head.bias", "lm_head.bias")
    return mapping


def load_phi_weights(
    model: PhiForCausalLM,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    validate: bool = True,
    use_mapping: bool = False,
) -> frozenset[str]:
    """Load a safetensors Phi checkpoint with the production bounded reader.

    Selected q/k/v tensors stream straight into the packed ``qkv_proj`` slices;
    ``use_mapping=True`` instead concatenates them through the declarative
    :class:`ExternMapping`. Rotary caches some exports ship are skipped: RoPE
    is computed by this module. Missing required tensors and shape mismatches
    raise; unknown tensors are never bound.
    """
    bindings: Mapping[str, WeightBinding] | ExternMapping = (
        phi_weight_mapping(model.config) if use_mapping else phi_weight_bindings(model.config)
    )
    return load_checkpoint(
        model,
        checkpoint_dir,
        expected=lambda dtype: phi_expected_weights(model.config, dtype),
        bindings=bindings,
        ignored=ROTARY_CACHE_KEYS,
        device=device,
        validate=validate,
    )


def write_phi_checkpoint(
    root: Path,
    config: PhiConfig,
    *,
    seed: int = 1234,
) -> Path:
    """Write a deterministic checkpoint in HF Phi naming order."""
    return write_checkpoint(
        root,
        architectures="PhiForCausalLM",
        config=config,
        specs=phi_expected_weights(config, config.resolved_dtype()),
        unit_weight_suffixes=("input_layernorm.weight", "final_layernorm.weight"),
        zero_bias_suffixes=("input_layernorm.bias", "final_layernorm.bias"),
        seed=seed,
    )


class _PhiAttention(nn.Module):
    def __init__(self, config: PhiConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.qkv_proj = QKVParallelLinear(
            hidden,
            self.head_dim,
            config.num_attention_heads,
            config.num_kv_heads,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.dense = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            hidden,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.rotary = get_rope(
            self.head_dim,
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta,
            True,
            rope_scaling=config.rope_scaling,
            device=device,
            dtype=dtype,
            backend=backend,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionCallback,
        layer_index: int,
    ) -> torch.Tensor:
        tokens = hidden.shape[0]
        packed = self.qkv_proj(hidden)
        assert isinstance(packed, torch.Tensor)
        parts = self.qkv_proj.split_output(packed)
        query = parts["q"].view(tokens, self.num_heads, self.head_dim)
        key = parts["k"].view(tokens, self.num_kv_heads, self.head_dim)
        value = parts["v"].view(tokens, self.num_kv_heads, self.head_dim)
        self.rotary(positions, query, key)
        output = attention(layer_index, query, key, value)
        projected = self.dense(output.reshape(tokens, -1))
        assert isinstance(projected, torch.Tensor)
        return projected


class _PhiMLP(nn.Module):
    def __init__(self, config: PhiConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.mlp_inner_size
        self.fc1 = ColumnParallelLinear(
            hidden,
            inner,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.fc2 = RowParallelLinear(
            inner,
            hidden,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.activation = get_act_fn(config.hidden_act, backend=backend)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.fc1(hidden)
        assert isinstance(projected, torch.Tensor)
        return self.fc2(self.activation(projected))


class _PhiBlock(nn.Module):
    def __init__(self, config: PhiConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.input_layernorm = LayerNorm(
            hidden,
            eps=config.layer_norm_eps,
            bias=True,
            device=device,
            dtype=dtype,
            backend=backend,
        )
        self.self_attn = _PhiAttention(config, device=device, dtype=dtype, backend=backend)
        self.mlp = _PhiMLP(config, device=device, dtype=dtype, backend=backend)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionCallback,
        layer_index: int,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        attended = self.self_attn(normalized, positions, attention, layer_index)
        return attended + self.mlp(normalized) + hidden


class _PhiModel(nn.Module):
    def __init__(self, config: PhiConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                _PhiBlock(config, device=device, dtype=dtype, backend=backend)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            bias=True,
            device=device,
            dtype=dtype,
            backend=backend,
        )

    @property
    def wte(self) -> nn.Embedding:
        """Alias for runtimes that embed tokens through ``transformer.wte``."""
        return self.embed_tokens

    def forward_hidden(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionCallback,
        *,
        skip_embed: bool = False,
        inputs_embeds: torch.Tensor | None = None,
        layer_start: int = 0,
        layer_end: int | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a half-open layer range of the decoder.

        ``layer_start``/``layer_end`` slice the transformer for one PP stage. A
        non-zero ``layer_start`` consumes the previous stage's ``state`` (the
        boundary hidden tensor) instead of token ids, and a ``layer_end`` below
        the layer count returns that boundary tensor so the next stage can
        resume. Phi's LayerNorm does not fuse the residual, so the boundary is a
        single tensor rather than Qwen's ``(hidden, residual)`` pair. Layer
        indices stay GLOBAL so attention callbacks index KV slots exactly as a
        full forward does; the final ``final_layernorm`` only runs on the last
        stage.

        ``skip_embed`` marks the first argument as a complete hidden state;
        ``inputs_embeds`` replaces the token lookup instead, which is how the
        paged runner supplies multimodal token embeddings.
        """
        num_layers = len(self.layers)
        end = resolve_layer_range(num_layers, layer_start, layer_end)
        if layer_start == 0:
            if state is not None:
                raise ValueError("layer_start 0 must not carry a boundary state")
            if skip_embed and inputs_embeds is not None:
                raise ValueError("skip_embed and inputs_embeds are mutually exclusive")
            if skip_embed:
                hidden = token_ids
            else:
                hidden = self.embed_tokens(token_ids) if inputs_embeds is None else inputs_embeds
        else:
            if state is None:
                raise ValueError("a non-first stage requires the previous stage's state")
            if inputs_embeds is not None or skip_embed:
                raise ValueError(
                    "inputs_embeds, skip_embed and a boundary state are mutually exclusive"
                )
            hidden = state
        for layer_index in range(layer_start, end):
            hidden = self.layers[layer_index](hidden, positions, attention, layer_index)
        if end < num_layers:
            return hidden
        return self.final_layernorm(hidden)


class PhiForCausalLM(CausalLM[PhiConfig]):
    """Dense Phi decoder whose attention is supplied by the runtime per forward."""

    model: _PhiModel
    _decoder_name = "model"

    def __init__(
        self,
        config: PhiConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: LayerBackend = "triton",
    ) -> None:
        super().__init__()
        self._attach_decoder(
            config,
            _PhiModel(config, device=device, dtype=dtype, backend=backend),
            lm_head_bias=True,
            device=device,
            dtype=dtype,
        )

    @property
    def transformer(self) -> DecoderModule:
        """Alias for runtimes that reach the decoder as ``transformer``."""
        return self._decoder()
