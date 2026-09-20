"""Inference-only LLaMA compatible with HuggingFace ``LlamaForCausalLM`` weights.

Architecture and tensor names follow the HF LLaMA layout (reference layout
credited to vLLM/SGLang, Apache-2.0). Tensor parallelism, RoPE, quantization,
checkpoint schema and the causal-LM outer contract are provided by
``ayaka.layers`` and ``ayaka.models._common``; attention execution stays a
runtime responsibility via the per-forward ``AttentionCallback``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from ayaka.distributed.parallel import ParallelContext
from ayaka.layers._common import LayerBackend
from ayaka.layers.activation import SiluAndMul
from ayaka.layers.embedding import VocabParallelEmbedding
from ayaka.layers.linear.attention import QKVParallelLinear
from ayaka.layers.linear.core import FusedGateUpLinear, RowParallelLinear
from ayaka.layers.linear.methods import LinearMethodBase
from ayaka.layers.norm import RMSNorm
from ayaka.layers.quantization.base import BaseQuantization
from ayaka.layers.rotary_embedding import get_rope
from ayaka.model_loader.mapping import ExternMapping
from ayaka.model_loader.module import WeightBinding
from ayaka.models._common import (
    ROTARY_CACHE_KEYS,
    AttentionCallback,
    CausalLM,
    DecoderModule,
    IgnoredCheckpointKeys,
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
    "LlamaConfig",
    "LlamaForCausalLM",
    "load_llama_weights",
    "llama_expected_weights",
    "llama_weight_bindings",
    "llama_weight_mapping",
    "write_llama_checkpoint",
]

QuantConfig = BaseQuantization | LinearMethodBase | str | None

#: Text-only loading tolerates multimodal exports sharing the text checkpoint.
_IGNORED_CHECKPOINT_KEYS = IgnoredCheckpointKeys(
    suffixes=ROTARY_CACHE_KEYS.suffixes,
    prefixes=("model.vision_tower", "vision_model", "multi_modal_projector"),
    substrings=("projector",),
)


@dataclass(frozen=True, slots=True)
class LlamaConfig(ModelConfigMixin):
    """HF ``LlamaConfig`` fields.

    Inference-irrelevant fields are preserved so a checkpoint's ``config.json``
    round-trips through the inherited :meth:`from_dict` / :meth:`arch_dict`.
    ``head_dim`` is normalized to an integer at construction so consumers never
    handle ``None``.
    """

    model_type: ClassVar[str] = "llama"

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    rope_theta: float = 10000.0
    rope_scaling: Mapping[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    bias: bool = False
    head_dim: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
        ):
            require_int(getattr(self, name), name, minimum=1)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_key_value_heads is not None:
            require_int(self.num_key_value_heads, "num_key_value_heads", minimum=1)
            if self.num_attention_heads % self.num_key_value_heads:
                raise ValueError(
                    "num_attention_heads must be divisible by num_key_value_heads for GQA"
                )
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)
        else:
            require_int(self.head_dim, "head_dim", minimum=1)
        if self.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {self.hidden_act}. Only silu is supported for now."
            )
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")

    @property
    def resolved_head_dim(self) -> int:
        """Non-optional view of ``head_dim`` after construction normalization."""
        assert self.head_dim is not None
        return self.head_dim

    @property
    def num_kv_heads(self) -> int:
        return (
            self.num_attention_heads
            if self.num_key_value_heads is None
            else self.num_key_value_heads
        )

    @property
    def attention_bias_enabled(self) -> bool:
        return self.attention_bias or self.bias

    @property
    def qkv_bias_enabled(self) -> bool:
        """Bias on the q/k/v projections."""
        return self.attention_bias_enabled

    @property
    def o_bias_enabled(self) -> bool:
        """Bias on the o projection; llama ties it to the qkv bias."""
        return self.attention_bias_enabled

    @property
    def scaling(self) -> float:
        return self.resolved_head_dim**-0.5


def llama_expected_weights(
    config: LlamaConfig,
    tensor_dtype: DType = DType.FP32,
) -> tuple[WeightSpec, ...]:
    """Every tensor an HF LLaMA checkpoint must provide, in HF naming order.

    Shapes are global; ``WeightSpec.shard`` stays at its default because the
    expected schema validates names/shapes/dtypes only. The separate q/k/v and
    gate/up tensors are bound into packed projections by
    :func:`llama_weight_bindings`.
    """
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
        if config.mlp_bias:
            specs += [
                spec(f"{prefix}.mlp.gate_proj.bias", (inner,)),
                spec(f"{prefix}.mlp.up_proj.bias", (inner,)),
                spec(f"{prefix}.mlp.down_proj.bias", (hidden,)),
            ]
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


def llama_weight_bindings(config: LlamaConfig) -> dict[str, WeightBinding]:
    """Checkpoint key -> module tensor, with logical shard ids for packed weights.

    ``q_proj``/``k_proj``/``v_proj`` bind the logical q/k/v parts of the packed
    QKV projection; ``gate_proj``/``up_proj`` bind the gate/up halves of the
    packed gate-up projection. Row projections and norms bind whole.
    """
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
        if config.mlp_bias:
            bindings.update(
                {
                    f"{prefix}.mlp.gate_proj.bias": WeightBinding(
                        f"{prefix}.mlp.gate_up_proj.bias", "gate"
                    ),
                    f"{prefix}.mlp.up_proj.bias": WeightBinding(
                        f"{prefix}.mlp.gate_up_proj.bias", "up"
                    ),
                    f"{prefix}.mlp.down_proj.bias": WeightBinding(f"{prefix}.mlp.down_proj.bias"),
                }
            )
    if not config.tie_word_embeddings:
        bindings["lm_head.weight"] = WeightBinding("lm_head.weight")
    return bindings


def llama_weight_mapping(config: LlamaConfig) -> ExternMapping:
    """Declarative ExternMapping for LLaMA checkpoints.

    Maps 1-1 tensors, concatenates q/k/v into ``qkv_proj`` and gate/up into
    ``gate_up_proj``, and whitelists rotary caches present in some exports.
    """
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
        if config.mlp_bias:
            mapping.add_mapping(
                f"{prefix}.mlp.gate_up_proj.bias",
                [f"{prefix}.mlp.gate_proj.bias", f"{prefix}.mlp.up_proj.bias"],
                func=lambda gate, up: torch.cat([gate, up], dim=0),
            )
            mapping.add_mapping(f"{prefix}.mlp.down_proj.bias", f"{prefix}.mlp.down_proj.bias")
        for suffix in ROTARY_CACHE_KEYS.suffixes:
            mapping.add_unused(f"{prefix}.self_attn.{suffix}")

    if not config.tie_word_embeddings:
        mapping.add_mapping("lm_head.weight", "lm_head.weight")
    return mapping


def load_llama_weights(
    model: LlamaForCausalLM,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    validate: bool = True,
    use_mapping: bool = False,
) -> frozenset[str]:
    """Load a safetensors LLaMA checkpoint with the production bounded reader.

    Rotary caches and multimodal tensors (``projector``, ``model.vision_tower``,
    HF Mllama's ``vision_model``) are skipped: they never feed the text decoder.
    Missing required tensors and shape mismatches raise; unknown text tensors
    are never bound.
    """
    bindings: Mapping[str, WeightBinding] | ExternMapping = (
        llama_weight_mapping(model.config) if use_mapping else llama_weight_bindings(model.config)
    )
    return load_checkpoint(
        model,
        checkpoint_dir,
        expected=lambda dtype: llama_expected_weights(model.config, dtype),
        bindings=bindings,
        ignored=_IGNORED_CHECKPOINT_KEYS,
        device=device,
        validate=validate,
    )


def write_llama_checkpoint(
    root: Path,
    config: LlamaConfig,
    *,
    seed: int = 1234,
) -> Path:
    """Write a deterministic checkpoint in HF LLaMA naming order."""
    return write_checkpoint(
        root,
        architectures="LlamaForCausalLM",
        config=config,
        specs=llama_expected_weights(config, config.resolved_dtype()),
        unit_weight_suffixes=("norm.weight",),
        seed=seed,
    )


class _LlamaAttention(nn.Module):
    """Packed GQA projection, RoPE and the attention callback boundary."""

    def __init__(
        self,
        config: LlamaConfig,
        *,
        prefix: str,
        device: torch.device | str | None,
        dtype: torch.dtype,
        backend: LayerBackend,
        quant_config: QuantConfig = None,
    ) -> None:
        super().__init__()
        self.head_dim = config.resolved_head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_kv_heads
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.qkv_bias_enabled,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.o_bias_enabled,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            device=device,
        )
        self.rotary = get_rope(
            self.head_dim,
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            True,
            rope_scaling=config.rope_scaling,
            device=device,
            dtype=dtype,
            backend=backend,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads

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
        projected = self.o_proj(output.reshape(tokens, -1))
        assert isinstance(projected, torch.Tensor)
        return projected


class _LlamaMLP(nn.Module):
    """SwiGLU MLP: ``down( silu(gate) * up )`` over the packed gate-up output."""

    def __init__(
        self,
        config: LlamaConfig,
        *,
        prefix: str,
        device: torch.device | str | None,
        dtype: torch.dtype,
        backend: LayerBackend,
        quant_config: QuantConfig = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = FusedGateUpLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            device=device,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            device=device,
        )
        self.activation = SiluAndMul(backend=backend)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        packed = self.gate_up_proj(hidden)
        assert isinstance(packed, torch.Tensor)
        projected = self.down_proj(self.activation(packed))
        assert isinstance(projected, torch.Tensor)
        return projected


class _LlamaDecoderLayer(nn.Module):
    """Pre-norm decoder block with fused residual additions."""

    def __init__(
        self,
        config: LlamaConfig,
        *,
        prefix: str,
        device: torch.device | str | None,
        dtype: torch.dtype,
        backend: LayerBackend,
        quant_config: QuantConfig = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.input_layernorm = RMSNorm(
            hidden, eps=config.rms_norm_eps, device=device, dtype=dtype, backend=backend
        )
        self.self_attn = _LlamaAttention(
            config,
            prefix=f"{prefix}.self_attn",
            device=device,
            dtype=dtype,
            backend=backend,
            quant_config=quant_config,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden, eps=config.rms_norm_eps, device=device, dtype=dtype, backend=backend
        )
        self.mlp = _LlamaMLP(
            config,
            prefix=f"{prefix}.mlp",
            device=device,
            dtype=dtype,
            backend=backend,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionCallback,
        layer_index: int,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_sum: torch.Tensor
        if residual is None:
            residual_sum = hidden
            normalized = self.input_layernorm(hidden)
        else:
            normalized, residual_sum = self.input_layernorm(hidden, residual)
        attended = self.self_attn(normalized, positions, attention, layer_index)
        normalized, residual_sum = self.post_attention_layernorm(attended, residual_sum)
        output = self.mlp(normalized)
        return output, residual_sum


class _LlamaModel(nn.Module):
    """Embedding, decoder layers and final norm over an HF-style module tree."""

    def __init__(
        self,
        config: LlamaConfig,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype,
        backend: LayerBackend,
        quant_config: QuantConfig = None,
        parallel_context: ParallelContext | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=dtype,
            device=device,
            backend=backend,
            parallel_context=parallel_context,
            prefix="model.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                _LlamaDecoderLayer(
                    config,
                    prefix=f"model.layers.{index}",
                    device=device,
                    dtype=dtype,
                    backend=backend,
                    quant_config=quant_config,
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
            backend=backend,
        )

    @property
    def wte(self) -> VocabParallelEmbedding:
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
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run a half-open layer range of the decoder.

        ``layer_start``/``layer_end`` slice the transformer for one PP stage:
        a non-zero ``layer_start`` requires ``state`` (the previous stage's
        ``(hidden, residual)`` boundary) instead of token ids, and a
        ``layer_end`` below the layer count returns the boundary tensors so
        the next stage can resume without re-normalizing. Layer indices stay
        GLOBAL so attention callbacks index KV slots exactly as a full
        forward does; the final ``norm`` only runs on the last stage.
        """
        num_layers = len(self.layers)
        end = resolve_layer_range(num_layers, layer_start, layer_end)
        residual_sum: torch.Tensor | None
        if layer_start == 0:
            if state is not None:
                raise ValueError("layer_start 0 must not carry a boundary state")
            if skip_embed and inputs_embeds is not None:
                raise ValueError("skip_embed and inputs_embeds are mutually exclusive")
            if skip_embed:
                hidden = token_ids
            else:
                hidden = self.embed_tokens(token_ids) if inputs_embeds is None else inputs_embeds
            residual_sum = None
        else:
            if state is None:
                raise ValueError("a non-first stage requires the previous stage's state")
            if inputs_embeds is not None or skip_embed:
                raise ValueError(
                    "inputs_embeds, skip_embed and a boundary state are mutually exclusive"
                )
            hidden, residual_sum = state
        for layer_index in range(layer_start, end):
            hidden, residual_sum = self.layers[layer_index](
                hidden, positions, attention, layer_index, residual_sum
            )
        if end < num_layers:
            assert residual_sum is not None
            return hidden, residual_sum
        assert residual_sum is not None
        normalized, _ = self.norm(hidden, residual_sum)
        return normalized


class LlamaForCausalLM(CausalLM[LlamaConfig]):
    """Dense LLaMA decoder whose attention is supplied by the runtime per forward."""

    model: _LlamaModel
    _decoder_name = "model"

    def __init__(
        self,
        config: LlamaConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: LayerBackend = "triton",
        quant_config: QuantConfig = None,
        parallel_context: ParallelContext | None = None,
    ) -> None:
        super().__init__()
        self._attach_decoder(
            config,
            _LlamaModel(
                config,
                device=device,
                dtype=dtype,
                backend=backend,
                quant_config=quant_config,
                parallel_context=parallel_context,
            ),
            lm_head_bias=False,
            device=device,
            dtype=dtype,
            parallel_context=parallel_context,
        )

    @property
    def transformer(self) -> DecoderModule:
        """Alias for runtimes that reach the decoder as ``transformer``."""
        return self._decoder()
