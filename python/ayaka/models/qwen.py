"""Inference-only QWen (original) compatible with HF ``QWenLMHeadModel``.

Architecture and checkpoint tensor names follow the HF QWen layout (reference
layout credited to vLLM, Apache-2.0). Tensor parallelism, RoPE, checkpoint
schema and the causal-LM outer contract are provided by ``ayaka.layers`` and
``ayaka.models._common``; attention execution stays a runtime responsibility
via the per-forward ``AttentionCallback``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rotary_embedding import get_rope
from ayaka.model_loader.mapping import ExternMapping
from ayaka.model_loader.module import WeightBinding
from ayaka.models._common import (
    ROTARY_CACHE_KEYS,
    AttentionCallback,
    CausalLM,
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
    "QwenConfig",
    "QwenForCausalLM",
    "load_qwen_weights",
    "qwen_expected_weights",
    "qwen_weight_bindings",
    "qwen_weight_mapping",
    "write_qwen_checkpoint",
]


@dataclass(frozen=True, slots=True)
class QwenConfig(ModelConfigMixin):
    """HF ``QWenConfig`` fields.

    Inference-irrelevant fields are preserved so a checkpoint's ``config.json``
    round-trips through the inherited :meth:`from_dict` / :meth:`arch_dict`.
    """

    model_type: ClassVar[str] = "qwen"

    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    emb_dropout_prob: float = 0.0
    attn_dropout_prob: float = 0.0
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    max_position_embeddings: int = 8192
    scale_attn_weights: bool = True
    use_cache: bool = True
    bf16: bool = False
    fp16: bool = False
    fp32: bool = False
    kv_channels: int | None = 128
    rotary_pct: float = 1.0
    rotary_emb_base: float = 10000
    use_dynamic_ntk: bool = True
    use_logn_attn: bool = True
    use_flash_attn: str | bool = "auto"
    intermediate_size: int = 22016
    no_bias: bool = True
    rope_scaling: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
        ):
            require_int(getattr(self, name), name, minimum=1)
        if self.kv_channels is not None:
            require_int(self.kv_channels, "kv_channels", minimum=1)
        elif self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.intermediate_size % 2:
            raise ValueError("intermediate_size must be even to split gate/up")
        if not math.isfinite(self.rotary_pct) or not 0 < self.rotary_pct <= 1:
            raise ValueError("rotary_pct must be in (0, 1]")
        if not math.isfinite(self.rotary_emb_base) or self.rotary_emb_base <= 0:
            raise ValueError("rotary_emb_base must be finite and positive")
        if not math.isfinite(self.layer_norm_epsilon) or self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be finite and positive")
        if self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even")

    @property
    def head_dim(self) -> int:
        if self.kv_channels is not None:
            return self.kv_channels
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_pct)

    @property
    def mlp_inner_size(self) -> int:
        """QWen's HF ``intermediate_size`` covers both gate and up halves."""
        return self.intermediate_size // 2

    @property
    def scaling(self) -> float:
        return self.head_dim**-0.5 if self.scale_attn_weights else 1.0

    def resolved_dtype(self) -> DType:
        """Checkpoint dtype from the HF flags, then ``torch_dtype``, else BF16."""
        if self.bf16:
            return DType.BF16
        if self.fp16:
            return DType.FP16
        if self.fp32:
            return DType.FP32
        if self.torch_dtype:
            return DType.from_str(self.torch_dtype)
        return DType.BF16


def qwen_expected_weights(
    config: QwenConfig,
    tensor_dtype: DType = DType.FP32,
) -> tuple[WeightSpec, ...]:
    """Every tensor a QWen checkpoint must provide, in stable HF naming order.

    Shapes are global; ``WeightSpec.shard`` stays at its default because the
    expected schema validates names/shapes/dtypes only. The fused ``c_attn``
    tensor is one entry — :func:`qwen_weight_bindings` splits it into q/k/v.
    """
    hidden = config.hidden_size
    attn_width = config.num_attention_heads * config.head_dim
    inner = config.mlp_inner_size
    vocab = config.vocab_size

    def spec(name: str, shape: tuple[int, ...], *, optional: bool = False) -> WeightSpec:
        return replace(weight_spec(name, shape, tensor_dtype), optional=optional)

    specs: list[WeightSpec] = [spec("transformer.wte.weight", (vocab, hidden))]
    for layer in range(config.num_hidden_layers):
        prefix = f"transformer.h.{layer}"
        specs += [
            spec(f"{prefix}.attn.c_attn.weight", (3 * attn_width, hidden), optional=True),
            spec(f"{prefix}.attn.c_attn.bias", (3 * attn_width,), optional=True),
            spec(f"{prefix}.attn.c_proj.weight", (hidden, attn_width)),
            spec(f"{prefix}.mlp.w1.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.w2.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.c_proj.weight", (hidden, inner)),
            spec(f"{prefix}.ln_1.weight", (hidden,)),
            spec(f"{prefix}.ln_2.weight", (hidden,)),
        ]
    specs.append(spec("transformer.ln_f.weight", (hidden,)))
    specs.append(
        WeightSpec(
            name="lm_head.weight",
            full_shape=(vocab, hidden),
            spec=TensorSpec(shape=(vocab, hidden), dtype=tensor_dtype, name="lm_head.weight"),
            tied_alias="transformer.wte.weight" if config.tie_word_embeddings else "",
        )
    )
    return tuple(specs)


def qwen_weight_bindings(config: QwenConfig) -> dict[str, WeightBinding]:
    """Checkpoint key -> module tensor, with logical shard ids for packed weights.

    ``c_attn`` binds whole (the QKV loader splits q/k/v); ``w2``/``w1`` bind the
    logical gate/up halves of the packed gate-up projection.
    """
    bindings: dict[str, WeightBinding] = {
        "transformer.wte.weight": WeightBinding("transformer.wte.weight"),
        "transformer.ln_f.weight": WeightBinding("transformer.ln_f.weight"),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"transformer.h.{layer}"
        bindings.update(
            {
                f"{prefix}.attn.c_attn.weight": WeightBinding(f"{prefix}.attn.c_attn.weight"),
                f"{prefix}.attn.c_attn.bias": WeightBinding(f"{prefix}.attn.c_attn.bias"),
                f"{prefix}.attn.c_proj.weight": WeightBinding(f"{prefix}.attn.c_proj.weight"),
                f"{prefix}.mlp.w2.weight": WeightBinding(
                    f"{prefix}.mlp.gate_up_proj.weight", "gate"
                ),
                f"{prefix}.mlp.w1.weight": WeightBinding(f"{prefix}.mlp.gate_up_proj.weight", "up"),
                f"{prefix}.mlp.c_proj.weight": WeightBinding(f"{prefix}.mlp.c_proj.weight"),
                f"{prefix}.ln_1.weight": WeightBinding(f"{prefix}.ln_1.weight"),
                f"{prefix}.ln_2.weight": WeightBinding(f"{prefix}.ln_2.weight"),
            }
        )
    if not config.tie_word_embeddings:
        bindings["lm_head.weight"] = WeightBinding("lm_head.weight")
    return bindings


def qwen_weight_mapping(config: QwenConfig) -> ExternMapping:
    """Declarative ExternMapping for Qwen checkpoints.

    Maps 1-1 tensors, fuses mlp.w2 (gate) and mlp.w1 (up) into gate_up_proj,
    and whitelists rotary caches present in some exports.
    """
    mapping = ExternMapping()
    mapping.add_mapping("transformer.wte.weight", "transformer.wte.weight")
    mapping.add_mapping("transformer.ln_f.weight", "transformer.ln_f.weight")
    for layer in range(config.num_hidden_layers):
        prefix = f"transformer.h.{layer}"
        mapping.add_mapping(f"{prefix}.attn.c_attn.weight", f"{prefix}.attn.c_attn.weight")
        mapping.add_mapping(f"{prefix}.attn.c_attn.bias", f"{prefix}.attn.c_attn.bias")
        mapping.add_mapping(f"{prefix}.attn.c_proj.weight", f"{prefix}.attn.c_proj.weight")
        mapping.add_mapping(
            f"{prefix}.mlp.gate_up_proj.weight",
            [f"{prefix}.mlp.w2.weight", f"{prefix}.mlp.w1.weight"],
            func=lambda w2, w1: torch.cat([w2, w1], dim=0),
        )
        mapping.add_mapping(f"{prefix}.mlp.c_proj.weight", f"{prefix}.mlp.c_proj.weight")
        mapping.add_mapping(f"{prefix}.ln_1.weight", f"{prefix}.ln_1.weight")
        mapping.add_mapping(f"{prefix}.ln_2.weight", f"{prefix}.ln_2.weight")
        for suffix in ROTARY_CACHE_KEYS.suffixes:
            mapping.add_unused(f"{prefix}.attn.{suffix}")

    if not config.tie_word_embeddings:
        mapping.add_mapping("lm_head.weight", "lm_head.weight")
    return mapping


def load_qwen_weights(
    model: QwenForCausalLM,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    validate: bool = True,
    use_mapping: bool = False,
) -> frozenset[str]:
    """Load a safetensors QWen checkpoint with the production bounded reader.

    Rotary caches some exports ship are skipped: RoPE is computed by this
    module. Missing required tensors and shape mismatches raise; unknown
    tensors are never bound.
    """
    bindings: Mapping[str, WeightBinding] | ExternMapping = (
        qwen_weight_mapping(model.config) if use_mapping else qwen_weight_bindings(model.config)
    )
    return load_checkpoint(
        model,
        checkpoint_dir,
        expected=lambda dtype: qwen_expected_weights(model.config, dtype),
        bindings=bindings,
        ignored=ROTARY_CACHE_KEYS,
        device=device,
        validate=validate,
    )


def write_qwen_checkpoint(
    root: Path,
    config: QwenConfig,
    *,
    seed: int = 1234,
) -> Path:
    """Write a deterministic checkpoint in HF QWen naming order."""
    return write_checkpoint(
        root,
        architectures="QWenLMHeadModel",
        config=config,
        specs=qwen_expected_weights(config, config.resolved_dtype()),
        unit_weight_suffixes=("ln_1.weight", "ln_2.weight", "ln_f.weight"),
        seed=seed,
    )


class _QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.c_attn = QKVParallelLinear(
            hidden,
            self.head_dim,
            self.num_heads,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.c_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.rotary = get_rope(
            self.head_dim,
            config.rotary_dim,
            config.max_position_embeddings,
            config.rotary_emb_base,
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
        packed = self.c_attn(hidden)
        assert isinstance(packed, torch.Tensor)
        query, key, value = packed.split(self.num_heads * self.head_dim, dim=-1)
        query = query.view(tokens, self.num_heads, self.head_dim)
        key = key.view(tokens, self.num_heads, self.head_dim)
        value = value.view(tokens, self.num_heads, self.head_dim)
        self.rotary(positions, query, key)
        output = attention(layer_index, query, key, value)
        reshaped = output.reshape(tokens, self.num_heads * self.head_dim)
        projected = self.c_proj(reshaped)
        assert isinstance(projected, torch.Tensor)
        return projected


class _QwenMLP(nn.Module):
    def __init__(self, config: QwenConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.mlp_inner_size
        self.gate_up_proj = FusedGateUpLinear(
            hidden,
            inner,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.c_proj = RowParallelLinear(
            inner,
            hidden,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.activation = SiluAndMul(backend=backend)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        fused = self.gate_up_proj(hidden)
        assert isinstance(fused, torch.Tensor)
        return self.c_proj(self.activation(fused))


class _QwenBlock(nn.Module):
    def __init__(self, config: QwenConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.ln_1 = RMSNorm(
            hidden, eps=config.layer_norm_epsilon, device=device, dtype=dtype, backend=backend
        )
        self.attn = _QwenAttention(config, device=device, dtype=dtype, backend=backend)
        self.ln_2 = RMSNorm(
            hidden, eps=config.layer_norm_epsilon, device=device, dtype=dtype, backend=backend
        )
        self.mlp = _QwenMLP(config, device=device, dtype=dtype, backend=backend)

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
            normalized = self.ln_1(hidden)
        else:
            normalized, residual_sum = self.ln_1(hidden, residual)
        attended = self.attn(normalized, positions, attention, layer_index)
        normalized, residual_sum = self.ln_2(attended, residual_sum)
        output = self.mlp(normalized)
        return output, residual_sum


class _QwenModel(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        *,
        device,
        dtype,
        backend: LayerBackend,
        parallel_context: ParallelContext | None = None,
    ) -> None:
        super().__init__()
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=dtype,
            device=device,
            backend=backend,
            parallel_context=parallel_context,
            prefix="transformer.wte",
        )
        self.h = nn.ModuleList(
            [
                _QwenBlock(config, device=device, dtype=dtype, backend=backend)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            device=device,
            dtype=dtype,
            backend=backend,
        )

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
        forward does; the final ``ln_f`` only runs on the last stage.
        """
        num_layers = len(self.h)
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
                hidden = self.wte(token_ids) if inputs_embeds is None else inputs_embeds
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
            hidden, residual_sum = self.h[layer_index](
                hidden, positions, attention, layer_index, residual_sum
            )
        if end < num_layers:
            assert residual_sum is not None
            return hidden, residual_sum
        assert residual_sum is not None
        normalized, _ = self.ln_f(hidden, residual_sum)
        return normalized


class QwenForCausalLM(CausalLM[QwenConfig]):
    """Dense QWen decoder whose attention is supplied by the runtime per forward."""

    transformer: _QwenModel
    _decoder_name = "transformer"

    def __init__(
        self,
        config: QwenConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: LayerBackend = "triton",
        parallel_context: ParallelContext | None = None,
    ) -> None:
        super().__init__()
        self._attach_decoder(
            config,
            _QwenModel(
                config,
                device=device,
                dtype=dtype,
                backend=backend,
                parallel_context=parallel_context,
            ),
            lm_head_bias=False,
            device=device,
            dtype=dtype,
            parallel_context=parallel_context,
        )

    @property
    def model(self) -> _QwenModel:
        """Alias for runtimes that reach the decoder as ``model``."""
        return self.transformer
