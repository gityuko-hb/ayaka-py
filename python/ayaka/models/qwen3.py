"""Inference-only Qwen3 compatible with HuggingFace ``Qwen3ForCausalLM``.

Qwen3 is structurally identical to Qwen2 with one addition: each attention
layer applies per-head RMSNorm to the query and key vectors after the QKV
projection and before RoPE.  That single change requires dedicated
``self_attn.q_norm`` and ``self_attn.k_norm`` tensors per layer in the
checkpoint schema and a small override in the attention module.

Everything else — decoder geometry, MLP, weight-loading plumbing, the
causal-LM outer contract — is shared with :mod:`ayaka.models.qwen2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from ayaka.distributed.parallel import ParallelContext
from ayaka.layers._common import LayerBackend
from ayaka.layers.embedding import VocabParallelEmbedding
from ayaka.layers.linear.methods import LinearMethodBase
from ayaka.layers.norm import RMSNorm
from ayaka.layers.quantization.base import BaseQuantization
from ayaka.model_loader.mapping import ExternMapping
from ayaka.model_loader.module import WeightBinding
from ayaka.models._common import (
    AttentionCallback,
    CausalLM,
    load_checkpoint,
    resolve_layer_range,
    weight_spec,
    write_checkpoint,
)
from ayaka.models.llama import LlamaConfig, _LlamaAttention, _LlamaMLP
from ayaka.models.qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM,
    qwen2_weight_bindings,
    qwen2_weight_mapping,
)
from ayaka.types import DType
from ayaka.weights.spec import TensorSpec, WeightSpec

__all__ = [
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "load_qwen3_weights",
    "qwen3_expected_weights",
    "qwen3_weight_bindings",
    "qwen3_weight_mapping",
    "write_qwen3_checkpoint",
]

QuantConfig = BaseQuantization | LinearMethodBase | str | None


@dataclass(frozen=True, slots=True)
class Qwen3Config(Qwen2Config):
    """HF ``Qwen3Config`` fields on the shared Qwen2 decoder geometry.

    The only structural difference between Qwen3 and Qwen2 is that Qwen3
    applies a per-head RMSNorm to ``q`` and ``k`` after the QKV projection.
    Qwen3 checkpoints ship without QKV bias (``qkv_bias: false``) and with an
    explicit ``head_dim: 128`` regardless of the hidden/head ratio.

    ``rope_scaling`` is permitted so the YaRN-extended context variants
    (e.g. Qwen3-30B-A3B) load without error.
    """

    model_type: ClassVar[str] = "qwen3"

    # Qwen3 defaults differ from Qwen2: no QKV bias, explicit 128-dim heads.
    qkv_bias: bool = False
    head_dim: int | None = 128

    def __post_init__(self) -> None:
        # Call LlamaConfig.__post_init__ directly to skip Qwen2's validation
        # that rejects rope_scaling (Qwen3 extended-context variants need it).
        LlamaConfig.__post_init__(self)
        if self.use_sliding_window:
            raise ValueError(
                "qwen3 sliding-window attention is outside the certified resident lane; "
                "checkpoints with use_sliding_window=true are rejected"
            )

    def resolved_dtype(self) -> DType:
        """Qwen3 checkpoints are bf16 unless the checkpoint states otherwise."""
        if self.torch_dtype:
            return DType.from_str(self.torch_dtype)
        return DType.BF16


# ---------------------------------------------------------------------------
# Checkpoint schema
# ---------------------------------------------------------------------------


def qwen3_expected_weights(
    config: Qwen3Config,
    tensor_dtype: DType = DType.BF16,
) -> tuple[WeightSpec, ...]:
    """Every tensor an HF Qwen3 checkpoint must provide, in HF naming order.

    The schema extends Qwen2's by adding ``self_attn.q_norm.weight`` and
    ``self_attn.k_norm.weight`` tensors of shape ``(head_dim,)`` for each
    decoder layer.
    """
    hidden = config.hidden_size
    head_dim = config.resolved_head_dim
    q_width = config.num_attention_heads * head_dim
    kv_width = config.num_kv_heads * head_dim
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
            spec(f"{prefix}.self_attn.q_norm.weight", (head_dim,)),
            spec(f"{prefix}.self_attn.k_proj.weight", (kv_width, hidden)),
            spec(f"{prefix}.self_attn.k_norm.weight", (head_dim,)),
            spec(f"{prefix}.self_attn.v_proj.weight", (kv_width, hidden)),
            spec(f"{prefix}.self_attn.o_proj.weight", (hidden, q_width)),
            spec(f"{prefix}.post_attention_layernorm.weight", (hidden,)),
            spec(f"{prefix}.mlp.gate_proj.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.up_proj.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.down_proj.weight", (hidden, inner)),
        ]
        # Qwen3 has no QKV bias by default; honour the config flag for
        # hypothetical fine-tuned variants.
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


def qwen3_weight_bindings(config: Qwen3Config) -> dict[str, WeightBinding]:
    """Checkpoint key -> module tensor for Qwen3.

    Extends the Qwen2 bindings with straight 1-1 mappings for the two new
    per-head norm weights ``q_norm.weight`` and ``k_norm.weight``.
    """
    # Start from the Qwen2 bindings (q/k/v fused into qkv_proj, gate/up fused
    # into gate_up_proj, rotary caches whitelisted, etc.).
    bindings: dict[str, WeightBinding] = dict(qwen2_weight_bindings(config))

    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        bindings[f"{prefix}.self_attn.q_norm.weight"] = WeightBinding(
            f"{prefix}.self_attn.q_norm.weight"
        )
        bindings[f"{prefix}.self_attn.k_norm.weight"] = WeightBinding(
            f"{prefix}.self_attn.k_norm.weight"
        )
    return bindings


def qwen3_weight_mapping(config: Qwen3Config) -> ExternMapping:
    """Declarative ExternMapping for Qwen3 checkpoints.

    Inherits Qwen2's fused-projection and rotary-cache handling and adds
    direct 1-1 mappings for ``q_norm.weight`` / ``k_norm.weight``.
    """
    mapping = qwen2_weight_mapping(config)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        mapping.add_mapping(
            f"{prefix}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.q_norm.weight",
        )
        mapping.add_mapping(
            f"{prefix}.self_attn.k_norm.weight",
            f"{prefix}.self_attn.k_norm.weight",
        )
    return mapping


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_qwen3_weights(
    model: Qwen3ForCausalLM,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    validate: bool = True,
    use_mapping: bool = False,
) -> frozenset[str]:
    """Load an HF Qwen3 safetensors checkpoint with the bounded reader."""
    config = model.config
    assert isinstance(config, Qwen3Config)
    bindings = qwen3_weight_mapping(config) if use_mapping else qwen3_weight_bindings(config)
    return load_checkpoint(
        model,
        checkpoint_dir,
        expected=lambda dtype: qwen3_expected_weights(config, dtype),
        bindings=bindings,
        tolerate_tied_in_checkpoint=True,
        device=device,
        validate=validate,
    )


def write_qwen3_checkpoint(
    root: Path,
    config: Qwen3Config,
    *,
    seed: int = 1234,
) -> Path:
    """Write a deterministic checkpoint in HF Qwen3 naming order."""
    return write_checkpoint(
        root,
        architectures="Qwen3ForCausalLM",
        config=config,
        specs=qwen3_expected_weights(config, config.resolved_dtype()),
        unit_weight_suffixes=("norm.weight",),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Attention module with per-head QK RMSNorm
# ---------------------------------------------------------------------------


class _Qwen3Attention(_LlamaAttention):
    """Qwen3 attention: packed QKV proj + per-head q/k RMSNorm + RoPE.

    Inherits projection, RoPE and the attention-callback boundary from
    :class:`~ayaka.models.llama._LlamaAttention`; overrides :meth:`forward`
    to inject head-wise RMSNorm on ``q`` and ``k`` before rotation.
    """

    def __init__(
        self,
        config: Qwen3Config,
        *,
        prefix: str,
        device: torch.device | str | None,
        dtype: torch.dtype,
        backend: LayerBackend,
        quant_config: QuantConfig = None,
    ) -> None:
        super().__init__(
            config,
            prefix=prefix,
            device=device,
            dtype=dtype,
            backend=backend,
            quant_config=quant_config,
        )
        head_dim = config.resolved_head_dim
        self.q_norm = RMSNorm(
            head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype, backend=backend
        )
        self.k_norm = RMSNorm(
            head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype, backend=backend
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

        # [tokens, num_heads, head_dim]
        query = parts["q"].view(tokens, self.num_heads, self.head_dim)
        key = parts["k"].view(tokens, self.num_kv_heads, self.head_dim)
        value = parts["v"].view(tokens, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm applied on the last (head_dim) axis.
        # The norm weight has shape (head_dim,) and broadcasts over the token
        # and head dimensions without any reshaping.
        query = self.q_norm(query.reshape(-1, self.head_dim)).view(
            tokens, self.num_heads, self.head_dim
        )
        key = self.k_norm(key.reshape(-1, self.head_dim)).view(
            tokens, self.num_kv_heads, self.head_dim
        )

        self.rotary(positions, query, key)
        output = attention(layer_index, query, key, value)
        projected = self.o_proj(output.reshape(tokens, -1))
        assert isinstance(projected, torch.Tensor)
        return projected


# ---------------------------------------------------------------------------
# Decoder layer that wires in the Qwen3 attention module
# ---------------------------------------------------------------------------


class _Qwen3DecoderLayer(nn.Module):
    """Pre-norm decoder block identical to LLaMA except for Qwen3 attention."""

    def __init__(
        self,
        config: Qwen3Config,
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
        self.self_attn = _Qwen3Attention(
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


# ---------------------------------------------------------------------------
# Inner model
# ---------------------------------------------------------------------------


class _Qwen3Model(nn.Module):
    """Embedding, Qwen3 decoder layers and final norm."""

    def __init__(
        self,
        config: Qwen3Config,
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
                _Qwen3DecoderLayer(
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
    def wte(self) -> nn.Module:
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


# ---------------------------------------------------------------------------
# Outer CausalLM
# ---------------------------------------------------------------------------


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    """Dense Qwen3 decoder with per-head QK RMSNorm.

    Structurally identical to :class:`~ayaka.models.qwen2.Qwen2ForCausalLM`
    except that the inner model uses :class:`_Qwen3DecoderLayer` (and thus
    :class:`_Qwen3Attention`) so that ``q_norm`` / ``k_norm`` are applied
    before RoPE in every attention layer.
    """

    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: str = "triton",
        quant_config: QuantConfig = None,
        parallel_context: ParallelContext | None = None,
    ) -> None:
        # Bypass Qwen2ForCausalLM.__init__ and go straight to CausalLM so we
        # can supply a _Qwen3Model instead of the Llama inner model.
        CausalLM.__init__(self)
        self._attach_decoder(
            config,
            _Qwen3Model(
                config,
                device=device,
                dtype=dtype,
                backend=backend,  # type: ignore[arg-type]
                quant_config=quant_config,
                parallel_context=parallel_context,
            ),
            lm_head_bias=False,
            device=device,
            dtype=dtype,
            parallel_context=parallel_context,
        )
