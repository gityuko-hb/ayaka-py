"""Inference-only GPT-2 compatible with HuggingFace ``GPT2LMHeadModel`` weights.

Reference layout credited to vLLM (Apache-2.0). HF checkpoints store ``Conv1D``
weights transposed relative to Ayaka's ``[out, in]`` linear storage, so every
projection weight is transposed by the declarative :func:`gpt2_weight_mapping`
during loading. Tensor parallelism, checkpoint schema and the causal-LM outer
contract are provided by ``ayaka.layers`` and ``ayaka.models._common``.
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
from ayaka.layers.activation import get_act_fn
from ayaka.layers.embedding import VocabParallelEmbedding
from ayaka.layers.linear.attention import QKVParallelLinear
from ayaka.layers.linear.core import ColumnParallelLinear, RowParallelLinear
from ayaka.layers.norm import LayerNorm
from ayaka.model_loader.mapping import ExternMapping
from ayaka.models._common import (
    AttentionCallback,
    CausalLM,
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
    "GPT2Config",
    "GPT2ForCausalLM",
    "gpt2_expected_weights",
    "gpt2_weight_mapping",
    "load_gpt2_weights",
    "write_gpt2_checkpoint",
]

#: Legacy exports register the causal mask as persistent buffers; never consumed.
_IGNORED_CHECKPOINT_KEYS = IgnoredCheckpointKeys(suffixes=(".attn.bias", ".attn.masked_bias"))


def _normalize_checkpoint_key(name: str) -> str:
    """Renamed hub exports drop the ``transformer.`` prefix; restore it."""
    if name == "lm_head.weight" or name.startswith("transformer."):
        return name
    return f"transformer.{name}"


_SUPPORTED_ACTIVATIONS = frozenset(
    {"gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh", "quick_gelu"}
)

_CONFIG_ALIASES = {
    "n_embd": "hidden_size",
    "n_layer": "num_hidden_layers",
    "n_head": "num_attention_heads",
    "n_positions": "max_position_embeddings",
    "n_inner": "intermediate_size",
    "layer_norm_eps": "layer_norm_epsilon",
}


@dataclass(frozen=True, slots=True)
class GPT2Config(ModelConfigMixin):
    """HF ``GPT2Config`` fields under the canonical long names Ayaka uses.

    Inference-irrelevant dropout fields are preserved so a checkpoint's
    ``config.json`` round-trips through the inherited :meth:`from_dict` /
    :meth:`arch_dict`, which accepts both canonical names and HF's short
    aliases (``n_embd``, ``n_layer``, ``n_head``, ``n_positions``, ``n_inner``).
    """

    model_type: ClassVar[str] = "gpt2"
    _CONFIG_ALIASES: ClassVar[Mapping[str, str]] = _CONFIG_ALIASES

    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 1024
    intermediate_size: int | None = None
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
        ):
            require_int(getattr(self, name), name, minimum=1)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.intermediate_size is not None:
            require_int(self.intermediate_size, "intermediate_size", minimum=1)
        if self.activation_function not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported activation_function {self.activation_function!r}; "
                f"expected one of {sorted(_SUPPORTED_ACTIVATIONS)}"
            )
        if not math.isfinite(self.layer_norm_epsilon) or self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be finite and positive")
        if self.scale_attn_by_inverse_layer_idx:
            raise ValueError("scale_attn_by_inverse_layer_idx is not supported")
        if self.reorder_and_upcast_attn:
            raise ValueError("reorder_and_upcast_attn is not supported")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def mlp_inner_size(self) -> int:
        """GPT-2's HF ``n_inner``, falling back to the canonical ``4 * hidden``."""
        if self.intermediate_size is not None:
            return self.intermediate_size
        return 4 * self.hidden_size

    @property
    def scaling(self) -> float:
        return self.head_dim**-0.5 if self.scale_attn_weights else 1.0


def gpt2_expected_weights(
    config: GPT2Config,
    tensor_dtype: DType = DType.FP32,
) -> tuple[WeightSpec, ...]:
    """Every tensor an HF GPT-2 checkpoint must provide, in HF naming order.

    Shapes are the checkpoint's ``Conv1D`` storage (``c_attn.weight`` is
    ``[hidden, 3 * attn_width]``), not Ayaka's transposed module layout; the
    loader performs the transpose. ``attn.bias``/``attn.masked_bias`` buffers
    are deliberately absent: not every export ships them and inference never
    consumes them.
    """
    hidden = config.hidden_size
    attn_width = config.num_attention_heads * config.head_dim
    inner = config.mlp_inner_size
    vocab = config.vocab_size

    def spec(name: str, shape: tuple[int, ...]) -> WeightSpec:
        return weight_spec(name, shape, tensor_dtype)

    specs: list[WeightSpec] = [
        spec("transformer.wte.weight", (vocab, hidden)),
        spec("transformer.wpe.weight", (config.max_position_embeddings, hidden)),
    ]
    for layer in range(config.num_hidden_layers):
        prefix = f"transformer.h.{layer}"
        specs += [
            spec(f"{prefix}.ln_1.weight", (hidden,)),
            spec(f"{prefix}.ln_1.bias", (hidden,)),
            spec(f"{prefix}.attn.c_attn.weight", (hidden, 3 * attn_width)),
            spec(f"{prefix}.attn.c_attn.bias", (3 * attn_width,)),
            spec(f"{prefix}.attn.c_proj.weight", (hidden, hidden)),
            spec(f"{prefix}.attn.c_proj.bias", (hidden,)),
            spec(f"{prefix}.ln_2.weight", (hidden,)),
            spec(f"{prefix}.ln_2.bias", (hidden,)),
            spec(f"{prefix}.mlp.c_fc.weight", (hidden, inner)),
            spec(f"{prefix}.mlp.c_fc.bias", (inner,)),
            spec(f"{prefix}.mlp.c_proj.weight", (inner, hidden)),
            spec(f"{prefix}.mlp.c_proj.bias", (hidden,)),
        ]
    specs.append(spec("transformer.ln_f.weight", (hidden,)))
    specs.append(spec("transformer.ln_f.bias", (hidden,)))
    specs.append(
        WeightSpec(
            name="lm_head.weight",
            full_shape=(vocab, hidden),
            spec=TensorSpec(shape=(vocab, hidden), dtype=tensor_dtype, name="lm_head.weight"),
            tied_alias="transformer.wte.weight" if config.tie_word_embeddings else "",
        )
    )
    return tuple(specs)


def _transpose_conv1d(weight: torch.Tensor) -> torch.Tensor:
    """HF ``Conv1D`` stores ``[in, out]``; Ayaka linears store ``[out, in]``."""
    return weight.transpose(0, 1)


def gpt2_weight_mapping(config: GPT2Config) -> ExternMapping:
    """Declarative ExternMapping translating HF Conv1D storage to module layout.

    GPT-2 has no binding-only path: a :class:`WeightBinding` cannot express the
    transpose, so it happens in each mapping's ``func`` before the destination
    loader shards the tensor according to its projection layout.
    """
    mapping = ExternMapping()
    mapping.add_mapping("transformer.wte.weight", "transformer.wte.weight")
    mapping.add_mapping("transformer.wpe.weight", "transformer.wpe.weight")
    mapping.add_mapping("transformer.ln_f.weight", "transformer.ln_f.weight")
    mapping.add_mapping("transformer.ln_f.bias", "transformer.ln_f.bias")
    for layer in range(config.num_hidden_layers):
        prefix = f"transformer.h.{layer}"
        mapping.add_mapping(f"{prefix}.ln_1.weight", f"{prefix}.ln_1.weight")
        mapping.add_mapping(f"{prefix}.ln_1.bias", f"{prefix}.ln_1.bias")
        mapping.add_mapping(
            f"{prefix}.attn.c_attn.weight",
            f"{prefix}.attn.c_attn.weight",
            _transpose_conv1d,
        )
        mapping.add_mapping(f"{prefix}.attn.c_attn.bias", f"{prefix}.attn.c_attn.bias")
        mapping.add_mapping(
            f"{prefix}.attn.c_proj.weight",
            f"{prefix}.attn.c_proj.weight",
            _transpose_conv1d,
        )
        mapping.add_mapping(f"{prefix}.attn.c_proj.bias", f"{prefix}.attn.c_proj.bias")
        mapping.add_mapping(f"{prefix}.ln_2.weight", f"{prefix}.ln_2.weight")
        mapping.add_mapping(f"{prefix}.ln_2.bias", f"{prefix}.ln_2.bias")
        mapping.add_mapping(
            f"{prefix}.mlp.c_fc.weight",
            f"{prefix}.mlp.c_fc.weight",
            _transpose_conv1d,
        )
        mapping.add_mapping(f"{prefix}.mlp.c_fc.bias", f"{prefix}.mlp.c_fc.bias")
        mapping.add_mapping(
            f"{prefix}.mlp.c_proj.weight",
            f"{prefix}.mlp.c_proj.weight",
            _transpose_conv1d,
        )
        mapping.add_mapping(f"{prefix}.mlp.c_proj.bias", f"{prefix}.mlp.c_proj.bias")
        mapping.add_unused(f"{prefix}.attn.bias", f"{prefix}.attn.masked_bias")
    if not config.tie_word_embeddings:
        mapping.add_mapping("lm_head.weight", "lm_head.weight")
    return mapping


def load_gpt2_weights(
    model: GPT2ForCausalLM,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    validate: bool = True,
) -> frozenset[str]:
    """Load an HF GPT-2 safetensors checkpoint, transposing Conv1D storage.

    A checkpoint that ships ``lm_head.weight`` while the config ties embeddings
    is tolerated (the tied storage stays authoritative), matching vLLM. The
    causal mask buffers of legacy exports are skipped, and hub exports without
    the ``transformer.`` prefix are accepted.
    """
    return load_checkpoint(
        model,
        checkpoint_dir,
        expected=lambda dtype: gpt2_expected_weights(model.config, dtype),
        bindings=gpt2_weight_mapping(model.config),
        ignored=_IGNORED_CHECKPOINT_KEYS,
        tolerate_tied_in_checkpoint=True,
        normalize_checkpoint_key=_normalize_checkpoint_key,
        device=device,
        validate=validate,
    )


def _checkpoint_document(config: GPT2Config) -> dict[str, Any]:
    document: dict[str, Any] = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "n_positions": config.max_position_embeddings,
        "n_embd": config.hidden_size,
        "n_layer": config.num_hidden_layers,
        "n_head": config.num_attention_heads,
        "n_inner": config.intermediate_size,
        "activation_function": config.activation_function,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "initializer_range": config.initializer_range,
        "resid_pdrop": config.resid_pdrop,
        "embd_pdrop": config.embd_pdrop,
        "attn_pdrop": config.attn_pdrop,
        "scale_attn_weights": config.scale_attn_weights,
        "scale_attn_by_inverse_layer_idx": config.scale_attn_by_inverse_layer_idx,
        "reorder_and_upcast_attn": config.reorder_and_upcast_attn,
        "use_cache": config.use_cache,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "tie_word_embeddings": config.tie_word_embeddings,
    }
    if config.torch_dtype:
        document["torch_dtype"] = config.torch_dtype
    return document


def write_gpt2_checkpoint(
    root: Path,
    config: GPT2Config,
    *,
    seed: int = 1234,
) -> Path:
    """Write a deterministic checkpoint in HF GPT-2 naming and Conv1D storage."""
    return write_checkpoint(
        root,
        architectures="GPT2LMHeadModel",
        config=config,
        specs=gpt2_expected_weights(config, config.resolved_dtype()),
        unit_weight_suffixes=("ln_1.weight", "ln_2.weight", "ln_f.weight"),
        zero_bias_suffixes=("ln_1.bias", "ln_2.bias", "ln_f.bias"),
        document=_checkpoint_document(config),
        seed=seed,
    )


class _GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, *, device, dtype, backend: LayerBackend) -> None:
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
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
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
        output = attention(layer_index, query, key, value)
        reshaped = output.reshape(tokens, self.num_heads * self.head_dim)
        projected = self.c_proj(reshaped)
        assert isinstance(projected, torch.Tensor)
        return projected


class _GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.mlp_inner_size
        self.c_fc = ColumnParallelLinear(
            hidden,
            inner,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.c_proj = RowParallelLinear(
            inner,
            hidden,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            device=device,
        )
        self.activation = get_act_fn(config.activation_function, backend=backend)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        fused = self.c_fc(hidden)
        assert isinstance(fused, torch.Tensor)
        return self.c_proj(self.activation(fused))


class _GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.ln_1 = LayerNorm(
            hidden,
            eps=config.layer_norm_epsilon,
            bias=True,
            device=device,
            dtype=dtype,
            backend=backend,
        )
        self.attn = _GPT2Attention(config, device=device, dtype=dtype, backend=backend)
        self.ln_2 = LayerNorm(
            hidden,
            eps=config.layer_norm_epsilon,
            bias=True,
            device=device,
            dtype=dtype,
            backend=backend,
        )
        self.mlp = _GPT2MLP(config, device=device, dtype=dtype, backend=backend)

    def forward(
        self,
        hidden: torch.Tensor,
        attention: AttentionCallback,
        layer_index: int,
    ) -> torch.Tensor:
        attended = self.attn(self.ln_1(hidden), attention, layer_index)
        hidden = hidden + attended
        fed_forward = self.mlp(self.ln_2(hidden))
        return hidden + fed_forward


class _GPT2Model(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
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
        self.wpe = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, device=device, dtype=dtype
        )
        self.h = nn.ModuleList(
            [
                _GPT2Block(config, device=device, dtype=dtype, backend=backend)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            bias=True,
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
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a half-open layer range of the decoder.

        ``layer_start``/``layer_end`` slice the transformer for one PP stage. A
        non-zero ``layer_start`` consumes the previous stage's ``state`` (the
        boundary hidden tensor) instead of token ids, and a ``layer_end`` below
        the layer count returns that boundary tensor so the next stage can
        resume. GPT-2's LayerNorm does not fuse the residual, so the boundary is
        a single tensor rather than Qwen's ``(hidden, residual)`` pair. Layer
        indices stay GLOBAL so attention callbacks index KV slots exactly as a
        full forward does; the final ``ln_f`` only runs on the last stage.

        ``inputs_embeds`` replaces the token lookup but still receives the
        learned position embedding; ``skip_embed`` marks the first argument as
        a complete hidden state instead, so neither embedding runs.
        """
        num_layers = len(self.h)
        end = resolve_layer_range(num_layers, layer_start, layer_end)
        if layer_start == 0:
            if state is not None:
                raise ValueError("layer_start 0 must not carry a boundary state")
            if skip_embed and inputs_embeds is not None:
                raise ValueError("skip_embed and inputs_embeds are mutually exclusive")
            if skip_embed:
                hidden = token_ids
            else:
                embeddings = self.wte(token_ids) if inputs_embeds is None else inputs_embeds
                hidden = embeddings + self.wpe(positions)
        else:
            if state is None:
                raise ValueError("a non-first stage requires the previous stage's state")
            if inputs_embeds is not None or skip_embed:
                raise ValueError(
                    "inputs_embeds, skip_embed and a boundary state are mutually exclusive"
                )
            hidden = state
        for layer_index in range(layer_start, end):
            hidden = self.h[layer_index](hidden, attention, layer_index)
        if end < num_layers:
            return hidden
        return self.ln_f(hidden)


class GPT2ForCausalLM(CausalLM[GPT2Config]):
    """Dense GPT-2 decoder whose attention is supplied by the runtime per forward."""

    transformer: _GPT2Model
    _decoder_name = "transformer"

    def __init__(
        self,
        config: GPT2Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: LayerBackend = "triton",
        parallel_context: ParallelContext | None = None,
    ) -> None:
        super().__init__()
        self._attach_decoder(
            config,
            _GPT2Model(
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
    def model(self) -> _GPT2Model:
        """Alias for runtimes that reach the decoder as ``model``."""
        return self.transformer
