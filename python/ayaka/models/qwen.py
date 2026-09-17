from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from ayaka.configs.model_source import ModelSourceConfig
from ayaka.layers._common import LayerBackend
from ayaka.layers.activation import SiluAndMul
from ayaka.layers.linear.attention import QKVParallelLinear
from ayaka.layers.linear.core import FusedGateUpLinear, RowParallelLinear
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rotary_embedding import get_rope
from ayaka.model_loader.manifest import build_manifest_from_source
from ayaka.model_loader.mapping import ExternMapping
from ayaka.model_loader.module import (
    WeightBinding,
    iter_checkpoint_tensors,
    load_module_weights,
)
from ayaka.model_loader.source import resolve_source
from ayaka.model_loader.validate import get_diff_weights
from ayaka.types import DType
from ayaka.utils.validation import require_int
from ayaka.weights.plan import CheckpointManifest
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

AttentionCallback = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_IGNORED_CHECKPOINT_SUFFIXES = ("rotary_emb.inv_freq",)


@dataclass(frozen=True, slots=True)
class QwenConfig:
    """HF ``QWenConfig`` fields.

    Inference-irrelevant fields are preserved so a checkpoint's ``config.json``
    round-trips through :meth:`from_dict` / :meth:`arch_dict`.
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
    tie_word_embeddings: bool = False
    torch_dtype: str = ""
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

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> QwenConfig:
        """Parse an HF ``config.json`` mapping, ignoring unrelated keys."""
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def arch_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        return WeightSpec(
            name=name,
            full_shape=shape,
            spec=TensorSpec(shape=shape, dtype=tensor_dtype, name=name),
            optional=optional,
        )

    specs: list[WeightSpec] = [spec("transformer.wte.weight", (vocab, hidden))]
    for layer in range(config.num_hidden_layers):
        prefix = f"transformer.h.{layer}"
        specs += [
            spec(f"{prefix}.attn.c_attn.weight", (3 * attn_width, hidden)),
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
    and whitelists rotary_emb.inv_freq.
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
        mapping.add_unused(f"{prefix}.attn.rotary_emb.inv_freq")

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

    ``rotary_emb.inv_freq`` tensors (present in some HF exports) are skipped:
    RoPE is computed by this module. Missing required tensors and shape
    mismatches raise; unknown tensors are never bound.
    """
    resolved = resolve_source(ModelSourceConfig(model=str(checkpoint_dir)))
    manifest = build_manifest_from_source(resolved)
    if validate:
        diff = get_diff_weights(
            qwen_expected_weights(model.config, _checkpoint_dtype(model, manifest)),
            manifest,
        )
        unexpected = tuple(
            name for name in diff.unexpected if not name.endswith(_IGNORED_CHECKPOINT_SUFFIXES)
        )
        replace(diff, unexpected=unexpected).raise_if_bad(context=str(checkpoint_dir))
    weights = (
        (name, tensor)
        for name, tensor in iter_checkpoint_tensors(manifest)
        if not name.endswith(_IGNORED_CHECKPOINT_SUFFIXES)
    )
    bindings = (
        qwen_weight_mapping(model.config) if use_mapping else qwen_weight_bindings(model.config)
    )
    return load_module_weights(model, weights, bindings=bindings, device=device)


def _checkpoint_dtype(model: QwenForCausalLM, manifest: CheckpointManifest) -> DType:
    entries = manifest.entries
    if entries:
        return entries[0].dtype
    return model.config.resolved_dtype()


def write_qwen_checkpoint(
    root: Path,
    config: QwenConfig,
    *,
    seed: int = 1234,
) -> Path:
    """Write a deterministic checkpoint in HF QWen naming order.

    Norm weights start at one; other tensors use a small seeded normal so the
    values are distinguishable from a never-written buffer.
    """
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    document = {
        "architectures": ["QWenLMHeadModel"],
        "model_type": config.model_type,
        **config.arch_dict(),
    }
    (root / "config.json").write_text(json.dumps(document, indent=2), encoding="utf-8")

    torch_dtype = config.resolved_dtype().torch_dtype
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for spec in qwen_expected_weights(config, config.resolved_dtype()):
        if spec.is_tied:
            continue
        values = torch.randn(spec.full_shape, generator=generator, dtype=torch.float32) * 0.02
        if (
            spec.name.endswith(("ln_1.weight", "ln_2.weight"))
            or spec.name == "transformer.ln_f.weight"
        ):
            values = 1.0 + values
        tensors[spec.name] = values.to(torch_dtype)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    return root


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
    def __init__(self, config: QwenConfig, *, device, dtype, backend: LayerBackend) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
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
        end = num_layers if layer_end is None else layer_end
        if not 0 <= layer_start < num_layers:
            raise ValueError(f"layer_start {layer_start} outside [0, {num_layers})")
        if layer_end is not None:
            if not layer_start < layer_end <= num_layers:
                raise ValueError(
                    f"layer_end {layer_end} must satisfy {layer_start} < layer_end <= {num_layers}"
                )
        residual_sum: torch.Tensor | None
        if layer_start == 0:
            if state is not None:
                raise ValueError("layer_start 0 must not carry a boundary state")
            hidden = self.wte(token_ids) if inputs_embeds is None else inputs_embeds
            residual_sum = None
        else:
            if state is None:
                raise ValueError("a non-first stage requires the previous stage's state")
            if inputs_embeds is not None:
                raise ValueError("inputs_embeds and a boundary state are mutually exclusive")
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


class QwenForCausalLM(nn.Module):
    """Dense QWen decoder whose attention is supplied by the runtime per forward."""

    def __init__(
        self,
        config: QwenConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: LayerBackend = "triton",
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer = _QwenModel(config, device=device, dtype=dtype, backend=backend)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward_hidden(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        attention: AttentionCallback,
        *,
        inputs_embeds: torch.Tensor | None = None,
        layer_start: int = 0,
        layer_end: int | None = None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            if inputs_embeds.shape != (token_ids.numel(), self.config.hidden_size):
                raise ValueError("inputs_embeds must match token count and hidden size")
            if inputs_embeds.device != token_ids.device:
                raise ValueError("inputs_embeds and tokens must share a device")
        return self.transformer.forward_hidden(
            token_ids,
            positions,
            attention,
            inputs_embeds=inputs_embeds,
            layer_start=layer_start,
            layer_end=layer_end,
            state=state,
        )

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        assert isinstance(logits, torch.Tensor)
        return logits

    def forward_dense(
        self,
        token_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Full-sequence oracle path with SDPA; used by acceptance tests only."""
        tokens = token_ids.shape[0]
        if positions is None:
            positions = torch.arange(tokens, device=token_ids.device, dtype=torch.int64)

        def attention(index: int, query, key, value):
            q = query.transpose(0, 1).unsqueeze(0)
            k = key.transpose(0, 1).unsqueeze(0)
            v = value.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, scale=self.config.scaling
            )
            return out.squeeze(0).transpose(0, 1)

        hidden = self.forward_hidden(token_ids, positions, attention)
        assert isinstance(hidden, torch.Tensor)
        return self.logits_from_hidden(hidden)
