"""Model-agnostic plumbing shared by Ayaka's dense decoder-only implementations.

Each model module keeps only what is specific to its architecture: config
fields and validation, the checkpoint tensor schema (expected weights,
bindings and mappings), and its attention/MLP/decoder modules. Parsing,
serialization, checkpoint I/O and the causal-LM outer contract live here so
four dense models cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, cast

import torch
import torch.nn.functional as F
from torch import nn

from ayaka.configs.model_source import ModelSourceConfig
from ayaka.distributed.parallel import ParallelContext
from ayaka.layers.embedding import ParallelLMHead, VocabParallelEmbedding
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
from ayaka.weights.plan import CheckpointManifest
from ayaka.weights.spec import TensorSpec, WeightSpec

__all__ = [
    "AttentionCallback",
    "CausalLM",
    "DecoderModule",
    "IgnoredCheckpointKeys",
    "ModelConfigMixin",
    "ROTARY_CACHE_KEYS",
    "checkpoint_dtype",
    "load_checkpoint",
    "resolve_layer_range",
    "weight_spec",
    "write_checkpoint",
]

AttentionCallback = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class ModelConfigMixin:
    """Shared HF config parsing, dtype resolution and serialization.

    Subclasses override the common fields with their architecture defaults and
    add their own; ``_CONFIG_ALIASES`` maps legacy HF keys to canonical names.
    """

    torch_dtype: str = ""
    tie_word_embeddings: bool = False
    hidden_size: int = 0
    vocab_size: int = 0

    model_type: ClassVar[str] = ""
    _CONFIG_ALIASES: ClassVar[Mapping[str, str]] = {}

    @property
    def scaling(self) -> float:
        raise NotImplementedError

    def resolved_dtype(self) -> DType:
        """Checkpoint dtype from ``torch_dtype``, else the HF FP32 default."""
        if self.torch_dtype:
            return DType.from_str(self.torch_dtype)
        return DType.FP32

    @classmethod
    def _normalize_config(cls, values: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in values.items():
            if key == "dtype" and "torch_dtype" not in values:
                # HF saves the checkpoint dtype as "dtype" from 5.x on.
                key = "torch_dtype"
            if key == "rope_parameters" and isinstance(value, Mapping):
                theta = value.get("rope_theta")
                if theta is not None:
                    normalized.setdefault("rope_theta", theta)
                partial = value.get("partial_rotary_factor")
                if partial is not None:
                    normalized.setdefault("partial_rotary_factor", partial)
                kind = value.get("rope_type", value.get("type", "default"))
                if kind != "default":
                    normalized["rope_scaling"] = {
                        "rope_type": kind,
                        "factor": value.get("factor", 1.0),
                    }
                continue
            normalized[cls._CONFIG_ALIASES.get(key, key)] = value
        return normalized

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Parse an HF ``config.json`` mapping, ignoring unrelated keys."""
        known = {field.name for field in fields(cls)}
        normalized = cls._normalize_config(values)
        return cls(**{key: value for key, value in normalized.items() if key in known})

    def arch_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IgnoredCheckpointKeys:
    """Checkpoint keys excluded from schema validation and from loading."""

    suffixes: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    substrings: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        if self.suffixes and name.endswith(self.suffixes):
            return True
        if self.prefixes and name.startswith(self.prefixes):
            return True
        return any(part in name for part in self.substrings)


#: Rotary caches some exports ship; RoPE is always computed by the model.
ROTARY_CACHE_KEYS = IgnoredCheckpointKeys(
    suffixes=("rotary_emb.inv_freq", "rotary_emb.cos_cached", "rotary_emb.sin_cached")
)

#: Default for loaders that consume every checkpoint tensor.
_NO_IGNORED_KEYS = IgnoredCheckpointKeys()


def weight_spec(name: str, shape: tuple[int, ...], tensor_dtype: DType) -> WeightSpec:
    """One expected checkpoint tensor with global shape and dtype."""
    return WeightSpec(
        name=name,
        full_shape=shape,
        spec=TensorSpec(shape=shape, dtype=tensor_dtype, name=name),
    )


def checkpoint_dtype(manifest: CheckpointManifest, config: ModelConfigMixin) -> DType:
    """The dtype the checkpoint actually stores, falling back to the config."""
    entries = manifest.entries
    if entries:
        return entries[0].dtype
    return config.resolved_dtype()


def resolve_layer_range(num_layers: int, layer_start: int, layer_end: int | None) -> int:
    """Validate a half-open decoder stage and return its exclusive end."""
    if not 0 <= layer_start < num_layers:
        raise ValueError(f"layer_start {layer_start} outside [0, {num_layers})")
    end = num_layers if layer_end is None else layer_end
    if layer_end is not None and not layer_start < layer_end <= num_layers:
        raise ValueError(
            f"layer_end {layer_end} must satisfy {layer_start} < layer_end <= {num_layers}"
        )
    return end


class DecoderModule(Protocol):
    """The inner decoder contract consumed by :class:`CausalLM`.

    ``wte`` may be a plain :class:`torch.nn.Embedding` or a vocabulary-parallel
    embedding; both accept token ids and expose a ``weight`` parameter that a
    tied LM head can share.
    """

    @property
    def wte(self) -> nn.Module: ...

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
        state: Any = None,
    ) -> Any: ...


class CausalLM[ConfigT: ModelConfigMixin](nn.Module):
    """Outer contract for a dense decoder whose attention is runtime-supplied.

    Subclasses register their decoder through :meth:`_attach_decoder`; the
    public attribute name (``model`` or ``transformer``) stays architecture
    specific and is declared by ``_decoder_name``.
    """

    config: ConfigT
    lm_head: nn.Module

    _decoder_name: ClassVar[str] = "model"

    def _attach_decoder(
        self,
        config: ConfigT,
        decoder: DecoderModule,
        *,
        lm_head_bias: bool,
        device: torch.device | str | None,
        dtype: torch.dtype,
        parallel_context: ParallelContext | None = None,
    ) -> None:
        """Register the decoder, the LM head and the tied/grad-free conventions.

        The head is vocabulary-parallel; with a single-rank context it is a
        dense projection whose output is already the full vocabulary.
        """
        self.config = config
        setattr(self, self._decoder_name, decoder)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=lm_head_bias,
            params_dtype=dtype,
            device=device,
            parallel_context=parallel_context,
            prefix="lm_head",
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(cast(VocabParallelEmbedding, decoder.wte))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _decoder(self) -> DecoderModule:
        return cast(DecoderModule, getattr(self, self._decoder_name))

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits, gathering shards when the head is sharded."""
        compute = getattr(self.lm_head, "logits", None)
        logits = compute(hidden) if callable(compute) else self.lm_head(hidden)
        assert isinstance(logits, torch.Tensor)
        return logits

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
        state: Any = None,
    ) -> Any:
        """Validate the embedding replacement, then run the decoder stage.

        ``skip_embed`` means the first argument already carries hidden states;
        it is mutually exclusive with ``inputs_embeds`` and with a boundary
        ``state``.
        """
        hidden_size = self.config.hidden_size
        if skip_embed:
            if inputs_embeds is not None:
                raise ValueError("skip_embed and inputs_embeds are mutually exclusive")
            if token_ids.ndim != 2 or token_ids.shape[-1] != hidden_size:
                raise ValueError("skip_embed requires hidden states shaped [tokens, hidden_size]")
        elif inputs_embeds is not None:
            if inputs_embeds.shape != (token_ids.numel(), hidden_size):
                raise ValueError("inputs_embeds must match token count and hidden size")
            if inputs_embeds.device != token_ids.device:
                raise ValueError("inputs_embeds and tokens must share a device")
        return self._decoder().forward_hidden(
            token_ids,
            positions,
            attention,
            skip_embed=skip_embed,
            inputs_embeds=inputs_embeds,
            layer_start=layer_start,
            layer_end=layer_end,
            state=state,
        )

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

        def attention(index: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
            q = query.transpose(0, 1).unsqueeze(0)
            k = key.transpose(0, 1).unsqueeze(0)
            v = value.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                scale=self.config.scaling,
                enable_gqa=q.shape[1] != k.shape[1],
            )
            return out.squeeze(0).transpose(0, 1)

        hidden = self.forward_hidden(token_ids, positions, attention)
        assert isinstance(hidden, torch.Tensor)
        return self.logits_from_hidden(hidden)


def load_checkpoint(
    model: CausalLM[Any],
    checkpoint_dir: str | Path,
    *,
    expected: Callable[[DType], Sequence[WeightSpec]],
    bindings: Mapping[str, WeightBinding] | ExternMapping,
    ignored: IgnoredCheckpointKeys = _NO_IGNORED_KEYS,
    tolerate_tied_in_checkpoint: bool = False,
    normalize_checkpoint_key: Callable[[str], str] | None = None,
    device: torch.device | str | None = None,
    validate: bool = True,
) -> frozenset[str]:
    """Load a safetensors checkpoint with the production bounded reader.

    ``expected`` receives the checkpoint dtype and returns the model's schema.
    Ignored keys are skipped in both validation and loading. With
    ``tolerate_tied_in_checkpoint`` a shipped ``lm_head.weight`` is accepted
    and dropped when the config ties embeddings (GPT-2 exports do this).
    ``normalize_checkpoint_key`` renames checkpoint keys before validation and
    binding, for exports whose module prefixes differ from the model's.
    Returns the checkpoint keys consumed.
    """
    resolved = resolve_source(ModelSourceConfig(model=str(checkpoint_dir)))
    manifest = build_manifest_from_source(resolved)
    if normalize_checkpoint_key is not None:
        manifest = replace(
            manifest,
            entries=tuple(
                replace(entry, tensor_key=normalize_checkpoint_key(entry.tensor_key))
                for entry in manifest.entries
            ),
        )
    if validate:
        diff = get_diff_weights(expected(checkpoint_dtype(manifest, model.config)), manifest)
        changes: dict[str, Any] = {
            "unexpected": tuple(name for name in diff.unexpected if not ignored.matches(name))
        }
        if tolerate_tied_in_checkpoint:
            changes["bad_tied"] = tuple(
                item for item in diff.bad_tied if item.name != "lm_head.weight"
            )
        replace(diff, **changes).raise_if_bad(context=str(checkpoint_dir))
    weights = (
        (name, tensor)
        for name, tensor in iter_checkpoint_tensors(manifest)
        if not ignored.matches(name)
        and not (
            tolerate_tied_in_checkpoint
            and model.config.tie_word_embeddings
            and name == "lm_head.weight"
        )
    )
    return load_module_weights(model, weights, bindings=bindings, device=device)


def write_checkpoint(
    root: Path,
    *,
    architectures: str,
    config: ModelConfigMixin,
    specs: Sequence[WeightSpec],
    unit_weight_suffixes: tuple[str, ...] = (),
    unit_weight_names: tuple[str, ...] = (),
    zero_bias_suffixes: tuple[str, ...] = (),
    document: Mapping[str, Any] | None = None,
    seed: int = 1234,
) -> Path:
    """Write a deterministic safetensors checkpoint for acceptance tests.

    Norm weights start at one and their biases at zero; other tensors use a
    small seeded normal so the values are distinguishable from a never-written
    buffer. Tied tensors are omitted, like every HF safetensors export.
    """
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    if document is None:
        document = {
            "architectures": [architectures],
            "model_type": config.model_type,
            **config.arch_dict(),
        }
    (root / "config.json").write_text(json.dumps(document, indent=2), encoding="utf-8")

    torch_dtype = config.resolved_dtype().torch_dtype
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for spec in specs:
        if spec.is_tied:
            continue
        values = torch.randn(spec.full_shape, generator=generator, dtype=torch.float32) * 0.02
        if spec.name in unit_weight_names or (
            unit_weight_suffixes and spec.name.endswith(unit_weight_suffixes)
        ):
            values = 1.0 + values
        elif zero_bias_suffixes and spec.name.endswith(zero_bias_suffixes):
            values = torch.zeros_like(values)
        tensors[spec.name] = values.to(torch_dtype)
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    return root
