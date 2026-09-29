"""PEFT and native tensor ingestion; validation completes before residency mutation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping

import torch
from safetensors import safe_open

from ayaka.lora.config import LoRAConfig
from ayaka.lora.mapping import ProjectionTarget
from ayaka.lora.peft import PeftConfig
from ayaka.lora.resolver import ResolvedAdapter
from ayaka.lora.variant import AdapterIdentity
from ayaka.lora.weights import AdapterWeights, LoRALayerWeights, LoRAWeights

_KEY = re.compile(r"^(.*)\.lora_([AB])(?:\.([^.]+))?\.weight$")
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class LoRALoader:
    def __init__(
        self,
        config: LoRAConfig,
        targets: tuple[ProjectionTarget, ...],
        mapping: Mapping[str, str],
        dtype: torch.dtype,
        *,
        base_model: str | None = None,
    ) -> None:
        self.config, self.targets, self.mapping, self.dtype = config, targets, dict(mapping), dtype
        self.base_model = base_model

    def load(self, resolved: ResolvedAdapter) -> AdapterWeights:
        peft = PeftConfig.read(resolved.config_path, max_rank=self.config.max_rank)
        if peft.base_model_name_or_path and not self.base_model:
            raise ValueError(
                "set LoRAConfig.base_model_name_or_path to validate the adapter's base model; "
                "the served API alias is not a checkpoint identity"
            )
        if (
            self.base_model
            and peft.base_model_name_or_path
            and peft.base_model_name_or_path != self.base_model
        ):
            raise ValueError("adapter is incompatible with the configured base model identity")

        def tensors() -> Iterable[tuple[str, torch.Tensor]]:
            for path in resolved.weight_files:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        yield key, handle.get_tensor(key)

        return self.from_peft(resolved.identity, peft, tensors())

    def from_peft(
        self,
        identity: AdapterIdentity,
        peft: PeftConfig,
        tensors: Iterable[tuple[str, torch.Tensor]],
    ) -> AdapterWeights:
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        seen: set[str] = set()
        for original, tensor in tensors:
            if original in seen:
                raise ValueError(f"duplicate checkpoint key {original!r}")
            seen.add(original)
            key = original.removeprefix("base_model.model.")
            match = _KEY.fullmatch(key)
            if match is None:
                if "lora_embedding_" in key:
                    raise ValueError("embedding LoRA requires P1 support")
                raise ValueError(f"unknown checkpoint key {original!r}")
            module, which, _adapter_name = match.groups()
            if not any(module == t or module.endswith("." + t) for t in peft.target_modules):
                raise ValueError(f"checkpoint module {module!r} is outside target_modules")
            if module not in self.mapping:
                raise ValueError(f"wrong or unsupported target module {module!r}")
            target = self.mapping[module]
            pair = pairs.setdefault(target, {})
            if which in pair:
                raise ValueError(f"duplicate module key for {target!r} LoRA {which}")
            pair[which] = tensor
        weights: dict[str, LoRAWeights] = {}
        for target, pair in pairs.items():
            if set(pair) != {"A", "B"}:
                raise ValueError(f"missing A/B tensor for {target!r}")
            if pair["A"].ndim != 2 or pair["A"].shape[0] != peft.r:
                raise ValueError(f"checkpoint rank differs from PEFT config for {target!r}")
            weights[target] = LoRAWeights(pair["A"], pair["B"], peft.scale)
        return self.from_native(identity, weights, alpha=peft.lora_alpha)

    def from_native(
        self,
        identity: AdapterIdentity,
        weights: Mapping[str, LoRAWeights],
        *,
        alpha: float | None = None,
    ) -> AdapterWeights:
        if not isinstance(identity, AdapterIdentity) or not weights:
            raise ValueError("adapter requires immutable identity and nonempty weights")
        target_by_key = {t.key: t for t in self.targets}
        expanded: dict[str, LoRAWeights] = {}
        for name, value in weights.items():
            if not isinstance(value, LoRAWeights):
                raise TypeError("weights must contain LoRAWeights")
            if name in target_by_key:
                destinations = [(target_by_key[name], value.b)]
            else:
                parts = [t for t in self.targets if t.module == name]
                if (
                    not parts
                    or value.b.ndim != 2
                    or value.b.shape[0] != sum(t.output_size for t in parts)
                ):
                    raise ValueError(f"wrong target module or packed B shape: {name!r}")
                destinations = [
                    (t, value.b.narrow(0, t.global_output_offset, t.output_size)) for t in parts
                ]
            for target, b in destinations:
                if target.key in expanded:
                    raise ValueError(f"duplicate logical projection {target.key!r}")
                expanded[target.key] = LoRAWeights(value.a, b, value.scale)
        layers = []
        digest = hashlib.sha256(identity.prefix_key.encode())
        digest.update(repr((self.config.modules, str(self.dtype), "lora-canonical-v1")).encode())
        for key, value in sorted(expanded.items()):
            target = target_by_key[key]
            a, b = value.a, value.b
            rank = a.shape[0] if a.ndim == 2 else 0
            if (
                not 1 <= rank <= self.config.max_rank
                or a.shape != (rank, target.input_size)
                or b.shape != (target.output_size, rank)
            ):
                raise ValueError(f"invalid LoRA rank/shape for {key}")
            if a.dtype not in _DTYPES or b.dtype != a.dtype or a.is_meta or b.is_meta:
                raise ValueError(f"invalid LoRA dtype/device for {key}")
            if not isinstance(value.scale, (int, float)) or not math.isfinite(value.scale):
                raise ValueError("LoRA scale must be finite")
            # Clone BEFORE hashing. Caller retains no mutable alias into either cache.
            a = a.detach().to(device="cpu", copy=True).contiguous()
            b = b.detach().to(device="cpu", copy=True).contiguous()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite adapter weights")
            runtime_a = a.to(self.dtype)
            runtime_b = (b.float() * value.scale).to(self.dtype)
            if not torch.isfinite(runtime_a).all() or not torch.isfinite(runtime_b).all():
                raise ValueError("nonfinite adapter weights after runtime cast/scaling")
            digest.update(
                repr((key, tuple(a.shape), tuple(b.shape), a.dtype, value.scale)).encode()
            )
            for tensor in (a, b):
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
            layers.append(
                LoRALayerWeights(
                    key,
                    rank,
                    rank * value.scale if alpha is None else alpha,
                    float(value.scale),
                    a,
                    b,
                    a.dtype,
                    self.dtype,
                    (target.output_size, target.input_size),
                    (target.local_output_size, target.local_input_size),
                    target.output_offset,
                )
            )
        result = AdapterWeights(identity, tuple(layers), digest.hexdigest())
        if result.nbytes > self.config.host_memory_bytes:
            raise MemoryError("adapter exceeds host_lora_bytes budget")
        return result
