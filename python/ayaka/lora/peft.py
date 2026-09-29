"""Strict inference subset of the PEFT LoRA configuration contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ayaka.utils.validation import require_int


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class PeftConfig:
    r: int
    lora_alpha: float
    target_modules: tuple[str, ...]
    use_rslora: bool = False
    bias: str = "none"
    modules_to_save: tuple[str, ...] = ()
    base_model_name_or_path: str | None = None

    @property
    def scale(self) -> float:
        return self.lora_alpha / (math.sqrt(self.r) if self.use_rslora else self.r)

    @classmethod
    def parse(cls, raw: Mapping[str, object], *, max_rank: int) -> PeftConfig:
        for key in ("r", "lora_alpha", "target_modules"):
            if key not in raw:
                raise ValueError(f"PEFT config missing {key}")
        rank = raw["r"]
        if not isinstance(rank, int):
            raise ValueError("PEFT rank must be an integer")
        require_int(rank, "PEFT rank", minimum=1)
        if rank > max_rank:
            raise ValueError("PEFT rank exceeds max_lora_rank")
        alpha = raw["lora_alpha"]
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (float, int))
            or not math.isfinite(alpha)
        ):
            raise ValueError("lora_alpha must be finite")
        targets = raw["target_modules"]
        if (
            not isinstance(targets, (list, tuple))
            or not targets
            or any(not isinstance(t, str) or not t.strip() for t in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("target_modules must be a nonempty unique list; regex is unsupported")
        if raw.get("peft_type", "LORA") != "LORA":
            raise ValueError("only PEFT LORA is supported")
        if raw.get("bias", "none") != "none":
            raise ValueError("adapter bias is unsupported; expected bias='none'")
        if raw.get("modules_to_save") not in (None, [], ()):
            raise ValueError("modules_to_save is unsupported in dense LoRA P0")
        for flag in ("use_rslora", "use_dora", "fan_in_fan_out", "lora_bias", "use_qalora"):
            if flag in raw and not isinstance(raw[flag], bool):
                raise ValueError(f"{flag} must be bool")
        for key in (
            "use_dora",
            "fan_in_fan_out",
            "lora_bias",
            "use_qalora",
            "rank_pattern",
            "alpha_pattern",
            "layer_replication",
            "layers_to_transform",
            "layers_pattern",
            "target_parameters",
            "trainable_token_indices",
            "alora_invocation_tokens",
            "megatron_config",
            "arrow_config",
            "eva_config",
            "corda_config",
            "exclude_modules",
        ):
            if raw.get(key):
                raise ValueError(f"unsupported PEFT feature {key}")
        # Benign training/export metadata are allowed; unknown semantics fail closed.
        allowed = {
            "r",
            "lora_alpha",
            "target_modules",
            "use_rslora",
            "bias",
            "modules_to_save",
            "base_model_name_or_path",
            "peft_type",
            "peft_version",
            "task_type",
            "inference_mode",
            "revision",
            "auto_mapping",
            "lora_dropout",
            "init_lora_weights",
            "loftq_config",
            "use_dora",
            "fan_in_fan_out",
            "lora_bias",
            "use_qalora",
            "qalora_group_size",
            "rank_pattern",
            "alpha_pattern",
            "layer_replication",
            "layers_to_transform",
            "layers_pattern",
            "target_parameters",
            "trainable_token_indices",
            "alora_invocation_tokens",
            "megatron_config",
            "megatron_core",
            "arrow_config",
            "eva_config",
            "corda_config",
            "exclude_modules",
            "ensure_weight_tying",
        }
        if unknown := set(raw) - allowed:
            raise ValueError(f"unknown PEFT config fields: {sorted(unknown)}")
        if raw.get("ensure_weight_tying"):
            raise ValueError("tied adapter weights are not certified in P0")
        base = raw.get("base_model_name_or_path")
        if base is not None and not isinstance(base, str):
            raise ValueError("base_model_name_or_path must be text")
        return cls(
            rank,
            float(alpha),
            tuple(targets),
            raw.get("use_rslora", False) is True,
            base_model_name_or_path=base,
        )

    @classmethod
    def read(cls, path: Path, *, max_rank: int) -> PeftConfig:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        if not isinstance(raw, dict):
            raise ValueError("adapter_config.json must contain an object")
        return cls.parse(raw, max_rank=max_rank)
