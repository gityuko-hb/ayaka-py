"""FP8 checkout adapter: recipe detection, scale contracts, and sharding.

The checkpoint only ships a ``quantization_config`` mapping and named scale
tensors; nothing in the storage layer knows what those names or directions
mean. This module is the single place that turns them into the explicit
:class:`~ayaka.weights.spec.FP8RecipeSpec` contract, so a recipe cannot be
inferred from a dtype alone and an inverse scale is inverted exactly once:

``quantization_config`` → :func:`parse_fp8_config` → :class:`ParsedFP8Config`
    → :class:`~ayaka.layers.linear.fp8.FP8Config` → linear method selection.

It covers the six recipes in ``docs/fp8-quantization.md``: static per-tensor,
dynamic per-tensor, rowwise, block128, MXFP8, W8A16, plus the KV-cache scale
contract owned by :class:`~ayaka.kvcache.storage.quantization.KVQuantization`.
Real model certification is a separate gate; this module only makes the
metadata and tensor contracts explicit and validated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ayaka.types import DType
from ayaka.utils.math_utils import div_ceil
from ayaka.weights.spec import (
    FP8ActivationScheme,
    FP8Recipe,
    FP8RecipeSpec,
    FP8ScaleDirection,
    FP8ScaleFormat,
    FP8ValueFormat,
    FP8ZeroPolicy,
    QuantSpec,
)

__all__ = [
    "FP8_SCALE_KEY_SUFFIXES",
    "ParsedFP8Config",
    "classify_scale_key",
    "expected_scale_keys",
    "normalize_scale",
    "parse_fp8_config",
    "recipe_from_config",
    "shard_scale_k",
    "validate_scale_key_set",
]

#: Amax floor the dynamic recipes use when the config does not declare one.
_DYNAMIC_FLOOR = 1e-10

#: Scale tensor suffixes a checkpoint may carry, mapped to the recipe that owns
#: them. ``weight_scale_inv`` is an inverse scale: the loader inverts it once and
#: the metadata keeps the original direction.
FP8_SCALE_KEY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "weight_scale": ("weight_scale",),
    "weight_scale_inv": ("weight_scale_inv",),
    "input_scale": ("input_scale",),
    "kv_scale": ("k_scale", "v_scale"),
}

_SCALE_FORMAT_BY_DTYPE: dict[torch.dtype, FP8ScaleFormat] = {
    torch.float32: FP8ScaleFormat.FP32,
    torch.bfloat16: FP8ScaleFormat.BF16,
    torch.uint8: FP8ScaleFormat.UE8M0,
}

_DTYPE_BY_SCALE_FORMAT: dict[FP8ScaleFormat, DType] = {
    FP8ScaleFormat.FP32: DType.FP32,
    FP8ScaleFormat.BF16: DType.BF16,
    FP8ScaleFormat.UE8M0: DType.UINT8,
}

_TORCH_DTYPE_BY_SCALE_FORMAT: dict[FP8ScaleFormat, torch.dtype] = {
    FP8ScaleFormat.FP32: torch.float32,
    FP8ScaleFormat.BF16: torch.bfloat16,
    FP8ScaleFormat.UE8M0: torch.uint8,
}


def _first(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _as_int_pair(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = value
        if all(isinstance(entry, int) and not isinstance(entry, bool) for entry in value):
            return (int(first), int(second))
    raise ValueError(f"{name} must be a (rows, K) integer pair, got {value!r}")


def _normalize_scale_format(value: Any) -> FP8ScaleFormat | None:
    if value is None:
        return None
    token = str(value).lower().replace("torch.", "")
    if token in ("ue8m0", "e8m0", "uint8"):
        return FP8ScaleFormat.UE8M0
    if token in ("bfloat16", "bf16"):
        return FP8ScaleFormat.BF16
    if token in ("float32", "fp32", "float"):
        return FP8ScaleFormat.FP32
    raise ValueError(f"unknown scale format {value!r}")


def _looks_inverse(config: Mapping[str, Any], explicit: Any) -> bool:
    if explicit is not None:
        return str(explicit).lower() in ("inverse", "inv", "scale_inv")
    return bool(_first(config, "use_scale_inv", "scale_inv", default=False)) or bool(
        _first(config, "weight_scale_inv", default=False)
    )


def _detect_recipe(config: Mapping[str, Any]) -> FP8Recipe:
    explicit = _first(config, "fp8_recipe", "recipe")
    if explicit is not None:
        try:
            return FP8Recipe(str(explicit))
        except ValueError as error:
            allowed = ", ".join(member.value for member in FP8Recipe)
            raise ValueError(
                f"unknown fp8 recipe {explicit!r}; expected one of: {allowed}"
            ) from error

    quant_algo = str(
        _first(config, "quant_algo", "quantization", "quant_method", "method", default="")
    ).lower()
    weight_block = _as_int_pair(
        _first(config, "weight_block_size", "weight_block", "weight_blocks"),
        "weight_block_size",
    )
    activation = str(_first(config, "activation_scheme", "activation", default="")).lower()
    scale_format = _normalize_scale_format(
        _first(config, "scale_format", "scale_dtype", "scale_fmt")
    )
    is_serialized = bool(
        _first(config, "is_checkpoint_fp8_serialized", "serialized", default=False)
    )
    kv_scheme = _first(config, "kv_cache_scheme", "kv_scheme")

    has_weight_quant = (
        bool(activation)
        or weight_block is not None
        or is_serialized
        or any(token in quant_algo for token in ("mxfp8", "w8a8", "w8a16"))
    )
    if kv_scheme is not None and not has_weight_quant:
        return FP8Recipe.KV_CACHE
    # Weight-only is decided before the format hints: a W8A16 checkpoint may
    # carry UE8M0 scales (and a weight block) without being an MXFP8 recipe.
    if activation in ("none", "weight_only", "weight-only") or "w8a16" in quant_algo:
        return FP8Recipe.W8A16
    if "mxfp8" in quant_algo or weight_block == (1, 32) or scale_format is FP8ScaleFormat.UE8M0:
        return FP8Recipe.MXFP8
    if weight_block == (128, 128) or "block" in quant_algo:
        if weight_block != (128, 128):
            raise ValueError("block FP8 requires weight_block_size=[128, 128]")
        return FP8Recipe.W8A8_BLOCK128
    if weight_block is not None:
        raise ValueError(
            f"unsupported weight_block_size {weight_block}; expected [128, 128] or [1, 32]"
        )
    # ``dynamic`` is the HF default; a rowwise checkpoint says so explicitly.
    rowwise_hint = str(
        _first(config, "weight_scheme", "weight_granularity", "weight_strategy", default="")
    ).lower()
    if "rowwise" in activation or "rowwise" in rowwise_hint:
        return FP8Recipe.W8A8_ROWWISE
    if activation in ("static", "calibrated"):
        return FP8Recipe.W8A8_STATIC_PER_TENSOR
    return FP8Recipe.W8A8_DYNAMIC_PER_TENSOR


def _scale_dtype_for(recipe: FP8Recipe, scale_format: FP8ScaleFormat | None) -> torch.dtype:
    if recipe is FP8Recipe.KV_CACHE:
        return torch.float32
    if recipe is FP8Recipe.MXFP8:
        return torch.uint8
    if scale_format is not None:
        return _TORCH_DTYPE_BY_SCALE_FORMAT[scale_format]
    return torch.bfloat16 if recipe is FP8Recipe.W8A8_BLOCK128 else torch.float32


def _activation_scheme_for(recipe: FP8Recipe) -> FP8ActivationScheme:
    if recipe is FP8Recipe.W8A16:
        return FP8ActivationScheme.NONE
    if recipe is FP8Recipe.W8A8_STATIC_PER_TENSOR:
        return FP8ActivationScheme.STATIC
    if recipe is FP8Recipe.KV_CACHE:
        return FP8ActivationScheme.CALIBRATED
    return FP8ActivationScheme.DYNAMIC


def _blocks_for(recipe: FP8Recipe) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    if recipe is FP8Recipe.W8A8_BLOCK128:
        return (1, 128), (128, 128)
    if recipe is FP8Recipe.MXFP8:
        return (1, 32), (1, 32)
    return None, None


@dataclass(frozen=True, slots=True)
class ParsedFP8Config:
    """Checkout metadata normalized into the explicit FP8 contract.

    ``quant`` is ``None`` for the KV-cache recipe: cache scales are owned by
    :class:`~ayaka.kvcache.storage.quantization.KVQuantization`, and a
    ``QuantSpec`` deliberately rejects that recipe.
    """

    recipe: FP8RecipeSpec
    quant: QuantSpec | None
    scale_dtype: torch.dtype
    ignored_layers: tuple[str, ...]

    @property
    def method_name(self) -> str:
        return self.recipe.recipe.value


def parse_fp8_config(config: Mapping[str, Any]) -> ParsedFP8Config:
    """Parse and validate one ``quantization_config`` mapping.

    Raises:
        TypeError: If ``config`` is not a mapping.
        ValueError: If the recipe, blocks, scale format, or calibration
            provenance are missing or contradictory.
    """
    if not isinstance(config, Mapping):
        raise TypeError("quantization_config must be a mapping")

    recipe = _detect_recipe(config)
    scale_format = _normalize_scale_format(
        _first(config, "scale_format", "scale_dtype", "scale_fmt")
    )
    if recipe is FP8Recipe.W8A8_BLOCK128 and scale_format is None:
        scale_format = FP8ScaleFormat.BF16
    scale_dtype = _scale_dtype_for(recipe, scale_format)
    # Re-derive the format from the resolved dtype so spec and storage agree.
    scale_format = (
        FP8ScaleFormat.FP32 if recipe is FP8Recipe.KV_CACHE else _SCALE_FORMAT_BY_DTYPE[scale_dtype]
    )

    activation_block, weight_block = _blocks_for(recipe)
    if recipe is FP8Recipe.W8A16:
        weight_block = _as_int_pair(
            _first(config, "weight_block_size", "weight_block", "weight_blocks"),
            "weight_block_size",
        )

    activation_scheme = _activation_scheme_for(recipe)
    if recipe is not FP8Recipe.W8A16 and recipe is not FP8Recipe.KV_CACHE:
        configured = str(_first(config, "activation_scheme", "activation", default="")).lower()
        if configured in ("none", "weight_only", "weight-only"):
            raise ValueError(f"recipe {recipe.value!r} requires an activation scheme, got 'none'")

    zero_policy = (
        FP8ZeroPolicy.CHECKPOINT_SCALE
        if activation_scheme in (FP8ActivationScheme.STATIC, FP8ActivationScheme.CALIBRATED)
        or recipe is FP8Recipe.W8A16
        else FP8ZeroPolicy.AMAX_FLOOR
    )
    amax_floor: float | None = None
    if zero_policy is FP8ZeroPolicy.AMAX_FLOOR:
        raw_floor = _first(config, "amax_floor", default=_DYNAMIC_FLOOR)
        amax_floor = float(raw_floor)

    calibration_source: str | None = None
    if (
        activation_scheme
        in (
            FP8ActivationScheme.STATIC,
            FP8ActivationScheme.CALIBRATED,
        )
        or recipe is FP8Recipe.W8A16
    ):
        raw_source = _first(
            config,
            "calibration_source",
            "calibration",
            "calibrated_by",
            default="quantization_config",
        )
        calibration_source = str(raw_source)

    direction = (
        FP8ScaleDirection.INVERSE
        if _looks_inverse(config, _first(config, "scale_direction"))
        else FP8ScaleDirection.DEQUANT
    )
    layout = str(_first(config, "logical_scale_layout", "scale_layout", default="row_major"))

    recipe_spec = FP8RecipeSpec(
        recipe=recipe,
        value_format=FP8ValueFormat.E4M3,
        activation_scheme=activation_scheme,
        activation_block=activation_block,
        weight_block=weight_block,
        scale_format=scale_format,
        scale_direction=direction,
        logical_scale_layout=layout,
        physical_scale_layout=str(_first(config, "physical_scale_layout", default=layout)),
        zero_policy=zero_policy,
        amax_floor=amax_floor,
        calibration_source=calibration_source,
        excluded_layers=tuple(
            str(name) for name in _first(config, "ignored_layers", "ignored", "exclude", default=())
        ),
    )
    quant = (
        None
        if recipe is FP8Recipe.KV_CACHE
        else QuantSpec(
            method="fp8",
            weight_dtype=DType.FP8_E4M3,
            scale_dtype=_DTYPE_BY_SCALE_FORMAT[recipe_spec.scale_format],
            fp8_recipe=recipe_spec,
        )
    )
    return ParsedFP8Config(
        recipe=recipe_spec,
        quant=quant,
        scale_dtype=scale_dtype,
        ignored_layers=recipe_spec.excluded_layers,
    )


def recipe_from_config(
    config: Mapping[str, Any],
) -> tuple[FP8Recipe, torch.dtype, tuple[str, ...]]:
    """Adapter entry point used by :meth:`ayaka.layers.linear.fp8.FP8Config.from_config`."""
    parsed = parse_fp8_config(config)
    return parsed.recipe.recipe, parsed.scale_dtype, parsed.ignored_layers


def expected_scale_keys(recipe: FP8Recipe) -> tuple[str, ...]:
    """Scale tensor suffixes a checkpoint of ``recipe`` must ship."""
    if recipe in (
        FP8Recipe.W8A8_STATIC_PER_TENSOR,
        FP8Recipe.W8A8_DYNAMIC_PER_TENSOR,
        FP8Recipe.W8A8_ROWWISE,
        FP8Recipe.W8A8_BLOCK128,
        FP8Recipe.MXFP8,
        FP8Recipe.W8A16,
    ):
        keys = ["weight_scale_inv"] if recipe is FP8Recipe.W8A8_BLOCK128 else ["weight_scale"]
        if recipe is FP8Recipe.W8A8_STATIC_PER_TENSOR:
            keys.append("input_scale")
        return tuple(keys)
    if recipe is FP8Recipe.KV_CACHE:
        return ("k_scale", "v_scale")
    raise ValueError(f"unknown FP8 recipe {recipe!r}")


def classify_scale_key(key: str) -> str | None:
    """Return the scale kind a checkpoint tensor name carries, or ``None``."""
    name = key.rsplit(".", 1)[-1]
    for kind, suffixes in FP8_SCALE_KEY_SUFFIXES.items():
        if name in suffixes:
            return kind
    return None


def validate_scale_key_set(keys: Sequence[str], recipe: FP8Recipe) -> None:
    """Reject a checkpoint whose scale tensors do not match the recipe."""
    present = {classify_scale_key(key) for key in keys}
    present.discard(None)
    expected = set(expected_scale_keys(recipe))
    missing = expected - present
    if missing:
        raise ValueError(f"{recipe.value} checkpoint is missing scale tensors: {sorted(missing)}")
    for kind in present:
        if kind not in expected and kind != "kv_scale":
            raise ValueError(
                f"{recipe.value} checkpoint carries an unexpected scale tensor kind {kind!r}"
            )


def normalize_scale(scale: torch.Tensor, *, direction: FP8ScaleDirection) -> torch.Tensor:
    """Return the dequant scale, inverting an inverse checkpoint scale exactly once.

    Raises:
        ValueError: If an inverse scale is non-finite or non-positive (it would
            become a non-finite/zero dequant scale).
    """
    if scale.dtype is torch.uint8:
        # UE8M0 scales are exponent codes; the kernel decodes them and direction
        # does not apply.
        return scale
    if direction is FP8ScaleDirection.DEQUANT:
        return scale
    values = scale.float()
    if not bool(torch.isfinite(values).all().item()) or bool((values <= 0).any().item()):
        raise ValueError("inverse scale must be finite and positive")
    return 1.0 / scale


def shard_scale_k(
    scale: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    block: int = 128,
) -> torch.Tensor:
    """Slice a block scale's K axis for one tensor-parallel rank.

    A rank's local K shard must cover whole scale blocks; a checkpoint whose
    K/``block`` does not divide by ``world_size`` cannot be sharded safely.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be smaller than world_size")
    if block < 1:
        raise ValueError("block must be positive")
    if scale.ndim != 2:
        raise ValueError("scale must be 2-D [rows, K / block]")
    columns = int(scale.shape[1])
    if columns % world_size:
        raise ValueError(f"scale K blocks {columns} must divide by world_size={world_size}")
    if world_size == 1:
        return scale
    local = columns // world_size
    return scale[:, rank * local : (rank + 1) * local]


def scale_blocks_for_k(inner: int, block: int) -> int:
    """Number of scale columns a ``block``-wide K axis needs."""
    return div_ceil(inner, block)
