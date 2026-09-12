"""RoPE configuration and per-module, non-persistent cosine/sine cache."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload

import torch

from ayaka.layers._common import LayerBackend, check_dtype, check_input, load_kernel, positive_float
from ayaka.layers.base import BaseLayer
from ayaka.utils.validation import require_int


class RotaryEmbedding(BaseLayer):
    """Apply default or linearly scaled RoPE in place on query and optional key.

    Positions are contiguous int64 [tokens] or [batch, sequence]. Q/K have the
    same leading dimensions and either flattened heads or explicit [heads,
    head_size]. Q head count must be divisible by K head count. Disjoint Q/K
    views into a packed QKV tensor are supported; overlapping Q/K are forbidden.
    Only the first ``rotary_dim`` values in each head are rotated.

    The cache has shape [max_position_embeddings, rotary_dim], with cosine in
    its first half and sine in its second. Frequencies are computed in FP32 at
    construction, then cast to dtype. ``.to()`` moves/casts this registered buffer
    normally; it is excluded from state_dict. Forward never resizes the cache.
    Position bounds are checked on device before invoking the rotary kernel;
    a CUDA assertion reports invalid values asynchronously.
    """

    cos_sin_cache: torch.Tensor

    def __init__(
        self,
        head_size: int,
        rotary_dim: int | None = None,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        is_neox_style: bool = True,
        *,
        scaling_factor: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: LayerBackend = "triton",
        **runtime: Any,
    ) -> None:
        super().__init__(**runtime)
        if self.quant_config is not None:
            raise ValueError("RotaryEmbedding does not support quant_config")
        require_int(head_size, "head_size", minimum=1)
        rotary_dim = head_size if rotary_dim is None else rotary_dim
        require_int(rotary_dim, "rotary_dim", minimum=2)
        require_int(max_position_embeddings, "max_position_embeddings", minimum=1)
        if rotary_dim % 2 or rotary_dim > head_size:
            raise ValueError("rotary_dim must be even and cannot exceed head_size")
        if not isinstance(is_neox_style, bool):
            raise TypeError("is_neox_style must be a bool")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = positive_float(base, "base")
        self.scaling_factor = positive_float(scaling_factor, "scaling_factor")
        self.is_neox_style = is_neox_style
        self.backend: LayerBackend = backend
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        check_dtype(dtype)
        exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim
        inv_freq = 1.0 / (self.base**exponent)
        positions = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        angles = torch.outer(positions / self.scaling_factor, inv_freq)
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        if not cache.is_meta:
            self.runtime_context(cache.device, dtype)
        self.op = load_kernel(backend, "rope", "rotary_embedding")

    def _check_tensor(self, x: torch.Tensor, positions: torch.Tensor, name: str) -> int:
        check_input(x, name, self.backend)
        if x.device != positions.device or x.device != self.cos_sin_cache.device:
            raise ValueError(f"{name}, positions, and cache must be on the same device")
        if x.dtype != self.cos_sin_cache.dtype:
            raise TypeError(f"{name} and cache must have the same dtype")
        dims = positions.ndim
        if x.ndim not in (dims + 1, dims + 2) or x.shape[:dims] != positions.shape:
            raise ValueError(f"{name} must match token dimensions and have flat or explicit heads")
        if x.stride(-1) != 1:
            raise ValueError(f"{name}'s final dimension must be contiguous")
        if x.ndim == dims + 2:
            if x.shape[-1] != self.head_size:
                raise ValueError(f"{name}'s head dimension must equal head_size")
            heads = x.shape[-2]
            head_stride = x.stride(-2)
        else:
            if x.shape[-1] % self.head_size:
                raise ValueError(f"{name}'s flattened width must be divisible by head_size")
            heads = x.shape[-1] // self.head_size
            head_stride = self.head_size
        if heads < 1 or (heads > 1 and head_stride < self.head_size):
            raise ValueError(f"{name} must have nonempty, nonoverlapping heads")
        token_stride = x.stride(dims - 1)
        if positions.numel() > 1 and token_stride < (heads - 1) * head_stride + self.head_size:
            raise ValueError(f"{name} tokens must not overlap")
        if dims == 2 and x.stride(0) != positions.shape[1] * token_stride:
            raise ValueError(f"{name} must be linearly addressable across batch and sequence")
        return heads

    def _apply_native(self, x: torch.Tensor, positions: torch.Tensor, heads: int) -> None:
        shaped = x.view(*positions.shape, heads, self.head_size)
        cos, sin = self.cos_sin_cache[positions].float().chunk(2, dim=-1)
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        half = self.rotary_dim // 2
        first = slice(0, half) if self.is_neox_style else slice(0, self.rotary_dim, 2)
        second = (
            slice(half, self.rotary_dim) if self.is_neox_style else slice(1, self.rotary_dim, 2)
        )
        x1, x2 = shaped[..., first].float(), shaped[..., second].float()
        # Compute both before either write: float() can preserve FP32 input aliases.
        result1 = x1 * cos - x2 * sin
        result2 = x2 * cos + x1 * sin
        shaped[..., first] = result1.to(x.dtype)
        shaped[..., second] = result2.to(x.dtype)

    @overload
    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: None = None
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rotate in place and return query, or the exact (query, key) aliases."""
        if not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64:
            raise TypeError("positions must be an int64 tensor")
        if positions.ndim not in (1, 2) or not positions.is_contiguous():
            raise ValueError("positions must be contiguous [tokens] or [batch, sequence]")
        q_heads = self._check_tensor(query, positions, "query")
        k_heads = 0
        if key is not None:
            k_heads = self._check_tensor(key, positions, "key")
            if q_heads % k_heads:
                raise ValueError("query head count must be divisible by key head count")
            if key is query:
                raise ValueError("query and key must not alias")
        torch.ops.aten._assert_async.msg(
            ((positions >= 0) & (positions < self.max_position_embeddings)).all(),
            "positions must be within the RoPE cache bounds",
        )
        if self.op is not None:
            return self.run_kernel(
                self.op,
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        self._apply_native(query, positions, q_heads)
        if key is None:
            return query
        self._apply_native(key, positions, k_heads)
        return query, key

    def extra_repr(self) -> str:
        return (
            f"head_size={self.head_size}, rotary_dim={self.rotary_dim}, "
            f"max_position_embeddings={self.max_position_embeddings}, base={self.base}, "
            f"is_neox_style={self.is_neox_style}, scaling_factor={self.scaling_factor}, "
            f"backend={self.backend!r}"
        )


def get_rope(
    head_size: int,
    rotary_dim: int | None = None,
    max_position_embeddings: int = 8192,
    base: float = 10000.0,
    is_neox_style: bool = True,
    *,
    rope_scaling: Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: LayerBackend = "triton",
    **runtime: Any,
) -> RotaryEmbedding:
    """Build an independent default/linear RoPE module from explicit configuration.

    Scaling accepts ``{'rope_type': 'linear', 'factor': 2.0}`` (legacy ``type``
    is also accepted). Unsupported types/fields raise rather than silently using
    default RoPE. ``max_position_embeddings`` is the actual cache capacity, not
    automatically multiplied by factor. There is no shared mutable module cache.
    """
    factor = 1.0
    if rope_scaling is not None:
        if not isinstance(rope_scaling, Mapping):
            raise TypeError("rope_scaling must be a mapping")
        if set(rope_scaling) - {"rope_type", "type", "factor"}:
            raise ValueError("unsupported rope_scaling fields")
        kind = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if "type" in rope_scaling and rope_scaling["type"] != kind:
            raise ValueError("conflicting rope_type and type")
        if kind == "linear":
            if "factor" not in rope_scaling:
                raise ValueError("linear RoPE requires factor")
            factor = positive_float(rope_scaling["factor"], "factor")
        elif kind != "default":
            raise ValueError(f"unsupported rope_type {kind!r}")
        elif "factor" in rope_scaling:
            raise ValueError("default RoPE does not accept factor")
    return RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        scaling_factor=factor,
        device=device,
        dtype=dtype,
        backend=backend,
        **runtime,
    )
