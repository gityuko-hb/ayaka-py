"""Analytical step cost for a dense decoder: estimates, never measurements (CT-O01).

The research asks for FLOPs/bytes models with explicit provenance and no
MFU or speed-of-light labels until device timing exists. Everything here is
derived from the model config and the step's per-request ``(q, p)`` shape: ``q``
query tokens computed this step on top of ``p`` tokens already in KV.

Counted: every weight matmul (2 FLOPs per multiply-add), causal attention
``QK^T`` and ``PV`` (query ``i`` of a slice attends ``p + i + 1`` keys), and the
LM head for each row that produces logits. Not counted: norms, activations,
softmax, RoPE, embedding gathers, sampling and padding. Bytes assume ideal reuse:
weights read once per step, each slice's ``p + q`` KV rows read once, ``q`` rows
written. Real kernels re-read and pad; treat these as lower bounds with
provenance, not achieved throughput.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ayaka.utils.validation import require_int

__all__ = ["DenseModelShape", "SliceShape", "StepCostEstimate", "estimate_step_cost"]

PROVENANCE = (
    "analytical dense-decoder model from the model config: weight matmuls, causal "
    "QK^T/PV and LM-head rows; excludes norms, activations, softmax, RoPE, gathers "
    "and padding; ideal-reuse bytes. An estimate, not a measurement"
)


@dataclass(frozen=True, slots=True)
class DenseModelShape:
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    #: SwiGLU-style MLP (gate + up + down) rather than up + down.
    gated_mlp: bool = True
    tied_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "hidden_size",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
        ):
            require_int(getattr(self, name), name, minimum=1)

    @classmethod
    def from_config(cls, config: object) -> DenseModelShape:
        """Read a Qwen2/Llama-style config (``resolved_head_dim`` when present)."""

        def get(*names: str):
            for name in names:
                value = getattr(config, name, None)
                if value is not None:
                    return value
            raise ValueError(f"model config has none of {names}")

        heads = int(get("num_attention_heads"))
        hidden = int(get("hidden_size"))
        head_dim = getattr(config, "resolved_head_dim", None) or getattr(config, "head_dim", None)
        return cls(
            num_layers=int(get("num_hidden_layers")),
            hidden_size=hidden,
            num_heads=heads,
            num_kv_heads=int(get("num_kv_heads", "num_key_value_heads")),
            head_dim=int(head_dim or hidden // heads),
            intermediate_size=int(get("intermediate_size")),
            vocab_size=int(get("vocab_size")),
            tied_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    @property
    def linear_params_per_layer(self) -> int:
        q = self.hidden_size * self.num_heads * self.head_dim
        kv = 2 * self.hidden_size * self.num_kv_heads * self.head_dim
        o = self.num_heads * self.head_dim * self.hidden_size
        mlp = (3 if self.gated_mlp else 2) * self.hidden_size * self.intermediate_size
        return q + kv + o + mlp

    @property
    def lm_head_params(self) -> int:
        return self.hidden_size * self.vocab_size

    @property
    def kv_values_per_token(self) -> int:
        """K and V values one token stores across every layer."""
        return 2 * self.num_kv_heads * self.head_dim * self.num_layers


@dataclass(frozen=True, slots=True)
class SliceShape:
    #: Tokens computed for this request in this step.
    query_tokens: int
    #: Tokens already in KV before this step (cached prefix or earlier chunks).
    prefix_tokens: int = 0
    #: Rows that produce logits: 1 for a sampling slice, 0 for a mid-prompt chunk.
    logits_rows: int = 1

    def __post_init__(self) -> None:
        require_int(self.query_tokens, "query_tokens")
        require_int(self.prefix_tokens, "prefix_tokens")
        require_int(self.logits_rows, "logits_rows")
        if self.logits_rows > self.query_tokens:
            raise ValueError("a slice cannot produce more logits rows than query tokens")

    @property
    def attended_keys(self) -> int:
        """Sum over queries of keys attended under a causal mask."""
        q, p = self.query_tokens, self.prefix_tokens
        return q * p + q * (q + 1) // 2


@dataclass(frozen=True, slots=True)
class StepCostEstimate:
    linear_flops: int
    attention_flops: int
    lm_head_flops: int
    weight_bytes: int
    kv_read_bytes: int
    kv_write_bytes: int
    #: Always True: no field of this record was measured.
    estimated: bool = True
    provenance: str = PROVENANCE

    @property
    def total_flops(self) -> int:
        return self.linear_flops + self.attention_flops + self.lm_head_flops

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "linear_flops": self.linear_flops,
            "attention_flops": self.attention_flops,
            "lm_head_flops": self.lm_head_flops,
            "total_flops": self.total_flops,
            "weight_bytes": self.weight_bytes,
            "kv_read_bytes": self.kv_read_bytes,
            "kv_write_bytes": self.kv_write_bytes,
            "total_bytes": self.total_bytes,
            "estimated": self.estimated,
            "provenance": self.provenance,
        }


def estimate_step_cost(
    shape: DenseModelShape,
    slices: Sequence[SliceShape],
    *,
    weight_bytes_per_value: int,
    kv_bytes_per_value: int,
) -> StepCostEstimate:
    """Estimate one step's work; an empty step costs nothing."""
    require_int(weight_bytes_per_value, "weight_bytes_per_value", minimum=1)
    require_int(kv_bytes_per_value, "kv_bytes_per_value", minimum=1)
    tokens = sum(s.query_tokens for s in slices)
    if tokens == 0:
        return StepCostEstimate(0, 0, 0, 0, 0, 0)
    attended = sum(s.attended_keys for s in slices)
    logits_rows = sum(s.logits_rows for s in slices)
    linear = 2 * shape.linear_params_per_layer * shape.num_layers * tokens
    attention = 4 * shape.num_heads * shape.head_dim * shape.num_layers * attended
    lm_head = 2 * shape.lm_head_params * logits_rows
    weights = shape.linear_params_per_layer * shape.num_layers
    weights += shape.lm_head_params if logits_rows else 0
    kv_rows_read = sum(s.prefix_tokens + s.query_tokens for s in slices if s.query_tokens)
    return StepCostEstimate(
        linear_flops=linear,
        attention_flops=attention,
        lm_head_flops=lm_head,
        weight_bytes=weights * weight_bytes_per_value,
        kv_read_bytes=kv_rows_read * shape.kv_values_per_token * kv_bytes_per_value,
        kv_write_bytes=tokens * shape.kv_values_per_token * kv_bytes_per_value,
    )
