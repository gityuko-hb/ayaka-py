"""Versioned operator intent and immutable per-phase execution policy.

Requested JSON is retained verbatim (including explicit defaults). Resolution
never changes an explicit bucket set or aliases one backend to another.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from ayaka.configs.base import validate_buckets
from ayaka.utils.validation import require_int

BACKENDS = ("eager", "full", "breakable", "torch_compile_piecewise")


def _object(value: Any, allowed: set[str], path: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {path} fields: {sorted(unknown)}")
    return value


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    backend: str = "eager"
    buckets: tuple[int, ...] = ()
    max_requests: int = 1
    memory_bytes: int = 0
    compile_seconds: int = 120
    max_compile_variants: int = 8
    max_padding_ratio: float = 8.0

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown execution backend: {self.backend}")
        for key in ("max_requests", "memory_bytes", "compile_seconds", "max_compile_variants"):
            require_int(getattr(self, key), key, minimum=0 if key == "memory_bytes" else 1)
        if self.memory_bytes >= 2**63:
            raise ValueError("memory_bytes exceeds int64 budget")
        if type(self.max_padding_ratio) not in (
            int,
            float,
        ) or not 1 <= self.max_padding_ratio < float("inf"):
            raise ValueError("max_padding_ratio must be finite and >= 1")
        object.__setattr__(self, "max_padding_ratio", float(self.max_padding_ratio))
        if self.buckets:
            validate_buckets(self.buckets)
        if self.backend != "eager" and (not self.buckets or not self.memory_bytes):
            raise ValueError("graph phases require explicit buckets and memory_bytes")

    @property
    def scope(self) -> str:
        return {
            "eager": "eager",
            "full": "model_forward_and_logits",
            "breakable": "transformer_body_graph_eager_logits",
            "torch_compile_piecewise": "eager_body_compiled_logits_graph",
        }[self.backend]


@dataclass(frozen=True, slots=True)
class ResolvedExecutionConfig:
    decode: PhaseConfig
    prefill: PhaseConfig
    memory_bytes: int
    max_captures: int
    provenance: tuple[tuple[str, str], ...]
    schema_version: int = 1

    @property
    def semantic_hash(self) -> str:
        values = asdict(self)
        values.pop("provenance")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def requires_rebuild(self, other: ResolvedExecutionConfig) -> bool:
        """Any effective change requires drain/rebuild; no live pointer retargeting."""
        return self.semantic_hash != other.semantic_hash


@dataclass(frozen=True, slots=True)
class RequestedOverrides:
    _json: str

    @classmethod
    def parse(cls, value: str | dict) -> RequestedOverrides:
        data = json.loads(value) if isinstance(value, str) else value
        _object(
            data, {"schema_version", "phases", "graph_memory", "full_prefill_max_req"}, "execution"
        )
        version = data.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("execution schema_version must be 1")
        phases = _object(data.get("phases", {}), {"decode", "prefill"}, "phases")
        for name, phase in phases.items():
            allowed = {
                "backend",
                "memory_bytes",
                "max_padding_ratio",
                "full",
                "breakable",
                "torch_compile_piecewise",
            }
            allowed |= (
                {"request_buckets"} if name == "decode" else {"token_buckets", "max_requests"}
            )
            _object(phase, allowed, name)
            backend = phase.get("backend", "eager")
            if backend not in BACKENDS:
                raise ValueError(f"unknown execution backend: {backend}")
            for option in ("full", "breakable", "torch_compile_piecewise"):
                if option in phase:
                    if option != backend:
                        raise ValueError(f"inactive {option} options under {backend}")
                    options = (
                        {"compile_seconds", "max_compile_variants"}
                        if option == "torch_compile_piecewise"
                        else set()
                    )
                    _object(phase[option], options, option)
        _object(data.get("graph_memory", {}), {"memory_bytes", "max_captures"}, "graph_memory")
        return cls(json.dumps(data, sort_keys=True, allow_nan=False))

    def to_dict(self) -> dict:
        return json.loads(self._json)

    def merge(self, invocation: RequestedOverrides) -> RequestedOverrides:
        """Config-file values precede explicit invocation overrides, recursively."""

        def merge(left: dict, right: dict) -> dict:
            result = dict(left)
            for key, value in right.items():
                result[key] = (
                    merge(result[key], value)
                    if isinstance(value, dict) and isinstance(result.get(key), dict)
                    else value
                )
            return result

        return self.parse(merge(self.to_dict(), invocation.to_dict()))

    def resolve(
        self,
        *,
        max_requests: int,
        max_tokens: int,
        flight_slots: int,
        legacy_decode: bool | None = None,
        legacy_buckets: tuple[int, ...] | None = None,
    ) -> ResolvedExecutionConfig:
        data = self.to_dict()
        phases = data.get("phases", {})
        resolved = {}
        provenance = []
        for name in ("decode", "prefill"):
            raw = dict(phases.get(name, {}))
            key = "request_buckets" if name == "decode" else "token_buckets"
            if name == "decode":
                if legacy_decode is not None:
                    backend = "full" if legacy_decode else "eager"
                    if "backend" in raw and (raw["backend"] != "eager") != legacy_decode:
                        raise ValueError("legacy decode_graph conflicts with phases.decode")
                    raw.setdefault("backend", backend)
                if legacy_buckets is not None:
                    if key in raw and tuple(raw[key]) != legacy_buckets:
                        raise ValueError("legacy graph_buckets conflicts with request_buckets")
                    raw.setdefault(key, list(legacy_buckets))
            elif "full_prefill_max_req" in data:
                alias = data["full_prefill_max_req"]
                if "max_requests" in raw and raw["max_requests"] != alias:
                    raise ValueError("full_prefill_max_req conflicts with max_requests")
                raw.setdefault("max_requests", alias)
            buckets = raw.get(key, [])
            if not isinstance(buckets, (list, tuple)):
                raise TypeError(f"{key} must be an array")
            backend = raw.get("backend", "eager")
            phase = PhaseConfig(
                backend=backend,
                buckets=tuple(buckets),
                max_requests=max_requests if name == "decode" else raw.get("max_requests", 1),
                memory_bytes=raw.get("memory_bytes", 0),
                compile_seconds=raw.get("torch_compile_piecewise", {}).get("compile_seconds", 120),
                max_compile_variants=raw.get("torch_compile_piecewise", {}).get(
                    "max_compile_variants", 8
                ),
                max_padding_ratio=raw.get("max_padding_ratio", 8.0),
            )
            ceiling = min(max_tokens, max_requests) if name == "decode" else max_tokens
            if phase.max_requests > max_requests or (phase.buckets and phase.buckets[-1] > ceiling):
                raise ValueError(f"{name} config exceeds runner capacity")
            resolved[name] = phase
            if (
                phase.backend == "torch_compile_piecewise"
                and len(phase.buckets) * flight_slots > phase.max_compile_variants
            ):
                raise ValueError(f"{name} exceeds compiler variant cache budget")
            provenance.extend((f"phases.{name}.{key}", "explicit") for key in raw)
        memory = data.get("graph_memory", {})
        cap = memory.get("memory_bytes", sum(p.memory_bytes for p in resolved.values()))
        count = memory.get("max_captures", 128)
        require_int(cap, "graph_memory.memory_bytes", minimum=0)
        require_int(count, "graph_memory.max_captures", minimum=1)
        if cap >= 2**63:
            raise ValueError("aggregate graph memory exceeds int64 budget")
        if sum(p.memory_bytes for p in resolved.values()) > cap:
            raise ValueError("phase reservations exceed aggregate graph memory cap")
        if (
            sum(len(p.buckets) for p in resolved.values() if p.backend != "eager") * flight_slots
            > count
        ):
            raise ValueError("phase buckets exceed max_captures")
        return ResolvedExecutionConfig(
            **resolved, memory_bytes=cap, max_captures=count, provenance=tuple(provenance)
        )
