from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """A validation failure with a stable code and a machine-readable path."""

    def __init__(
        self,
        path: str,
        code: str,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.code = code
        self.message = message
        self.context = dict(context or {})
        super().__init__(f"{path}: {message} [{code}]")

class FrozenMapping(tuple):
    """An immutable frozen mapping represented as sorted ((key, value), ...) pairs."""

def freeze_mapping(
    value: Mapping[str, Any] | tuple[tuple[str, Any], ...] | None,
    *,
    path: str = "extra"
) -> FrozenMapping:
    """Return a deterministic immutable representation of a string-keyed map."""

    if value is None:
        return FrozenMapping()
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    keys = [key for key, _ in items]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ConfigError(path, "MAP_KEY_INVALID", "mapping keys must be non-empty strings")
    if len(keys) != len(set(keys)):
        raise ConfigError(path, "MAP_KEY_DUPLICATE", "mapping contains duplicate keys")
    return FrozenMapping(sorted(items, key=lambda item: item[0]))

def _canonical(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if field.repr
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, FrozenMapping):
        return {key: _canonical(item) for key, item in value}
    if bool(value) and isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    ):
        return {key: _canonical(item) for key, item in value}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_canonical(item) for item in value]
    return value

@dataclass(frozen=True, slots=True)
class ConfigMixin:
    """Canonical serialization and fingerprints for immutable config records."""

    def to_dict(self) -> dict[str, Any]:
        payload = _canonical(self)
        if not isinstance(payload, dict):  # pragma: no cover - defensive
            raise TypeError("a dataclass config must serialize to an object")
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
