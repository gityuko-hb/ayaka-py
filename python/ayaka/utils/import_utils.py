from __future__ import annotations

import importlib
import importlib.util
from functools import cache
from types import ModuleType
from typing import Any

__all__ = [
    "CapabilityError",
    "LazyModule",
    "has_module",
    "require_module",
    "resolve_qualname",
]


class CapabilityError(ImportError):
    """Raised when an optional dependency required by a capability is unavailable."""

    def __init__(self, capability: str, *, detail: str, remedy: str) -> None:
        self.capability = capability
        self.detail = detail
        self.remedy = remedy
        super().__init__(
            f"capability {capability!r} unavailable: {detail}. Remedy: {remedy}"
        )


@cache
def has_module(name: str) -> bool:
    """True if `name` is importable, without importing it.

    ``find_spec`` walks the finders but does not execute module code, so
    this is safe to call before the fork/spawn decision.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A parent package that itself fails to import raises rather than
        # returning None.
        return False


def require_module(
    name: str, *, capability: str | None = None, remedy: str | None = None
) -> ModuleType:
    """Import `name` or raise `CapabilityError`.

    Args:
        name: Module to import, e.g. ``"triton"``.
        capability: Stable identifier for the feature that needs it.
            Defaults to the module name.
        remedy: Install hint, e.g. ``"pip install triton>=3.0"``.

    Examples:
        >>> module = require_module("json")
        >>> module.dumps({"ok": True})
        '{"ok": true}'

        >>> require_module("nonexistent_pkg", capability="fp8_gemm",  # doctest: +ELLIPSIS
        ...                remedy="pip install ayaka-kernels")
        Traceback (most recent call last):
            ...
        CapabilityError: ...

    """

    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise CapabilityError(
            capability or name,
            detail=f"cannot import {name!r}: {exc}",
            remedy=remedy or f"install {name}",
        ) from exc


class LazyModule(ModuleType):
    """A module proxy that imports on first attribute access.

    Assign it at module scope where you would otherwise write a top-level
    import of something expensive::

        triton = LazyModule("triton")

    Type checkers see a ModuleType, and every attribute access after the
    first is a plain instance-dict lookup because the real module's
    ``__dict__`` is spliced in.
    """

    def __init__(
        self,
        name: str,
        *,
        capability: str | None = None,
        remedy: str | None = None,
    ) -> None:
        super().__init__(name)

        self._capability = capability
        self._remedy = remedy
        self._loader: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._loader is None:
            module = require_module(
                self.__name__, capability=self._capability, remedy=self._remedy
            )
            self._loader = module

            # Splice the real module in so subsequent lookups skip
            # __getattr__ entirely.
            self.__dict__.update(module.__dict__)
        return self._loader

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_ayaka_"):
            raise AttributeError(item)
        return getattr(self._load(), item)

    def __dir__(self):
        return dir(self._load())


def resolve_qualname(path: str) -> Any:
    """Resolve ``"pkg.module:attr"`` or ``"pkg.module.attr"`` to an object.

    Args:
        path: A dotted or colon-separated path to a module and attribute.
    Examples:
        >>> resolve_qualname("ayaka.utils.errors:CapabilityError")
        <class 'ayaka.utils.errors.CapabilityError'>
    """

    if ":" in path:
        module_name, _, attr = path.partition(":")
        module = importlib.import_module(module_name)
        obj: Any = module
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj

    parts = path.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"{path!r} must contain a module and an attribute, "
            f"e.g. 'ayaka.utils.hashing:stable_hash'"
        )

    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        obj = module
        try:
            for part in parts[split:]:
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj

    raise ImportError(f"cannot resolve {path!r}")
