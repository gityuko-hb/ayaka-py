# Declarative parameter and quantization mapping for checkpoint ingestion.

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch

type MapFunc = Callable[..., torch.Tensor]
type QuantizeFunc = Callable[
    [torch.Tensor], dict[str, torch.Tensor] | list[torch.Tensor] | tuple[torch.Tensor, ...]
]


@dataclass
class ExternMapping:
    """Mapping from module parameter names to source checkpoint tensor names.

    Attributes:
        param_map: Maps internal module parameter path -> list of source checkpoint keys.
        map_func: Maps internal parameter path -> combinator function combining source tensors.
        unused_params: Set of checkpoint tensor keys explicitly ignored during loading.
    """

    param_map: dict[str, list[str]] = field(default_factory=dict)
    map_func: dict[str, MapFunc] = field(default_factory=dict)
    unused_params: set[str] = field(default_factory=set)

    def add_mapping(
        self,
        target: str,
        sources: str | Iterable[str],
        func: MapFunc | None = None,
    ) -> None:
        """Register a mapping from external checkpoint tensor(s) to a module parameter.

        If a single source name is given and func is omitted, the identity transform is used.
        If multiple sources are given without a func, a ValueError is raised.
        """
        source_list = [sources] if isinstance(sources, str) else list(sources)
        if not source_list:
            raise ValueError(f"Mapping for target {target!r} must specify at least one source key")

        if func is None:
            if len(source_list) == 1:
                func = lambda x: x  # noqa: E731
            else:
                raise ValueError(
                    f"Target {target!r} has multiple sources ({source_list}), "
                    f"a combination func must be provided"
                )

        self.param_map[target] = source_list
        self.map_func[target] = func

    def add_unused(self, *names: str) -> None:
        """Register checkpoint tensor names that are safely ignored without raising errors."""
        for name in names:
            self.unused_params.add(name)

    @property
    def all_source_keys(self) -> set[str]:
        """Return all unique checkpoint tensor keys referenced across all mappings."""
        return {src for sources in self.param_map.values() for src in sources}

    @property
    def target_keys(self) -> set[str]:
        """Return all target module parameter paths in this mapping."""
        return set(self.param_map.keys())

    def source_to_targets(self) -> dict[str, list[str]]:
        """Invert mapping to provide a lookup from source checkpoint key to target parameter(s)."""
        rev: dict[str, list[str]] = {}
        for target, sources in self.param_map.items():
            for src in sources:
                rev.setdefault(src, []).append(target)
        return rev


@dataclass
class QuantizeMapping:
    """Mapping from a dense parameter to its eventual destination names and quantization function.

    Used for on-the-fly quantization as weights stream into memory.
    """

    param_map: dict[str, list[str]] = field(default_factory=dict)
    map_func: dict[str, QuantizeFunc] = field(default_factory=dict)

    def add_mapping(
        self,
        target: str,
        destinations: Iterable[str],
        func: QuantizeFunc,
    ) -> None:
        """Register an on-the-fly quantization transformation for a parameter.

        Args:
            target: Parameter name prior to quantization.
            destinations: Destination parameter/buffer names resulting from quantization.
            func: Function taking unquantized tensor and returning split components.
        """
        dest_list = list(destinations)
        if not dest_list:
            raise ValueError(
                f"Quantize mapping for {target!r} must specify at least one destination"
            )
        self.param_map[target] = dest_list
        self.map_func[target] = func


__all__ = [
    "ExternMapping",
    "MapFunc",
    "QuantizeFunc",
    "QuantizeMapping",
]
