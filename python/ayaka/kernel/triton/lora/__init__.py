"""Triton LoRA shrink/expand kernels.

Exports are lazy: importing this package must stay safe on CPU-only installs
that lack Triton, so the kernel modules load on first attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ayaka.kernel.triton.lora.expand import bgmv_expand, sgmv_expand
    from ayaka.kernel.triton.lora.shrink import bgmv_shrink, sgmv_shrink

__all__ = ["bgmv_expand", "bgmv_shrink", "sgmv_expand", "sgmv_shrink"]

_LAZY = {
    "bgmv_shrink": "ayaka.kernel.triton.lora.shrink",
    "sgmv_shrink": "ayaka.kernel.triton.lora.shrink",
    "bgmv_expand": "ayaka.kernel.triton.lora.expand",
    "sgmv_expand": "ayaka.kernel.triton.lora.expand",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
