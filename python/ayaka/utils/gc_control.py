"""GC control during CUDA-graph capture and latency-sensitive phases.

Python's cyclic GC can run mid-capture and mutate or free objects a capturing
kernel launch depends on existing at a stable address for the rest of the
session -- most "it worked eager but broke under capture" bugs trace back to
something like this.

freeze_gc() collects once up front, then excludes every object alive at that
point from every future GC generation for the duration of capture, and disables
automatic cyclic GC so newly allocated objects in Generation 0 cannot trigger a
collection pass mid-stream.
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Generator
from contextlib import contextmanager

__all__ = [
    "disable_gc",
    "freeze_gc",
    "is_gc_frozen",
]

_LOCK = threading.Lock()
_FREEZE_DEPTH = 0
_FREEZE_WAS_ENABLED = True

_DISABLE_DEPTH = 0
_DISABLE_WAS_ENABLED = True


def is_gc_frozen() -> bool:
    """Return whether freeze_gc is currently active."""
    with _LOCK:
        return _FREEZE_DEPTH > 0


@contextmanager
def freeze_gc(
    enabled: bool = True,
    *,
    collect_on_exit: bool = True,
) -> Generator[None]:
    """Freeze existing objects and disable cyclic GC during CUDA-graph capture.

    Why both?
    - ``gc.freeze()`` moves existing objects into a permanent generation so
      cyclic GC will never touch or inspect them again.
    - ``gc.disable()`` is required because newly created objects during
      capture (Generation 0) could otherwise trigger an automatic collection
      pass mid-stream if allocation thresholds are reached, invoking finalizers
      or weakref callbacks that corrupt stream capture.

    Re-entrancy / nesting is tracked safely: only the outermost enter freezes
    and only the outermost exit unfreezes and restores GC state.

    Args:
        enabled: Set False to make this a no-op (useful when reproducing bugs).
        collect_on_exit: Whether to collect garbage after unfreezing on exit.
    """
    if not enabled:
        yield
        return

    global _FREEZE_DEPTH, _FREEZE_WAS_ENABLED

    with _LOCK:
        if _FREEZE_DEPTH == 0:
            _FREEZE_WAS_ENABLED = gc.isenabled()
            gc.collect()
            gc.freeze()
            gc.disable()
        _FREEZE_DEPTH += 1

    try:
        yield
    finally:
        with _LOCK:
            _FREEZE_DEPTH -= 1
            if _FREEZE_DEPTH == 0:
                gc.unfreeze()
                if _FREEZE_WAS_ENABLED:
                    gc.enable()
                if collect_on_exit:
                    gc.collect()


@contextmanager
def disable_gc(enabled: bool = True) -> Generator[None]:
    """Temporarily disable automatic cyclic GC without freezing generations.

    Lighter than freeze_gc(), suitable for latency-critical eager inference steps
    or micro-benchmarks. Re-entrant and preserves previous GC enabled state.

    Args:
        enabled: Set False to make this a no-op.
    """
    if not enabled:
        yield
        return

    global _DISABLE_DEPTH, _DISABLE_WAS_ENABLED

    with _LOCK:
        if _DISABLE_DEPTH == 0:
            _DISABLE_WAS_ENABLED = gc.isenabled()
            gc.disable()
        _DISABLE_DEPTH += 1

    try:
        yield
    finally:
        with _LOCK:
            _DISABLE_DEPTH -= 1
            if _DISABLE_DEPTH == 0 and _DISABLE_WAS_ENABLED:
                gc.enable()
