"""Sampler backend registry and factory management.

Analogous to the backend registration pattern in SGLang/vLLM, this module
allows packaging alternative `Sampler` implementations (e.g., custom subclasses
overriding dispatch ladders, injecting probability warpers, or integrating
hardware-specific sampling kernels) under registered backend names. The inference
engine constructs samplers dynamically by name instead of hard-coding concrete
classes.

Examples:
    Register a custom sampler backend and initialize a coordinator:

        register_sampler_backend("custom", lambda: CustomSampler())
        coordinator = SamplingCoordinator(8, backend="custom")

Notes:
    Built-in backend names (`triton`, `reference`) always resolve to the
    canonical `Sampler`. Kernel selection (such as Triton versus FlashInfer)
    is handled internally by the dispatch ladder of `topk_topp_sample`. The
    backend registry selects the outer `Sampler` class orchestration, not the
    internal operator ladder.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ayaka.sampling.plan import Sampler

__all__ = [
    "BUILT_IN_SAMPLER_BACKENDS",
    "create_sampler",
    "register_sampler_backend",
    "registered_sampler_backends",
]

BUILT_IN_SAMPLER_BACKENDS: frozenset[str] = frozenset({"triton", "reference"})
_FACTORIES: dict[str, Callable[[], Sampler]] = {}

logger = logging.getLogger(__name__)


def register_sampler_backend(backend: str, factory: Callable[[], Sampler]) -> None:
    """Register a custom sampler factory for a backend identifier.

    The factory callable must adhere to a zero-argument convention and return an
    object satisfying the `Sampler` subclass contract. If a backend with the given
    identifier is already registered, the existing factory is overwritten with a
    warning, matching SGLang registration semantics.

    Args:
        backend: Unique string identifier for the backend.
        factory: Zero-argument callable returning a `Sampler`-compatible instance.

    Raises:
        ValueError: If `backend` is an empty string.
    """
    # Enforce non-empty backend identifier.
    if not backend:
        raise ValueError("backend must be a non-empty string")
    # Log warning when overwriting an existing backend factory.
    if backend in _FACTORIES:
        logger.warning("Overriding existing sampler factory for backend '%s'", backend)
    _FACTORIES[backend] = factory


def registered_sampler_backends() -> tuple[str, ...]:
    """Return all currently registered custom sampler backend names.

    Returns:
        Sorted tuple of registered custom backend identifiers, excluding
        built-in backend names.
    """
    return tuple(sorted(_FACTORIES))


def create_sampler(
    backend: str | None = None,
    *,
    penalty_state: Any | None = None,
    need_stats: bool = False,
    bias_state: Any | None = None,
) -> Sampler:
    """Instantiate a sampler for the specified backend identifier.

    When `backend` is `None` or matches a built-in backend (`triton`, `reference`),
    the canonical `Sampler` is instantiated with the provided execution states.
    When a custom backend is requested, its registered zero-argument factory is
    invoked; custom factories are responsible for their own state wiring.

    Args:
        backend: Optional backend identifier. Defaults to None (canonical Sampler).
        penalty_state: Optional state container tracking repetition/frequency penalties.
        need_stats: Whether to compute sampling distribution statistics.
        bias_state: Optional state container tracking logit biases.

    Returns:
        Instantiated `Sampler` instance.

    Raises:
        TypeError: If a custom backend factory returns an object that does not
            subclass `Sampler`.
        ValueError: If `backend` is neither registered nor a recognized built-in.
    """
    from ayaka.sampling.plan import Sampler

    # Dispatch to custom factory if registered.
    if backend in _FACTORIES:
        sampler = _FACTORIES[backend]()
        # Assert subclass contract for custom sampler instances.
        if not isinstance(sampler, Sampler):
            raise TypeError(f"Sampler factory for backend '{backend}' must return a Sampler")
        return sampler
    # Default and built-in backends resolve to the canonical Sampler implementation.
    if backend is None or backend in BUILT_IN_SAMPLER_BACKENDS:
        return Sampler(penalty_state=penalty_state, need_stats=need_stats, bias_state=bias_state)
    # Reject unknown backend identifiers fail-closed.
    raise ValueError(
        f"Unknown sampling backend {backend!r}. Register it via register_sampler_backend()."
    )
