"""Compatibility re-export; the oracles live in :mod:`ayaka.speculative.acceptance`."""

from __future__ import annotations

from ayaka.speculative.acceptance.greedy import Acceptance, accept_greedy
from ayaka.speculative.acceptance.rejection import accept_stochastic

__all__ = ["Acceptance", "accept_greedy", "accept_stochastic"]
