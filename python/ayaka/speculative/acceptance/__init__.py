"""Speculative acceptance algorithms: host oracles and batched device forms."""

from ayaka.speculative.acceptance.greedy import Acceptance, accept_greedy, greedy_accept
from ayaka.speculative.acceptance.rejection import accept_stochastic, residual_distribution
from ayaka.speculative.acceptance.tree import TreeAcceptance, accept_tree_greedy

__all__ = [
    "Acceptance",
    "TreeAcceptance",
    "accept_greedy",
    "accept_stochastic",
    "accept_tree_greedy",
    "greedy_accept",
    "residual_distribution",
]
