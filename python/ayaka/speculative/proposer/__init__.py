"""Draft sources. Neural drafters are resources of the target worker."""

from ayaka.speculative.proposer.base import SpeculativeProposer
from ayaka.speculative.proposer.ngram import NGramProposer, PublicNGramPool

__all__ = ["NGramProposer", "PublicNGramPool", "SpeculativeProposer"]
