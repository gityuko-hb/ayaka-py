"""Tier-2 sampling custom ops — matcher adapters (A4/B3)."""

from __future__ import annotations

try:  # pragma: no cover - environment dependent
    from ayaka.sampling.custom.xgrammar_matcher import (
        XGrammarMatcher,
        has_xgrammar,
    )

    __all__ = ["XGrammarMatcher", "has_xgrammar"]
except Exception:  # pragma: no cover
    __all__ = []
