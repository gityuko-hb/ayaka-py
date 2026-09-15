"""Tier-1 bitmask subsystem for constrained decoding.

Submodules:

* :mod:`ayaka.sampling.mask.producer`: :class:`MaskRows` views and the
  :class:`MaskProducer` protocol.
* :mod:`ayaka.sampling.mask.arena`: double-buffered pinned staging area
  (:class:`BitmaskArena`, :class:`MaskHandle`).
* :mod:`ayaka.sampling.mask.pipeline`: parallel emit plus AND-gather
  (:class:`MaskPipeline`, :class:`MaskEntry`).
* :mod:`ayaka.sampling.mask.grammar`: adapter from constraint matchers to
  producers (:class:`GrammarMaskProducer`).
* :mod:`ayaka.sampling.mask.trie`: CSR trie backend (:class:`CsrTrie`,
  :class:`TrieMatcher`).

Import from submodules directly; this package intentionally re-exports
nothing to keep imports light and cycle-free.
"""
