"""Sampling ops: pure logit transforms and filter/sample primitives.

Submodules:

* :mod:`ayaka.sampling.ops.penalties`: first half of the pipeline — logit
  adjustment before mask/temperature/sample. Merged from four files
  (``fused_stats.py``, ``bias.py``, ``penalties.py``, ``dry.py``) into one
  module with sections ``FUSED_STATS -> BIAS -> PENALTIES -> DRY``.
* :mod:`ayaka.sampling.ops.sampling`: second half — mask, temperature,
  filter/sample, and reporting. Merged from five files (``bitmask.py``,
  ``topk_topp.py``, ``gumbel.py``, ``mirostat.py``, ``support_capture.py``)
  into one module with sections ``BITMASK -> TOPK_TOPP -> GUMBEL ->
  MIROSTAT -> SUPPORT_CAPTURE``.

Canonical per-step order is ``bias -> penalties -> DRY -> mask ->
temperature -> filter/sample -> report``. Import from submodules directly;
this package intentionally re-exports nothing to keep imports light and
cycle-free.
"""
