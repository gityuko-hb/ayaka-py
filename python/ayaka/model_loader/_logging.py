"""Human-facing logging for the model loader.

A single ``logging.Logger`` under the ``ayaka`` namespace plus formatters that
render load statistics the way vLLM/transformers do: one banner at the end of a
load, terse progress lines during the download.  Importing this module must
never import hub/source/manifest code — it sits below the loader so both the
loader and a bootstrap can use it without a cycle.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

from ayaka.weights.plan import ModelLoadReport

__all__ = ["format_bytes", "format_load_report", "format_seconds", "logger"]

logger = logging.getLogger("ayaka.model_loader")


# The banner uses U+2500 box-drawing rules, but a Windows console running a
# legacy codepage (cp1258 and friends) cannot encode them and logging would
# print "--- Logging error ---" noise on every banner.  Pick the character the
# actual output stream can render, once, at import.
def _pick_rule_char(encodings: Sequence[str]) -> str:
    for encoding in encodings:
        try:
            "─".encode(encoding)
            return "─"
        except (UnicodeEncodeError, LookupError):
            continue
    return "-"


_RULE_CHAR = _pick_rule_char(
    [getattr(s, "encoding", "") or "utf-8" for s in (sys.stdout, sys.stderr)]
)
_RULE = _RULE_CHAR * 51


def format_bytes(n: int | float) -> str:
    """Human-readable byte count, decimal (1 GB = 10^9) like the Hub's own UI."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1000.0 or unit == "PB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
        value /= 1000.0
    return f"{value:.0f} PB"


def format_seconds(ns: int) -> str:
    """Nanoseconds to a human duration: ``0.1s``, ``12.3s``, ``1m23s``."""
    seconds = ns / 1e9
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m{seconds:.0f}s"


def _humanize_stage(name: str, ns: int) -> str:
    return f"{name} {format_seconds(ns)}" if ns else ""


def format_load_report(report: ModelLoadReport) -> str:
    """Render a :class:`ModelLoadReport` as a box-drawing summary banner.

    Every field shown is already measured on the report; nothing is re-derived
    here, so the banner cannot disagree with the numbers a caller logs.
    """
    timing_parts = [
        part
        for part in (
            _humanize_stage("discover", report.discover_ns),
            _humanize_stage("read", report.read_ns),
            _humanize_stage("transform", report.transform_ns),
            _humanize_stage("h2d", report.h2d_ns),
            _humanize_stage("bind", report.bind_ns),
        )
        if part
    ]
    lines = [
        f"{_RULE_CHAR * 2} Weights loaded {_RULE_CHAR * 34}",
        f"  tensors     {report.weight_count}"
        + (f" ({format_bytes(report.checkpoint_bytes)})" if report.checkpoint_bytes else ""),
        f"  timing      {' | '.join(timing_parts) if timing_parts else 'n/a'}",
    ]
    if report.read_bytes and report.device_bytes:
        lines.append(f"  io          amplification {report.read_amplification:.2f}x")
    if report.total_ns:
        lines.append(f"  total       {format_seconds(report.total_ns)}")
    lines.append(_RULE)
    return "\n".join(lines)
