"""CPU-only microbenchmark of the output-collection step (research tasks O01 / V01).

Compares the legacy pull pattern (``OutputProcessor.events()`` then slice from the
consumer cursor) with ``OutputProcessor.events_since``. It times only the
retrieval calls on one process: no model, no GPU, no HTTP, no detokenizer. It
therefore measures the host cost the O(E^2) analysis predicts, not serving
latency or throughput; treat any number as an upper bound on what this change can
save, and re-measure end to end before claiming a serving speedup.

Usage::

    python tools/bench_output_events.py
    python tools/bench_output_events.py --events 128 1024 8192 --repeat 9 --json out.json

Raw per-repetition samples are written with ``--json`` so a run can be audited.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from ayaka.request.schema import Request, RequestId, StopCriteria
from ayaka.runtime.output import OutputProcessor

_PROMPT = (1, 2, 3)


def _processor(request_ids: list[str], *, max_tokens: int, history_limit: int | None = None):
    output = OutputProcessor(history_limit=history_limit)
    for request_id in request_ids:
        output.register(
            Request(
                request_id=RequestId(request_id),
                prompt_token_ids=_PROMPT,
                stop=StopCriteria(max_tokens=max_tokens),
            )
        )
    return output


def _publish(output: OutputProcessor, request_id: str, generated: int) -> None:
    output.on_published(
        request_id,
        7,
        sequence_epoch=1,
        generated_tokens=generated,
        prompt_tokens=len(_PROMPT),
    )


def poll_every_event(events: int, mode: str) -> int:
    """Nanoseconds spent polling once after each of ``events`` published events."""
    output = _processor(["r"], max_tokens=events + 1)
    cursor = spent = 0
    for generated in range(1, events + 1):
        _publish(output, "r", generated)
        started = time.perf_counter_ns()
        if mode == "legacy":
            new = output.events("r")[cursor:]
        else:
            new = output.events_since("r", cursor)
        spent += time.perf_counter_ns() - started
        cursor += len(new)
    assert cursor == events
    return spent


def collector_pass(active: int, history: int, mode: str) -> int:
    """Nanoseconds for one collector pass over ``active`` caught-up requests."""
    ids = [f"r{i}" for i in range(active)]
    output = _processor(ids, max_tokens=history + 2)
    for request_id in ids:
        for generated in range(1, history + 1):
            _publish(output, request_id, generated)
    for request_id in ids:  # one more decode step per request, as a live engine would
        _publish(output, request_id, history + 1)
    started = time.perf_counter_ns()
    for request_id in ids:
        if mode == "legacy":
            output.events(request_id)[history:]
        else:
            output.events_since(request_id, history)
    return time.perf_counter_ns() - started


def retained_after(events: int, history_limit: int | None) -> int:
    output = _processor(["r"], max_tokens=events + 1, history_limit=history_limit)
    for generated in range(1, events + 1):
        _publish(output, "r", generated)
    return len(output.events("r"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, nargs="+", default=[128, 1024, 8192])
    parser.add_argument("--passes", type=int, nargs=2, action="append", metavar=("N", "H"))
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--history-limit", type=int, default=256)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    passes = args.passes or [(16, 256), (64, 1024), (256, 1024), (256, 4096)]

    report: dict = {
        "scope": "CPU-only retrieval microbenchmark; not an end-to-end serving benchmark",
        "python": sys.version.split()[0],
        "machine": platform.platform(),
        "repeat": args.repeat,
        "poll_every_event": [],
        "collector_pass": [],
    }

    print(f"python {report['python']}  {report['machine']}  (median of {args.repeat})")
    print("\npoll after every event (ms per request)")
    print(f"{'E':>7} {'legacy':>12} {'events_since':>14} {'ratio':>8}")
    for events in args.events:
        samples = {
            mode: [poll_every_event(events, mode) for _ in range(args.repeat)]
            for mode in ("legacy", "delta")
        }
        legacy = statistics.median(samples["legacy"]) / 1e6
        delta = statistics.median(samples["delta"]) / 1e6
        print(f"{events:>7} {legacy:>12.3f} {delta:>14.3f} {legacy / delta:>7.1f}x")
        report["poll_every_event"].append({"events": events, "ns": samples})

    print("\none collector pass over N caught-up requests holding H events each (ms)")
    print(f"{'N':>7} {'H':>7} {'legacy':>12} {'events_since':>14} {'ratio':>8}")
    for active, history in passes:
        samples = {
            mode: [collector_pass(active, history, mode) for _ in range(args.repeat)]
            for mode in ("legacy", "delta")
        }
        legacy = statistics.median(samples["legacy"]) / 1e6
        delta = statistics.median(samples["delta"]) / 1e6
        print(f"{active:>7} {history:>7} {legacy:>12.3f} {delta:>14.3f} {legacy / delta:>7.1f}x")
        report["collector_pass"].append({"active": active, "history": history, "ns": samples})

    print(f"\nevents retained per request after {max(args.events)} events")
    for limit in (None, args.history_limit):
        label = "unbounded" if limit is None else f"history_limit={limit}"
        retained = retained_after(max(args.events), limit)
        print(f"  {label:<22} {retained:>7}")
        report.setdefault("retained", {})[label] = retained

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nraw samples written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
