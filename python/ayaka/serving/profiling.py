"""Bounded profiling sessions driven by the serving owner (research profiler_manager, CT-P).

SGLang's profiler manager mixes stage counters, activity modes and HTTP control;
the research found invalid step counts accepted, a second START able to replace
an active recorder and partial failures leaving recorders alive. Here one
session at a time wraps ``torch.profiler`` around engine steps on the owner
thread:

* options are validated before any state changes (CT-P01, CT-P03);
* a START while a session is active is refused, a STOP without one is refused,
  and session ids increase monotonically (CT-P02);
* ``max_steps`` stops the session by itself after that many engine steps;
* a recorder that fails to start, stop or export is torn down and the session
  ends ``FAILED`` with the reason, never leaving an active recorder (CT-P05);
* shutdown ends an active session instead of leaking it.

Profiling perturbs timing; results carry their step count and wall time so a
reader can judge the window, and kernel extrema are measured only from raw events.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ayaka.kernel.profiler.kernel_profiler import KernelStats, _parse_profiler_events

__all__ = [
    "ProfileSession",
    "ProfileSessionResult",
    "ProfileSessionStatus",
    "ProfilerBusyError",
    "validate_profile_options",
]

#: Upper bound on a session's step window.
MAX_PROFILE_STEPS = 100_000


class ProfilerBusyError(RuntimeError):
    """A START while a session is active, or a STOP without one."""


class ProfileSessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProfileSessionResult:
    session: int
    status: ProfileSessionStatus
    #: Engine steps observed while recording.
    steps: int
    wall_seconds: float
    trace_path: str | None = None
    stats: tuple[KernelStats, ...] = ()
    #: Why the session ended: ``stop``, ``max_steps``, ``shutdown`` or a failure.
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "session": self.session,
            "status": self.status.value,
            "steps": self.steps,
            "wall_seconds": self.wall_seconds,
            "trace_path": self.trace_path,
            "kernels": len(self.stats),
            "reason": self.reason,
        }


def validate_profile_options(
    max_steps: object, trace_path: object, activities: Sequence[str]
) -> tuple[int | None, Path | None, tuple[str, ...]]:
    """Check every option before a session exists; raise without side effects."""
    if max_steps is not None and (
        type(max_steps) is not int or not 1 <= max_steps <= MAX_PROFILE_STEPS
    ):
        raise ValueError(f"max_steps must be None or an integer in [1, {MAX_PROFILE_STEPS}]")
    path = None
    if trace_path is not None:
        if not isinstance(trace_path, (str, Path)) or not str(trace_path):
            raise ValueError("trace_path must be a nonempty path")
        path = Path(trace_path)
        if not path.parent.is_dir():
            raise ValueError(f"trace directory {str(path.parent)!r} does not exist")
    chosen = tuple(activities)
    if not chosen or any(activity not in ("cpu", "cuda") for activity in chosen):
        raise ValueError("activities must be a nonempty subset of ('cpu', 'cuda')")
    if "cuda" in chosen:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("cuda profiling requested but CUDA is not available")
    return max_steps, path, chosen


def _torch_recorder(activities: tuple[str, ...]) -> Any:
    from torch.profiler import ProfilerActivity, profile

    kinds = [
        ProfilerActivity.CPU if name == "cpu" else ProfilerActivity.CUDA for name in activities
    ]
    return profile(activities=kinds)


@dataclass(slots=True)
class ProfileSession:
    """One recording window; created, stepped and finished on the owner thread."""

    session: int
    max_steps: int | None
    trace_path: Path | None
    activities: tuple[str, ...]
    recorder_factory: Callable[[tuple[str, ...]], Any] = _torch_recorder
    steps: int = 0
    _recorder: Any = field(default=None, repr=False)
    _started: float = 0.0

    def start(self) -> None:
        """Start recording; on failure nothing is left running and the error propagates."""
        recorder = self.recorder_factory(self.activities)
        try:
            recorder.start()
        except BaseException:
            _quiet_stop(recorder)
            raise
        self._recorder = recorder
        self._started = time.perf_counter()

    def on_step(self) -> bool:
        """Count one engine step; True when the window is full."""
        self.steps += 1
        return self.max_steps is not None and self.steps >= self.max_steps

    def finish(self, reason: str) -> ProfileSessionResult:
        """Stop, export and parse; any failure still releases the recorder."""
        recorder, self._recorder = self._recorder, None
        wall = time.perf_counter() - self._started
        if recorder is None:
            return self._result(ProfileSessionStatus.FAILED, wall, reason="never started")
        try:
            recorder.stop()
        except BaseException as exc:  # noqa: BLE001 - reported, never re-raised mid-loop
            _quiet_stop(recorder)
            return self._result(ProfileSessionStatus.FAILED, wall, reason=f"stop failed: {exc}")
        trace = None
        try:
            if self.trace_path is not None:
                recorder.export_chrome_trace(str(self.trace_path))
                trace = str(self.trace_path)
            stats = tuple(_parse_profiler_events(recorder))
        except BaseException as exc:  # noqa: BLE001
            return self._result(ProfileSessionStatus.FAILED, wall, reason=f"export failed: {exc}")
        return self._result(
            ProfileSessionStatus.COMPLETED, wall, trace=trace, stats=stats, reason=reason
        )

    def _result(
        self,
        status: ProfileSessionStatus,
        wall: float,
        *,
        trace: str | None = None,
        stats: tuple[KernelStats, ...] = (),
        reason: str,
    ) -> ProfileSessionResult:
        return ProfileSessionResult(
            session=self.session,
            status=status,
            steps=self.steps,
            wall_seconds=wall,
            trace_path=trace,
            stats=stats,
            reason=reason,
        )


def _quiet_stop(recorder: Any) -> None:
    try:
        recorder.stop()
    except BaseException:  # noqa: BLE001 - best-effort teardown of a broken recorder
        pass
