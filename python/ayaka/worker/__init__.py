"""Worker boundary: device ownership behind an executor-facing protocol.

The executor keeps tickets, fences and drain bookkeeping; a worker owns the
device context, streams, runner, COW ordering and completion proofs. Import
:class:`~ayaka.worker.base.StepWorker` for the contract and
:class:`~ayaka.worker.local.LocalWorker` for the resident local implementation.
"""

from ayaka.worker.base import StepWorker, WorkerOutcome, WorkerStep
from ayaka.worker.fences import FlightFence, ImmediateFence, RecordingFence
from ayaka.worker.lifecycle import FlightKey, WorkerLifecycle, WorkerState
from ayaka.worker.local import LocalWorker

__all__ = [
    "FlightFence",
    "FlightKey",
    "ImmediateFence",
    "LocalWorker",
    "RecordingFence",
    "StepWorker",
    "WorkerLifecycle",
    "WorkerOutcome",
    "WorkerState",
    "WorkerStep",
]
