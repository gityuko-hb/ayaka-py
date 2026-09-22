"""torch.distributed lifecycle for the R11 runtime wiring.

The runtime/launcher owns process-group membership: this module initializes
the collective backend described by a resolved distributed plan and adapts it
to the :class:`~ayaka.distributed.process_group.TorchDistributedKVProcessGroup`
control plane the step coordinator drives. Initialization is fail-closed: an
unavailable backend raises instead of degrading to single-process execution,
because a silent singleton would mask a broken launch.

Shutdown is symmetric: the group runs one final agreement and barrier with an
explicit deadline, and only then does the process that initialized the default
group destroy it. A refused agreement or an expired deadline propagates and
the communicator is left alone — no rank may tear down a group a peer still
expects, and a wedged peer must never look like a clean teardown.
"""

from __future__ import annotations

from typing import Any

from ayaka.configs.distributed import CollectiveBackend, ResolvedDistributedPlan
from ayaka.distributed.process_group import (
    DistributedStepError,
    TorchDistributedKVProcessGroup,
)
from ayaka.distributed.topology import _optional_torch
from ayaka.exceptions import StorageUnavailableError
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.validation import require_int

__all__ = ["init_distributed_process_group", "shutdown_distributed_process_group"]

_BACKEND_NAMES: dict[CollectiveBackend, str] = {
    CollectiveBackend.NCCL: "nccl",
    CollectiveBackend.GLOO: "gloo",
}


def init_distributed_process_group(
    plan: ResolvedDistributedPlan,
    *,
    group: Any | None = None,
    timeout_s: float | None = None,
    device: str | None = None,
    max_metadata_bytes: int | None = None,
) -> TorchDistributedKVProcessGroup:
    """Initialize (or adopt) the collective backend and return the control plane.

    A process group already initialized by an external launcher is adopted as
    WORLD; otherwise this initializes the backend named by the plan. NCCL pins
    the control device to the local rank's CUDA device; Gloo stays on CPU. A
    plan-level world size must agree with the initialized group.

    Ownership is tracked for the shutdown helper: only a group this call
    initialized (and that uses the default process group, not an explicitly
    supplied subgroup) may be destroyed by
    :func:`shutdown_distributed_process_group`. An adopted or externally
    supplied group belongs to its launcher.
    """
    torch = _optional_torch()
    if torch is None or not torch.distributed.is_available():
        raise StorageUnavailableError("torch.distributed is unavailable")
    dist = torch.distributed
    if plan.collective_backend not in _BACKEND_NAMES:
        raise CapabilityError(
            "distributed.collective_backend",
            detail=f"collective backend {plan.collective_backend.value!r} has no runtime adapter",
            remedy="select CollectiveBackend.NCCL or CollectiveBackend.GLOO",
        )
    timeout = plan.init_timeout_s if timeout_s is None else float(timeout_s)
    initialized_here = False
    if not dist.is_initialized():
        if plan.collective_backend is CollectiveBackend.MPI:
            raise CapabilityError(
                "distributed.collective_backend",
                detail="MPI process groups have no runtime adapter",
                remedy="launch through NCCL or Gloo",
            )
        from ayaka.distributed.env import rank as env_rank

        rank = env_rank()
        require_int(rank, "RANK")
        dist.init_process_group(
            _BACKEND_NAMES[plan.collective_backend],
            init_method=plan.init_method,
            world_size=plan.world_size,
            rank=rank,
            timeout=_timedelta(timeout),
        )
        initialized_here = True
    kwargs: dict[str, Any] = {
        "group": group,
        "device": device if device is not None else _default_control_device(plan, torch),
        "default_timeout_s": timeout,
        "owns_process_group": initialized_here and group is None,
    }
    if max_metadata_bytes is not None:
        kwargs["max_metadata_bytes"] = max_metadata_bytes
    process_group = TorchDistributedKVProcessGroup(**kwargs)
    if process_group.world_size != plan.world_size:
        raise StorageUnavailableError(
            f"initialized process group world {process_group.world_size} does not match "
            f"the resolved plan world {plan.world_size}"
        )
    return process_group


def shutdown_distributed_process_group(
    process_group: TorchDistributedKVProcessGroup,
    *,
    local_accepting: bool = True,
    timeout_s: float | None = None,
) -> bool:
    """Run the final coordinated shutdown, then destroy a group this process owns.

    The agreement vote and barrier are the last collectives of the job: every
    rank must accept (``local_accepting``) before any rank proceeds, so no rank
    destroys or frees state while a peer still expects a collective. The
    deadline defaults to the control plane's configured ``default_timeout_s``
    and is applied to both the vote and the barrier.

    Fail-closed: a refused agreement raises :class:`DistributedStepFailed` and
    a deadline expiry raises :class:`DistributedCollectiveTimeout` (both
    :class:`DistributedStepError`); neither destroys the process group, so a
    broken job cannot be mistaken for a clean teardown. Only after a completed
    handshake is the default process group destroyed, and only when this
    process initialized it — adopted groups stay under their launcher.
    """
    timeout = process_group.default_timeout_s if timeout_s is None else float(timeout_s)
    if timeout <= 0:
        raise ValueError("shutdown timeout must be positive")
    agreed = process_group.shutdown(bool(local_accepting), timeout_s=timeout)
    if not agreed:
        raise DistributedStepError(
            "distributed process group shutdown was refused by at least one rank; "
            "no clean teardown and the process group is left intact"
        )
    process_group.destroy()
    return True


def _timedelta(timeout_s: float) -> Any:
    from datetime import timedelta

    return timedelta(seconds=timeout_s)


def _default_control_device(plan: ResolvedDistributedPlan, torch: Any) -> str:
    """Control device for the process group: local CUDA rank for NCCL, CPU otherwise."""
    if plan.collective_backend is CollectiveBackend.NCCL:
        from ayaka.distributed.env import local_rank as env_local_rank

        return f"cuda:{env_local_rank()}"
    return "cpu"
