"""Two-rank signal/epoch communication kernels (DC4/F4)."""

from ayaka.kernel.comm_layout import WorkspaceLayout
from ayaka.kernel.triton.comm.signal_epoch import (
    BLOCK_ELEMENTS,
    DEFAULT_SPIN_BUDGET,
    POINTER_TABLE_BYTES,
    SIGNAL_PAD_BYTES,
    STATUS_PAD_BYTES,
    STATUS_TIMEOUT,
    STATUS_UNSET,
    available,
    launch_signal_epoch_sum,
    signal_epoch_sum,
    warmup_signal_epoch_sum,
)

__all__ = [
    "BLOCK_ELEMENTS",
    "DEFAULT_SPIN_BUDGET",
    "POINTER_TABLE_BYTES",
    "SIGNAL_PAD_BYTES",
    "STATUS_PAD_BYTES",
    "STATUS_TIMEOUT",
    "STATUS_UNSET",
    "WorkspaceLayout",
    "available",
    "launch_signal_epoch_sum",
    "signal_epoch_sum",
    "warmup_signal_epoch_sum",
]
