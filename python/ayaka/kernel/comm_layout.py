"""Protocol layout for the two-rank communication workspace.

Kept free of Triton and torch so capability probing and agreement can compute
the workspace budget on any host, while
:mod:`ayaka.kernel.triton.comm.signal_epoch` consumes the same offsets for the
kernel and :mod:`ayaka.distributed.custom_all_reduce` for the views.
"""

from __future__ import annotations

import dataclasses

__all__ = [
    "POINTER_TABLE_BYTES",
    "SIGNAL_PAD_BYTES",
    "STATUS_PAD_BYTES",
    "STATUS_TIMEOUT",
    "STATUS_UNSET",
    "WorkspaceLayout",
]

#: Signal/status pads are one cache line apart
SIGNAL_PAD_BYTES = 128
STATUS_PAD_BYTES = 128
#: Reserved for the DC5 graph pointer bindings; unused by the eager v1 kernel,
#: which passes peer pointers as typed tensor arguments instead.
POINTER_TABLE_BYTES = 128

#: Status word value written when the spin budget was exhausted.
STATUS_TIMEOUT = 0
#: Status word value before any collective ran. Epochs start at 1.
STATUS_UNSET = -1


@dataclasses.dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """Byte offsets inside one rank's dedicated communication workspace.

    The allocation holds two payload slots (ping-pong), one signal pad, one
    status pad and a reserved pointer-table region. Payload starts at offset 0
    so an imported byte view can be reinterpreted as any supported dtype
    without an offset bias.
    """

    bucket_bytes: int
    payload_bytes: int
    signal_offset: int
    status_offset: int
    pointer_table_offset: int
    total_bytes: int

    def __post_init__(self) -> None:
        if type(self.bucket_bytes) is not int or self.bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be a positive integer")
        if self.bucket_bytes % 16:
            raise ValueError("bucket_bytes must be a multiple of 16 bytes")
        expected = (
            2 * self.bucket_bytes,
            2 * self.bucket_bytes,
            2 * self.bucket_bytes + SIGNAL_PAD_BYTES,
            2 * self.bucket_bytes + SIGNAL_PAD_BYTES + STATUS_PAD_BYTES,
            2 * self.bucket_bytes + SIGNAL_PAD_BYTES + STATUS_PAD_BYTES + POINTER_TABLE_BYTES,
        )
        actual = (
            self.payload_bytes,
            self.signal_offset,
            self.status_offset,
            self.pointer_table_offset,
            self.total_bytes,
        )
        if actual != expected:
            raise ValueError("WorkspaceLayout fields do not match the fixed DC4 layout")

    @classmethod
    def for_bucket(cls, bucket_bytes: int) -> WorkspaceLayout:
        """Build the layout for one payload bucket capacity in bytes."""
        payload = 2 * bucket_bytes
        signal = payload
        status = signal + SIGNAL_PAD_BYTES
        pointer_table = status + STATUS_PAD_BYTES
        return cls(
            bucket_bytes=bucket_bytes,
            payload_bytes=payload,
            signal_offset=signal,
            status_offset=status,
            pointer_table_offset=pointer_table,
            total_bytes=pointer_table + POINTER_TABLE_BYTES,
        )

    def slot_stride_elements(self, element_size: int) -> int:
        """Elements per payload slot for one element size."""
        if element_size not in (2, 4):
            raise ValueError(f"unsupported element size {element_size}")
        return self.bucket_bytes // element_size
