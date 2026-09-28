"""Device capacity and graph price, resolved from the owners that measure them.

Two facts are easy to get wrong in opposite directions here, so this module is
explicit about both.

The first is capacity. CUDA enumerates a MIG compute instance as a device in
its own right, so what ``torch`` reports for memory is the partition this
process was actually given. NVML reports the parent card. Sizing a partition
from the card's capacity overstates the budget by the size of the slices it does
not own, and the failure surfaces much later as an OOM that looks like a leak.
Capacity therefore comes from torch; the driver snapshot is only ever used to
detect a discrepancy, never to supply a number.

The second is the graph price. A per-bucket price is a real input, not
arithmetic to invent. The Triton metadata builder already exposes
``estimate_graph_state_bytes(max_batch_size=...)``, so a backend that can price
its own persistent state returns a callable. A backend that cannot returns
``None``, and the profile resolver then admits only the smallest set rather than
guessing that a wider one would fit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ayaka.execution.execution_capabilities import DeviceIdentity
from ayaka.utils.validation import require_int

__all__ = [
    "DeviceCapacityFacts",
    "PartitionKind",
    "resolve_device_capacity",
    "triton_graph_state_pricer",
]

#: A MIG instance is assigned a GPU UUID with this prefix from driver R470
#: onwards, which is a measured signal rather than an inference from a name.
_MIG_UUID_PREFIX = "MIG-"

#: How much smaller than the card a partition must be before it counts as a
#: slice. CUDA and the driver disagree slightly on a whole device -- a laptop
#: card reports 4294443008 through torch and 4096 MiB through nvidia-smi, a
#: difference of well under a megabyte. Treating that as a container slice
#: would mislabel ordinary hardware.
_PARTITION_GAP_TOLERANCE: float = 0.05


class PartitionKind(StrEnum):
    """What slice of the card this process was given.

    Recorded because the same compute capability and the same nominal memory can
    describe a whole card, one MIG instance, or a container slice, and those
    three have different correct budgets.
    """

    WHOLE_DEVICE = "whole_device"
    MIG = "mig"
    #: A vGPU, an MPS slice or a container device limit. Indicated by the
    #: partition reporting less memory than the card the driver describes.
    VGPU_OR_CONTAINER = "vgpu_or_container"
    #: Capacity could not be established at all.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DeviceCapacityFacts:
    """What this process may budget against, and how we know.

    ``budget_bytes`` is the only field a caller should size anything from.
    ``driver_card_bytes`` exists so a discrepancy is visible; feeding it into a
    budget is the mistake this module exists to prevent.
    """

    budget_bytes: int
    partition: PartitionKind
    source: str
    #: Parent-card capacity as the driver reports it, when a snapshot supplied
    #: one. Never a budget input.
    driver_card_bytes: int | None = None
    #: Why the partition is smaller than the card, when it is.
    note: str = ""

    @property
    def known(self) -> bool:
        return self.budget_bytes > 0

    @property
    def measured(self) -> bool:
        """Whether a real device reported this, rather than a default."""
        return self.known and self.partition is not PartitionKind.UNKNOWN


def resolve_device_capacity(
    device: DeviceIdentity, *, hardware: object | None = None
) -> DeviceCapacityFacts:
    """Resolve the budget capacity for ``device``.

    Args:
        device: The resolved device record. Its ``capacity_bytes`` came from
            torch and is therefore already partition-correct.
        hardware: An optional driver-level snapshot. Used only to notice that
            the partition is smaller than the card, which is what distinguishes
            a container slice from a whole device.

    Returns:
        Facts whose ``budget_bytes`` is the torch-reported capacity. When the
        device is not CUDA, the budget is zero and the partition is ``UNKNOWN``:
        there is no device budget on a CPU reference lane.
    """
    require_int(device.capacity_bytes, "device capacity_bytes", minimum=0)
    if not device.is_cuda:
        return DeviceCapacityFacts(
            budget_bytes=0,
            partition=PartitionKind.UNKNOWN,
            source="cpu-reference",
            note="no device memory budget applies to a CPU lane",
        )

    driver_card = _driver_card_bytes(hardware)
    if device.uuid.startswith(_MIG_UUID_PREFIX):
        return DeviceCapacityFacts(
            budget_bytes=device.capacity_bytes,
            partition=PartitionKind.MIG,
            source="torch",
            driver_card_bytes=driver_card,
            note="CUDA reports the MIG instance capacity, not the parent card",
        )
    if _is_material_gap(driver_card, device.capacity_bytes):
        return DeviceCapacityFacts(
            budget_bytes=device.capacity_bytes,
            partition=PartitionKind.VGPU_OR_CONTAINER,
            source="torch",
            driver_card_bytes=driver_card,
            note=(
                f"partition reports {device.capacity_bytes} bytes against a "
                f"{driver_card}-byte card; budgeting the card would overstate headroom"
            ),
        )
    if device.capacity_bytes > 0:
        return DeviceCapacityFacts(
            budget_bytes=device.capacity_bytes,
            partition=PartitionKind.WHOLE_DEVICE,
            source="torch",
            driver_card_bytes=driver_card,
        )
    return DeviceCapacityFacts(
        budget_bytes=0,
        partition=PartitionKind.UNKNOWN,
        source="torch",
        driver_card_bytes=driver_card,
        note="the device reported no capacity; treat every budget as unmeasured",
    )


def _is_material_gap(driver_card: int | None, budget: int) -> bool:
    """Whether the partition is meaningfully smaller than the card.

    A whole device shows a small disagreement between what CUDA reports and
    what the driver reports, because they account for the same memory slightly
    differently. Only a gap large enough to be a real slice counts.
    """
    if driver_card is None or budget <= 0 or driver_card <= budget:
        return False
    return (driver_card - budget) / driver_card > _PARTITION_GAP_TOLERANCE


def _driver_card_bytes(hardware: object | None) -> int | None:
    """Parent-card capacity from a driver snapshot, when one was supplied.

    Deliberately indirect attribute access: this accepts the existing
    ``HardwareConfig`` without importing it, so a caller that already has one
    does not have to hand over a live NVML session.
    """
    if hardware is None:
        return None
    capabilities: Mapping[str, object] | tuple = getattr(hardware, "capabilities", ())  # type: ignore[assignment]
    if not capabilities:
        return None
    first = capabilities[0]  # type: ignore[index]
    value = getattr(first, "hbm_bytes", 0)
    return int(value) if isinstance(value, int) and value > 0 else None


def triton_graph_state_pricer(
    *,
    max_seq_len: int,
    page_size: int,
    num_qo_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    kv_partition_size: int,
    output_dtype: Any,
    sliding_window: int | None = None,
) -> Callable[[int], int] | None:
    """Adapt the Triton metadata builder's estimator to the pricer contract.

    Returns ``None`` when the Triton backend cannot be imported, which is the
    normal CPU-reference case. A ``None`` pricer is an honest "I cannot price
    this", and the profile resolver degrades accordingly; it is never a
    substitute for a real estimate.
    """
    try:
        from ayaka.attention.backend.triton_backend import (
            DEFAULT_KV_PARTITION_SIZE,
            TritonAttentionMetadataBuilder,
        )
    except ImportError:
        return None

    require_int(max_seq_len, "max_seq_len", minimum=1)
    require_int(page_size, "page_size", minimum=1)
    require_int(num_qo_heads, "num_qo_heads", minimum=1)

    partition = kv_partition_size or DEFAULT_KV_PARTITION_SIZE

    def price(largest_bucket: int) -> int:
        require_int(largest_bucket, "largest_bucket", minimum=1)
        return int(
            TritonAttentionMetadataBuilder.estimate_graph_state_bytes(
                max_batch_size=largest_bucket,
                max_seq_len=max_seq_len,
                page_size=page_size,
                num_qo_heads=num_qo_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                kv_partition_size=partition,
                output_dtype=output_dtype,
                sliding_window=sliding_window,
            )
        )

    return price
