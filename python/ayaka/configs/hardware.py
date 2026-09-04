from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from typing import Final, TypedDict

from ayaka.distributed.device import DeviceCapability, DeviceRef, LinkKind
from ayaka.types import DeviceKind
from ayaka.utils.torch_memory import (
    _host_total_ram_bytes,
    _numa_nodes,
    _pci_numa_node
)
from ayaka.utils.nvml_utils import (
    get_driver_version,
    nvml_session,
    probe_device,
    probe_nvlink_matrix,
)

CC_LIMITS: dict[tuple[int, int], dict[str, int]] = {
    # Pascal
    # GP100 (Tesla P100)
    (6, 0): dict(
        shared_per_block=64 << 10,
        shared_per_sm=64 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GP102/104/106 (GTX 1080 Ti, Tesla P40, P4)
    (6, 1): dict(
        shared_per_block=48 << 10,
        shared_per_sm=96 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GP10B (Tegra Parker)
    (6, 2): dict(
        shared_per_block=48 << 10,
        shared_per_sm=48 << 10,
        regs_per_sm=32768,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # Volta 
    # GV100 (Tesla V100, Titan V)
    (7, 0): dict(
        shared_per_block=96 << 10,
        shared_per_sm=96 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GV10B (Jetson AGX Xavier)
    (7, 2): dict(
        shared_per_block=96 << 10,
        shared_per_sm=96 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # Turing 
    # TU102/104/106/116 (RTX 2080 Ti, Tesla T4)
    (7, 5): dict(
        shared_per_block=64 << 10,
        shared_per_sm=64 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1024,
        max_threads_per_block=1024,
        max_blocks_per_sm=16,
    ),
    # Ampere 
    # GA100 (A100, A30)
    (8, 0): dict(
        shared_per_block=163 << 10,
        shared_per_sm=164 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GA102/104/106 (RTX 3090, A10, A40, RTX A6000)
    (8, 6): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=16,
    ),
    # GA10B (Jetson AGX Orin)
    (8, 7): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=16,
    ),
    # Ada Lovelace 
    # AD102/103/104 (RTX 4090, L40, L40S, L4)
    (8, 9): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=24,
    ),
    # Hopper 
    # GH100 (H100, H200, GH200)
    (9, 0): dict(
        shared_per_block=227 << 10,
        shared_per_sm=228 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # Blackwell
    # GB100/GB200 (B100, B200, GB200 - Data Center)
    (10, 0): dict(
        shared_per_block=227 << 10,
        shared_per_sm=228 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        max_blocks_per_sm=32,
    ),
    # GB20x (RTX 5090, RTX 5080 - Client / Workstation)
    (12, 0): dict(
        shared_per_block=99 << 10,
        shared_per_sm=100 << 10,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        max_blocks_per_sm=24,
    ),
}

_L2_BYTES: dict[str, int] = {
    "RTX 3050": 2 << 20,
    "RTX 3060": 3 << 20,
    "RTX 3070": 4 << 20,
    "RTX 3080": 5 << 20,
    "RTX 3090": 6 << 20,
    "RTX 4060": 24 << 20,
    "RTX 4070": 36 << 20,
    "RTX 4080": 64 << 20,
    "RTX 4090": 72 << 20,
    "A100": 40 << 20,
    "H100": 50 << 20,
    "H200": 50 << 20,
    "B100": 128 << 20,
    "B200": 128 << 20,
    "GB200": 128 << 20,
    "L40S": 96 << 20,
    "L40": 96 << 20,
    "L4": 48 << 20,
    "V100": 6 << 20,
    "T4": 4 << 20,
}


def _l2_for(name: str) -> int:
    name_upper = name.upper()
    for key, value in _L2_BYTES.items():
        if key.upper() in name_upper:
            return value
    return 0


def _fill_cc_limits(cap: DeviceCapability) -> DeviceCapability:
    limits = CC_LIMITS.get(cap.compute_capability, {})
    return replace(
        cap,
        shared_mem_per_block=cap.shared_mem_per_block or limits.get("shared_per_block", 0),
        shared_mem_per_sm=cap.shared_mem_per_sm or limits.get("shared_per_sm", 0),
        registers_per_sm=cap.registers_per_sm or limits.get("regs_per_sm", 0),
        max_threads_per_sm=cap.max_threads_per_sm or limits.get("max_threads_per_sm", 0),
        l2_bytes=cap.l2_bytes or _l2_for(cap.name),
    )
    
@dataclass(frozen=True, slots=True)
class HardwareConfig:
    devices: tuple[DeviceRef, ...] = ()
    capabilities: tuple[DeviceCapability, ...] = ()
    host_ram_bytes: int = 0
    numa_nodes: int = 1
    cpu_count: int = 1
    links: tuple[tuple[LinkKind, ...], ...] = ()
    driver_version: str = ""
    detected: bool = False
    notes: tuple[str, ...] = field(default=(), compare=False)
    peer_access: tuple[tuple[bool | None, ...], ...] = ()
    
    def __post_init__(self) -> None:
        if len(self.devices) != len(self.capabilities):
            raise ValueError("devices/capabilities length mismatch")
        if self.links:
            size = len(self.devices)
            if len(self.links) != size or any(len(row) != size for row in self.links):
                raise ValueError("links must be a square matrix over devices")
            for src in range(size):
                if self.links[src][src] is not LinkKind.SELF:
                    raise ValueError("links must carry SELF on the diagonal")
                for dst in range(src + 1, size):
                    if self.links[src][dst] is not self.links[dst][src]:
                        raise ValueError("links must be symmetric")

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def has_cuda(self) -> bool:
        return any(d.kind is DeviceKind.CUDA for d in self.devices)

    def capability(self, index: int = 0) -> DeviceCapability:
        if not self.capabilities:
            raise RuntimeError("no device detected; pass a HardwareConfig explicitly")
        return self.capabilities[index]

    def link(self, src: int, dst: int) -> LinkKind:
        if not self.links:
            return LinkKind.SELF if src == dst else LinkKind.PCIE
        return self.links[src][dst]

    @classmethod
    def cpu_only(cls, *, note: str = "") -> HardwareConfig:
        return cls(
            host_ram_bytes=_host_total_ram_bytes() or 0,
            numa_nodes=_numa_nodes(),
            cpu_count=os.cpu_count() or 1,
            detected=False,
            notes=(note,) if note else (),
        )

    @classmethod
    def synthetic(cls, cap: DeviceCapability, count: int = 1) -> HardwareConfig:
        cap = _fill_cc_limits(cap)
        devices = tuple(DeviceRef(kind=DeviceKind.CUDA, index=i) for i in range(count))
        links = tuple(
            tuple(LinkKind.SELF if i == j else LinkKind.PCIE for j in range(count))
            for i in range(count)
        )
        return cls(
            devices=devices,
            capabilities=tuple(cap for _ in range(count)),
            host_ram_bytes=_host_total_ram_bytes() or 0,
            numa_nodes=_numa_nodes(),
            cpu_count=os.cpu_count() or 1,
            links=links,
            detected=False,
            notes=("synthetic",),
        )
        
def _detect_via_pynvml() -> HardwareConfig | None:
    with nvml_session() as pynvml:
        if pynvml is None:
            return None
        try:
            count = int(pynvml.nvmlDeviceGetCount())
            devices: list[DeviceRef] = []
            caps: list[DeviceCapability] = []
            handles = []

            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                handles.append(h)
                raw = probe_device(pynvml, i, h)

                devices.append(DeviceRef(kind=DeviceKind.CUDA, index=i, uuid=raw.uuid))
                caps.append(
                    _fill_cc_limits(
                        DeviceCapability(
                            name=raw.name,
                            sm_major=raw.sm_major,
                            sm_minor=raw.sm_minor,
                            num_sms=raw.num_sms,
                            hbm_bytes=raw.hbm_bytes,
                            pci_bus_id=raw.pci_bus_id,
                            numa_node=_pci_numa_node(raw.pci_bus_id),
                        )
                    )
                )

            nvlink_mask = probe_nvlink_matrix(pynvml, handles)
            matrix: list[tuple[LinkKind, ...]] = []
            for i in range(count):
                row: list[LinkKind] = []
                for j in range(count):
                    if i == j:
                        row.append(LinkKind.SELF)
                    elif nvlink_mask[i][j]:
                        row.append(LinkKind.NVLINK)
                    else:
                        row.append(LinkKind.PCIE)
                matrix.append(tuple(row))

            return HardwareConfig(
                devices=tuple(devices),
                capabilities=tuple(caps),
                host_ram_bytes=_host_total_ram_bytes() or 0,
                numa_nodes=_numa_nodes(),
                cpu_count=os.cpu_count() or 1,
                links=tuple(matrix),
                driver_version=get_driver_version(pynvml),
                detected=True,
                notes=("pynvml",),
            )
        except Exception:
            return None


def _detect_via_smi() -> HardwareConfig | None:
    if not shutil.which("nvidia-smi"):
        return None
    query = "index,name,memory.total,compute_cap,uuid,pci.bus_id"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if out.returncode != 0 or not out.stdout.strip():
        return None

    devices: list[DeviceRef] = []
    caps: list[DeviceCapability] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, mem_mib, cc, uuid, bus = parts[:6]
        major, _, minor = cc.partition(".")
        try:
            device_index = int(idx)
            hbm = int(float(mem_mib)) * (1 << 20)
            cc_pair = (int(major), int(minor or 0))
        except ValueError:
            continue

        devices.append(DeviceRef(kind=DeviceKind.CUDA, index=device_index, uuid=uuid))
        caps.append(
            _fill_cc_limits(
                DeviceCapability(
                    name=name,
                    sm_major=cc_pair[0],
                    sm_minor=cc_pair[1],
                    hbm_bytes=hbm,
                    pci_bus_id=bus,
                    numa_node=_pci_numa_node(bus),
                )
            )
        )

    if not devices:
        return None

    n = len(devices)
    links = tuple(
        tuple(LinkKind.SELF if i == j else LinkKind.PCIE for j in range(n)) for i in range(n)
    )
    return HardwareConfig(
        devices=tuple(devices),
        capabilities=tuple(caps),
        host_ram_bytes=_host_total_ram_bytes() or 0,
        numa_nodes=_numa_nodes(),
        cpu_count=os.cpu_count() or 1,
        links=links,
        detected=True,
        notes=("nvidia-smi", "num_sms and NVLink topology unknown via smi"),
    )


def detect_hardware() -> HardwareConfig:
    if os.environ.get("AYAKA_DISABLE_HW_DETECT") not in (None, "", "0"):
        return HardwareConfig.cpu_only(note="detection disabled by env")
    for probe in (_detect_via_pynvml, _detect_via_smi):
        result = probe()
        if result is not None and result.num_devices:
            return result
    return HardwareConfig.cpu_only(note="no CUDA device found")