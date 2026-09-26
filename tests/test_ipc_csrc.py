"""Contract tests for the CUDA/HIP IPC extension wiring.

Coverage is split so every host can run something useful:

- CPU paths (no device needed): the fail-closed probes
  (:func:`ipc_available`, :func:`legacy_ipc_capable`), the
  ``AYAKA_DISABLE_CUDA_IPC`` kill-switch, and the pure-Python
  :class:`IpcRegionDescriptor` / :class:`IpcRegionRegistry` bookkeeping.
- GPU paths (``pytest.mark.gpu``, Linux + real device): the JIT-built
  extension field contract and the producer-side export/release cycle.

A true two-process share is intentionally *not* tested here; it needs two
OS processes and is left for a ``slow`` multi-proc harness.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch
from ayaka.distributed.ipc import IpcRegionDescriptor, IpcRegionRegistry
from ayaka.kernel import ipc as ipc_mod
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_utils import cuda_available

_HAS_CUDA = cuda_available()


@pytest.fixture
def ipc_disabled(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("AYAKA_DISABLE_CUDA_IPC", "1")
    ipc_mod.reset_cache()
    yield
    ipc_mod.reset_cache()


def _descriptor(handle: bytes = b"handle", **overrides: object) -> IpcRegionDescriptor:
    fields: dict[str, object] = {
        "handle": handle,
        "offset": 0,
        "nbytes": 64,
        "allocation_nbytes": 4096,
        "device": 0,
    }
    fields.update(overrides)
    return IpcRegionDescriptor(**fields)  # type: ignore[arg-type]


# ── CPU: fail-closed probes ──────────────────────────────────────────────────


def test_ipc_unavailable_when_disabled(ipc_disabled: None) -> None:
    assert ipc_mod.ipc_available() is False
    assert ipc_mod.legacy_ipc_capable(torch.zeros(4)) is False
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.export_allocation(torch.zeros(4))
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.open_allocation(b"handle", 4096, 0)


@pytest.mark.skipif(_HAS_CUDA, reason="requires a host without CUDA")
def test_ipc_unavailable_without_cuda() -> None:
    assert ipc_mod.ipc_available() is False
    assert ipc_mod.legacy_ipc_capable(torch.zeros(4)) is False
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.export_allocation(torch.zeros(4))


def test_legacy_ipc_capable_rejects_cpu_tensor() -> None:
    # Never triggers a JIT build: CPU tensors answer False through the probe
    # on every platform (Windows gates even earlier on os.name).
    assert ipc_mod.legacy_ipc_capable(torch.zeros(16, dtype=torch.float32)) is False


def test_legacy_ipc_capable_rejects_non_tensor() -> None:
    assert ipc_mod.legacy_ipc_capable(object()) is False  # type: ignore[arg-type]


# ── CPU: descriptor + registry bookkeeping ───────────────────────────────────


def test_descriptor_rejects_bad_ranges() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _descriptor(handle=b"")
    with pytest.raises(ValueError, match="non-negative"):
        _descriptor(offset=-1)
    with pytest.raises(ValueError, match="nbytes must be positive"):
        _descriptor(nbytes=0)
    with pytest.raises(ValueError, match="outside its"):
        _descriptor(offset=4000, nbytes=128, allocation_nbytes=4096)


def test_registry_release_close_unknown_key_raise() -> None:
    registry = IpcRegionRegistry()
    descriptor = _descriptor()
    with pytest.raises(KeyError, match="not exported"):
        registry.release(descriptor)
    with pytest.raises(KeyError, match="not opened"):
        registry.close(descriptor)


@pytest.mark.skipif(_HAS_CUDA, reason="export succeeds on real CUDA hosts")
def test_registry_export_requires_capability_without_cuda() -> None:
    registry = IpcRegionRegistry()
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        registry.export(torch.zeros(64, dtype=torch.uint8))


# ── GPU: JIT extension field contract ────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA unavailable")
def test_ipc_handle_size_positive() -> None:
    assert ipc_mod.ipc_available() is True
    assert ipc_mod.ipc_handle_size() > 0


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA unavailable")
def test_legacy_ipc_capable_true_for_cuda_tensor() -> None:
    tensor = torch.zeros(64, dtype=torch.uint8, device="cuda")
    assert ipc_mod.legacy_ipc_capable(tensor) is True


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA unavailable")
def test_export_allocation_field_contract() -> None:
    tensor = torch.arange(256, dtype=torch.float32, device="cuda")
    handle, offset, allocation_nbytes, device = ipc_mod.export_allocation(tensor)
    assert len(handle) == ipc_mod.ipc_handle_size()
    assert offset >= 0
    assert allocation_nbytes >= tensor.numel() * tensor.element_size()
    assert offset + tensor.numel() * tensor.element_size() <= allocation_nbytes
    assert device == tensor.device.index


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA unavailable")
def test_registry_producer_export_release_cycle() -> None:
    registry = IpcRegionRegistry()
    tensor = torch.zeros(1024, dtype=torch.bfloat16, device="cuda")
    try:
        first = registry.export(tensor)
        second = registry.export(tensor, offset=128, nbytes=256)
        assert registry.exports == 1
        # Same allocation: identical handle, second descriptor shifted.
        assert first.handle == second.handle
        assert second.offset == first.offset + 128
        assert second.nbytes == 256
        assert first.offset + tensor.numel() * tensor.element_size() <= first.allocation_nbytes
        registry.release(first)
        assert registry.exports == 1
        registry.release(second)
        assert registry.exports == 0
    finally:
        registry.close_all()


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA unavailable")
def test_open_garbage_handle_raises_capability_error() -> None:
    size = ipc_mod.ipc_handle_size()
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.open_allocation(b"\x00" * size, 4096, 0)
