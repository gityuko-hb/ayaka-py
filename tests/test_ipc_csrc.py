"""Contract tests for the CUDA/HIP IPC extension wiring.

Coverage is split so every host can run something useful:

- CPU paths (no device needed): the fail-closed probes
  (:func:`ipc_available`, :func:`legacy_ipc_capable`), the
  ``AYAKA_DISABLE_CUDA_IPC`` kill-switch, the unsupported-platform branch and
  the pure-Python :class:`IpcRegionDescriptor` / :class:`IpcRegionRegistry`
  bookkeeping.
- GPU paths (``pytest.mark.gpu``, Linux + real device): the JIT-built
  extension field contract, the producer-side export/release cycle, native
  error recovery (device restore, owner-retaining byte view) and a clean
  JIT build from the packaged native source.

A true two-process share lives in ``tests/test_ipc_processes.py``; registry
lifetime semantics without a device live in ``tests/test_ipc_registry.py``.
"""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from ayaka.distributed.ipc import IpcRegionDescriptor, IpcRegionRegistry
from ayaka.kernel import ipc as ipc_mod
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_utils import cuda_available

_HAS_CUDA = cuda_available()
_IPC_HOST = _HAS_CUDA and sys.platform.startswith("linux")
_skip_gpu = pytest.mark.skipif(not _IPC_HOST, reason="Linux + CUDA required for native IPC")

#: First JIT build compiles torch's extension headers; subsequent calls reuse
#: the torch extensions cache.
_GPU_TIMEOUT = pytest.mark.timeout(900)


@pytest.fixture
def ipc_disabled(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setenv("AYAKA_DISABLE_CUDA_IPC", "1")
    ipc_mod.reset_cache()
    yield
    ipc_mod.reset_cache()


@pytest.fixture
def isolated_extension_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[Path]:
    build_dir = tmp_path / "torch_extensions"
    build_dir.mkdir()
    monkeypatch.setenv("AYAKA_IPC_BUILD_DIR", str(build_dir))
    ipc_mod.reset_cache()
    yield build_dir
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
    # on every platform (Windows gates even earlier on sys.platform).
    assert ipc_mod.legacy_ipc_capable(torch.zeros(16, dtype=torch.float32)) is False


def test_legacy_ipc_capable_rejects_non_tensor() -> None:
    assert ipc_mod.legacy_ipc_capable(object()) is False  # type: ignore[arg-type]


def test_unsupported_platform_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ipc_mod, "_platform_supported", lambda: False)
    ipc_mod.reset_cache()
    try:
        assert ipc_mod.ipc_available() is False
        assert ipc_mod.legacy_ipc_capable(torch.zeros(4)) is False
        with pytest.raises(CapabilityError, match="cuda_ipc"):
            ipc_mod.export_allocation(torch.zeros(4, device="cpu"))
        with pytest.raises(CapabilityError, match="cuda_ipc"):
            ipc_mod.ipc_handle_size()
    finally:
        ipc_mod.reset_cache()


def test_export_rejects_non_tensor() -> None:
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.export_allocation(object())  # type: ignore[arg-type]


def test_export_rejects_bad_sizes_before_native() -> None:
    # Argument validation rejects these before any JIT/native call on every
    # host, so this test is safe in the CPU lane.
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.byte_view(torch.zeros(4), -1)
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.byte_view(torch.zeros(4), 4096)
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.open_allocation(b"handle", -1, 0)
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.open_allocation(b"handle", 4096, -1)


def test_native_source_ships_with_package() -> None:
    import ayaka.kernel

    source = Path(ayaka.kernel.__file__).resolve().parent / "csrc" / "ipc" / "ipc_ext.cpp"
    assert source.is_file(), f"native IPC source missing from the package: {source}"


def test_cpu_import_needs_no_compiler_and_no_cuda(tmp_path: Path) -> None:
    """G-DC1.7: importing the wrapper on a GPU-less host never builds."""
    import subprocess

    import ayaka

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CC"] = "/nonexistent-cc"
    env["CXX"] = "/nonexistent-cxx"
    env["TORCH_EXTENSIONS_DIR"] = str(tmp_path / "torch_extensions")
    env.pop("AYAKA_DISABLE_CUDA_IPC", None)
    package_root = str(Path(ayaka.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import torch\n"
        "from ayaka.kernel import ipc\n"
        "print(ipc.ipc_available())\n"
        "print(ipc.legacy_ipc_capable(torch.zeros(4)))\n"
        "print(ipc.ipc_handle_size.__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["False", "False", "ipc_handle_size"]


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


def test_descriptor_rejects_bad_identity_and_schema() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _descriptor(schema_version=2)
    with pytest.raises(ValueError, match="session_generation"):
        _descriptor(session_generation=(1, 0))
    with pytest.raises(TypeError, match="must be an integer"):
        _descriptor(offset=True)


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


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_ipc_handle_size_positive() -> None:
    assert ipc_mod.ipc_available() is True
    assert ipc_mod.ipc_handle_size() > 0


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_legacy_ipc_capable_true_for_cuda_tensor() -> None:
    tensor = torch.zeros(64, dtype=torch.uint8, device="cuda")
    assert ipc_mod.legacy_ipc_capable(tensor) is True


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_export_allocation_field_contract() -> None:
    tensor = torch.arange(256, dtype=torch.float32, device="cuda")
    handle, offset, allocation_nbytes, device = ipc_mod.export_allocation(tensor)
    assert len(handle) == ipc_mod.ipc_handle_size()
    assert offset >= 0
    assert allocation_nbytes >= tensor.numel() * tensor.element_size()
    assert offset + tensor.numel() * tensor.element_size() <= allocation_nbytes
    assert device == tensor.device.index


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
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
        registry.close_all(force=True)


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_open_garbage_handle_raises_capability_error() -> None:
    size = ipc_mod.ipc_handle_size()
    with pytest.raises(CapabilityError, match="cuda_ipc"):
        ipc_mod.open_allocation(b"\x00" * size, 4096, 0)


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_open_wrong_handle_size_is_rejected() -> None:
    with pytest.raises(CapabilityError, match="handle size"):
        ipc_mod.open_allocation(b"\x00" * 8, 4096, 0)


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_open_failure_restores_current_device() -> None:
    size = ipc_mod.ipc_handle_size()
    before = torch.cuda.current_device()
    other = (before + 1) % torch.cuda.device_count()
    torch.cuda.set_device(other)
    try:
        with pytest.raises(CapabilityError, match="cuda_ipc"):
            ipc_mod.open_allocation(b"\x00" * size, 4096, before)
        assert torch.cuda.current_device() == other
    finally:
        torch.cuda.set_device(before)


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_byte_view_keeps_its_owner_alive() -> None:
    tensor = torch.arange(64, dtype=torch.float32, device="cuda")
    expected = tensor[:16].clone()
    view = ipc_mod.byte_view(tensor, 64)
    del tensor
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.equal(view.view(torch.float32), expected)


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_registry_rejects_strided_and_expanded_views() -> None:
    registry = IpcRegionRegistry()
    tensor = torch.zeros(256, dtype=torch.float32, device="cuda")
    try:
        with pytest.raises(ValueError, match="contiguous"):
            registry.export(tensor[::2])
        with pytest.raises(ValueError, match="contiguous"):
            registry.export(tensor.expand(4, 256))
        with pytest.raises(ValueError, match="non-empty"):
            registry.export(tensor[:0])
    finally:
        registry.close_all(force=True)


@_skip_gpu
@pytest.mark.gpu
@_GPU_TIMEOUT
def test_clean_jit_build_from_packaged_source(isolated_extension_cache: Path) -> None:
    """G-DC1.7: a fresh extension cache builds the shipped source end to end."""
    assert ipc_mod.ipc_available() is True
