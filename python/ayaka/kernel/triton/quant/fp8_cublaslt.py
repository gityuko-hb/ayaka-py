"""cuBLASLt-backed FP8 W8A8 for per-tensor and per-row scales.

``torch._scaled_mm`` lowers to a cuBLASLt FP8 GEMM with the scales folded into
the epilogue. On a part with FP8 tensor cores it is substantially faster than
the tiled Triton kernel in :mod:`ayaka.kernel.triton.gemm.fp8_gemm` at the batch
sizes between decode and prefill: that kernel's grid is
``cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)``, so at ``M < BLOCK_M`` the M axis
contributes a single block row and the runtime tracks K while ignoring N.

This module owns three things and nothing else:

1. **Capability probes.** :func:`scaled_mm_available` and
   :func:`rowwise_scaled_mm_available` decide once per process, on the default
   stream, whether the installed PyTorch build can run the tensor-wise /
   row-wise kernels on this device. Some builds (Windows wheels, pre-2.12
   sm_89) ship without the row-wise kernel, and a probe is the only honest
   answer. Probes never run under CUDA-graph capture, which forbids the
   synchronize that makes them meaningful.
2. **Layout normalization.** ``b`` may arrive as a row-major ``[K, N]`` or as a
   transposed ``[N, K]`` weight view. cuBLASLt wants column-major ``[K, N]``:
   a stride change for the second form, a copy for the first.
3. **The call.** :func:`scaled_mm` computes into fp32, adds the bias in fp32 and
   casts once, exactly where the Triton kernel casts.

Numerics: cuBLASLt applies the scales in its epilogue, the Triton kernel applies
them to the fp32 accumulator. The products are identical up to accumulation
order; both accumulate in fp32.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from ayaka.kernel.triton.fp8_compat import e4m3_native
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_utils import cuda_available

__all__ = [
    "rowwise_scaled_mm_available",
    "scaled_mm",
    "scaled_mm_available",
    "usable",
]

FP8 = torch.float8_e4m3fn

#: Private ATen op; ``getattr`` keeps the reference out of the public API and
#: lets an older PyTorch answer "unavailable" instead of failing at import.
_scaled_mm: Any = getattr(torch, "_scaled_mm", None)

#: Probe shape. Small enough to be free, big enough to satisfy the cuBLASLt
#: alignment checks (K and N multiples of 16).
_PROBE_M = 16
_PROBE_K = 32
_PROBE_N = 32


def _probe(rowwise: bool) -> bool:
    if not cuda_available() or not e4m3_native() or _scaled_mm is None:
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
            a = torch.zeros(_PROBE_M, _PROBE_K, dtype=FP8, device=device)
            b = torch.zeros(_PROBE_N, _PROBE_K, dtype=FP8, device=device)
            if rowwise:
                scale_a = torch.ones(_PROBE_M, 1, device=device)
                scale_b = torch.ones(1, _PROBE_N, device=device)
            else:
                scale_a = torch.ones((), device=device)
                scale_b = torch.ones((), device=device)
            _scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
            torch.cuda.synchronize(device)
    except RuntimeError:
        return False
    return True


@functools.cache
def scaled_mm_available() -> bool:
    """Whether ``torch._scaled_mm`` runs a tensor-wise W8A8 GEMM on this device.

    Decided once per process. False below sm_89, without a usable PyTorch build,
    or when the installed wheel has no kernel for this architecture.
    """
    return _probe(rowwise=False)


@functools.cache
def rowwise_scaled_mm_available() -> bool:
    """Whether the row-wise kernel (``scale_a [M, 1]``, ``scale_b [1, N]``) exists."""
    return scaled_mm_available() and _probe(rowwise=True)


def usable(rowwise: bool) -> bool:
    """Availability, without ever probing under CUDA-graph capture.

    Capture forbids the synchronize a probe needs, so an unresolved probe during
    capture answers False and the caller falls back to the Triton kernel. A
    process that probes before capture keeps the fast path for its graphs.
    """
    if cuda_available() and torch.cuda.is_current_stream_capturing():
        if scaled_mm_available.cache_info().currsize == 0:
            return False
        if rowwise and rowwise_scaled_mm_available.cache_info().currsize == 0:
            return False
    return rowwise_scaled_mm_available() if rowwise else scaled_mm_available()


def _require_capability() -> None:
    if not scaled_mm_available():
        raise CapabilityError(
            "fp8_cublaslt",
            detail=(
                "torch._scaled_mm requires sm_89 or newer with a PyTorch build that "
                "ships the FP8 kernels, and it is unavailable on this device"
            ),
            remedy="use backend='triton' or backend='auto' to fall back to the Triton kernel",
        )


def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``(a @ b) * scale_a * scale_b + bias`` through cuBLASLt.

    Args:
        a: ``[M, K]`` row-major ``torch.float8_e4m3fn``.
        b: ``[K, N]`` row-major, or the transposed ``[N, K]`` weight view
            (``stride(0) == 1``).
        scale_a: Scalar or ``[M]`` fp32 scales.
        scale_b: Scalar or ``[N]`` fp32 scales.
        bias: Optional ``[N]`` additive term, applied in fp32.
        out: Optional ``[M, N]`` output buffer (fp32, fp16, or bf16).

    Returns:
        The ``[M, N]`` output tensor, the same object as ``out`` when provided.

    Raises:
        CapabilityError: When the tensor-wise or row-wise kernel is unavailable
            for this device, build, or scale layout.
    """

    _require_capability()
    rows, columns = int(a.shape[0]), int(b.shape[1])
    rowwise = scale_a.numel() > 1 or scale_b.numel() > 1
    if rowwise and not rowwise_scaled_mm_available():
        raise CapabilityError(
            "fp8_cublaslt.rowwise",
            detail=(
                "this PyTorch build has no row-wise torch._scaled_mm kernel; "
                "per-row/per-column scales need it"
            ),
            remedy="use backend='triton' or backend='auto' to fall back to the Triton kernel",
        )
    mat2 = b if (b.stride(0) == 1 and b.stride(1) != 1) else b.t().contiguous().t()
    sa = scale_a.float()
    sb = scale_b.float()
    if rowwise:
        sa2 = sa.reshape(-1, 1) if sa.numel() > 1 else sa.reshape(1, 1).expand(rows, 1).contiguous()
        sb2 = (
            sb.reshape(1, -1)
            if sb.numel() > 1
            else sb.reshape(1, 1).expand(1, columns).contiguous()
        )
        result = _scaled_mm(a, mat2, scale_a=sa2, scale_b=sb2, out_dtype=torch.float32)
    else:
        result = _scaled_mm(
            a, mat2, scale_a=sa.reshape(()), scale_b=sb.reshape(()), out_dtype=torch.float32
        )
    if bias is not None:
        result = result + bias.to(torch.float32)
    if out is None:
        return result
    out.copy_(result)
    return out
