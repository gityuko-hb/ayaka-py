// CUDA/HIP IPC extension, JIT-built by `ayaka.kernel.ipc` through
// `torch.utils.cpp_extension.load`. Linux-first; never imported on hosts
// that cannot run IPC (see `_require_supported` in `ipc.py`).
//
// Contract (must match `python/ayaka/kernel/ipc.py`):
//   ipc_supported() -> bool
//   legacy_ipc_capable(tensor) -> bool  (never throws for probes)
//   export_allocation(tensor) -> (bytes, offset, allocation_nbytes, device)
//   open_allocation(handle, allocation_nbytes, device) -> uint8 CUDA tensor
//   byte_view(tensor, nbytes) -> uint8 view without ownership
//   ipc_handle_size() -> int
//
// The registry in `python/ayaka/distributed/ipc.py` narrows the opened
// allocation to `(offset, nbytes)` itself, so `open_allocation` maps the
// whole allocation and owns exactly that mapping.

#include <torch/extension.h>

#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>

// The JIT loader defines AYAKA_USE_HIP on ROCm; TRITON_AR_USE_HIP is
// accepted as a legacy alias so older build caches keep working.
#if defined(AYAKA_USE_HIP) || defined(TRITON_AR_USE_HIP)
#define AYAKA_IPC_HIP 1
#include <hip/hip_runtime.h>
using IpcMemHandle = hipIpcMemHandle_t;
#define GPU_CHECK(expr)                                     \
  do {                                                      \
    hipError_t e = (expr);                                  \
    if (e != hipSuccess)                                    \
      throw std::runtime_error(std::string("HIP error: ") + \
                               hipGetErrorString(e));       \
  } while (0)
#else
#include <cuda.h>
#include <cuda_runtime_api.h>
using IpcMemHandle = cudaIpcMemHandle_t;
#define GPU_CHECK(expr)                                      \
  do {                                                       \
    cudaError_t e = (expr);                                  \
    if (e != cudaSuccess)                                    \
      throw std::runtime_error(std::string("CUDA error: ") + \
                               cudaGetErrorString(e));       \
  } while (0)
#endif

namespace py = pybind11;

struct AddressRange {
  void* base;
  size_t size;
};

// Throws when `ptr` is not inside a shareable allocation. VMM-backed
// allocations (expandable_segments / cudaMallocAsync pools) fail here, which
// is exactly how `legacy_ipc_capable` detects them.
static AddressRange get_address_range(void* ptr) {
#ifdef AYAKA_IPC_HIP
  hipDeviceptr_t base_addr = 0;
  size_t allocation_size = 0;
  GPU_CHECK(hipMemGetAddressRange(&base_addr, &allocation_size,
                                 reinterpret_cast<hipDeviceptr_t>(ptr)));
  return {reinterpret_cast<void*>(base_addr), allocation_size};
#else
  CUdeviceptr base_addr = 0;
  size_t allocation_size = 0;
  CUresult cr = cuMemGetAddressRange(&base_addr, &allocation_size,
                                    reinterpret_cast<CUdeviceptr>(ptr));
  if (cr != CUDA_SUCCESS)
    throw std::runtime_error("cuMemGetAddressRange failed");
  return {reinterpret_cast<void*>(base_addr), allocation_size};
#endif
}

static bool ipc_supported() {
  int count = 0;
#ifdef AYAKA_IPC_HIP
  if (hipGetDeviceCount(&count) != hipSuccess) return false;
#else
  if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
#endif
  return count > 0;
}

// Fail-closed probe: any unexpected state answers `false` instead of
// raising, so capability checks never take the process down.
static bool legacy_ipc_capable(torch::Tensor tensor) {
  try {
    if (!tensor.defined() || !tensor.is_cuda() || tensor.numel() == 0)
      return false;
    (void)get_address_range(tensor.data_ptr());
    return true;
  } catch (...) {
    return false;
  }
}

static std::tuple<py::bytes, int64_t, int64_t, int64_t> export_allocation(
    torch::Tensor tensor) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda(),
              "export_allocation tensor must be CUDA/HIP");
  TORCH_CHECK(tensor.numel() > 0, "export_allocation tensor must be non-empty");
  void* ptr = tensor.data_ptr();
  AddressRange range = get_address_range(ptr);

  IpcMemHandle handle{};
#ifdef AYAKA_IPC_HIP
  GPU_CHECK(hipIpcGetMemHandle(&handle, range.base));
#else
  GPU_CHECK(cudaIpcGetMemHandle(&handle, range.base));
#endif
  const int64_t offset =
      reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(range.base);
  return {py::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle)),
          offset, static_cast<int64_t>(range.size), tensor.get_device()};
}

static torch::Tensor open_allocation(py::bytes handle_bytes,
                                     int64_t allocation_nbytes,
                                     int64_t device) {
  TORCH_CHECK(allocation_nbytes > 0, "invalid IPC allocation size");
  TORCH_CHECK(device >= 0, "invalid IPC device");
  std::string raw = handle_bytes;
  TORCH_CHECK(raw.size() == sizeof(IpcMemHandle),
              "unexpected IPC handle size");

  int old_device = 0;
#ifdef AYAKA_IPC_HIP
  GPU_CHECK(hipGetDevice(&old_device));
  GPU_CHECK(hipSetDevice(static_cast<int>(device)));
#else
  GPU_CHECK(cudaGetDevice(&old_device));
  GPU_CHECK(cudaSetDevice(static_cast<int>(device)));
#endif

  IpcMemHandle handle{};
  std::memcpy(&handle, raw.data(), sizeof(handle));
  void* base = nullptr;
#ifdef AYAKA_IPC_HIP
  GPU_CHECK(hipIpcOpenMemHandle(&base, handle, hipIpcMemLazyEnablePeerAccess));
#else
  GPU_CHECK(
      cudaIpcOpenMemHandle(&base, handle, cudaIpcMemLazyEnablePeerAccess));
#endif

#ifdef AYAKA_IPC_HIP
  GPU_CHECK(hipSetDevice(old_device));
#else
  GPU_CHECK(cudaSetDevice(old_device));
#endif

  // The storage owns the mapping: the deleter receives the exact base
  // pointer the open call returned. Callers narrow to their byte range.
  auto deleter = [device](void* p) {
    if (!p) return;
    int previous = 0;
#ifdef AYAKA_IPC_HIP
    if (hipGetDevice(&previous) == hipSuccess) {
      (void)hipSetDevice(static_cast<int>(device));
      (void)hipIpcCloseMemHandle(p);
      (void)hipSetDevice(previous);
    }
#else
    if (cudaGetDevice(&previous) == cudaSuccess) {
      (void)cudaSetDevice(static_cast<int>(device));
      (void)cudaIpcCloseMemHandle(p);
      (void)cudaSetDevice(previous);
    }
#endif
  };
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device);
  return torch::from_blob(base, {allocation_nbytes}, deleter, options);
}

static torch::Tensor byte_view(torch::Tensor tensor, int64_t nbytes) {
  TORCH_CHECK(tensor.is_cuda(), "byte_view tensor must be CUDA/HIP");
  TORCH_CHECK(nbytes >= 0 && nbytes <= tensor.numel() * tensor.element_size(),
              "invalid byte_view size");
  const auto device = tensor.get_device();
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device);
  // The Python RegisteredTensor keeps the owning tensor alive. This view must
  // never free the pointer, hence the no-op deleter.
  return torch::from_blob(tensor.data_ptr(), {nbytes}, [](void*) {}, options);
}

static int64_t ipc_handle_size() { return sizeof(IpcMemHandle); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ipc_supported", &ipc_supported,
        "Probe whether IPC is usable on this host");
  m.def("legacy_ipc_capable", &legacy_ipc_capable,
        "Whether a tensor sits in an exportable (non-VMM) allocation");
  m.def("export_allocation", &export_allocation,
        "Export (handle, offset, allocation_nbytes, device)");
  m.def("open_allocation", &open_allocation,
        "Open an IPC handle as an owning CUDA uint8 tensor");
  m.def("byte_view", &byte_view,
        "Create a flat uint8 view at tensor.data_ptr without ownership");
  m.def("ipc_handle_size", &ipc_handle_size, "Runtime IPC handle size");
}
