// CUDA/HIP IPC extension, JIT-built by `ayaka.kernel.ipc` through
// `torch.utils.cpp_extension.load`. Linux-first; never imported on hosts
// that cannot run IPC (see `_unavailable_reason` in `ipc.py`).
//
// Contract (must match `python/ayaka/kernel/ipc.py`):
//   ipc_supported() -> bool
//   legacy_ipc_capable(tensor) -> bool  (never throws for probes)
//   export_allocation(tensor) -> (bytes, offset, allocation_nbytes, device)
//   open_allocation(handle, allocation_nbytes, device) -> uint8 CUDA tensor
//   byte_view(tensor, nbytes) -> uint8 view that keeps its owner alive
//   ipc_handle_size() -> int
//
// The registry in `python/ayaka/distributed/ipc.py` narrows the opened
// allocation to `(offset, nbytes)` itself, so `open_allocation` maps the
// whole allocation and owns exactly that mapping.
//
// Native safety contract:
//   * `DeviceGuard` restores the current device on every exit path,
//     including exceptions thrown by the open call itself.
//   * A failure after `cudaIpcOpenMemHandle` succeeded closes the mapping
//     before rethrowing: mappings never leak.
//   * `byte_view` captures the owning tensor in the storage deleter, so a
//     live view can never point at a freed/reused allocation.
//   * `cuMemGetAddressRange` is resolved dynamically through the CUDA
//     runtime, so the extension links against libcudart only and does not
//     depend on a host driver stub or `nvcc` toolchain layout.
//
// The loader defines AYAKA_USE_HIP on ROCm; TRITON_AR_USE_HIP is accepted as
// a legacy alias so older build caches keep working. The v1 target is
// Linux/NVIDIA; the HIP branch mirrors the same contract.

#include <torch/extension.h>

#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

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

// Test-only accounting: how many IPC mappings this process currently owns.
// A failed open after a successful cudaIpcOpenMemHandle must leave it flat.
static std::atomic<int64_t> g_open_mappings{0};

static bool test_fail_after_open() {
  const char* value = std::getenv("AYAKA_IPC_TEST_FAIL_AFTER_OPEN");
  return value != nullptr && *value != '\0' && std::string(value) != "0";
}

struct AddressRange {
  void* base;
  size_t size;
};

// ---------------------------------------------------------------------------
// Device guard / small platform shims
// ---------------------------------------------------------------------------

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    GPU_CHECK(get_current(&previous_));
    if (previous_ != device) GPU_CHECK(set_device(device));
  }

  ~DeviceGuard() {
    // Never throw from a destructor: a torn-down context at interpreter
    // shutdown legitimately refuses set_device.
    if (previous_ >= 0) (void)set_device(previous_);
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
#ifdef AYAKA_IPC_HIP
  static hipError_t get_current(int* device) { return hipGetDevice(device); }
  static hipError_t set_device(int device) { return hipSetDevice(device); }
#else
  static cudaError_t get_current(int* device) { return cudaGetDevice(device); }
  static cudaError_t set_device(int device) { return cudaSetDevice(device); }
#endif
  int previous_ = -1;
};

static int device_count() {
  int count = 0;
#ifdef AYAKA_IPC_HIP
  if (hipGetDeviceCount(&count) != hipSuccess) return 0;
#else
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
#endif
  return count;
}

static void get_handle(IpcMemHandle* handle, void* base) {
#ifdef AYAKA_IPC_HIP
  GPU_CHECK(hipIpcGetMemHandle(handle, base));
#else
  GPU_CHECK(cudaIpcGetMemHandle(handle, base));
#endif
}

static void open_handle(void** base, const IpcMemHandle& handle) {
#ifdef AYAKA_IPC_HIP
  GPU_CHECK(hipIpcOpenMemHandle(base, handle, hipIpcMemLazyEnablePeerAccess));
#else
  GPU_CHECK(cudaIpcOpenMemHandle(base, handle, cudaIpcMemLazyEnablePeerAccess));
#endif
}

static void close_handle(void* base) {
#ifdef AYAKA_IPC_HIP
  (void)hipIpcCloseMemHandle(base);
#else
  (void)cudaIpcCloseMemHandle(base);
#endif
}

// Best-effort consume of the runtime's sticky "last error" slot. Without
// this, a failed IPC probe would surface as an unrelated accelerator error
// on the caller's next CUDA operation.
static void clear_last_error() {
#ifdef AYAKA_IPC_HIP
  (void)hipGetLastError();
#else
  (void)cudaGetLastError();
#endif
}

// Deleter path: the owning tensor may be dropped from any stream context, so
// make the device current around the close and never propagate an error.
static void close_mapping_noexcept(void* base, int device) {
  if (base == nullptr) return;
  try {
    DeviceGuard guard(device);
    close_handle(base);
  } catch (...) {
    // Best effort: closing a mapping must not terminate the process.
  }
  clear_last_error();
}

// ---------------------------------------------------------------------------
// Address range: dynamic driver entry point, no libcuda link dependency
// ---------------------------------------------------------------------------

#ifndef AYAKA_IPC_HIP
using CuMemGetAddressRangeFn = CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr);

static CuMemGetAddressRangeFn resolve_address_range_fn() {
  static CuMemGetAddressRangeFn fn = []() -> CuMemGetAddressRangeFn {
    void* symbol = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSuccess;
  #if defined(CUDART_VERSION) && CUDART_VERSION >= 12050
    cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuMemGetAddressRange", &symbol, 12000, cudaEnableDefault, &status);
  #else
    cudaError_t err = cudaGetDriverEntryPoint("cuMemGetAddressRange", &symbol,
                                              cudaEnableDefault, &status);
  #endif
    if (err != cudaSuccess || status != cudaDriverEntryPointSuccess ||
        symbol == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<CuMemGetAddressRangeFn>(symbol);
  }();
  return fn;
}
#endif

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
  CuMemGetAddressRangeFn fn = resolve_address_range_fn();
  TORCH_CHECK(fn != nullptr,
              "cuMemGetAddressRange is unavailable from the CUDA driver");
  CUdeviceptr base_addr = 0;
  size_t allocation_size = 0;
  CUresult cr =
      fn(&base_addr, &allocation_size, reinterpret_cast<CUdeviceptr>(ptr));
  TORCH_CHECK(cr == CUDA_SUCCESS,
              "cuMemGetAddressRange failed with driver error ",
              static_cast<int>(cr));
  TORCH_CHECK(allocation_size > 0, "driver reported an empty allocation");
  return {reinterpret_cast<void*>(base_addr), allocation_size};
#endif
}

// ---------------------------------------------------------------------------
// Public primitives
// ---------------------------------------------------------------------------

static bool ipc_supported() { return device_count() > 0; }

// Fail-closed probe: any unexpected state answers `false` instead of
// raising, so capability checks never take the process down. Dense
// contiguous is part of the probe because `export_allocation` rejects
// strided/expanded views whose byte span is not a linear region.
static bool legacy_ipc_capable(torch::Tensor tensor) {
  try {
    if (!tensor.defined() || !tensor.is_cuda() || tensor.numel() == 0)
      return false;
    if (!tensor.is_contiguous()) return false;
    (void)get_address_range(tensor.data_ptr());
    return true;
  } catch (...) {
    clear_last_error();
    return false;
  }
}

static std::tuple<py::bytes, int64_t, int64_t, int64_t> export_allocation(
    torch::Tensor tensor) {
  try {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda(),
                "export_allocation tensor must be CUDA/HIP");
    TORCH_CHECK(tensor.numel() > 0,
                "export_allocation tensor must be non-empty");
    TORCH_CHECK(tensor.is_contiguous(),
                "export_allocation requires a dense contiguous tensor");
    TORCH_CHECK(tensor.storage_offset() >= 0,
                "export_allocation tensor has a negative storage offset");
    void* ptr = tensor.data_ptr();
    AddressRange range = get_address_range(ptr);
    TORCH_CHECK(range.size > 0, "export_allocation found an empty allocation");

    const int64_t offset =
        reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(range.base);
    TORCH_CHECK(offset >= 0,
                "export_allocation computed a negative byte offset");
    TORCH_CHECK(static_cast<size_t>(offset) < range.size,
                "export_allocation tensor starts outside its allocation");

    IpcMemHandle handle{};
    get_handle(&handle, range.base);
    return {py::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle)),
            offset, static_cast<int64_t>(range.size), tensor.get_device()};
  } catch (...) {
    clear_last_error();
    throw;
  }
}

static torch::Tensor open_allocation(py::bytes handle_bytes,
                                     int64_t allocation_nbytes,
                                     int64_t device) {
  TORCH_CHECK(allocation_nbytes > 0, "invalid IPC allocation size");
  TORCH_CHECK(device >= 0, "invalid IPC device");
  const int count = device_count();
  TORCH_CHECK(device < count, "IPC device ", device, " is outside the ", count,
              " visible devices");
  std::string raw = handle_bytes;
  TORCH_CHECK(raw.size() == sizeof(IpcMemHandle),
              "unexpected IPC handle size: got ", raw.size(), ", expected ",
              sizeof(IpcMemHandle));

  // Every exit path from here restores the caller's current device, and a
  // failure after a successful open closes the mapping before rethrowing.
  DeviceGuard guard(static_cast<int>(device));
  IpcMemHandle handle{};
  std::memcpy(&handle, raw.data(), sizeof(handle));
  void* base = nullptr;
  bool opened = false;
  try {
    open_handle(&base, handle);
    opened = true;
    ++g_open_mappings;
    if (test_fail_after_open()) {
      throw std::runtime_error("test hook: simulated failure after open");
    }
    // A cross-device mapping is imported into `device` but CUDA still
    // attributes the peer pointer to the exporting GPU. Tag the tensor with
    // the importing device through `target_device` and pass a device type
    // without an index in the options, which is the supported way to wrap
    // external memory that is accessed from a device other than the one
    // cudaPointerGetAttributes reports. Kernels launched on `device` then
    // reach the peer allocation through the peer access enabled by
    // cudaIpcMemLazyEnablePeerAccess.
    auto options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    return at::from_blob(
        base, {allocation_nbytes},
        [device](void* pointer) {
          close_mapping_noexcept(pointer, static_cast<int>(device));
          --g_open_mappings;
        },
        options,
        c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  } catch (...) {
    if (opened) {
      close_mapping_noexcept(base, static_cast<int>(device));
      --g_open_mappings;
    }
    clear_last_error();
    throw;
  }
}

static torch::Tensor byte_view(torch::Tensor tensor, int64_t nbytes) {
  TORCH_CHECK(tensor.is_cuda(), "byte_view tensor must be CUDA/HIP");
  TORCH_CHECK(tensor.is_contiguous(),
              "byte_view requires a dense contiguous tensor");
  const int64_t limit = tensor.numel() * tensor.element_size();
  TORCH_CHECK(nbytes >= 0 && nbytes <= limit, "invalid byte_view size ", nbytes,
              " for a ", limit, "-byte tensor");
  const auto device = tensor.get_device();
  void* pointer = tensor.data_ptr();
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device);
  // The closure captures the owning tensor, so the from_blob storage holds a
  // reference until the view dies. A view therefore never dangles after the
  // caller drops its own tensor reference.
  auto deleter = [owner = std::move(tensor)](void*) mutable { owner.reset(); };
  return torch::from_blob(pointer, {nbytes}, std::move(deleter), options);
}

static int64_t ipc_handle_size() { return sizeof(IpcMemHandle); }

static int64_t open_mapping_count() { return g_open_mappings.load(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ipc_supported", &ipc_supported,
        "Probe whether IPC is usable on this host");
  m.def("legacy_ipc_capable", &legacy_ipc_capable,
        "Whether a tensor sits in an exportable (non-VMM), dense allocation");
  m.def("export_allocation", &export_allocation,
        "Export (handle, offset, allocation_nbytes, device)");
  m.def("open_allocation", &open_allocation,
        "Open an IPC handle as an owning CUDA uint8 tensor");
  m.def("byte_view", &byte_view,
        "Create a flat uint8 view at tensor.data_ptr that keeps its owner");
  m.def("ipc_handle_size", &ipc_handle_size, "Runtime IPC handle size");
  m.def("_open_mapping_count", &open_mapping_count,
        "Test-only: number of live IPC mappings owned by this process");
}
