
#include "gpu_sort.h"

#if GPU_SORT_HAS_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

// Convert CUDA runtime failures into C++ exceptions with file and line context.
#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::ostringstream oss__; \
        oss__ << "CUDA error " << cudaGetErrorString(err__) << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error(oss__.str()); \
    } \
} while (0)

// Round group sizes up to a power of two for block-level bitonic sorting.
[[maybe_unused]] static int next_power_of_two(int x) {
    int p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

// Allocate typed device buffers through one checked helper.
template <typename T>
static T* device_alloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

// Copy host vectors into device memory for GPU benchmark paths.
template <typename T>
static void copy_to_device(T* dst, const std::vector<T>& src) {
    CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

// Copy compact top-k output back to host vectors for validation.
template <typename T>
static void copy_to_host(std::vector<T>& dst, const T* src) {
    CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

// Own the device input and compact output buffers for a benchmark run.
struct DeviceBuffers {
    float* d_keys = nullptr;
    int* d_values = nullptr;
    float* d_out_keys = nullptr;
    int* d_out_values = nullptr;
    size_t input_count = 0;
    size_t output_count = 0;

    DeviceBuffers(const std::vector<float>& keys, const std::vector<int>& values, int groups, int topk)
        : input_count(keys.size()), output_count(static_cast<size_t>(groups) * topk) {
        d_keys = device_alloc<float>(input_count);
        d_values = device_alloc<int>(input_count);
        d_out_keys = device_alloc<float>(output_count);
        d_out_values = device_alloc<int>(output_count);
        copy_to_device(d_keys, keys);
        copy_to_device(d_values, values);
    }

    ~DeviceBuffers() {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_out_keys);
        cudaFree(d_out_values);
    }
};

// Touch device memory once so later timing is less affected by first-use overhead.
__global__ void warmup_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 0.0f;
    }
}

// Run a small warmup kernel before measuring sort kernels.
void run_warmup_kernel(const std::vector<float>& keys) {
    CUDA_CHECK(cudaFree(nullptr));
    float* d_tmp = device_alloc<float>(keys.size());
    copy_to_device(d_tmp, keys);
    int threads = 256;
    int blocks = static_cast<int>((keys.size() + threads - 1) / threads);
    warmup_kernel<<<blocks, threads>>>(d_tmp, static_cast<int>(keys.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_tmp));
}

#endif
