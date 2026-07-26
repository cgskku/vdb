
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

// Select top-k values for each group using one lightweight insertion path per block.
__global__ void segmented_insertion_topk_kernel(
    const float* keys,
    const int* values,
    int group_size,
    int topk,
    int group_offset,
    float* out_keys,
    int* out_values) {
    int local_group = blockIdx.x;
    int group = group_offset + local_group;
    if (threadIdx.x != 0) {
        return;
    }

    float best_keys[GPU_SORT_MAX_TOPK];
    int best_values[GPU_SORT_MAX_TOPK];
    for (int k = 0; k < topk; ++k) {
        best_keys[k] = INFINITY;
        best_values[k] = -1;
    }

    const int base = group * group_size;
    for (int i = 0; i < group_size; ++i) {
        float key = keys[base + i];
        int value = values[base + i];
        if (key > best_keys[topk - 1] || (key == best_keys[topk - 1] && value >= best_values[topk - 1])) {
            continue;
        }
        int pos = topk - 1;
        while (pos > 0 && (key < best_keys[pos - 1] || (key == best_keys[pos - 1] && value < best_values[pos - 1]))) {
            best_keys[pos] = best_keys[pos - 1];
            best_values[pos] = best_values[pos - 1];
            --pos;
        }
        best_keys[pos] = key;
        best_values[pos] = value;
    }

    const int out_base = group * topk;
    for (int k = 0; k < topk; ++k) {
        out_keys[out_base + k] = best_keys[k];
        out_values[out_base + k] = best_values[k];
    }
}

// Execute and validate the insertion-based segmented GPU top-k path.
BenchResult run_gpu_insertion(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    std::vector<float>* final_keys,
    std::vector<int>* final_values) {
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    double best_ms = std::numeric_limits<double>::infinity();
    for (int r = 0; r < opt.repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        segmented_insertion_topk_kernel<<<opt.groups, 32>>>(
            buffers.d_keys, buffers.d_values, opt.group_size, opt.topk, 0,
            buffers.d_out_keys, buffers.d_out_values);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        best_ms = std::min(best_ms, now_ms() - start);
    }
    std::vector<float> got_keys(static_cast<size_t>(opt.groups) * opt.topk);
    std::vector<int> got_values(static_cast<size_t>(opt.groups) * opt.topk);
    copy_to_host(got_keys, buffers.d_out_keys);
    copy_to_host(got_values, buffers.d_out_values);
    bool ok = validate_topk(ref_keys, ref_values, got_keys, got_values, opt.groups, opt.topk);
    if (final_keys) {
        *final_keys = got_keys;
    }
    if (final_values) {
        *final_values = got_values;
    }
    return {"gpu_insertion_segmented_topk", best_ms, ok};
}

#endif
