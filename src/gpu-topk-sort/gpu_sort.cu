
#include "gpu_sort.h"

#if GPU_SORT_HAS_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

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

// Let each thread maintain a local insertion top-k list for one independent group.
template <int TOPK_CAPACITY>
__global__ void segmented_parallel_insertion_topk_kernel(
    const float* keys,
    const int* values,
    int group_size,
    int topk,
    int group_offset,
    int group_count,
    float* out_keys,
    int* out_values) {
    int local_group = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_group >= group_count) {
        return;
    }
    int group = group_offset + local_group;
    float best_keys[TOPK_CAPACITY];
    int best_values[TOPK_CAPACITY];
    for (int rank = 0; rank < topk; ++rank) {
        best_keys[rank] = INFINITY;
        best_values[rank] = -1;
    }

    int input_base = group * group_size;
    for (int candidate = 0; candidate < group_size; ++candidate) {
        float key = keys[input_base + candidate];
        int value = values[input_base + candidate];
        if (key > best_keys[topk - 1] ||
            (key == best_keys[topk - 1] && value >= best_values[topk - 1])) {
            continue;
        }
        int position = topk - 1;
        while (position > 0 &&
               (key < best_keys[position - 1] ||
                (key == best_keys[position - 1] && value < best_values[position - 1]))) {
            best_keys[position] = best_keys[position - 1];
            best_values[position] = best_values[position - 1];
            --position;
        }
        best_keys[position] = key;
        best_values[position] = value;
    }

    int output_base = group * topk;
    for (int rank = 0; rank < topk; ++rank) {
        out_keys[output_base + rank] = best_keys[rank];
        out_values[output_base + rank] = best_values[rank];
    }
}

// Select the smallest local-array specialization that can hold the requested top-k.
static void launch_parallel_insertion(
    const float* keys,
    const int* values,
    int group_size,
    int topk,
    int group_offset,
    int group_count,
    float* out_keys,
    int* out_values,
    cudaStream_t stream = 0) {
    auto launch = [&](auto capacity, int threads) {
        constexpr int topk_capacity = decltype(capacity)::value;
        int blocks = (group_count + threads - 1) / threads;
        segmented_parallel_insertion_topk_kernel<topk_capacity><<<blocks, threads, 0, stream>>>(
            keys, values, group_size, topk, group_offset, group_count, out_keys, out_values);
    };
    if (topk <= 1) {
        launch(std::integral_constant<int, 1>{}, 256);
    } else if (topk <= 4) {
        launch(std::integral_constant<int, 4>{}, 256);
    } else if (topk <= 8) {
        launch(std::integral_constant<int, 8>{}, 128);
    } else if (topk <= 16) {
        launch(std::integral_constant<int, 16>{}, 64);
    } else if (topk <= 32) {
        launch(std::integral_constant<int, 32>{}, 32);
    } else if (topk <= 64) {
        launch(std::integral_constant<int, 64>{}, 16);
    } else {
        launch(std::integral_constant<int, GPU_SORT_MAX_TOPK>{}, 8);
    }
}

// Benchmark and validate the thread-per-group parallel insertion implementation.
// CUDA events measure the device interval; host timing remains a separate metric.
class KernelTimer {
public:
    KernelTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~KernelTimer() {
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop() { CUDA_CHECK(cudaEventRecord(stop_)); }
    double elapsed() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }
private:
    cudaEvent_t start_, stop_;
};

BenchResult run_gpu_parallel_insertion(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    std::vector<float>* final_keys,
    std::vector<int>* final_values) {
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    double best_ms = std::numeric_limits<double>::infinity();
    double best_kernel_ms = std::numeric_limits<double>::infinity();
    KernelTimer timer;
    for (int repeat = 0; repeat < opt.repeats; ++repeat) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        timer.start();
        launch_parallel_insertion(
            buffers.d_keys, buffers.d_values, opt.group_size, opt.topk, 0, opt.groups,
            buffers.d_out_keys, buffers.d_out_values);
        CUDA_CHECK(cudaGetLastError());
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        best_ms = std::min(best_ms, now_ms() - start);
        best_kernel_ms = std::min(best_kernel_ms, timer.elapsed());
    }
    std::vector<float> got_keys(static_cast<size_t>(opt.groups) * opt.topk);
    std::vector<int> got_values(static_cast<size_t>(opt.groups) * opt.topk);
    copy_to_host(got_keys, buffers.d_out_keys);
    copy_to_host(got_values, buffers.d_out_values);
    bool valid = validate_topk(
        ref_keys, ref_values, got_keys, got_values, opt.groups, opt.topk);
    if (final_keys) {
        *final_keys = got_keys;
    }
    if (final_values) {
        *final_values = got_values;
    }
    return {"gpu_parallel_insertion_topk", best_ms, valid, best_kernel_ms};
}

// Sort groups of at most 64 candidates with all lanes of one warp participating.
__global__ void segmented_warp_micro_topk_kernel(
    const float* keys,
    const int* values,
    int group_size,
    int padded_size,
    int topk,
    int group_offset,
    float* out_keys,
    int* out_values) {
    __shared__ float shared_keys[64];
    __shared__ int shared_values[64];
    int lane = threadIdx.x;
    int group = group_offset + blockIdx.x;
    int input_base = group * group_size;

    for (int i = lane; i < padded_size; i += 32) {
        if (i < group_size) {
            shared_keys[i] = keys[input_base + i];
            shared_values[i] = values[input_base + i];
        } else {
            shared_keys[i] = INFINITY;
            shared_values[i] = -1;
        }
    }
    __syncwarp();

    for (int sequence = 2; sequence <= padded_size; sequence <<= 1) {
        for (int stride = sequence >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < padded_size; i += 32) {
                int peer = i ^ stride;
                if (peer > i) {
                    bool ascending = (i & sequence) == 0;
                    float a = shared_keys[i];
                    float b = shared_keys[peer];
                    int av = shared_values[i];
                    int bv = shared_values[peer];
                    bool greater = (a > b) || (a == b && av > bv);
                    bool less = (a < b) || (a == b && av < bv);
                    if (ascending ? greater : less) {
                        shared_keys[i] = b;
                        shared_keys[peer] = a;
                        shared_values[i] = bv;
                        shared_values[peer] = av;
                    }
                }
            }
            __syncwarp();
        }
    }

    int output_base = group * topk;
    for (int i = lane; i < topk; i += 32) {
        out_keys[output_base + i] = shared_keys[i];
        out_values[output_base + i] = shared_values[i];
    }
}

// Execute and validate the warp-cooperative segmented GPU top-k path.
BenchResult run_gpu_insertion(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    std::vector<float>* final_keys,
    std::vector<int>* final_values) {
    if (opt.group_size > 64) {
        throw std::runtime_error("warp micro-sort supports group_size <= 64");
    }
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    int padded_size = next_power_of_two(opt.group_size);
    double best_ms = std::numeric_limits<double>::infinity();
    double best_kernel_ms = std::numeric_limits<double>::infinity();
    KernelTimer timer;
    for (int r = 0; r < opt.repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        timer.start();
        segmented_warp_micro_topk_kernel<<<opt.groups, 32>>>(
            buffers.d_keys, buffers.d_values, opt.group_size, padded_size, opt.topk, 0,
            buffers.d_out_keys, buffers.d_out_values);
        CUDA_CHECK(cudaGetLastError());
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        best_ms = std::min(best_ms, now_ms() - start);
        best_kernel_ms = std::min(best_kernel_ms, timer.elapsed());
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
    return {"gpu_warp_micro_topk", best_ms, ok, best_kernel_ms};
}

// Sort one group per block in shared memory with a bitonic network.
__global__ void segmented_bitonic_topk_kernel(
    const float* keys,
    const int* values,
    int group_size,
    int topk,
    int group_offset,
    float* out_keys,
    int* out_values) {
    extern __shared__ unsigned char shared_raw[];
    float* s_keys = reinterpret_cast<float*>(shared_raw);
    int* s_values = reinterpret_cast<int*>(s_keys + blockDim.x);

    int tid = threadIdx.x;
    int local_group = blockIdx.x;
    int group = group_offset + local_group;
    int input_idx = group * group_size + tid;

    if (tid < group_size) {
        s_keys[tid] = keys[input_idx];
        s_values[tid] = values[input_idx];
    } else {
        s_keys[tid] = INFINITY;
        s_values[tid] = -1;
    }
    __syncthreads();

    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                float a = s_keys[tid];
                float b = s_keys[ixj];
                int av = s_values[tid];
                int bv = s_values[ixj];
                bool pair_gt = (a > b) || (a == b && av > bv);
                bool pair_lt = (a < b) || (a == b && av < bv);
                bool swap = ascending ? pair_gt : pair_lt;
                if (swap) {
                    s_keys[tid] = b;
                    s_keys[ixj] = a;
                    s_values[tid] = bv;
                    s_values[ixj] = av;
                }
            }
            __syncthreads();
        }
    }

    if (tid < topk) {
        int out_idx = group * topk + tid;
        out_keys[out_idx] = s_keys[tid];
        out_values[out_idx] = s_values[tid];
    }
}

// Launch bitonic sorting for a contiguous range of segmented groups.
static void launch_bitonic_range(
    const DeviceBuffers& buffers,
    int group_size,
    int topk,
    int group_offset,
    int group_count,
    cudaStream_t stream = 0) {
    int threads = next_power_of_two(group_size);
    if (threads < topk) {
        threads = next_power_of_two(topk);
    }
    if (threads > 1024) {
        throw std::runtime_error("bitonic path supports group_size <= 1024 in this implementation");
    }
    size_t shmem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int));
    segmented_bitonic_topk_kernel<<<group_count, threads, shmem, stream>>>(
        buffers.d_keys, buffers.d_values, group_size, topk, group_offset,
        buffers.d_out_keys, buffers.d_out_values);
}

// Execute and validate the block-level bitonic segmented top-k path.
BenchResult run_gpu_bitonic(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    std::vector<float>* final_keys,
    std::vector<int>* final_values) {
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    double best_ms = std::numeric_limits<double>::infinity();
    double best_kernel_ms = std::numeric_limits<double>::infinity();
    KernelTimer timer;
    for (int r = 0; r < opt.repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        timer.start();
        launch_bitonic_range(buffers, opt.group_size, opt.topk, 0, opt.groups);
        CUDA_CHECK(cudaGetLastError());
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        best_ms = std::min(best_ms, now_ms() - start);
        best_kernel_ms = std::min(best_kernel_ms, timer.elapsed());
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
    return {"gpu_bitonic_segmented_topk", best_ms, ok, best_kernel_ms};
}

// Convert a float and signed candidate id into one radix-sortable tie-broken key.
__device__ unsigned long long encode_sort_key(float key, int value) {
    unsigned int bits = __float_as_uint(key == 0.0f ? 0.0f : key);
    unsigned int ordered_key = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    unsigned int ordered_value = static_cast<unsigned int>(value) ^ 0x80000000u;
    return (static_cast<unsigned long long>(ordered_key) << 32) | ordered_value;
}

// Encode every input pair before the segmented radix sort.
__global__ void encode_sort_pairs_kernel(
    const float* keys,
    const int* values,
    unsigned long long* encoded,
    size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        encoded[index] = encode_sort_key(keys[index], values[index]);
    }
}

// Decode only the first top-k entries from each fully sorted radix segment.
__global__ void compact_radix_topk_kernel(
    const unsigned long long* sorted_keys,
    const int* sorted_values,
    int groups,
    int group_size,
    int topk,
    float* out_keys,
    int* out_values) {
    size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t output_count = static_cast<size_t>(groups) * topk;
    if (output_index >= output_count) {
        return;
    }
    int group = static_cast<int>(output_index / topk);
    int rank = static_cast<int>(output_index % topk);
    size_t sorted_index = static_cast<size_t>(group) * group_size + rank;
    unsigned int ordered = static_cast<unsigned int>(sorted_keys[sorted_index] >> 32);
    unsigned int bits = ordered ^ ((ordered & 0x80000000u) ? 0x80000000u : 0xffffffffu);
    out_keys[output_index] = __uint_as_float(bits);
    out_values[output_index] = sorted_values[sorted_index];
}

// Use CUB segmented radix sort when one block cannot hold an entire group.
BenchResult run_gpu_segmented_radix(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    std::vector<float>* final_keys,
    std::vector<int>* final_values) {
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    size_t count = keys.size();
    auto* d_encoded = device_alloc<unsigned long long>(count);
    auto* d_sorted_keys = device_alloc<unsigned long long>(count);
    auto* d_sorted_values = device_alloc<int>(count);
    auto* d_offsets = device_alloc<int>(static_cast<size_t>(opt.groups) + 1);
    std::vector<int> offsets(static_cast<size_t>(opt.groups) + 1);
    for (int group = 0; group <= opt.groups; ++group) {
        offsets[group] = group * opt.group_size;
    }
    copy_to_device(d_offsets, offsets);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_encoded, d_sorted_keys,
        buffers.d_values, d_sorted_values, static_cast<int>(count), opt.groups,
        d_offsets, d_offsets + 1));
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    double best_ms = std::numeric_limits<double>::infinity();
    double best_kernel_ms = std::numeric_limits<double>::infinity();
    KernelTimer timer;
    for (int r = 0; r < opt.repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        timer.start();
        int threads = 256;
        int encode_blocks = static_cast<int>((count + threads - 1) / threads);
        encode_sort_pairs_kernel<<<encode_blocks, threads>>>(
            buffers.d_keys, buffers.d_values, d_encoded, count);
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_encoded, d_sorted_keys,
            buffers.d_values, d_sorted_values, static_cast<int>(count), opt.groups,
            d_offsets, d_offsets + 1));
        size_t output_count = static_cast<size_t>(opt.groups) * opt.topk;
        int compact_blocks = static_cast<int>((output_count + threads - 1) / threads);
        compact_radix_topk_kernel<<<compact_blocks, threads>>>(
            d_sorted_keys, d_sorted_values, opt.groups, opt.group_size, opt.topk,
            buffers.d_out_keys, buffers.d_out_values);
        CUDA_CHECK(cudaGetLastError());
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        best_ms = std::min(best_ms, now_ms() - start);
        best_kernel_ms = std::min(best_kernel_ms, timer.elapsed());
    }

    std::vector<float> got_keys(static_cast<size_t>(opt.groups) * opt.topk);
    std::vector<int> got_values(static_cast<size_t>(opt.groups) * opt.topk);
    copy_to_host(got_keys, buffers.d_out_keys);
    copy_to_host(got_values, buffers.d_out_values);
    bool ok = validate_topk(
        ref_keys, ref_values, got_keys, got_values, opt.groups, opt.topk);
    if (final_keys) {
        *final_keys = got_keys;
    }
    if (final_values) {
        *final_values = got_values;
    }
    cudaFree(d_temp_storage);
    cudaFree(d_offsets);
    cudaFree(d_sorted_values);
    cudaFree(d_sorted_keys);
    cudaFree(d_encoded);
    return {"gpu_cub_segmented_radix_topk", best_ms, ok, best_kernel_ms};
}

// Choose the cheaper insertion path for small groups or very small k.
bool use_insertion_path(const Options& opt) {
    return opt.group_size <= 64;
}

// Dispatch each group range to insertion or bitonic sorting based on workload shape.
static void launch_adaptive_range(
    const DeviceBuffers& buffers,
    int group_size,
    int topk,
    int group_offset,
    int group_count,
    cudaStream_t stream = 0) {
    if (group_size <= 64) {
        segmented_warp_micro_topk_kernel<<<group_count, 32, 0, stream>>>(
            buffers.d_keys, buffers.d_values, group_size,
            next_power_of_two(group_size), topk, group_offset,
            buffers.d_out_keys, buffers.d_out_values);
    } else {
        launch_bitonic_range(buffers, group_size, topk, group_offset, group_count, stream);
    }
}

// Execute and validate the adaptive GPU segmented top-k path.
BenchResult run_gpu_adaptive(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    std::vector<float>* final_keys,
    std::vector<int>* final_values) {
    if (opt.group_size > 1024) {
        return run_gpu_segmented_radix(
            opt, keys, values, ref_keys, ref_values, final_keys, final_values);
    }
    if (opt.group_size <= 128) {
        std::vector<float> insertion_keys;
        std::vector<int> insertion_values;
        std::vector<float> warp_keys;
        std::vector<int> warp_values;
        std::vector<float> bitonic_keys;
        std::vector<int> bitonic_values;
        BenchResult insertion = run_gpu_parallel_insertion(
            opt, keys, values, ref_keys, ref_values,
            final_keys ? &insertion_keys : nullptr,
            final_values ? &insertion_values : nullptr);
        BenchResult best = insertion;
        std::vector<float>* best_keys = &insertion_keys;
        std::vector<int>* best_values = &insertion_values;
        bool all_valid = insertion.valid;

        if (opt.group_size <= 64) {
            BenchResult warp = run_gpu_insertion(
                opt, keys, values, ref_keys, ref_values,
                final_keys ? &warp_keys : nullptr,
                final_values ? &warp_values : nullptr);
            all_valid = all_valid && warp.valid;
            if (warp.milliseconds < best.milliseconds) {
                best = warp;
                best_keys = &warp_keys;
                best_values = &warp_values;
            }
        }
        BenchResult bitonic = run_gpu_bitonic(
            opt, keys, values, ref_keys, ref_values,
            final_keys ? &bitonic_keys : nullptr,
            final_values ? &bitonic_values : nullptr);
        all_valid = all_valid && bitonic.valid;
        if (bitonic.milliseconds < best.milliseconds) {
            best = bitonic;
            best_keys = &bitonic_keys;
            best_values = &bitonic_values;
        }
        if (final_keys) {
            *final_keys = std::move(*best_keys);
        }
        if (final_values) {
            *final_values = std::move(*best_values);
        }
        best.name = "gpu_autotuned_" + best.name.substr(4);
        best.valid = all_valid;
        return best;
    }
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    double best_ms = std::numeric_limits<double>::infinity();
    double best_kernel_ms = std::numeric_limits<double>::infinity();
    KernelTimer timer;
    for (int r = 0; r < opt.repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        timer.start();
        launch_adaptive_range(buffers, opt.group_size, opt.topk, 0, opt.groups);
        CUDA_CHECK(cudaGetLastError());
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        best_ms = std::min(best_ms, now_ms() - start);
        best_kernel_ms = std::min(best_kernel_ms, timer.elapsed());
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
    return {"gpu_adaptive_segmented_topk", best_ms, ok, best_kernel_ms};
}

// Describe a contiguous group range that can be scheduled independently.
struct SortRequest {
    int group_offset = 0;
    int group_count = 0;
    int group_size = 0;
    int topk = 0;
};

// Split segmented rows into near-even requests for asynchronous scheduling.
static std::vector<SortRequest> make_even_requests(const Options& opt) {
    int chunks = std::max(1, std::min(opt.streams, opt.groups));
    std::vector<SortRequest> requests;
    int base = 0;
    for (int i = 0; i < chunks; ++i) {
        int remaining = opt.groups - base;
        int take = (remaining + (chunks - i) - 1) / (chunks - i);
        requests.push_back({base, take, opt.group_size, opt.topk});
        base += take;
    }
    return requests;
}

// Manage non-blocking CUDA streams used by scheduled sort requests.
class StreamPool {
public:
    explicit StreamPool(int count) : streams_(std::max(1, count)) {
        for (auto& stream : streams_) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
    }
    ~StreamPool() {
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }
    cudaStream_t get(int index) const { return streams_[static_cast<size_t>(index) % streams_.size()]; }
    void synchronize() const {
        for (auto stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
private:
    std::vector<cudaStream_t> streams_;
};

// Execute adaptive sort requests over multiple CUDA streams and validate output.
BenchResult run_gpu_scheduler(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    std::vector<float>* final_keys,
    std::vector<int>* final_values) {
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    auto requests = make_even_requests(opt);
    double best_ms = std::numeric_limits<double>::infinity();
    for (int r = 0; r < opt.repeats; ++r) {
        StreamPool pool(opt.streams);
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& req = requests[i];
            launch_adaptive_range(buffers, req.group_size, req.topk, req.group_offset, req.group_count, pool.get(static_cast<int>(i)));
        }
        CUDA_CHECK(cudaGetLastError());
        pool.synchronize();
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
    return {"gpu_async_scheduler_topk", best_ms, ok};
}

// Measure allocation, host transfers, scheduled kernels, and validation output separately.
PipelineTiming run_gpu_scheduler_pipeline(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values) {
    PipelineTiming best;
    best.total_ms = std::numeric_limits<double>::infinity();
    auto requests = make_even_requests(opt);

    for (int r = 0; r < opt.repeats; ++r) {
        double total_start = now_ms();
        double h2d_start = now_ms();
        DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
        CUDA_CHECK(cudaDeviceSynchronize());
        double h2d_ms = now_ms() - h2d_start;

        StreamPool pool(opt.streams);
        double kernel_start = now_ms();
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& req = requests[i];
            launch_adaptive_range(
                buffers, req.group_size, req.topk, req.group_offset,
                req.group_count, pool.get(static_cast<int>(i)));
        }
        CUDA_CHECK(cudaGetLastError());
        pool.synchronize();
        double kernel_ms = now_ms() - kernel_start;

        std::vector<float> got_keys(static_cast<size_t>(opt.groups) * opt.topk);
        std::vector<int> got_values(static_cast<size_t>(opt.groups) * opt.topk);
        double d2h_start = now_ms();
        copy_to_host(got_keys, buffers.d_out_keys);
        copy_to_host(got_values, buffers.d_out_values);
        double d2h_ms = now_ms() - d2h_start;
        double total_ms = now_ms() - total_start;

        if (total_ms < best.total_ms) {
            best.h2d_ms = h2d_ms;
            best.kernel_ms = kernel_ms;
            best.d2h_ms = d2h_ms;
            best.total_ms = total_ms;
            best.valid = validate_topk(
                ref_keys, ref_values, got_keys, got_values, opt.groups, opt.topk);
        }
    }
    return best;
}

// Establish a correctness and latency baseline by executing variable requests one at a time.
BenchResult run_gpu_heterogeneous_sequential(
    const Options& opt,
    const PackedSortWorkload& workload) {
    double best_ms = std::numeric_limits<double>::infinity();
    bool valid = true;
    for (int repeat = 0; repeat < opt.repeats; ++repeat) {
        double start = now_ms();
        bool repeat_valid = true;
        for (const auto& request : workload.requests) {
            size_t input_count = static_cast<size_t>(request.groups) * request.group_size;
            size_t output_count = static_cast<size_t>(request.groups) * request.topk;
            std::vector<float> keys(
                workload.keys.begin() + static_cast<std::ptrdiff_t>(request.input_offset),
                workload.keys.begin() + static_cast<std::ptrdiff_t>(request.input_offset + input_count));
            std::vector<int> values(
                workload.values.begin() + static_cast<std::ptrdiff_t>(request.input_offset),
                workload.values.begin() + static_cast<std::ptrdiff_t>(request.input_offset + input_count));
            std::vector<float> reference_keys(
                workload.reference_keys.begin() + static_cast<std::ptrdiff_t>(request.output_offset),
                workload.reference_keys.begin() + static_cast<std::ptrdiff_t>(request.output_offset + output_count));
            std::vector<int> reference_values(
                workload.reference_values.begin() + static_cast<std::ptrdiff_t>(request.output_offset),
                workload.reference_values.begin() + static_cast<std::ptrdiff_t>(request.output_offset + output_count));
            Options request_options = opt;
            request_options.groups = request.groups;
            request_options.group_size = request.group_size;
            request_options.topk = request.topk;
            request_options.repeats = 1;
            BenchResult result = run_gpu_adaptive(
                request_options, keys, values, reference_keys, reference_values);
            repeat_valid = repeat_valid && result.valid;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double elapsed = now_ms() - start;
        if (elapsed < best_ms) {
            best_ms = elapsed;
            valid = repeat_valid;
        }
    }
    return {"gpu_heterogeneous_sequential", best_ms, valid};
}

// Own one packed input allocation and its compact variable-size output allocation.
struct PackedDeviceBuffers {
    float* d_keys = nullptr;
    int* d_values = nullptr;
    float* d_out_keys = nullptr;
    int* d_out_values = nullptr;

    explicit PackedDeviceBuffers(const PackedSortWorkload& workload) {
        d_keys = device_alloc<float>(workload.keys.size());
        d_values = device_alloc<int>(workload.values.size());
        d_out_keys = device_alloc<float>(workload.reference_keys.size());
        d_out_values = device_alloc<int>(workload.reference_values.size());
        copy_to_device(d_keys, workload.keys);
        copy_to_device(d_values, workload.values);
    }

    ~PackedDeviceBuffers() {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_out_keys);
        cudaFree(d_out_values);
    }
};

// Reserve independent CUB storage for a large request that may overlap other streams.
struct RadixRequestWorkspace {
    unsigned long long* d_encoded = nullptr;
    unsigned long long* d_sorted_keys = nullptr;
    int* d_sorted_values = nullptr;
    int* d_offsets = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    explicit RadixRequestWorkspace(const SortRequestDescriptor& request) {
        size_t count = static_cast<size_t>(request.groups) * request.group_size;
        d_encoded = device_alloc<unsigned long long>(count);
        d_sorted_keys = device_alloc<unsigned long long>(count);
        d_sorted_values = device_alloc<int>(count);
        d_offsets = device_alloc<int>(static_cast<size_t>(request.groups) + 1);
        std::vector<int> offsets(static_cast<size_t>(request.groups) + 1);
        for (int group = 0; group <= request.groups; ++group) {
            offsets[group] = group * request.group_size;
        }
        copy_to_device(d_offsets, offsets);
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_encoded, d_sorted_keys,
            static_cast<const int*>(nullptr), d_sorted_values,
            static_cast<int>(count), request.groups, d_offsets, d_offsets + 1));
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    ~RadixRequestWorkspace() {
        cudaFree(d_temp_storage);
        cudaFree(d_offsets);
        cudaFree(d_sorted_values);
        cudaFree(d_sorted_keys);
        cudaFree(d_encoded);
    }
};

// Assign requests to the currently least-loaded stream using an n log n cost estimate.
static std::vector<int> assign_request_streams(
    const PackedSortWorkload& workload,
    int stream_count) {
    stream_count = std::max(1, stream_count);
    std::vector<double> loads(static_cast<size_t>(stream_count), 0.0);
    std::vector<int> assignments(workload.requests.size(), 0);
    for (size_t i = 0; i < workload.requests.size(); ++i) {
        const auto& request = workload.requests[i];
        int stream = static_cast<int>(
            std::min_element(loads.begin(), loads.end()) - loads.begin());
        assignments[i] = stream;
        double candidates = static_cast<double>(request.groups) * request.group_size;
        loads[stream] += candidates * (std::log2(static_cast<double>(request.group_size)) + 1.0);
    }
    return assignments;
}

enum class DeviceSortAlgorithm {
    ParallelInsertion,
    WarpMicro,
    BlockBitonic,
    SegmentedRadix
};

struct DeviceShape {
    int groups = 0;
    int group_size = 0;
    int topk = 0;

    bool operator==(const DeviceShape& other) const {
        return groups == other.groups &&
               group_size == other.group_size &&
               topk == other.topk;
    }
};

struct DeviceShapeHash {
    size_t operator()(const DeviceShape& shape) const {
        size_t hash = std::hash<int>{}(shape.groups);
        hash ^= std::hash<int>{}(shape.group_size) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(shape.topk) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        return hash;
    }
};

// Build an exact cache key from every shape dimension that affects kernel cost.
static DeviceShape shape_key(const DeviceTopkRequest& request) {
    return {request.groups, request.group_size, request.topk};
}

// Launch one caller-owned device request with the selected algorithm.
static void launch_device_request(
    const DeviceTopkRequest& request,
    RadixRequestWorkspace* radix,
    cudaStream_t stream,
    DeviceSortAlgorithm algorithm) {
    if (algorithm == DeviceSortAlgorithm::ParallelInsertion) {
        launch_parallel_insertion(
            request.keys, request.values, request.group_size, request.topk, 0,
            request.groups, request.out_keys, request.out_values, stream);
        return;
    }
    if (algorithm == DeviceSortAlgorithm::WarpMicro) {
        segmented_warp_micro_topk_kernel<<<request.groups, 32, 0, stream>>>(
            request.keys, request.values, request.group_size,
            next_power_of_two(request.group_size), request.topk, 0,
            request.out_keys, request.out_values);
        return;
    }
    if (algorithm == DeviceSortAlgorithm::BlockBitonic) {
        int threads = next_power_of_two(request.group_size);
        size_t shared_bytes = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int));
        segmented_bitonic_topk_kernel<<<request.groups, threads, shared_bytes, stream>>>(
            request.keys, request.values, request.group_size, request.topk, 0,
            request.out_keys, request.out_values);
        return;
    }

    size_t count = static_cast<size_t>(request.groups) * request.group_size;
    int threads = 256;
    int encode_blocks = static_cast<int>((count + threads - 1) / threads);
    encode_sort_pairs_kernel<<<encode_blocks, threads, 0, stream>>>(
        request.keys, request.values, radix->d_encoded, count);
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        radix->d_temp_storage, radix->temp_storage_bytes,
        radix->d_encoded, radix->d_sorted_keys,
        request.values, radix->d_sorted_values,
        static_cast<int>(count), request.groups,
        radix->d_offsets, radix->d_offsets + 1, 0, 64, stream));
    size_t output_count = static_cast<size_t>(request.groups) * request.topk;
    int compact_blocks = static_cast<int>((output_count + threads - 1) / threads);
    compact_radix_topk_kernel<<<compact_blocks, threads, 0, stream>>>(
        radix->d_sorted_keys, radix->d_sorted_values,
        request.groups, request.group_size, request.topk,
        request.out_keys, request.out_values);
}

// Time every feasible small-group kernel and retain the fastest choice for this shape.
static DeviceSortAlgorithm tune_device_algorithm(
    const DeviceTopkRequest& request,
    cudaStream_t stream) {
    if (request.group_size > 1024) {
        return DeviceSortAlgorithm::SegmentedRadix;
    }
    if (request.group_size > 128) {
        return DeviceSortAlgorithm::BlockBitonic;
    }

    auto measure = [&](DeviceSortAlgorithm algorithm) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        float best_ms = std::numeric_limits<float>::infinity();
        for (int repeat = 0; repeat < 3; ++repeat) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            launch_device_request(request, nullptr, stream, algorithm);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
            best_ms = std::min(best_ms, elapsed);
        }
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaEventDestroy(start));
        return best_ms;
    };

    DeviceSortAlgorithm best_algorithm = DeviceSortAlgorithm::ParallelInsertion;
    float best_ms = measure(best_algorithm);
    if (request.group_size <= 64) {
        float warp_ms = measure(DeviceSortAlgorithm::WarpMicro);
        if (warp_ms < best_ms) {
            best_ms = warp_ms;
            best_algorithm = DeviceSortAlgorithm::WarpMicro;
        }
    }
    float bitonic_ms = measure(DeviceSortAlgorithm::BlockBitonic);
    if (bitonic_ms < best_ms) {
        best_algorithm = DeviceSortAlgorithm::BlockBitonic;
    }
    return best_algorithm;
}

// Translate offsets in a packed workload into the public device-pointer request contract.
static void launch_packed_request(
    const PackedDeviceBuffers& buffers,
    const SortRequestDescriptor& request,
    RadixRequestWorkspace* radix,
    cudaStream_t stream,
    DeviceSortAlgorithm algorithm) {
    DeviceTopkRequest device_request;
    device_request.keys = buffers.d_keys + request.input_offset;
    device_request.values = buffers.d_values + request.input_offset;
    device_request.out_keys = buffers.d_out_keys + request.output_offset;
    device_request.out_values = buffers.d_out_values + request.output_offset;
    device_request.groups = request.groups;
    device_request.group_size = request.group_size;
    device_request.topk = request.topk;
    launch_device_request(device_request, radix, stream, algorithm);
}

// Keep stream load and per-request radix storage behind the public C++ API.
struct GpuTopkScheduler::Impl {
    explicit Impl(int stream_count)
        : pool(std::max(1, stream_count)),
          loads(static_cast<size_t>(std::max(1, stream_count)), 0.0) {}

    StreamPool pool;
    std::vector<double> loads;
    std::vector<std::unique_ptr<RadixRequestWorkspace>> pending_radix;
    std::unordered_map<DeviceShape, DeviceSortAlgorithm, DeviceShapeHash> algorithms;
};

GpuTopkScheduler::GpuTopkScheduler(int stream_count)
    : impl_(new Impl(stream_count)) {}

GpuTopkScheduler::~GpuTopkScheduler() {
    if (impl_) {
        try {
            impl_->pool.synchronize();
        } catch (...) {
        }
        delete impl_;
    }
}

void GpuTopkScheduler::submit(const DeviceTopkRequest& request) {
    if (!request.keys || !request.values || !request.out_keys || !request.out_values) {
        throw std::runtime_error("device top-k request contains a null pointer");
    }
    if (request.groups <= 0 || request.group_size <= 0 || request.topk <= 0 ||
        request.topk > request.group_size || request.topk > GPU_SORT_MAX_TOPK) {
        throw std::runtime_error("device top-k request has an invalid shape");
    }
    if (static_cast<size_t>(request.groups) * request.group_size >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("device top-k request exceeds the 32-bit index range");
    }
    int stream_index = static_cast<int>(
        std::min_element(impl_->loads.begin(), impl_->loads.end()) - impl_->loads.begin());
    RadixRequestWorkspace* radix = nullptr;
    if (request.group_size > 1024) {
        SortRequestDescriptor descriptor;
        descriptor.groups = request.groups;
        descriptor.group_size = request.group_size;
        descriptor.topk = request.topk;
        impl_->pending_radix.push_back(
            std::make_unique<RadixRequestWorkspace>(descriptor));
        radix = impl_->pending_radix.back().get();
    }
    DeviceShape key = shape_key(request);
    auto found = impl_->algorithms.find(key);
    if (found == impl_->algorithms.end()) {
        DeviceSortAlgorithm selected = tune_device_algorithm(
            request, impl_->pool.get(stream_index));
        found = impl_->algorithms.emplace(key, selected).first;
    }
    launch_device_request(
        request, radix, impl_->pool.get(stream_index), found->second);
    double candidates = static_cast<double>(request.groups) * request.group_size;
    impl_->loads[stream_index] +=
        candidates * (std::log2(static_cast<double>(request.group_size)) + 1.0);
}

void GpuTopkScheduler::synchronize() {
    impl_->pool.synchronize();
    impl_->pending_radix.clear();
    std::fill(impl_->loads.begin(), impl_->loads.end(), 0.0);
}

// Share completion state between the producer-facing ticket and the queue worker.
struct GpuTopkCompletion::State {
    mutable std::mutex mutex;
    mutable std::condition_variable completed_cv;
    bool completed = false;
    std::exception_ptr error;
};

GpuTopkCompletion::GpuTopkCompletion(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

void GpuTopkCompletion::wait() const {
    if (!state_) {
        throw std::runtime_error("cannot wait on an empty GPU top-k completion");
    }
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->completed_cv.wait(lock, [&] { return state_->completed; });
    if (state_->error) {
        std::rethrow_exception(state_->error);
    }
}

bool GpuTopkCompletion::ready() const {
    if (!state_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->completed;
}

GpuTopkCompletion::operator bool() const {
    return static_cast<bool>(state_);
}

// Keep host queue synchronization and the GPU scheduler behind one public queue object.
struct GpuTopkRequestQueue::Impl {
    struct PendingRequest {
        DeviceTopkRequest request;
        std::shared_ptr<GpuTopkCompletion::State> completion;
    };

    Impl(int stream_count, size_t maximum_pending)
        : scheduler(stream_count), max_pending(std::max<size_t>(1, maximum_pending)) {
        CUDA_CHECK(cudaGetDevice(&device));
        worker = std::thread([this] { worker_loop(); });
    }

    ~Impl() {
        try {
            flush();
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work_cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    std::shared_ptr<GpuTopkCompletion::State> enqueue(const DeviceTopkRequest& request) {
        auto completion = std::make_shared<GpuTopkCompletion::State>();
        std::unique_lock<std::mutex> lock(mutex);
        capacity_cv.wait(lock, [&] { return stopping || outstanding < max_pending; });
        if (stopping) {
            throw std::runtime_error("cannot enqueue into a stopped GPU top-k request queue");
        }
        pending.push_back({request, completion});
        ++outstanding;
        lock.unlock();
        work_cv.notify_one();
        return completion;
    }

    void flush() {
        std::unique_lock<std::mutex> lock(mutex);
        drained_cv.wait(lock, [&] { return outstanding == 0; });
        if (first_error) {
            std::rethrow_exception(first_error);
        }
    }

    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(mutex);
        return outstanding;
    }

    void worker_loop() {
        std::exception_ptr device_error;
        try {
            CUDA_CHECK(cudaSetDevice(device));
        } catch (...) {
            device_error = std::current_exception();
        }

        for (;;) {
            std::vector<PendingRequest> batch;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work_cv.wait(lock, [&] { return stopping || !pending.empty(); });
                if (stopping && pending.empty()) {
                    return;
                }
                work_cv.wait_for(lock, std::chrono::microseconds(200), [&] {
                    return stopping || pending.size() >= max_pending;
                });
                while (!pending.empty()) {
                    batch.push_back(std::move(pending.front()));
                    pending.pop_front();
                }
            }

            std::exception_ptr batch_error = device_error;
            if (!batch_error) {
                try {
                    for (const auto& item : batch) {
                        scheduler.submit(item.request);
                    }
                    scheduler.synchronize();
                } catch (...) {
                    batch_error = std::current_exception();
                    try {
                        scheduler.synchronize();
                    } catch (...) {
                    }
                }
            }

            for (const auto& item : batch) {
                {
                    std::lock_guard<std::mutex> state_lock(item.completion->mutex);
                    item.completion->error = batch_error;
                    item.completion->completed = true;
                }
                item.completion->completed_cv.notify_all();
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                outstanding -= batch.size();
                if (batch_error && !first_error) {
                    first_error = batch_error;
                }
            }
            capacity_cv.notify_all();
            drained_cv.notify_all();
        }
    }

    GpuTopkScheduler scheduler;
    size_t max_pending = 1;
    int device = 0;
    mutable std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable capacity_cv;
    std::condition_variable drained_cv;
    std::deque<PendingRequest> pending;
    size_t outstanding = 0;
    bool stopping = false;
    std::exception_ptr first_error;
    std::thread worker;
};

GpuTopkRequestQueue::GpuTopkRequestQueue(int stream_count, size_t max_pending)
    : impl_(new Impl(stream_count, max_pending)) {}

GpuTopkRequestQueue::~GpuTopkRequestQueue() {
    delete impl_;
}

GpuTopkCompletion GpuTopkRequestQueue::enqueue(const DeviceTopkRequest& request) {
    return GpuTopkCompletion(impl_->enqueue(request));
}

void GpuTopkRequestQueue::flush() {
    impl_->flush();
}

size_t GpuTopkRequestQueue::pending() const {
    return impl_->pending_count();
}

// Compare serialized and cost-balanced submissions on the same resident device buffers.
SchedulerComparison run_gpu_heterogeneous_scheduler(
    const Options& opt,
    const PackedSortWorkload& workload) {
    PackedDeviceBuffers buffers(workload);
    StreamPool pool(opt.streams);
    std::vector<std::unique_ptr<RadixRequestWorkspace>> radix_workspaces;
    radix_workspaces.reserve(workload.requests.size());
    for (const auto& request : workload.requests) {
        if (request.group_size > 1024) {
            radix_workspaces.push_back(std::make_unique<RadixRequestWorkspace>(request));
        } else {
            radix_workspaces.push_back(nullptr);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> assignments = assign_request_streams(workload, opt.streams);
    std::vector<DeviceSortAlgorithm> algorithms;
    std::unordered_map<DeviceShape, DeviceSortAlgorithm, DeviceShapeHash> algorithm_cache;
    algorithms.reserve(workload.requests.size());
    for (const auto& request : workload.requests) {
        DeviceTopkRequest device_request;
        device_request.keys = buffers.d_keys + request.input_offset;
        device_request.values = buffers.d_values + request.input_offset;
        device_request.out_keys = buffers.d_out_keys + request.output_offset;
        device_request.out_values = buffers.d_out_values + request.output_offset;
        device_request.groups = request.groups;
        device_request.group_size = request.group_size;
        device_request.topk = request.topk;
        DeviceShape key = shape_key(device_request);
        auto found = algorithm_cache.find(key);
        if (found == algorithm_cache.end()) {
            DeviceSortAlgorithm selected = tune_device_algorithm(
                device_request, pool.get(0));
            found = algorithm_cache.emplace(key, selected).first;
        }
        algorithms.push_back(found->second);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    bool policies_valid = true;
    auto measure = [&](bool asynchronous) {
        double best_ms = std::numeric_limits<double>::infinity();
        for (int repeat = 0; repeat < opt.repeats; ++repeat) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(buffers.d_out_values, 0xff,
                workload.reference_values.size() * sizeof(int)));
            CUDA_CHECK(cudaDeviceSynchronize());
            double start = now_ms();
            for (size_t i = 0; i < workload.requests.size(); ++i) {
                int stream_index = asynchronous ? assignments[i] : 0;
                launch_packed_request(
                    buffers, workload.requests[i], radix_workspaces[i].get(),
                    pool.get(stream_index), algorithms[i]);
            }
            CUDA_CHECK(cudaGetLastError());
            pool.synchronize();
            best_ms = std::min(best_ms, now_ms() - start);
            std::vector<float> checked_keys(workload.reference_keys.size());
            std::vector<int> checked_values(workload.reference_values.size());
            copy_to_host(checked_keys, buffers.d_out_keys);
            copy_to_host(checked_values, buffers.d_out_values);
            policies_valid = validate_topk(workload.reference_keys, workload.reference_values,
                checked_keys, checked_values, 1, static_cast<int>(checked_keys.size())) && policies_valid;
        }
        return best_ms;
    };

    SchedulerComparison comparison;
    comparison.sequential_ms = measure(false);
    comparison.asynchronous_ms = measure(true);
    std::vector<float> got_keys(workload.reference_keys.size());
    std::vector<int> got_values(workload.reference_values.size());
    copy_to_host(got_keys, buffers.d_out_keys);
    copy_to_host(got_values, buffers.d_out_values);
    comparison.valid = policies_valid;
    for (size_t i = 0; i < got_keys.size(); ++i) {
        if (std::fabs(got_keys[i] - workload.reference_keys[i]) > 1e-4f ||
            got_values[i] != workload.reference_values[i]) {
            comparison.valid = false;
            break;
        }
    }
    return comparison;
}

// Exercise the public API with resident distance buffers as an integration contract test.
BenchResult run_gpu_device_api_validation(
    const Options& opt,
    const PackedSortWorkload& workload) {
    PackedDeviceBuffers buffers(workload);
    GpuTopkScheduler scheduler(opt.streams);
    CUDA_CHECK(cudaDeviceSynchronize());
    double start = now_ms();
    for (const auto& request : workload.requests) {
        DeviceTopkRequest device_request;
        device_request.keys = buffers.d_keys + request.input_offset;
        device_request.values = buffers.d_values + request.input_offset;
        device_request.out_keys = buffers.d_out_keys + request.output_offset;
        device_request.out_values = buffers.d_out_values + request.output_offset;
        device_request.groups = request.groups;
        device_request.group_size = request.group_size;
        device_request.topk = request.topk;
        scheduler.submit(device_request);
    }
    scheduler.synchronize();
    double elapsed = now_ms() - start;

    std::vector<float> got_keys(workload.reference_keys.size());
    std::vector<int> got_values(workload.reference_values.size());
    copy_to_host(got_keys, buffers.d_out_keys);
    copy_to_host(got_values, buffers.d_out_values);
    bool valid = true;
    for (size_t i = 0; i < got_keys.size(); ++i) {
        if (std::fabs(got_keys[i] - workload.reference_keys[i]) > 1e-4f ||
            got_values[i] != workload.reference_values[i]) {
            valid = false;
            break;
        }
    }
    return {"gpu_device_pointer_api", elapsed, valid};
}

// Validate producer-side queueing, bounded pending work, and completion tracking.
BenchResult run_gpu_request_queue_validation(
    const Options& opt,
    const PackedSortWorkload& workload) {
    PackedDeviceBuffers buffers(workload);
    GpuTopkRequestQueue queue(
        opt.streams, static_cast<size_t>(opt.queue_depth));
    CUDA_CHECK(cudaDeviceSynchronize());

    double best_ms = std::numeric_limits<double>::infinity();
    for (int repeat = 0; repeat < opt.repeats; ++repeat) {
        std::vector<GpuTopkCompletion> completions;
        completions.reserve(workload.requests.size());
        double start = now_ms();
        for (const auto& request : workload.requests) {
            DeviceTopkRequest device_request;
            device_request.keys = buffers.d_keys + request.input_offset;
            device_request.values = buffers.d_values + request.input_offset;
            device_request.out_keys = buffers.d_out_keys + request.output_offset;
            device_request.out_values = buffers.d_out_values + request.output_offset;
            device_request.groups = request.groups;
            device_request.group_size = request.group_size;
            device_request.topk = request.topk;
            completions.push_back(queue.enqueue(device_request));
        }
        for (const auto& completion : completions) {
            completion.wait();
        }
        queue.flush();
        best_ms = std::min(best_ms, now_ms() - start);
    }

    std::vector<float> got_keys(workload.reference_keys.size());
    std::vector<int> got_values(workload.reference_values.size());
    copy_to_host(got_keys, buffers.d_out_keys);
    copy_to_host(got_values, buffers.d_out_values);
    bool valid = true;
    for (size_t index = 0; index < got_keys.size(); ++index) {
        if (std::fabs(got_keys[index] - workload.reference_keys[index]) > 1e-4f ||
            got_values[index] != workload.reference_values[index]) {
            valid = false;
            break;
        }
    }
    return {"gpu_async_request_queue", best_ms, valid};
}

// Measure the full heterogeneous path from packed host input to compact host output.
PipelineTiming run_gpu_heterogeneous_pipeline(
    const Options& opt,
    const PackedSortWorkload& workload) {
    PipelineTiming best;
    best.total_ms = std::numeric_limits<double>::infinity();
    for (int repeat = 0; repeat < opt.repeats; ++repeat) {
        double total_start = now_ms();
        double h2d_start = now_ms();
        PackedDeviceBuffers buffers(workload);
        CUDA_CHECK(cudaDeviceSynchronize());
        double h2d_ms = now_ms() - h2d_start;

        GpuTopkScheduler scheduler(opt.streams);
        double kernel_start = now_ms();
        for (const auto& request : workload.requests) {
            DeviceTopkRequest device_request;
            device_request.keys = buffers.d_keys + request.input_offset;
            device_request.values = buffers.d_values + request.input_offset;
            device_request.out_keys = buffers.d_out_keys + request.output_offset;
            device_request.out_values = buffers.d_out_values + request.output_offset;
            device_request.groups = request.groups;
            device_request.group_size = request.group_size;
            device_request.topk = request.topk;
            scheduler.submit(device_request);
        }
        scheduler.synchronize();
        double kernel_ms = now_ms() - kernel_start;

        std::vector<float> got_keys(workload.reference_keys.size());
        std::vector<int> got_values(workload.reference_values.size());
        double d2h_start = now_ms();
        copy_to_host(got_keys, buffers.d_out_keys);
        copy_to_host(got_values, buffers.d_out_values);
        double d2h_ms = now_ms() - d2h_start;
        double total_ms = now_ms() - total_start;

        bool valid = true;
        for (size_t i = 0; i < got_keys.size(); ++i) {
            if (std::fabs(got_keys[i] - workload.reference_keys[i]) > 1e-4f ||
                got_values[i] != workload.reference_values[i]) {
                valid = false;
                break;
            }
        }
        if (total_ms < best.total_ms) {
            best.h2d_ms = h2d_ms;
            best.kernel_ms = kernel_ms;
            best.d2h_ms = d2h_ms;
            best.total_ms = total_ms;
            best.valid = valid;
        }
    }
    return best;
}

// Reject distance-tile inputs that do not match the configured row layout.
static void validate_distance_tile_input(
    const Options& opt,
    const std::vector<float>& tile_distances,
    const std::vector<int>& candidate_ids) {
    const size_t expected_count = static_cast<size_t>(opt.groups) * opt.group_size;
    if (tile_distances.size() != expected_count || candidate_ids.size() != expected_count) {
        throw std::invalid_argument(
            "distance tile and candidate ids must contain groups * group_size elements");
    }
}

// Adapt row-wise distance tiles into the same segmented top-k scheduler path.
BenchResult run_distance_tile_topk_adapter(
    const Options& opt,
    const std::vector<float>& tile_distances,
    const std::vector<int>& candidate_ids,
    std::vector<float>& out_keys,
    std::vector<int>& out_values) {
    validate_distance_tile_input(opt, tile_distances, candidate_ids);
    std::vector<float> ref_keys;
    std::vector<int> ref_values;
    cpu_segmented_topk(
        tile_distances, candidate_ids, opt.groups, opt.group_size, opt.topk,
        ref_keys, ref_values);
    BenchResult result = run_gpu_scheduler(
        opt, tile_distances, candidate_ids, ref_keys, ref_values,
        &out_keys, &out_values);
    result.name = "gpu_distance_tile_topk_adapter";
    return result;
}

// Measure the complete host-to-host distance-tile adapter execution.
BenchResult run_distance_tile_topk_adapter_end_to_end(
    const Options& opt,
    const std::vector<float>& tile_distances,
    const std::vector<int>& candidate_ids) {
    validate_distance_tile_input(opt, tile_distances, candidate_ids);
    std::vector<float> ref_keys;
    std::vector<int> ref_values;
    cpu_segmented_topk(
        tile_distances, candidate_ids, opt.groups, opt.group_size, opt.topk,
        ref_keys, ref_values);
    PipelineTiming pipeline = run_gpu_scheduler_pipeline(
        opt, tile_distances, candidate_ids, ref_keys, ref_values);
    return {"gpu_distance_tile_adapter_total", pipeline.total_ms, pipeline.valid};
}

#endif
