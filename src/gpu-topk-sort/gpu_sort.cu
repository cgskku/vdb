
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

// Let each thread maintain the insertion top-k list for one independent group.
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

// Limit threads as the per-thread top-k state grows to avoid excessive register pressure.
template <int TOPK_CAPACITY>
static void launch_parallel_insertion_with_capacity(
    const DeviceBuffers& buffers,
    int group_size,
    int topk,
    int group_offset,
    int group_count,
    int threads,
    cudaStream_t stream) {
    int blocks = (group_count + threads - 1) / threads;
    segmented_parallel_insertion_topk_kernel<TOPK_CAPACITY><<<blocks, threads, 0, stream>>>(
        buffers.d_keys, buffers.d_values, group_size, topk, group_offset, group_count,
        buffers.d_out_keys, buffers.d_out_values);
}

// Select the smallest local-array specialization that can contain the requested top-k.
static void launch_insertion_range(
    const DeviceBuffers& buffers,
    int group_size,
    int topk,
    int group_offset,
    int group_count,
    cudaStream_t stream = 0) {
    if (topk <= 1) {
        launch_parallel_insertion_with_capacity<1>(
            buffers, group_size, topk, group_offset, group_count, 256, stream);
    } else if (topk <= 4) {
        launch_parallel_insertion_with_capacity<4>(
            buffers, group_size, topk, group_offset, group_count, 256, stream);
    } else if (topk <= 8) {
        launch_parallel_insertion_with_capacity<8>(
            buffers, group_size, topk, group_offset, group_count, 128, stream);
    } else if (topk <= 16) {
        launch_parallel_insertion_with_capacity<16>(
            buffers, group_size, topk, group_offset, group_count, 64, stream);
    } else if (topk <= 32) {
        launch_parallel_insertion_with_capacity<32>(
            buffers, group_size, topk, group_offset, group_count, 32, stream);
    } else if (topk <= 64) {
        launch_parallel_insertion_with_capacity<64>(
            buffers, group_size, topk, group_offset, group_count, 16, stream);
    } else {
        launch_parallel_insertion_with_capacity<GPU_SORT_MAX_TOPK>(
            buffers, group_size, topk, group_offset, group_count, 8, stream);
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
        launch_insertion_range(buffers, opt.group_size, opt.topk, 0, opt.groups);
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
    for (int r = 0; r < opt.repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        launch_bitonic_range(buffers, opt.group_size, opt.topk, 0, opt.groups);
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
    return {"gpu_bitonic_segmented_topk", best_ms, ok};
}

// Choose the cheaper insertion path for small groups or very small k.
bool use_insertion_path(const Options& opt) {
    return opt.group_size <= 64 || opt.topk <= 8;
}

// Dispatch each group range to insertion or bitonic sorting based on workload shape.
static void launch_adaptive_range(
    const DeviceBuffers& buffers,
    int group_size,
    int topk,
    int group_offset,
    int group_count,
    cudaStream_t stream = 0) {
    if (group_size <= 64 || topk <= 8) {
        launch_insertion_range(
            buffers, group_size, topk, group_offset, group_count, stream);
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
    DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
    double best_ms = std::numeric_limits<double>::infinity();
    for (int r = 0; r < opt.repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double start = now_ms();
        launch_adaptive_range(buffers, opt.group_size, opt.topk, 0, opt.groups);
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
    return {"gpu_adaptive_segmented_topk", best_ms, ok};
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
    BenchResult result = run_gpu_end_to_end(
        opt, tile_distances, candidate_ids, ref_keys, ref_values);
    result.name = "gpu_distance_tile_adapter_total";
    return result;
}


// Measure one complete scheduled request from device preparation through host output.
BenchResult run_gpu_end_to_end(
    const Options& opt,
    const std::vector<float>& keys,
    const std::vector<int>& values,
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values) {
    double best_ms = std::numeric_limits<double>::infinity();
    bool best_valid = false;
    auto requests = make_even_requests(opt);
    for (int r = 0; r < opt.repeats; ++r) {
        double start = now_ms();
        DeviceBuffers buffers(keys, values, opt.groups, opt.topk);
        CUDA_CHECK(cudaDeviceSynchronize());
        StreamPool pool(opt.streams);
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& req = requests[i];
            launch_adaptive_range(
                buffers, req.group_size, req.topk, req.group_offset,
                req.group_count, pool.get(static_cast<int>(i)));
        }
        CUDA_CHECK(cudaGetLastError());
        pool.synchronize();
        std::vector<float> got_keys(static_cast<size_t>(opt.groups) * opt.topk);
        std::vector<int> got_values(static_cast<size_t>(opt.groups) * opt.topk);
        copy_to_host(got_keys, buffers.d_out_keys);
        copy_to_host(got_values, buffers.d_out_values);
        double elapsed = now_ms() - start;
        if (elapsed < best_ms) {
            best_ms = elapsed;
            best_valid = validate_topk(
                ref_keys, ref_values, got_keys, got_values, opt.groups, opt.topk);
        }
    }
    return {"gpu_end_to_end", best_ms, best_valid};
}

#endif
