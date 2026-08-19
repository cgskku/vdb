#ifndef GPU_SORT_H
#define GPU_SORT_H

#include <string>
#include <vector>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define GPU_SORT_HAS_CUDA 1
#else
#define GPU_SORT_HAS_CUDA 0
#endif

#ifndef GPU_SORT_MAX_TOPK
#define GPU_SORT_MAX_TOPK 128
#endif

// Runtime options that define the segmented top-k workload.
struct Options {
    int groups = 1024;
    int group_size = 128;
    int topk = 16;
    int streams = 4;
    int repeats = 1;
    int seed = 1234;
    bool profile_cpu = false;
    std::string keys_bin_path;
    std::string values_bin_path;
};

// Common result record for CPU and GPU benchmark paths.
struct BenchResult {
    std::string name;
    double milliseconds = 0.0;
    bool valid = true;
};

// Per-phase CPU timing used to identify the baseline bottleneck.
struct CpuPhaseProfile {
    double total_ms = 0.0;
    double output_init_ms = 0.0;
    double group_index_ms = 0.0;
    double row_fill_ms = 0.0;
    double topk_sort_ms = 0.0;
    double output_store_ms = 0.0;
};

void print_usage(const char* prog);
Options parse_options(int argc, char** argv);
std::vector<float> make_synthetic_keys(const Options& opt);
std::vector<int> make_synthetic_values(const Options& opt);
std::vector<float> load_or_make_keys(const Options& opt);
std::vector<int> load_or_make_values(const Options& opt);
double now_ms();
void cpu_segmented_topk(const std::vector<float>& keys, const std::vector<int>& values, int groups, int group_size, int topk, std::vector<float>& out_keys, std::vector<int>& out_values);
CpuPhaseProfile cpu_segmented_topk_profiled(const std::vector<float>& keys, const std::vector<int>& values, int groups, int group_size, int topk, std::vector<float>& out_keys, std::vector<int>& out_values);
bool validate_topk(const std::vector<float>& ref_keys, const std::vector<int>& ref_values, const std::vector<float>& got_keys, const std::vector<int>& got_values, int groups, int topk, float eps = 1e-4f);
void print_first_group(const std::vector<float>& keys, const std::vector<int>& values, int topk);
void print_cpu_profile(const CpuPhaseProfile& best, const CpuPhaseProfile& avg);

// CUDA entrypoints are declared only when the runtime headers are available.
#if GPU_SORT_HAS_CUDA
void run_warmup_kernel(const std::vector<float>& keys);
BenchResult run_gpu_end_to_end(const Options& opt, const std::vector<float>& keys, const std::vector<int>& values, const std::vector<float>& ref_keys, const std::vector<int>& ref_values);
BenchResult run_gpu_insertion(const Options& opt, const std::vector<float>& keys, const std::vector<int>& values, const std::vector<float>& ref_keys, const std::vector<int>& ref_values, std::vector<float>* final_keys = nullptr, std::vector<int>* final_values = nullptr);
BenchResult run_gpu_bitonic(const Options& opt, const std::vector<float>& keys, const std::vector<int>& values, const std::vector<float>& ref_keys, const std::vector<int>& ref_values, std::vector<float>* final_keys = nullptr, std::vector<int>* final_values = nullptr);
bool use_insertion_path(const Options& opt);
BenchResult run_gpu_adaptive(const Options& opt, const std::vector<float>& keys, const std::vector<int>& values, const std::vector<float>& ref_keys, const std::vector<int>& ref_values, std::vector<float>* final_keys = nullptr, std::vector<int>* final_values = nullptr);
BenchResult run_gpu_scheduler(const Options& opt, const std::vector<float>& keys, const std::vector<int>& values, const std::vector<float>& ref_keys, const std::vector<int>& ref_values, std::vector<float>* final_keys = nullptr, std::vector<int>* final_values = nullptr);
BenchResult run_distance_tile_topk_adapter(const Options& opt, const std::vector<float>& tile_distances, std::vector<float>& out_keys, std::vector<int>& out_values);
#endif

int run_gpu_sort_demo(int argc, char** argv);

#endif
