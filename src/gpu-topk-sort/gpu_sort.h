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
    int groups = 64;
    int group_size = 64;
    int topk = 8;
    int streams = 1;
    int repeats = 1;
    int seed = 1234;
    std::string keys_bin_path;
    std::string values_bin_path;
};

// Common result record for CPU and GPU benchmark paths.
struct BenchResult {
    std::string name;
    double milliseconds = 0.0;
    bool valid = true;
};

void print_usage(const char* prog);
Options parse_options(int argc, char** argv);
std::vector<float> make_synthetic_keys(const Options& opt);
std::vector<int> make_synthetic_values(const Options& opt);
std::vector<float> load_or_make_keys(const Options& opt);
std::vector<int> load_or_make_values(const Options& opt);
double now_ms();
void cpu_segmented_topk(const std::vector<float>& keys, const std::vector<int>& values, int groups, int group_size, int topk, std::vector<float>& out_keys, std::vector<int>& out_values);
bool validate_topk(const std::vector<float>& ref_keys, const std::vector<int>& ref_values, const std::vector<float>& got_keys, const std::vector<int>& got_values, int groups, int topk, float eps = 1e-4f);
void print_first_group(const std::vector<float>& keys, const std::vector<int>& values, int topk);
int run_gpu_sort_demo(int argc, char** argv);

#endif
