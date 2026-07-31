
#include "gpu_sort.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

// Print the supported benchmark arguments for this executable.
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--groups N] [--group-size N] [--topk K] [--streams N] [--repeats N]"
              << " [--keys-bin path] [--values-bin path] [--profile-cpu]\n";
}

// Parse command-line options and normalize invalid benchmark values early.
Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--groups") {
            opt.groups = std::atoi(need_value("--groups"));
        }
        else if (arg == "--group-size") {
            opt.group_size = std::atoi(need_value("--group-size"));
        }
        else if (arg == "--topk") {
            opt.topk = std::atoi(need_value("--topk"));
        }
        else if (arg == "--streams") {
            opt.streams = std::atoi(need_value("--streams"));
        }
        else if (arg == "--repeats") {
            opt.repeats = std::atoi(need_value("--repeats"));
        }
        else if (arg == "--keys-bin") {
            opt.keys_bin_path = need_value("--keys-bin");
        }
        else if (arg == "--values-bin") {
            opt.values_bin_path = need_value("--values-bin");
        }
        else if (arg == "--profile-cpu") {
            opt.profile_cpu = true;
        }
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.topk <= 0 || opt.group_size <= 0 || opt.groups <= 0) {
        throw std::runtime_error("groups, group-size, and topk must be positive");
    }
    if (opt.topk > opt.group_size) {
        opt.topk = opt.group_size;
    }
    if (opt.topk > GPU_SORT_MAX_TOPK) {
        throw std::runtime_error("topk exceeds GPU_SORT_MAX_TOPK; rebuild with a larger limit");
    }
    if (opt.streams <= 0) {
        opt.streams = 1;
    }
    if (opt.repeats <= 0) {
        opt.repeats = 1;
    }
    return opt;
}

// Generate deterministic distance keys for repeatable local benchmarks.
std::vector<float> make_synthetic_keys(const Options& opt) {
    std::mt19937 rng(static_cast<unsigned>(opt.seed));
    std::uniform_real_distribution<float> noise(0.0f, 1.0f);
    std::vector<float> keys(static_cast<size_t>(opt.groups) * opt.group_size);
    for (int g = 0; g < opt.groups; ++g) {
        float group_bias = static_cast<float>((g * 17) % 113) * 0.001f;
        for (int i = 0; i < opt.group_size; ++i) {
            float locality = static_cast<float>((i * 31 + g * 7) % 257) * 0.0001f;
            keys[static_cast<size_t>(g) * opt.group_size + i] = noise(rng) + group_bias + locality;
        }
    }
    return keys;
}

// Generate candidate ids that stay aligned with the synthetic distance keys.
std::vector<int> make_synthetic_values(const Options& opt) {
    std::vector<int> values(static_cast<size_t>(opt.groups) * opt.group_size);
    for (int g = 0; g < opt.groups; ++g) {
        for (int i = 0; i < opt.group_size; ++i) {
            values[static_cast<size_t>(g) * opt.group_size + i] = g * opt.group_size + i;
        }
    }
    return values;
}

template <typename T>
// Load a compact binary vector and verify that its shape matches the workload.
static std::vector<T> load_binary_vector(const std::string& path, size_t expected_count, const char* label) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("failed to open ") + label + " file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes < 0 || static_cast<size_t>(bytes) != expected_count * sizeof(T)) {
        std::ostringstream oss;
        oss << label << " file size mismatch: got " << bytes
            << " bytes, expected " << expected_count * sizeof(T);
        throw std::runtime_error(oss.str());
    }
    std::vector<T> out(expected_count);
    in.read(reinterpret_cast<char*>(out.data()), bytes);
    if (!in) {
        throw std::runtime_error(std::string("failed to read ") + label + " file: " + path);
    }
    return out;
}

// Select external distance keys when provided, otherwise fall back to synthetic input.
std::vector<float> load_or_make_keys(const Options& opt) {
    size_t count = static_cast<size_t>(opt.groups) * opt.group_size;
    if (!opt.keys_bin_path.empty()) {
        return load_binary_vector<float>(opt.keys_bin_path, count, "keys");
    }
    return make_synthetic_keys(opt);
}

// Select external candidate ids when provided, otherwise generate aligned ids.
std::vector<int> load_or_make_values(const Options& opt) {
    size_t count = static_cast<size_t>(opt.groups) * opt.group_size;
    if (!opt.values_bin_path.empty()) {
        return load_binary_vector<int>(opt.values_bin_path, count, "values");
    }
    return make_synthetic_values(opt);
}

// Return a millisecond timestamp for lightweight benchmark timing.
double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// Keep only the nearest top-k pairs from each independent group.
void cpu_segmented_topk(
    const std::vector<float>& keys,
    const std::vector<int>& values,
    int groups,
    int group_size,
    int topk,
    std::vector<float>& out_keys,
    std::vector<int>& out_values) {
    out_keys.assign(static_cast<size_t>(groups) * topk, std::numeric_limits<float>::infinity());
    out_values.assign(static_cast<size_t>(groups) * topk, -1);
    std::vector<std::pair<float, int>> row;
    row.reserve(group_size);
    for (int g = 0; g < groups; ++g) {
        row.clear();
        size_t base = static_cast<size_t>(g) * group_size;
        for (int i = 0; i < group_size; ++i) {
            row.emplace_back(keys[base + i], values[base + i]);
        }
        if (topk < group_size) {
            std::partial_sort(row.begin(), row.begin() + topk, row.end());
        } else {
            std::sort(row.begin(), row.end());
        }
        for (int k = 0; k < topk; ++k) {
            out_keys[static_cast<size_t>(g) * topk + k] = row[k].first;
            out_values[static_cast<size_t>(g) * topk + k] = row[k].second;
        }
    }
}

// Convert high-resolution clock intervals into milliseconds.
static double elapsed_ms(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Measure the CPU baseline by phase without changing the output contract.
CpuPhaseProfile cpu_segmented_topk_profiled(
    const std::vector<float>& keys,
    const std::vector<int>& values,
    int groups,
    int group_size,
    int topk,
    std::vector<float>& out_keys,
    std::vector<int>& out_values) {
    using clock = std::chrono::high_resolution_clock;
    CpuPhaseProfile profile;
    auto total_start = clock::now();

    auto init_start = clock::now();
    out_keys.assign(static_cast<size_t>(groups) * topk, std::numeric_limits<float>::infinity());
    out_values.assign(static_cast<size_t>(groups) * topk, -1);
    std::vector<std::pair<float, int>> row;
    row.reserve(group_size);
    auto init_end = clock::now();
    profile.output_init_ms += elapsed_ms(init_start, init_end);

    for (int g = 0; g < groups; ++g) {
        auto index_start = clock::now();
        size_t input_base = static_cast<size_t>(g) * group_size;
        size_t output_base = static_cast<size_t>(g) * topk;
        auto index_end = clock::now();
        profile.group_index_ms += elapsed_ms(index_start, index_end);

        auto fill_start = clock::now();
        row.clear();
        for (int i = 0; i < group_size; ++i) {
            row.emplace_back(keys[input_base + i], values[input_base + i]);
        }
        auto fill_end = clock::now();
        profile.row_fill_ms += elapsed_ms(fill_start, fill_end);

        auto sort_start = clock::now();
        if (topk < group_size) {
            std::partial_sort(row.begin(), row.begin() + topk, row.end());
        } else {
            std::sort(row.begin(), row.end());
        }
        auto sort_end = clock::now();
        profile.topk_sort_ms += elapsed_ms(sort_start, sort_end);

        auto store_start = clock::now();
        for (int k = 0; k < topk; ++k) {
            out_keys[output_base + k] = row[k].first;
            out_values[output_base + k] = row[k].second;
        }
        auto store_end = clock::now();
        profile.output_store_ms += elapsed_ms(store_start, store_end);
    }

    auto total_end = clock::now();
    profile.total_ms = elapsed_ms(total_start, total_end);
    return profile;
}

// Compare a candidate top-k output against the CPU reference pairs.
bool validate_topk(
    const std::vector<float>& ref_keys,
    const std::vector<int>& ref_values,
    const std::vector<float>& got_keys,
    const std::vector<int>& got_values,
    int groups,
    int topk,
    float eps) {
    if (ref_keys.size() != got_keys.size() || ref_values.size() != got_values.size()) {
        return false;
    }
    for (int g = 0; g < groups; ++g) {
        for (int k = 0; k < topk; ++k) {
            size_t idx = static_cast<size_t>(g) * topk + k;
            if (std::fabs(ref_keys[idx] - got_keys[idx]) > eps || ref_values[idx] != got_values[idx]) {
                std::cerr << "Mismatch at group=" << g << " k=" << k
                          << " ref=(" << ref_keys[idx] << "," << ref_values[idx]
                          << ") got=(" << got_keys[idx] << "," << got_values[idx] << ")\n";
                return false;
            }
        }
    }
    return true;
}

// Print a small preview so result ordering can be inspected quickly.
void print_first_group(const std::vector<float>& keys, const std::vector<int>& values, int topk) {
    std::cout << "First group top-" << topk << ":";
    for (int k = 0; k < std::min(topk, 8); ++k) {
        std::cout << " (" << std::fixed << std::setprecision(5) << keys[k] << "," << values[k] << ")";
    }
    std::cout << "\n";
}

// Accumulate CPU phase timings across repeated runs.
static CpuPhaseProfile operator+(const CpuPhaseProfile& a, const CpuPhaseProfile& b) {
    return {
        a.total_ms + b.total_ms,
        a.output_init_ms + b.output_init_ms,
        a.group_index_ms + b.group_index_ms,
        a.row_fill_ms + b.row_fill_ms,
        a.topk_sort_ms + b.topk_sort_ms,
        a.output_store_ms + b.output_store_ms,
    };
}

// Convert accumulated phase timings into an average profile.
static CpuPhaseProfile scale_profile(const CpuPhaseProfile& p, double scale) {
    return {
        p.total_ms * scale,
        p.output_init_ms * scale,
        p.group_index_ms * scale,
        p.row_fill_ms * scale,
        p.topk_sort_ms * scale,
        p.output_store_ms * scale,
    };
}

// Report CPU timing by phase to show where baseline time is spent.
void print_cpu_profile(const CpuPhaseProfile& best, const CpuPhaseProfile& avg) {
    auto measured_sum = [](const CpuPhaseProfile& p) {
        return p.output_init_ms + p.group_index_ms + p.row_fill_ms + p.topk_sort_ms + p.output_store_ms;
    };
    auto print_row = [](const char* label, double best_ms, double avg_ms, double total_ms) {
        double pct = total_ms > 0.0 ? best_ms / total_ms * 100.0 : 0.0;
        std::cout << "  " << std::left << std::setw(24) << label
                  << std::right << std::fixed << std::setprecision(6)
                  << best_ms << " ms best, "
                  << avg_ms << " ms avg, "
                  << std::setprecision(2) << pct << "% of best total\n";
    };

    std::cout << "\nCPU phase profile\n";
    print_row("output initialization", best.output_init_ms, avg.output_init_ms, best.total_ms);
    print_row("group index compute", best.group_index_ms, avg.group_index_ms, best.total_ms);
    print_row("row fill", best.row_fill_ms, avg.row_fill_ms, best.total_ms);
    print_row("partial sort", best.topk_sort_ms, avg.topk_sort_ms, best.total_ms);
    print_row("output store", best.output_store_ms, avg.output_store_ms, best.total_ms);
    double best_unaccounted = best.total_ms - measured_sum(best);
    double avg_unaccounted = avg.total_ms - measured_sum(avg);
    print_row("loop/timer overhead", best_unaccounted, avg_unaccounted, best.total_ms);
    std::cout << "  " << std::left << std::setw(24) << "total"
              << std::right << std::fixed << std::setprecision(6)
              << best.total_ms << " ms best, "
              << avg.total_ms << " ms avg\n";
}

// Summarize the flat segmented workload shape before running benchmarks.
static void print_workload_layout(const Options& opt) {
    size_t candidates = static_cast<size_t>(opt.groups) * opt.group_size;
    size_t outputs = static_cast<size_t>(opt.groups) * opt.topk;
    std::cout << "Workload layout: " << candidates << " candidate pairs, "
              << outputs << " retained top-k pairs\n";
}

// Report the input and output buffer sizes implied by the workload.
static void print_buffer_footprint(const Options& opt) {
    size_t input_bytes = static_cast<size_t>(opt.groups) * opt.group_size * (sizeof(float) + sizeof(int));
    size_t output_bytes = static_cast<size_t>(opt.groups) * opt.topk * (sizeof(float) + sizeof(int));
    std::cout << "Buffer footprint: input=" << input_bytes
              << " bytes output=" << output_bytes << " bytes\n";
}

// Explain how many top-k pairs are checked against the reference output.
static void print_validation_scope(const Options& opt) {
    std::cout << "Validation scope: compare " << static_cast<size_t>(opt.groups) * opt.topk
              << " top-k distance/id pairs against CPU reference\n";
}

// Show the active top-k bounds used by the current executable.
static void print_topk_guard(const Options& opt) {
    std::cout << "Top-k guard: requested k=" << opt.topk
              << " within group_size=" << opt.group_size
              << " and max_k=" << GPU_SORT_MAX_TOPK << "\n";
}

// Drive input loading, reference generation, optional GPU paths, and reporting.
int run_gpu_sort_demo(int argc, char** argv) {
    try {
        Options opt = parse_options(argc, argv);
        std::cout << "GPU sorting benchmark: Micro-sort tuning parameters\n";
        std::cout << "groups=" << opt.groups << " group_size=" << opt.group_size
                  << " topk=" << opt.topk << " streams=" << opt.streams
                  << " repeats=" << opt.repeats << "\n";
        print_workload_layout(opt);
        print_buffer_footprint(opt);
        if (!opt.keys_bin_path.empty()) {
            std::cout << "OpenAI binary sort workload: " << opt.keys_bin_path << "\n";
        }

        auto keys = load_or_make_keys(opt);
        auto values = load_or_make_values(opt);

        std::vector<float> cpu_keys;
        std::vector<int> cpu_values;
        double cpu_ms = 0.0;
        if (opt.profile_cpu) {
            CpuPhaseProfile best_profile;
            CpuPhaseProfile profile_sum;
            bool first = true;
            for (int r = 0; r < opt.repeats; ++r) {
                std::vector<float> run_keys;
                std::vector<int> run_values;
                CpuPhaseProfile profile = cpu_segmented_topk_profiled(
                    keys, values, opt.groups, opt.group_size, opt.topk, run_keys, run_values);
                profile_sum = profile_sum + profile;
                if (first || profile.total_ms < best_profile.total_ms) {
                    best_profile = profile;
                    cpu_keys = std::move(run_keys);
                    cpu_values = std::move(run_values);
                    first = false;
                }
            }
            CpuPhaseProfile avg_profile = scale_profile(profile_sum, 1.0 / static_cast<double>(opt.repeats));
            cpu_ms = best_profile.total_ms;
            print_cpu_profile(best_profile, avg_profile);
        } else {
            double cpu_start = now_ms();
            cpu_segmented_topk(keys, values, opt.groups, opt.group_size, opt.topk, cpu_keys, cpu_values);
            cpu_ms = now_ms() - cpu_start;
        }
        std::vector<BenchResult> results;
        results.push_back({"cpu_partial_sort", cpu_ms, true});
        print_first_group(cpu_keys, cpu_values, opt.topk);
        print_validation_scope(opt);
        print_topk_guard(opt);

#if GPU_SORT_HAS_CUDA
        run_warmup_kernel(keys);
        std::cout << "GPU warmup completed.\n";
#else
        std::cout << "CUDA runtime was not available at build time; GPU sections skipped.\n";
#endif

#if GPU_SORT_HAS_CUDA
        results.push_back(run_gpu_insertion(opt, keys, values, cpu_keys, cpu_values));
#endif

        std::cout << "\nBenchmark summary\n";
        for (const auto& result : results) {
            std::cout << "  " << std::left << std::setw(32) << result.name
                      << std::right << std::fixed << std::setprecision(3)
                      << result.milliseconds << " ms"
                      << " valid=" << (result.valid ? "yes" : "no") << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
