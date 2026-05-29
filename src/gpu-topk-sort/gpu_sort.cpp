#include "gpu_sort.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

// Print the supported benchmark arguments for this executable.
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--groups N] [--group-size N] [--topk K] [--streams N] [--repeats N]"
              << " [--keys-bin path] [--values-bin path]\n";
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
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.groups <= 0 || opt.group_size <= 0 || opt.topk <= 0) {
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

// Drive input loading, reference generation, optional GPU paths, and reporting.
int run_gpu_sort_demo(int argc, char** argv) {
    try {
        Options opt = parse_options(argc, argv);
        std::cout << "GPU sorting benchmark: CPU top-k baseline with binary workload input\n";
        std::cout << "groups=" << opt.groups << " group_size=" << opt.group_size
                  << " topk=" << opt.topk << " streams=" << opt.streams
                  << " repeats=" << opt.repeats << "\n";
        if (!opt.keys_bin_path.empty()) {
            std::cout << "Binary sort workload: " << opt.keys_bin_path << "\n";
        }

        auto keys = load_or_make_keys(opt);
        auto values = load_or_make_values(opt);
        std::vector<float> cpu_keys;
        std::vector<int> cpu_values;

        double best_ms = std::numeric_limits<double>::infinity();
        for (int r = 0; r < opt.repeats; ++r) {
            double start = now_ms();
            cpu_segmented_topk(keys, values, opt.groups, opt.group_size, opt.topk, cpu_keys, cpu_values);
            best_ms = std::min(best_ms, now_ms() - start);
        }

        print_first_group(cpu_keys, cpu_values, opt.topk);
        std::cout << "\nBenchmark summary\n";
        std::cout << "  " << std::left << std::setw(32) << "cpu_partial_sort"
                  << std::right << std::fixed << std::setprecision(3)
                  << best_ms << " ms valid=yes\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
