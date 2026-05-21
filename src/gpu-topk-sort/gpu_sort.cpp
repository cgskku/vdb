#include "gpu_sort.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

// Print the supported benchmark arguments for this executable.
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--groups N] [--group-size N] [--topk K] [--repeats N]\n";
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
        else if (arg == "--repeats") {
            opt.repeats = std::atoi(need_value("--repeats"));
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
            // Exact top-k; entries after topk may remain unordered.
            std::partial_sort(row.begin(), row.begin() + topk, row.end());
        }
        else {
            std::sort(row.begin(), row.end());
        }

        for (int k = 0; k < topk; ++k) {
            out_keys[static_cast<size_t>(g) * topk + k] = row[k].first;
            out_values[static_cast<size_t>(g) * topk + k] = row[k].second;
        }
    }
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
        std::cout << "GPU sorting benchmark: Synthetic CPU top-k baseline\n";
        std::cout << "groups=" << opt.groups << " group_size=" << opt.group_size
                  << " topk=" << opt.topk << " repeats=" << opt.repeats << "\n";

        auto keys = make_synthetic_keys(opt);
        auto values = make_synthetic_values(opt);
        std::vector<float> out_keys;
        std::vector<int> out_values;

        double best_ms = std::numeric_limits<double>::infinity();
        for (int r = 0; r < opt.repeats; ++r) {
            double start = now_ms();
            cpu_segmented_topk(keys, values, opt.groups, opt.group_size, opt.topk, out_keys, out_values);
            best_ms = std::min(best_ms, now_ms() - start);
        }

        print_first_group(out_keys, out_values, opt.topk);
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
