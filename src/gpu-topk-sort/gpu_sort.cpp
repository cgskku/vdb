#include "gpu_sort.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr int kGroups = 8;
constexpr int kGroupSize = 16;
constexpr int kTopK = 4;

double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

std::vector<float> make_keys() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> keys(kGroups * kGroupSize);
    for (float& key : keys) {
        key = dist(rng);
    }
    return keys;
}

std::vector<int> make_values() {
    std::vector<int> values(kGroups * kGroupSize);
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        values[i] = i;
    }
    return values;
}

void cpu_segmented_topk(
    const std::vector<float>& keys,
    const std::vector<int>& values,
    std::vector<float>& out_keys,
    std::vector<int>& out_values) {
    out_keys.assign(kGroups * kTopK, std::numeric_limits<float>::infinity());
    out_values.assign(kGroups * kTopK, -1);

    std::vector<std::pair<float, int>> row;
    row.reserve(kGroupSize);
    for (int group = 0; group < kGroups; ++group) {
        row.clear();
        int input_base = group * kGroupSize;
        int output_base = group * kTopK;

        for (int i = 0; i < kGroupSize; ++i) {
            row.emplace_back(keys[input_base + i], values[input_base + i]);
        }

        // Exact top-k; entries after kTopK may remain unordered.
        std::partial_sort(row.begin(), row.begin() + kTopK, row.end());

        for (int k = 0; k < kTopK; ++k) {
            out_keys[output_base + k] = row[k].first;
            out_values[output_base + k] = row[k].second;
        }
    }
}

void print_first_group(const std::vector<float>& keys, const std::vector<int>& values) {
    std::cout << "First group top-" << kTopK << ":";
    for (int k = 0; k < kTopK; ++k) {
        std::cout << " (" << std::fixed << std::setprecision(5)
                  << keys[k] << "," << values[k] << ")";
    }
    std::cout << "\n";
}

}  // namespace

int run_gpu_sort_demo() {
    std::cout << "Segmented top-k CPU reference baseline\n";
    std::cout << "groups=" << kGroups
              << " group_size=" << kGroupSize
              << " topk=" << kTopK << "\n";

    auto keys = make_keys();
    auto values = make_values();

    std::vector<float> out_keys;
    std::vector<int> out_values;
    double start = now_ms();
    cpu_segmented_topk(keys, values, out_keys, out_values);
    double elapsed = now_ms() - start;

    print_first_group(out_keys, out_values);
    std::cout << "cpu_partial_sort " << std::fixed << std::setprecision(6)
              << elapsed << " ms\n";
    return 0;
}
