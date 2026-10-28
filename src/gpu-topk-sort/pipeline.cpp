#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Reverse distance order but retain IDs, so consecutive batches need different outputs.
static PackedSortWorkload alternate_input(const PackedSortWorkload& input) {
    auto result = input;
    for (const auto& request : input.requests) {
        for (int group = 0; group < request.groups; ++group) {
            size_t offset = request.input_offset + static_cast<size_t>(group) * request.group_size;
            std::reverse(result.keys.begin() + offset,
                         result.keys.begin() + offset + request.group_size);
        }
        size_t count = static_cast<size_t>(request.groups) * request.group_size;
        std::vector<float> keys(result.keys.begin() + request.input_offset,
                                result.keys.begin() + request.input_offset + count);
        std::vector<int> values(result.values.begin() + request.input_offset,
                               result.values.begin() + request.input_offset + count);
        std::vector<float> output_keys;
        std::vector<int> output_values;
        cpu_segmented_topk(keys, values, request.groups, request.group_size,
                           request.topk, output_keys, output_values);
        std::copy(output_keys.begin(), output_keys.end(),
                  result.reference_keys.begin() + request.output_offset);
        std::copy(output_values.begin(), output_values.end(),
                  result.reference_values.begin() + request.output_offset);
    }
    return result;
}

int run_pipeline_experiments(const Options& opt, const PackedSortWorkload& workload) {
    if (workload.keys.empty() || workload.requests.empty()) {
        throw std::invalid_argument("pipeline requires nonempty input");
    }
    for (float key : workload.keys) {
        if (!std::isfinite(key)) {
            throw std::invalid_argument("pipeline distances must be finite");
        }
    }
    std::vector<PackedSortWorkload> inputs{workload, alternate_input(workload)};
    std::vector<PipelineMode> modes{
        {"pageable_serial", false, 1},
        {"pinned_serial", true, 1},
        {"pinned_overlap", true, opt.pipeline_slots},
        {"prepared_serial", true, 1, true},
        {"prepared_overlap", true, opt.pipeline_slots, true},
        {"adaptive_serial", true, 1, true, true},
        {"adaptive_overlap", true, opt.pipeline_slots, true, true},
        {"graph_serial", true, 1, true, false, true},
        {"graph_overlap", true, opt.pipeline_slots, true, false, true},
        {"adaptive_graph_overlap", true, opt.pipeline_slots, true, true, true}
    };
    std::rotate(modes.begin(), modes.begin() + opt.pipeline_order % modes.size(), modes.end());
    bool valid = true;
    std::cout << "Pipeline input: pairs=" << workload.keys.size()
              << " output_pairs=" << workload.reference_keys.size()
              << " requests=" << workload.requests.size() << " batches=" << opt.batches
              << " repeats=" << opt.repeats << " variants=2\n";
#if GPU_SORT_HAS_CUDA
    run_warmup_kernel(workload.keys);
    int executed = 0;
    for (const auto& mode : modes) {
        if (!opt.pipeline_mode.empty() && opt.pipeline_mode != mode.name) {
            continue;
        }
        ++executed;
        auto report = benchmark_pipeline(opt, inputs, mode);
        valid = report.valid && valid;
        std::cout << std::fixed << std::setprecision(6)
                  << "PIPELINE mode=" << mode.name
                  << " setup_ms=" << report.setup_ms
                  << " batch_mean_ms=" << report.batch_mean_ms
                  << " batch_best_ms=" << report.batch_best_ms
                  << " latency_mean_ms=" << report.latency_mean_ms
                  << " latency_p95_ms=" << report.latency_p95_ms
                  << " h2d_event_ms=" << report.h2d_event_ms
                  << " sort_event_ms=" << report.sort_event_ms
                  << " d2h_event_ms=" << report.d2h_event_ms
                  << " valid=" << (report.valid ? "yes" : "no") << "\n";
    }
    if (!executed) {
        throw std::invalid_argument("unknown pipeline mode: " + opt.pipeline_mode);
    }
#else
    throw std::runtime_error("pipeline experiments require CUDA");
#endif
    return valid ? 0 : 1;
}
