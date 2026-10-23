#ifndef PIPELINE_H
#define PIPELINE_H

#include "gpu_sort.h"

struct PipelineMode {
    std::string name;
    bool pinned = true;
    int slots = 1;
    bool prepared = false;
    bool adaptive = false;
};

struct PipelineReport {
    double setup_ms = 0;
    double batch_mean_ms = 0;
    double batch_best_ms = 0;
    double latency_mean_ms = 0;
    double latency_p95_ms = 0;
    double h2d_event_ms = 0;
    double sort_event_ms = 0;
    double d2h_event_ms = 0;
    bool valid = true;
};

#if GPU_SORT_HAS_CUDA
// Bind caller-owned buffers to one stream; buffers must outlive the batch object.
class BoundTopkBatch {
public:
    BoundTopkBatch(const std::vector<DeviceTopkRequest>& requests, cudaStream_t stream,
                  const PipelineMode& mode);
    ~BoundTopkBatch();
    BoundTopkBatch(const BoundTopkBatch&) = delete;
    BoundTopkBatch& operator=(const BoundTopkBatch&) = delete;
    void enqueue();
    std::string description() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

PipelineReport benchmark_pipeline(const Options& opt,
    const std::vector<PackedSortWorkload>& inputs, const PipelineMode& mode);
#endif

int run_pipeline_experiments(const Options& opt, const PackedSortWorkload& workload);

#endif
