#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

static void check_impl(cudaError_t error, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(cudaGetErrorString(error)) + " at pipeline.cu:" + std::to_string(line));
    }
}

#define check(call) check_impl((call), __LINE__)

template<class T>
struct Buffer {
    T* data = nullptr;
    int kind;
    Buffer(size_t count, int allocation_kind) : kind(allocation_kind) {
        if (kind == 2) {
            check(cudaMalloc(&data, count * sizeof(T)));
        } else if (kind == 1) {
            check(cudaMallocHost(&data, count * sizeof(T)));
        } else {
            data = new T[count];
        }
    }
    ~Buffer() {
        if (kind == 2) {
            cudaFree(data);
        } else if (kind == 1) {
            cudaFreeHost(data);
        } else {
            delete[] data;
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
    cudaStream_t value;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};

struct Event {
    cudaEvent_t value;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};

struct BatchEvents {
    Event start, transferred, sorted, done;
    double submitted = 0;
    double observed = 0;
    double h2d_ms = 0, sort_ms = 0, d2h_ms = 0;
};

struct Graph {
    cudaGraph_t definition = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) {
            cudaGraphExecDestroy(executable);
        }
        if (definition) {
            cudaGraphDestroy(definition);
        }
    }
};

struct Slot {
    Stream stream;
    Buffer<float> host_keys, host_output_keys, device_keys, device_output_keys;
    Buffer<int> host_values, host_output_values, device_values, device_output_values;
    std::unique_ptr<BoundTopkBatch> batch;
    Event graph_start, graph_transferred, graph_sorted, graph_done;
    Graph graph;
    int active = -1;

    Slot(const PackedSortWorkload& input, const PipelineMode& mode)
        : host_keys(input.keys.size(), mode.pinned),
          host_output_keys(input.reference_keys.size(), mode.pinned),
          device_keys(input.keys.size(), 2), device_output_keys(input.reference_keys.size(), 2),
          host_values(input.values.size(), mode.pinned),
          host_output_values(input.reference_values.size(), mode.pinned),
          device_values(input.values.size(), 2), device_output_values(input.reference_values.size(), 2) {
        check(cudaMemcpy(device_keys.data, input.keys.data(), input.keys.size()*sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_values.data, input.values.data(), input.values.size()*sizeof(int), cudaMemcpyHostToDevice));
        std::vector<DeviceTopkRequest> requests;
        for (const auto& request : input.requests) {
            requests.push_back({device_keys.data + request.input_offset,
                device_values.data + request.input_offset,
                device_output_keys.data + request.output_offset,
                device_output_values.data + request.output_offset,
                request.groups, request.group_size, request.topk});
        }
        batch = std::make_unique<BoundTopkBatch>(requests, stream.value, mode);
    }
    ~Slot() { cudaStreamSynchronize(stream.value); }

    void capture(size_t count, size_t outputs) {
        check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
        check(cudaEventRecordWithFlags(graph_start.value, stream.value, cudaEventRecordExternal));
        check(cudaMemcpyAsync(device_keys.data, host_keys.data, count*sizeof(float), cudaMemcpyHostToDevice, stream.value));
        check(cudaMemcpyAsync(device_values.data, host_values.data, count*sizeof(int), cudaMemcpyHostToDevice, stream.value));
        check(cudaEventRecordWithFlags(graph_transferred.value, stream.value, cudaEventRecordExternal));
        batch->enqueue();
        check(cudaEventRecordWithFlags(graph_sorted.value, stream.value, cudaEventRecordExternal));
        check(cudaMemcpyAsync(host_output_keys.data, device_output_keys.data, outputs*sizeof(float), cudaMemcpyDeviceToHost, stream.value));
        check(cudaMemcpyAsync(host_output_values.data, device_output_values.data, outputs*sizeof(int), cudaMemcpyDeviceToHost, stream.value));
        check(cudaEventRecordWithFlags(graph_done.value, stream.value, cudaEventRecordExternal));
        check(cudaStreamEndCapture(stream.value, &graph.definition));
        check(cudaGraphInstantiate(&graph.executable, graph.definition, 0));
    }
};

static double elapsed(const Event& begin, const Event& end) {
    float ms;
    check(cudaEventElapsedTime(&ms, begin.value, end.value));
    return ms;
}

PipelineReport benchmark_pipeline(const Options& opt,
    const std::vector<PackedSortWorkload>& inputs, const PipelineMode& mode) {
    PipelineReport report;
    double setup = now_ms();
    const size_t count = inputs.front().keys.size();
    const size_t outputs = inputs.front().reference_keys.size();
    std::vector<std::unique_ptr<Slot>> slots;
    for (int index = 0; index < mode.slots; ++index) {
        slots.push_back(std::make_unique<Slot>(inputs.front(), mode));
        slots.back()->batch->enqueue();
        check(cudaStreamSynchronize(slots.back()->stream.value));
        if (mode.graph) {
            slots.back()->capture(count, outputs);
        }
    }
    std::cout << "PLAN mode=" << mode.name << " " << slots.front()->batch->description() << "\n";
    std::vector<std::unique_ptr<BatchEvents>> events;
    for (int index = 0; index < opt.batches; ++index) {
        events.push_back(std::make_unique<BatchEvents>());
    }
    std::vector<std::vector<float>> saved_keys(opt.batches, std::vector<float>(outputs));
    std::vector<std::vector<int>> saved_values(opt.batches, std::vector<int>(outputs));
    report.setup_ms = now_ms() - setup;
    report.batch_best_ms = std::numeric_limits<double>::infinity();
    std::vector<double> latencies;

    for (int repeat = -1; repeat < opt.repeats; ++repeat) {
        auto finish = [&](Slot& slot) {
            if (slot.active < 0) {
                return;
            }
            auto& event = *events[slot.active];
            check(cudaEventSynchronize(event.done.value));
            event.observed = now_ms();
            event.h2d_ms = mode.graph ? elapsed(slot.graph_start, slot.graph_transferred)
                                      : elapsed(event.start, event.transferred);
            event.sort_ms = mode.graph ? elapsed(slot.graph_transferred, slot.graph_sorted)
                                       : elapsed(event.transferred, event.sorted);
            event.d2h_ms = mode.graph ? elapsed(slot.graph_sorted, slot.graph_done)
                                      : elapsed(event.sorted, event.done);
            // Preserve every batch output before the ring slot is reused.
            std::memcpy(saved_keys[slot.active].data(), slot.host_output_keys.data, outputs*sizeof(float));
            std::memcpy(saved_values[slot.active].data(), slot.host_output_values.data, outputs*sizeof(int));
            slot.active = -1;
        };
        double start = now_ms();
        for (int index = 0; index < opt.batches; ++index) {
            auto& slot = *slots[index % slots.size()];
            finish(slot);
            const auto& input = inputs[(index + repeat + 1) % inputs.size()];
            auto& event = *events[index];
            event.submitted = now_ms();
            std::memcpy(slot.host_keys.data, input.keys.data(), count*sizeof(float));
            std::memcpy(slot.host_values.data, input.values.data(), count*sizeof(int));
            if (mode.graph) {
                check(cudaGraphLaunch(slot.graph.executable, slot.stream.value));
            } else {
                check(cudaEventRecord(event.start.value, slot.stream.value));
                check(cudaMemcpyAsync(slot.device_keys.data, slot.host_keys.data, count*sizeof(float), cudaMemcpyHostToDevice, slot.stream.value));
                check(cudaMemcpyAsync(slot.device_values.data, slot.host_values.data, count*sizeof(int), cudaMemcpyHostToDevice, slot.stream.value));
                check(cudaEventRecord(event.transferred.value, slot.stream.value));
                slot.batch->enqueue();
                check(cudaEventRecord(event.sorted.value, slot.stream.value));
                check(cudaMemcpyAsync(slot.host_output_keys.data, slot.device_output_keys.data, outputs*sizeof(float), cudaMemcpyDeviceToHost, slot.stream.value));
                check(cudaMemcpyAsync(slot.host_output_values.data, slot.device_output_values.data, outputs*sizeof(int), cudaMemcpyDeviceToHost, slot.stream.value));
            }
            check(cudaEventRecord(event.done.value, slot.stream.value));
            slot.active = index;
        }
        for (auto& slot : slots) {
            finish(*slot);
        }
        double total_ms = now_ms() - start;
        for (int index = 0; index < opt.batches; ++index) {
            const auto& reference = inputs[(index + repeat + 1) % inputs.size()];
            report.valid = validate_topk(reference.reference_keys, reference.reference_values,
                saved_keys[index], saved_values[index], 1, static_cast<int>(outputs)) && report.valid;
            if (repeat >= 0) {
                auto& event = *events[index];
                latencies.push_back(event.observed - event.submitted);
                report.h2d_event_ms += event.h2d_ms;
                report.sort_event_ms += event.sort_ms;
                report.d2h_event_ms += event.d2h_ms;
            }
        }
        if (repeat >= 0) {
            double per_batch = total_ms / opt.batches;
            report.batch_mean_ms += per_batch / opt.repeats;
            report.batch_best_ms = std::min(report.batch_best_ms, per_batch);
        }
    }
    const double samples = static_cast<double>(latencies.size());
    for (double value : latencies) {
        report.latency_mean_ms += value / samples;
    }
    std::sort(latencies.begin(), latencies.end());
    report.latency_p95_ms = latencies[static_cast<size_t>(std::ceil(samples * .95)) - 1];
    report.h2d_event_ms /= samples;
    report.sort_event_ms /= samples;
    report.d2h_event_ms /= samples;
    return report;
}
