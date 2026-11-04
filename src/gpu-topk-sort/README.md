# GPU Top-k Sort for Vector Candidate Lists

This project implements a **segmented top-k sorting module** for vector database candidate lists.
It selects the nearest `k` candidates from each independent group of precomputed distance values.
Distance ties are ordered by candidate ID. Inputs are expected to contain finite float32 distances and int32 IDs.

## Quickstart

Follow these steps to build and run the current implementation.

### 1. Build the Top-k Sort Program

```bash
cd <path_to_repo>/src/gpu-topk-sort
make
```

The Makefile defaults to CUDA architecture `sm_86`. Override `NVCC_ARCH` for a different GPU.

### 2. Run with Synthetic Data

```bash
./gpu_sort --groups 1024 --group-size 128 --topk 32 --repeats 3 --profile-cpu
```

This creates deterministic synthetic distance values and validates GPU results against the CPU reference.

### 3. Run with the OpenAI Benchmark Workload

The program reads precomputed distance/ID binary arrays, not parquet files directly.

```bash
./gpu_sort --groups <num_groups> --group-size <candidates_per_group> --topk <topk> --repeats <num_repeats> --streams <num_streams> --keys-bin <path_to_float_distance_keys> --values-bin <path_to_int_candidate_ids> --profile-cpu
```

### 4. Run with Variable-Size Requests

```bash
./gpu_sort --heterogeneous --requests 20 --streams 4 --repeats 5
```

```bash
./gpu_sort --request-manifest <path_to_request_manifest> --streams 4 --repeats 5
```

Each manifest row contains `name keys_file values_file groups group_size topk`. Relative file paths are resolved from the manifest directory.

Prepare inverted-list requests from downloaded OpenAI embeddings using NumPy and PyArrow:

```bash
python prepare_ivf_sort_workload.py --data-root <path_to_openai_dataset> --output-dir <path_to_workload_directory> --vectors 16384 --clusters 64 --iterations 8 --topk 32 --stress-streams 4
```

#### Arguments

| Argument | Description |
|---|---|
| `--groups <num_groups>` | Number of independent groups. |
| `--group-size <candidates_per_group>` | Candidates per group. |
| `--topk <topk>` | Retained candidates per group, clamped to group size and limited to 128. |
| `--repeats <num_repeats>` | Repetitions; benchmark summaries report minima, not averages. |
| `--streams <num_streams>` | CUDA streams for independent requests. |
| `--keys-bin <path_to_float_distance_keys>` | Binary float32 distances, exactly groups * group_size values. |
| `--values-bin <path_to_int_candidate_ids>` | Aligned binary int32 IDs. |
| `--profile-cpu` | Separately measure CPU initialization, indexing, row fill, partial sort and result storage. |
| `--csv <path_to_output_csv>` | Append legacy benchmark summary rows. Detailed event/pipeline metrics are printed to stdout. |
| `--heterogeneous` | Generate variable-size synthetic requests. |
| `--requests <num_requests>` | Number of generated variable-size requests. |
| `--request-manifest <path_to_request_manifest>` | Read variable-size requests from a manifest. |
| `--queue-depth <max_outstanding>` | Backpressure limit including pending and executing requests. |
| `--queue-wait-us <microseconds>` | Maximum batching wait (default 20); work/depth thresholds may trigger earlier submission. |
| `--queue-batch-pairs <candidate_count>` | Submit early when queued candidates reach this threshold (default 65536). |
| `--producers <num_producers>` | Benchmark producer threads (default 1). |

Without binary files, deterministic synthetic data is generated. `*_kernel_only` is the CUDA-event device interval; scheduler times include host submission and synchronization. Pipeline total includes H2D, execution and D2H but excludes teardown, file I/O and distance generation. The reusable pipeline reports one-time setup/warmup separately.


## Repeated Batch Pipeline

Build includes `pipeline.cpp`, `pipeline.cu`, and `pipeline.h` alongside the sorting sources.

```bash
./gpu_sort --pipeline-bench --request-manifest <path_to_request_manifest> --batches 16 --repeats 5 --pipeline-slots 2
./gpu_sort --pipeline-bench --groups 62496 --group-size 8 --topk 1 --batches 16 --repeats 5
```

| Argument | Description |
|---|---|
| `--pipeline-bench` | Compare available repeated-batch modes instead of the legacy benchmark. |
| `--batches <num_batches>` | Batches per measurement window, 2..1024 (default 16). |
| `--pipeline-slots <num_slots>` | Independent buffer/stream slots for overlap modes, 1..16 (default 2). |
| `--pipeline-mode <name>` | Run only the specified mode; omitted means all modes. |
| `--pipeline-order <offset>` | Rotate mode order without changing the input (default 0). |

Available modes: `pageable_serial`, `pinned_serial`, `pinned_overlap`, `prepared_serial`, `prepared_overlap`, `adaptive_serial`, `adaptive_overlap`, `graph_serial`, `graph_overlap`, `adaptive_graph_overlap`, `hierarchy_512`, `hierarchy_1024_t128`, `hierarchy_1024`, `hierarchy_2048`, `chunk_tuned`, `chunk_tuned_graph`.

The CPU reference is computed for the original input and a version with distances reversed within each group. IDs remain unchanged. Every warmup and measured batch is checked after the timed interval.

`batch_mean_ms` is total processing time divided by batch count, averaged across repetitions. It includes CPU staging and preserving each output before slot reuse. It is not individual-batch latency. `latency_p95_ms` is host-observed completion latency from staging start, excluding the wait for a free slot; it may exceed physical GPU completion time. Phase events represent device timeline intervals, not pure kernel busy time. Setup, an additional warmup pass, reference generation/validation and teardown are excluded from steady-state timing.

`BoundTopkBatch` binds caller-owned device pointers and shapes to one CUDA stream. Keep all buffers alive until execution completes. Prepared modes require fixed pointers and shapes; construct a new batch object if the layout changes.
