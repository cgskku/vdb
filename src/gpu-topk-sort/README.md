# GPU Top-k Sort for Vector Candidate Lists

This project implements a **segmented top-k sorting module** for vector database candidate lists.
This implementation adds binary workload loading for precomputed distance and candidate-id arrays.

## Quickstart

Follow these steps to build and run this implementation.

### 1. Build the Top-k Sort Program

```bash
cd <path_to_repo>/src/gpu-topk-sort
make
```

### 2. Run with Synthetic Data

```bash
./gpu_sort --groups 512 --group-size 128 --topk 16 --repeats 3
```

This creates deterministic synthetic distance values and runs the segmented top-k baseline.

### 3. Run with the OpenAI Benchmark Workload

The staged OpenAI workload uses a compact binary input generated from the downloaded parquet dataset.

```bash
./gpu_sort \
  --groups <num_groups> \
  --group-size <candidates_per_group> \
  --topk <topk> \
  --repeats <num_repeats> \
  --streams <num_streams> \
  --keys-bin <path_to_float_distance_keys> \
  --values-bin <path_to_int_candidate_ids>
```

To print a detailed CPU phase breakdown:

```bash
./gpu_sort \
  --groups <num_groups> \
  --group-size <candidates_per_group> \
  --topk <topk> \
  --repeats <num_repeats> \
  --streams <num_streams> \
  --keys-bin <path_to_float_distance_keys> \
  --values-bin <path_to_int_candidate_ids> \
  --profile-cpu
```

#### Arguments

The program treats the input as a set of independent candidate groups.
For each group, it sorts the candidate distance/id pairs and keeps only the nearest `topk` results.

| Argument | Description |
|---|---|
| `--groups <num_groups>` | Number of independent candidate groups. Each group is processed separately. |
| `--group-size <candidates_per_group>` | Number of candidate distance/id pairs in each group. |
| `--topk <topk>` | Number of nearest candidates to keep per group. |
| `--repeats <num_repeats>` | Number of benchmark repetitions. The best timing is used for the reported result. |
| `--streams <num_streams>` | Reserved stream-count option kept for compatibility with later GPU implementations. |
| `--keys-bin <path_to_float_distance_keys>` | Path to a binary `float32` file containing precomputed distance keys. The file must contain `groups * group_size` values. |
| `--values-bin <path_to_int_candidate_ids>` | Path to a binary `int32` file containing candidate vector IDs aligned with `keys-bin`. The file must contain `groups * group_size` values. |
| `--profile-cpu` | Prints a phase-level CPU timing breakdown for the selected CPU path. |

If `--keys-bin` and `--values-bin` are omitted, the program generates deterministic synthetic data.
