# Synthetic CPU Segmented Top-k Baseline

This project implements a **segmented top-k sorting module** for vector database candidate lists.
This implementation adds a configurable synthetic-data CPU segmented top-k baseline.

## Quickstart

Follow these steps to build and run this implementation.

### 1. Build the Top-k Sort Program

```bash
cd <path_to_repo>/src/gpu-topk-sort
make
```

### 2. Run with Synthetic Data

```bash
./gpu_sort --groups 64 --group-size 64 --topk 8 --repeats 2
```

This creates deterministic synthetic distance values and runs the segmented top-k baseline.

#### Arguments

The program treats the input as a set of independent candidate groups.
For each group, it sorts the candidate distance/id pairs and keeps only the nearest `topk` results.

| Argument | Description |
|---|---|
| `--groups <num_groups>` | Number of independent candidate groups. Each group is processed separately. |
| `--group-size <candidates_per_group>` | Number of candidate distance/id pairs in each group. |
| `--topk <topk>` | Number of nearest candidates to keep per group. |
| `--repeats <num_repeats>` | Number of benchmark repetitions. The best timing is used for the reported result. |

The program generates deterministic synthetic data for local benchmarking.
