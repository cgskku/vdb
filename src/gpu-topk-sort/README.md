# GPU Top-k Sort for Vector Candidate Lists

This project implements a **segmented top-k sorting module** for vector database candidate lists.
It is designed to select the nearest `k` candidates from each independent group of precomputed distance values.

## Quickstart

Follow these steps to build and run the current implementation.

### 1. Build the Top-k Sort Program

```bash
make
```

### 2. Run with Synthetic Data

```bash
./gpu_sort
```

This creates a small deterministic synthetic distance/id workload and runs the segmented top-k baseline.

## Input Layout

The current implementation uses a fixed-size synthetic workload generated inside the program.

```text
groups = 8
group_size = 16
topk = 4
```

The input is treated as independent candidate groups:

```text
group 0: 16 candidate distance/id pairs
group 1: 16 candidate distance/id pairs
...
group 7: 16 candidate distance/id pairs
```

For each group, the program keeps only the nearest `topk` candidates.

```text
total candidates = 8 * 16 = 128
output results = 8 * 4 = 32
```

## Implementation

The program generates two arrays:

| Array | Description |
|---|---|
| `keys` | Synthetic `float` distance values. Smaller values are treated as nearer candidates. |
| `values` | Integer candidate IDs aligned with the distance values. |

Each group is copied into a temporary row of `(distance, candidate_id)` pairs.
The row is processed with `std::partial_sort`, which exactly sorts only the nearest `topk` entries.
The remaining entries do not need to be fully sorted.

## Output

The program prints:

- the fixed workload shape
- the nearest candidates from the first group
- the elapsed CPU time for the segmented top-k baseline

Example output format:

```text
Segmented top-k CPU reference scaffold
groups=8 group_size=16 topk=4
First group top-4: (...)
cpu_partial_sort <elapsed_ms> ms
```
