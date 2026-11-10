---
title: GPU Top-k Sorting and Asynchronous Task Scheduling
layout: default
permalink: /vdb/web/gpu_sort/
---

# GPU-Accelerated Segmented Top-k Sorting and Asynchronous Task Scheduling

This project develops a **GPU-accelerated sorting module** for vector database candidate lists, complementing the clustering and distance-computation modules.
It selects the nearest `k` candidates from each independent group of precomputed distance/ID pairs and manages concurrent requests with different sizes.

---

## Features

- **Scalable Candidate-List Sorting**\
  Supports many small groups as well as groups containing large numbers of candidates.\
  Combines insertion, warp micro-sort, block-level bitonic, and hierarchical top-k implementations rather than relying on a single algorithm.

- **Workload-Aware Algorithm Selection**\
  Measures supported sorting implementations for the candidate count, group count, and top-k configuration.\
  Reuses prepared execution choices for repeated workloads to reduce selection and setup overhead.

- **Asynchronous Sorting Task Scheduler**\
  Distributes independent requests across CUDA streams using estimated sorting costs.\
  A bounded producer-facing queue manages batching, completion, error propagation, and backpressure.

- **Reusable Transfer-Compute Pipeline**\
  Uses pinned host memory and independent buffer/stream slots to overlap transfers and sorting across batches.\
  Prepared execution plans and optional CUDA Graph replay reduce recurring submission costs for fixed layouts.

---

## Implementation Details

### Segmented Input and Top-k Output

- Inputs contain finite `float32` distances and aligned `int32` candidate IDs, not the original embedding vectors.
- Each group is processed independently; outputs are ordered by increasing distance, with candidate ID as the tie-breaker.
- A uniform request contains `groups * group_size` input pairs and produces `groups * topk` output pairs.
- Multiple requests can use different group counts, candidate counts, and top-k values through request descriptors or a workload manifest.
- The default maximum top-k is 128; the effective top-k cannot exceed the number of candidates in its group.

### Sorting Algorithms

| Implementation | Processing Strategy |
|---|---|
| GPU insertion | Maintains a small ordered set of the nearest candidates while scanning each group; top-k-specialized variants reduce unnecessary local storage and work. |
| Warp micro-sort | Uses warp-level cooperation for small candidate groups, reducing the need for block-wide synchronization. |
| Block-level bitonic | Loads one group's candidates into shared memory and performs parallel compare-swap stages, padding incomplete power-of-two layouts. |
| CUB segmented radix | Fully sorts each group independently and extracts its first k entries; provides a large-group implementation and a comparison baseline. |
| Hierarchical top-k | Divides each group into chunks, retains each chunk's top-k using CUB block radix sorting, and repeatedly reduces the survivors to the final top-k. |

Hierarchical top-k is an exact selection algorithm, not a dispatcher between insertion, warp micro-sort, and bitonic.
A candidate discarded from a chunk already has at least k better candidates in that chunk and therefore cannot enter the original group's top-k.
The implementation also compares chunk sizes and thread configurations for repeated large-group workloads.

### Small-Request Batching and Memory Reuse

- Groups with similar supported sizes can share a kernel launch while retaining separate input ranges and outputs.
- Reusable request metadata, temporary storage, and execution plans reduce repeated allocation and preparation costs.
- Batching does not merge separate groups into a global top-k problem.

---

## Architectural Summary

- **Device-Resident Request Interface**\
  Accepts device pointers and group metadata so an upstream distance stage can supply candidates without an unnecessary host round trip.\
  The caller retains buffer ownership and must keep buffers valid until the request completes.

- **Bounded Asynchronous Request Queue**\
  A worker collects requests from producer threads and submits batches according to pending request count, accumulated candidates, and a maximum batching wait.\
  The outstanding-request limit includes queued and executing work; completion handles expose status and errors.

- **Cost-Balanced CUDA Stream Scheduling**\
  Estimates request work from its shape and sorting method, prioritizing larger work and assigning requests to streams with lower accumulated estimated load.\
  Scheduling operates at the stream and launch-configuration level, not by reserving physical GPU cores.

- **Independent Pipeline Slots**\
  Each slot owns its host/device buffers and stream so one batch's transfers can overlap another batch's execution.\
  Results are preserved before a slot is reused.

- **Prepared Batches and CUDA Graph Replay**\
  Fixed shapes and device addresses allow request preparation and scratch storage to be reused.\
  Optional graph capture records the transfer-sort-transfer sequence for repeated execution; changed layouts require rebuilding the prepared batch.

---

## Validation and Measurement

- A CPU `std::partial_sort` reference checks the selected distance/ID pairs and their ordering for the same input groups.
- Workloads include deterministic synthetic data, graph-shaped pruning/second-neighbor requests, and variable-length IVF lists prepared from OpenAI embeddings.
- Scale tests use up to 4,999,936 precomputed distance/ID pairs derived from the 1,536-dimensional OpenAI dataset.
- Boundary, multi-producer, and Compute Sanitizer tests check supported shapes, error handling, and memory access behavior.
- GPU event intervals, host-side pipeline totals, and repeated-batch throughput are reported separately; initial setup and warmup are excluded from steady-state pipeline measurements.

The module accelerates candidate sorting, not embedding-distance generation or an entire vector database index build.
Exact top-k agreement within the supplied candidates does not establish ANN search recall, and no single sorting or scheduling policy is assumed to win for every workload.

---

Implementation, build instructions, workload preparation, and validation scripts are available in the [GPU Top-k Sort module](https://github.com/cgskku/vdb/tree/main/src/gpu-topk-sort).

---
