#!/usr/bin/env python3
"""Build multi-query graph candidate-list sorting workloads from OpenAI embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train", default="train-00-of-10.parquet")
    parser.add_argument("--query", default="test.parquet")
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--candidate-pool", type=int, default=8192)
    parser.add_argument("--graph-nodes", type=int, default=2048)
    parser.add_argument("--graph-neighbors", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def list_array_to_matrix(array) -> np.ndarray:
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    lengths = offsets[1:] - offsets[:-1]
    if len(lengths) == 0:
        return np.empty((0, 0), dtype=np.float32)
    if not np.all(lengths == lengths[0]):
        raise RuntimeError("embedding column contains variable-length rows")
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(lengths), int(lengths[0]))


def read_queries(path: Path, count: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    remaining = count
    for batch in pq.ParquetFile(path).iter_batches(batch_size=min(4096, count), columns=["emb"]):
        matrix = list_array_to_matrix(batch.column(0))
        take = min(remaining, len(matrix))
        parts.append(matrix[:take])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(f"query file contains fewer than {count} rows")
    return np.concatenate(parts)


def read_candidates(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    id_parts: list[np.ndarray] = []
    embedding_parts: list[np.ndarray] = []
    remaining = count
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["id", "emb"]):
        ids = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int32)
        embeddings = list_array_to_matrix(batch.column(1))
        take = min(remaining, len(ids))
        id_parts.append(ids[:take])
        embedding_parts.append(embeddings[:take])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(f"training shard contains fewer than {count} rows")
    return np.concatenate(id_parts), np.concatenate(embedding_parts)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def write_request(
    output_dir: Path,
    name: str,
    distances: np.ndarray,
    candidate_ids: np.ndarray,
    group_size: int,
    topk: int,
    rng: np.random.Generator,
) -> str:
    groups = len(distances)
    request_keys = np.empty((groups, group_size), dtype=np.float32)
    request_values = np.empty((groups, group_size), dtype=np.int32)
    for group in range(groups):
        nearest = np.argpartition(distances[group], group_size - 1)[:group_size]
        rng.shuffle(nearest)
        request_keys[group] = distances[group, nearest]
        request_values[group] = candidate_ids[nearest]
    keys_name = f"{name}_keys.f32"
    values_name = f"{name}_values.i32"
    request_keys.tofile(output_dir / keys_name)
    request_values.tofile(output_dir / values_name)
    return f"{name} {keys_name} {values_name} {groups} {group_size} {topk}"


def write_request_batches(
    output_dir: Path,
    name: str,
    distances: np.ndarray,
    candidate_ids: np.ndarray,
    group_size: int,
    topk: int,
    batch_count: int,
    rng: np.random.Generator,
) -> list[str]:
    batches = np.array_split(distances, min(batch_count, len(distances)))
    return [
        write_request(
            output_dir,
            f"{name}_{index:02d}",
            batch,
            candidate_ids,
            group_size,
            topk,
            rng,
        )
        for index, batch in enumerate(batches)
        if len(batch)
    ]


def write_precomputed_batches(
    output_dir: Path,
    name: str,
    keys: np.ndarray,
    values: np.ndarray,
    topk: int,
    batch_count: int,
) -> list[str]:
    lines: list[str] = []
    key_batches = np.array_split(keys, min(batch_count, len(keys)))
    value_batches = np.array_split(values, min(batch_count, len(values)))
    for index, (key_batch, value_batch) in enumerate(zip(key_batches, value_batches)):
        if not len(key_batch):
            continue
        batch_name = f"{name}_{index:02d}"
        keys_name = f"{batch_name}_keys.f32"
        values_name = f"{batch_name}_values.i32"
        np.asarray(key_batch, dtype=np.float32).tofile(output_dir / keys_name)
        np.asarray(value_batch, dtype=np.int32).tofile(output_dir / values_name)
        lines.append(
            f"{batch_name} {keys_name} {values_name} "
            f"{len(key_batch)} {key_batch.shape[1]} {topk}"
        )
    return lines


def build_second_neighbor_rows(
    embeddings: np.ndarray,
    candidate_ids: np.ndarray,
    anchor_count: int,
    graph_neighbors: int,
    group_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if graph_neighbors >= len(embeddings):
        raise RuntimeError("graph-neighbors must be smaller than graph-nodes")
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)
    adjacency = np.argpartition(
        -similarities, graph_neighbors - 1, axis=1
    )[:, :graph_neighbors]

    output_keys = np.empty((anchor_count, group_size), dtype=np.float32)
    output_values = np.empty((anchor_count, group_size), dtype=np.int32)
    all_nodes = np.arange(len(embeddings), dtype=np.int32)
    for row, anchor in enumerate(range(anchor_count)):
        second_neighbors = np.unique(adjacency[adjacency[anchor]].reshape(-1))
        second_neighbors = second_neighbors[second_neighbors != anchor]
        if len(second_neighbors) < group_size:
            supplement = np.setdiff1d(all_nodes, second_neighbors, assume_unique=False)
            supplement = supplement[supplement != anchor]
            second_neighbors = np.concatenate(
                [second_neighbors, supplement[: group_size - len(second_neighbors)]]
            )
        distances = 1.0 - embeddings[second_neighbors] @ embeddings[anchor]
        selected = np.argpartition(distances, group_size - 1)[:group_size]
        selected_nodes = second_neighbors[selected]
        selected_distances = distances[selected].astype(np.float32)
        order = np.arange(group_size)
        rng.shuffle(order)
        output_keys[row] = selected_distances[order]
        output_values[row] = candidate_ids[selected_nodes[order]]
    return output_keys, output_values


def main() -> int:
    args = parse_args()
    if args.query_count <= 0 or args.candidate_pool < args.graph_nodes:
        raise RuntimeError("query-count must be positive and candidate-pool must cover graph-nodes")
    if args.graph_nodes < 1537:
        raise RuntimeError("graph-nodes must be at least 1537 for the large second-neighbor request")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = normalize_rows(read_queries(data_root / args.query, args.query_count))
    candidate_ids, candidates = read_candidates(data_root / args.train, args.candidate_pool)
    candidates = normalize_rows(candidates)
    distances = (1.0 - queries @ candidates.T).astype(np.float32)
    rng = np.random.default_rng(args.seed)

    lines: list[str] = []
    lines.extend(write_request_batches(
        output_dir, "pruning_micro", distances, candidate_ids, 64, 16, 4, rng))
    lines.extend(write_request_batches(
        output_dir, "pruning_regular", distances, candidate_ids, 128, 32, 4, rng))
    second_keys, second_values = build_second_neighbor_rows(
        candidates[: args.graph_nodes],
        candidate_ids[: args.graph_nodes],
        min(16, args.query_count),
        args.graph_neighbors,
        1536,
        rng,
    )
    lines.extend(write_precomputed_batches(
        output_dir,
        "nn_descent_second_neighbor",
        second_keys,
        second_values,
        64,
        4,
    ))
    manifest = output_dir / "openai_graph_requests.txt"
    manifest.write_text("\n".join(lines) + "\n")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
