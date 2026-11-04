#!/usr/bin/env python3
"""Build variable-length IVF sorting requests from OpenAI embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train", default="train-00-of-10.parquet")
    parser.add_argument("--vectors", type=int, default=16384)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--assignment-batch", type=int, default=2048)
    parser.add_argument("--stress-streams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def list_array_to_matrix(array) -> np.ndarray:
    if hasattr(array, "offsets"):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        lengths = offsets[1:] - offsets[:-1]
        if len(lengths) == 0:
            return np.empty((0, 0), dtype=np.float32)
        if not np.all(lengths == lengths[0]):
            raise RuntimeError("embedding column contains variable-length rows")
        values = array.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(len(lengths), int(lengths[0]))

    width = array.type.list_size
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(array), width)


def read_training_vectors(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    id_parts: list[np.ndarray] = []
    embedding_parts: list[np.ndarray] = []
    remaining = count
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=min(4096, count), columns=["id", "emb"]):
        raw_ids = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
        embeddings = list_array_to_matrix(batch.column(1))
        take = min(remaining, len(raw_ids))
        id_parts.append(raw_ids[:take])
        embedding_parts.append(embeddings[:take])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(f"training shard contains fewer than {count} rows")

    ids = np.concatenate(id_parts)
    int32 = np.iinfo(np.int32)
    if ids.min() < int32.min or ids.max() > int32.max:
        raise RuntimeError("candidate IDs do not fit in the int32 workload format")
    return ids.astype(np.int32), np.concatenate(embedding_parts)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float32)


def assign_vectors(
    vectors: np.ndarray,
    centroids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty(len(vectors), dtype=np.int32)
    similarities = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), batch_size):
        stop = min(start + batch_size, len(vectors))
        scores = vectors[start:stop] @ centroids.T
        batch_labels = np.argmax(scores, axis=1)
        labels[start:stop] = batch_labels
        similarities[start:stop] = scores[np.arange(stop - start), batch_labels]
    return labels, similarities


def spherical_kmeans(
    vectors: np.ndarray,
    clusters: int,
    iterations: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = rng.choice(len(vectors), size=clusters, replace=False)
    centroids = vectors[initial].copy()
    labels = np.zeros(len(vectors), dtype=np.int32)
    similarities = np.zeros(len(vectors), dtype=np.float32)

    for iteration in range(iterations):
        labels, similarities = assign_vectors(vectors, centroids, batch_size)
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, vectors)
        counts = np.bincount(labels, minlength=clusters)

        empty = np.flatnonzero(counts == 0)
        if len(empty):
            farthest = np.argsort(similarities)[: len(empty)]
            sums[empty] = vectors[farthest]
            counts[empty] = 1

        updated = normalize_rows(sums)
        shift = float(np.max(1.0 - np.sum(centroids * updated, axis=1)))
        objective = float(np.mean(1.0 - similarities))
        print(
            f"iteration={iteration + 1} mean_cosine_distance={objective:.6f} "
            f"max_centroid_shift={shift:.6f}"
        )
        centroids = updated

    labels, similarities = assign_vectors(vectors, centroids, batch_size)
    return centroids, labels, similarities


def write_ivf_requests(
    output_dir: Path,
    vectors: np.ndarray,
    candidate_ids: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    topk: int,
    stress_streams: int,
    rng: np.random.Generator,
) -> tuple[Path, Path, list[int]]:
    manifest_records: list[tuple[str, int]] = []
    list_sizes: list[int] = []
    for cluster in range(len(centroids)):
        members = np.flatnonzero(labels == cluster)
        if len(members) == 0:
            continue

        distances = np.asarray(
            1.0 - vectors[members] @ centroids[cluster], dtype=np.float32
        )
        order = rng.permutation(len(members))
        distances = distances[order]
        member_ids = candidate_ids[members[order]]

        name = f"ivf_list_{cluster:04d}"
        keys_name = f"{name}_keys.f32"
        values_name = f"{name}_values.i32"
        distances.tofile(output_dir / keys_name)
        np.asarray(member_ids, dtype=np.int32).tofile(output_dir / values_name)
        request_topk = min(topk, len(members))
        manifest_records.append((
            f"{name} {keys_name} {values_name} 1 {len(members)} {request_topk}",
            len(members),
        ))
        list_sizes.append(len(members))

    manifest = output_dir / "openai_ivf_requests.txt"
    manifest.write_text("\n".join(line for line, _ in manifest_records) + "\n")

    # Put the largest requests at positions assigned to stream zero by round-robin.
    ordered = sorted(manifest_records, key=lambda record: record[1], reverse=True)
    stress_records: list[tuple[str, int] | None] = [None] * len(ordered)
    stream_zero_slots = list(range(0, len(ordered), stress_streams))
    for slot, record in zip(stream_zero_slots, ordered[: len(stream_zero_slots)]):
        stress_records[slot] = record
    remaining = iter(ordered[len(stream_zero_slots) :])
    for index, record in enumerate(stress_records):
        if record is None:
            stress_records[index] = next(remaining)
    stress_manifest = output_dir / "openai_ivf_requests_skewed.txt"
    stress_manifest.write_text(
        "\n".join(record[0] for record in stress_records if record is not None) + "\n"
    )
    return manifest, stress_manifest, list_sizes


def main() -> int:
    args = parse_args()
    if args.vectors <= 0 or args.clusters <= 0 or args.iterations <= 0:
        raise RuntimeError("vectors, clusters, and iterations must be positive")
    if args.clusters > args.vectors:
        raise RuntimeError("clusters cannot exceed vectors")
    if args.topk <= 0 or args.topk > 128:
        raise RuntimeError("topk must be in the range [1, 128]")
    if args.assignment_batch <= 0:
        raise RuntimeError("assignment-batch must be positive")
    if args.stress_streams <= 0:
        raise RuntimeError("stress-streams must be positive")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_root / args.train
    candidate_ids, vectors = read_training_vectors(train_path, args.vectors)
    vectors = normalize_rows(vectors)
    rng = np.random.default_rng(args.seed)
    centroids, labels, similarities = spherical_kmeans(
        vectors,
        args.clusters,
        args.iterations,
        args.assignment_batch,
        rng,
    )
    manifest, stress_manifest, list_sizes = write_ivf_requests(
        output_dir,
        vectors,
        candidate_ids,
        centroids,
        labels,
        args.topk,
        args.stress_streams,
        rng,
    )

    sizes = np.asarray(list_sizes, dtype=np.int64)
    metadata = {
        "source": str(train_path),
        "vectors": int(len(vectors)),
        "dimensions": int(vectors.shape[1]),
        "clusters_requested": int(args.clusters),
        "nonempty_lists": int(len(sizes)),
        "iterations": int(args.iterations),
        "topk": int(args.topk),
        "seed": int(args.seed),
        "metric": "cosine distance on L2-normalized embeddings",
        "mean_cosine_distance": float(np.mean(1.0 - similarities)),
        "list_size_min": int(sizes.min()),
        "list_size_median": float(np.median(sizes)),
        "list_size_mean": float(np.mean(sizes)),
        "list_size_max": int(sizes.max()),
        "list_size_stddev": float(np.std(sizes)),
        "list_imbalance_max_over_mean": float(sizes.max() / np.mean(sizes)),
    }
    metadata_path = output_dir / "ivf_workload_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    np.savetxt(output_dir / "ivf_list_sizes.csv", sizes, fmt="%d", header="list_size")
    print(f"wrote {manifest}")
    print(f"wrote {stress_manifest}")
    print(f"wrote {metadata_path}")
    print(
        f"lists={len(sizes)} min={sizes.min()} median={np.median(sizes):.1f} "
        f"mean={np.mean(sizes):.1f} max={sizes.max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
