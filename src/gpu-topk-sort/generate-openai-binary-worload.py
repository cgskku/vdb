"""Generate compact binary sort inputs from OpenAI benchmark parquet shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/vdb/openai_large_5m")
    parser.add_argument("--output-dir", default="/data/vdb/openai_large_5m/sort_bench")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--train", nargs="+", required=True, help="Training parquet shard names.")
    parser.add_argument("--query", default="test.parquet")
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32768)
    return parser.parse_args()


def list_array_to_matrix(array, dtype: np.dtype) -> np.ndarray:
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    lengths = offsets[1:] - offsets[:-1]
    if len(lengths) == 0:
        return np.empty((0, 0), dtype=dtype)
    dim = int(lengths[0])
    if not np.all(lengths == dim):
        raise RuntimeError("embedding column contains variable-length rows")
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype).reshape(len(lengths), dim)


def first_query(data_root: Path, query_name: str, query_index: int) -> np.ndarray:
    parquet = pq.ParquetFile(data_root / query_name)
    seen = 0
    for batch in parquet.iter_batches(batch_size=max(1, query_index + 1), columns=["emb"]):
        matrix = list_array_to_matrix(batch.column(0), np.float32)
        if seen + len(matrix) > query_index:
            query = matrix[query_index - seen].copy()
            norm = np.linalg.norm(query)
            if norm > 0.0:
                query /= norm
            return query
        seen += len(matrix)
    raise RuntimeError(f"query index {query_index} is out of range")


def append_distances(
    shard_path: Path,
    query: np.ndarray,
    remaining: int,
    batch_size: int,
    keys_parts: list[np.ndarray],
    ids_parts: list[np.ndarray],
) -> int:
    parquet = pq.ParquetFile(shard_path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "emb"]):
        if remaining <= 0:
            break
        ids = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int32)
        embs = list_array_to_matrix(batch.column(1), np.float32)
        take = min(remaining, len(ids))
        ids = ids[:take]
        embs = embs[:take]
        norms = np.linalg.norm(embs, axis=1)
        norms[norms == 0.0] = 1.0
        cosine = (embs @ query) / norms
        distances = (1.0 - cosine).astype(np.float32)
        keys_parts.append(distances)
        ids_parts.append(ids)
        remaining -= take
    return remaining


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = args.groups * args.group_size
    query = first_query(data_root, args.query, args.query_index)

    keys_parts: list[np.ndarray] = []
    ids_parts: list[np.ndarray] = []
    remaining = requested
    used_shards: list[str] = []
    for shard in args.train:
        if remaining <= 0:
            break
        used_shards.append(shard)
        remaining = append_distances(
            data_root / shard,
            query,
            remaining,
            args.batch_size,
            keys_parts,
            ids_parts,
        )

    keys = np.concatenate(keys_parts) if keys_parts else np.empty(0, dtype=np.float32)
    ids = np.concatenate(ids_parts) if ids_parts else np.empty(0, dtype=np.int32)
    usable = (len(keys) // args.group_size) * args.group_size
    if usable == 0:
        raise RuntimeError("no complete candidate groups were generated")
    keys = keys[:usable]
    ids = ids[:usable]
    groups = usable // args.group_size

    keys_path = output_dir / f"{args.prefix}_keys_g{groups}_s{args.group_size}.f32"
    ids_path = output_dir / f"{args.prefix}_ids_g{groups}_s{args.group_size}.i32"
    manifest_path = output_dir / f"{args.prefix}_manifest.txt"

    keys.tofile(keys_path)
    ids.tofile(ids_path)
    manifest_path.write_text(
        "\n".join(
            [
                f"source_train={','.join(used_shards)}",
                f"source_query={data_root / args.query}",
                f"query_index={args.query_index}",
                f"groups={groups}",
                f"group_size={args.group_size}",
                f"rows={usable}",
                "dim=1536",
                "metric=cosine_distance",
                f"keys={keys_path}",
                f"values={ids_path}",
                "",
            ]
        )
    )
    print(f"wrote {keys_path}")
    print(f"wrote {ids_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
