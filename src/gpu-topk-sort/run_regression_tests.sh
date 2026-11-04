#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/gpu_sort"

shapes=(
  "64 1 1"
  "4096 8 1"
  "2048 16 4"
  "1024 31 7"
  "512 64 64"
  "256 65 8"
  "256 128 1"
  "64 1024 128"
  "32 1025 128"
  "16 2048 64"
)

for shape in "${shapes[@]}"; do
  read -r groups group_size topk <<<"${shape}"
  echo "[test] groups=${groups} group_size=${group_size} topk=${topk}"
  "${BINARY}" \
    --groups "${groups}" \
    --group-size "${group_size}" \
    --topk "${topk}" \
    --streams 4 \
    --repeats 3 >/dev/null
done

echo "All segmented top-k regression shapes passed."
