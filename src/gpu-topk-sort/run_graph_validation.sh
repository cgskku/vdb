#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/data/vdb/openai_large_5m}"
OUTPUT_DIR="${2:-${DATA_ROOT}/graph_sort_bench}"

python "${SCRIPT_DIR}/prepare_graph_sort_workload.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

make -C "${SCRIPT_DIR}" clean all
"${SCRIPT_DIR}/gpu_sort" \
  --request-manifest "${OUTPUT_DIR}/openai_graph_requests.txt" \
  --streams 4 \
  --repeats 5
