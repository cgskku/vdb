#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/data/vdb/openai_large_5m}"
OUTPUT_DIR="${2:-${DATA_ROOT}/ivf_sort_bench}"
PYTHON_BIN="${PYTHON:-python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_ivf_sort_workload.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --vectors 16384 \
  --clusters 64 \
  --iterations 8 \
  --topk 32 \
  --stress-streams 4

make -C "${SCRIPT_DIR}" clean all
"${SCRIPT_DIR}/gpu_sort" \
  --request-manifest "${OUTPUT_DIR}/openai_ivf_requests.txt" \
  --streams 4 \
  --queue-depth 64 \
  --repeats 5

"${SCRIPT_DIR}/gpu_sort" \
  --request-manifest "${OUTPUT_DIR}/openai_ivf_requests_skewed.txt" \
  --streams 4 \
  --queue-depth 64 \
  --repeats 5
