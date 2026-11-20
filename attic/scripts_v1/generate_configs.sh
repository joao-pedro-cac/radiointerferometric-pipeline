#!/usr/bin/env bash
# Generate the configuration JSON files for the float16 analysis

set -euo pipefail

PYTHON="python3"

ROOT="${STAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO="$ROOT/code"
cd "$REPO"
PROGRAM="$REPO/src/generate_configs.py"

DATASET_ROOT="$ROOT/data/simulated/hogbom_experiments"
RESULTS_ROOT="$ROOT/results/analysis-float16"
JSON_OUTPUT_DIR="$REPO/configs/analysis-float16"

# optional virtualenv: set VENV_PATH to activate it
if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then source "${VENV_PATH}/bin/activate"; fi

echo "[$(date +%H:%M:%S)] Creating JSON configs for float16 vs float64 analysis"

python "$PROGRAM" \
    --datasets-root $DATASET_ROOT \
    --out-dir       $JSON_OUTPUT_DIR \
    --results-root  $RESULTS_ROOT

echo "[$(date +%H:%M:%S)] Done. Config JSON files at $JSON_OUTPUT_DIR"
