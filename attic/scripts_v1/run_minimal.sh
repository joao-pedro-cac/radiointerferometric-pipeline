#!/usr/bin/env bash
# Run the minimal CLEAN-tuning experiments (configs/minimal/*.json), one at a time.
# Usage: bash scripts/run_minimal.sh [config-name ...]   (default: all)
#        VENV_PATH=/path/to/venv bash scripts/run_minimal.sh
set -euo pipefail
shopt -s nullglob

ROOT="${STAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT/code"

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then source "${VENV_PATH}/bin/activate"; fi
export PYTHONPATH="$ROOT/lib/pfb-imaging/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LOG_DIR="$ROOT/logs/minimal"
mkdir -p "$LOG_DIR"

if (( $# > 0 )); then
    configs=("$@")
else
    configs=( configs/minimal/*.json )
fi

for cfg in "${configs[@]}"; do
    name="$(basename "${cfg%.json}")"
    echo "=== $name ==="
    if python -u src/main.py "$cfg" > "${LOG_DIR}/${name}.log" 2>&1; then
        echo "DONE  $name"
    else
        echo "FAIL  $name (see ${LOG_DIR}/${name}.log)" >&2
    fi
done
