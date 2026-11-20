#!/usr/bin/env bash
# Generate the configuration JSON files for the float16 analysis

set -euo pipefail

PYTHON="python3"

ROOT="${STAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO="$ROOT/code"
cd "$REPO"
CONFIGS_ROOT="$REPO/configs/analysis-float16"
VENV_PATH="${VENV_PATH:-}"
PROGRAM="$REPO/src/main.py"

PRELUDE=""
if [[ -n "$VENV_PATH" ]]; then PRELUDE="source '${VENV_PATH}/bin/activate'; "; fi
echo "[$(date +%H:%M:%S)] Simulating data..."

# for conf in $REPO/configs/analysis-float16/*; do
#     python $PROGRAM $conf
# done

screen -dmS "float64" bash -c "
        ${PRELUDE}$PYTHON '$PROGRAM' '$CONFIGS_ROOT/gaussian_dr10000_s0__q-baseline-f64__vf100.json'
        echo '[Done] float64'
        exec bash
    "
echo "  -> launched session: float64"


screen -dmS "float16" bash -c "
        ${PRELUDE}$PYTHON '$PROGRAM' '$CONFIGS_ROOT/gaussian_dr10000_s0__q-dirty-f16__vf100.json'
        echo '[Done] float16'
        exit
    "
echo "  -> launched session: float16"




# for conf in "$CONFIGS_ROOT"/*; do
#     name=$(basename "$conf")
#     session="float16_${name%.*}"

#     screen -dmS "$session" bash -c "
#         source '${VENV_PATH}/bin/activate'
#         $PYTHON '$PROGRAM' '$conf'
#         echo '[Done] $name'
#         exec bash
#     "
#     echo "  -> launched session: $session"
# done


echo "[$(date +%H:%M:%S)] Done. Results succesfully saved!"
