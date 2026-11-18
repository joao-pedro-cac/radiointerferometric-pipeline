#!/bin/bash

PROGRAM="python3"

ROOT="${STAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIGS_DIR="$ROOT/code/configs/analysis-float16"
SRC_DIR="$ROOT/code/src"

float16config="$CONFIGS_DIR/gaussian_dr10000_s0__q-dirty-f16__vf100.json"
float64config="$CONFIGS_DIR/gaussian_dr10000_s0__q-baseline-f64__vf100.json"

PRELUDE=""
if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then PRELUDE="source ${VENV_PATH}/bin/activate; "; fi

screen -dmS float16 bash -c "
        ${PRELUDE}$PROGRAM $SRC_DIR/main.py $float16config
    "
screen -dmS float64 bash -c "
        ${PRELUDE}$PROGRAM $SRC_DIR/main.py $float64config
    "
