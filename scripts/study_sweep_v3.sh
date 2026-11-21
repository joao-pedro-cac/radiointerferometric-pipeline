#!/usr/bin/env bash
# v3 sweep study: scheme x sky x noise-seed runs with per-iteration probes on.
# One process per run, MAXJOBS in parallel, self-copied into the timestamped
# results dir. Schemes encode their rescale ceiling (0 = raw cast, overflow).
#
#   SKIES="S2_mixed_dr100 S5_points_dr1e6 S7_point_peak1e6" NOISE_SEEDS="101" \
#     MAXJOBS=4 bash scripts/study_sweep_v3.sh
#
# Scheme spec format: name=q_vis:q_dirty:q_psf:q_model:rescale_max
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "$HERE")"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
STAGE_ROOT="${STAGE_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/tmp/joao-venv}"
PY="$VENV_PATH/bin/python"
DATASET_ROOT="${DATASET_PATH:-$STAGE_ROOT/data/simulated/v2}"
RESULTS="${RESULTS_PATH:-$CODE_ROOT/results}"
STUDY="${STUDY:-sweep_v3_$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS/$STUDY"
MAXJOBS="${MAXJOBS:-4}"
NTHREADS="${NTHREADS:-2}"

SCHEMES="${SCHEMES:-f64=float64:float64:float64:float64:65503 allf16=float16:float16:float16:float16:65503 allf16raw=float16:float16:float16:float16:0 allbf16=bfloat16:bfloat16:bfloat16:bfloat16:65503 modelf16=float64:float64:float64:float16:65503}"
SKIES="${SKIES:-S2_mixed_dr100 S5_points_dr1e6 S7_point_peak1e6}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"

mkdir -p "$OUT"
cp "${BASH_SOURCE[0]}" "$OUT/study_sweep_v3.sh"

cd "$CODE_ROOT"
for sky in $SKIES; do
  for nseed in $NOISE_SEEDS; do
    ds="$DATASET_ROOT/${sky}_n${nseed}"
    [ -f "$ds/obs.npz" ] || { echo "missing dataset $ds (run scripts/make_ladder_v3.sh first)"; exit 1; }
    for spec in $SCHEMES; do
      name="${spec%%=*}"
      rest="${spec#*=}"
      IFS=: read -r qv qd qp qm rmax <<< "$rest"
      run="$OUT/${sky}_n${nseed}_${name}"
      if [ ! -f "$run/results.npz" ]; then
        "$PY" experiments/run_clean.py --dataset "$ds" --out "$run" \
          --gamma 0.1 --pf 1e-4 --maxiter 10000 --loop-max 10 \
          --nthreads "$NTHREADS" --rescale-max "$rmax" \
          --q-vis "$qv" --q-dirty "$qd" --q-psf "$qp" --q-model "$qm" \
          > "$run.log" 2>&1 &
        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 2; done
      fi
    done
  done
done
wait
echo "study complete: $OUT"
