#!/usr/bin/env bash
# Quantization-scheme study: 16 schemes x sky ladder x noise seeds, vs the
# f64 baseline. One process per run, MAXJOBS in parallel, self-copied into
# the timestamped results dir.
#
#   SKIES="S0_point1 S3_gauss_dr1000" NOISE_SEEDS="101" MAXJOBS=4 \
#     bash scripts/study_quant_schemes.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"
DATASET_ROOT="$DATASET_PATH"
RESULTS="$RESULTS_PATH"
STUDY="quant_schemes_$(date +%Y%m%d_%H%M%S)"
OUT="$RESULTS/$STUDY"
MAXJOBS="${MAXJOBS:-4}"

# 16 schemes as vis:dirty:psf:model
SCHEMES=(
  "float64:float64:float64:float64"   # 0 baseline (f64)
  "float32:float64:float64:float64"
  "float64:float32:float64:float64"
  "float64:float64:float32:float64"
  "float64:float64:float64:float32"
  "float16:float64:float64:float64"   # single-stage f16
  "float64:float16:float64:float64"
  "float64:float64:float16:float64"
  "float64:float64:float64:float16"
  "bfloat16:float64:float64:float64"  # single-stage bf16
  "float64:bfloat16:float64:float64"
  "float64:float64:bfloat16:float64"
  "float64:float64:float64:bfloat16"
  "float16:float16:float16:float16"   # all f16
  "bfloat16:bfloat16:bfloat16:bfloat16" # all bf16
  "float16:bfloat16:float16:bfloat16"   # mixed low precision
)
SKIES="${SKIES:-S0_point1 S1_points_dr10 S2_mixed_dr100 S3_gauss_dr1000}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"

mkdir -p "$OUT"
cp "${BASH_SOURCE[0]}" "$OUT/study_quant_schemes.sh"

cd "$CODE_ROOT"
for sky in $SKIES; do
  for nseed in $NOISE_SEEDS; do
    ds="$DATASET_ROOT/${sky}_n${nseed}"
    [ -f "$ds/obs.npz" ] || { echo "missing dataset $ds (run study_f64_baseline.sh first)"; exit 1; }
    for i in "${!SCHEMES[@]}"; do
      s="${SCHEMES[$i]}"
      IFS=: read -r qv qd qp qm <<< "$s"
      run="$OUT/${sky}_n${nseed}_s${i}_${qv}-${qd}-${qp}-${qm}"
      if [ ! -f "$run/results.npz" ]; then
        "$PY" experiments/run_clean.py --dataset "$ds" --out "$run" \
          --gamma 0.1 --pf 1e-4 --maxiter 10000 --loop-max 10 \
          --nthreads 2 --q-vis "$qv" --q-dirty "$qd" --q-psf "$qp" --q-model "$qm" \
          > "$run.log" 2>&1 &
        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 2; done
      fi
    done
  done
done
wait
echo "study complete: $OUT"
