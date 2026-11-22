#!/usr/bin/env bash
# f64 CLEAN baseline over the v2 dataset ladder, several noise seeds.
# One process per (sky, noise-seed); MAXJOBS in parallel.
# Self-copies into the timestamped results dir (reproducibility record).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"
DATASET_ROOT="$DATASET_PATH"
RESULTS="$RESULTS_PATH"
STUDY="f64_baseline_$(date +%Y%m%d_%H%M%S)"
OUT="$RESULTS/$STUDY"
MAXJOBS="${MAXJOBS:-4}"

# sky ladder: name:n_point:n_gauss:dr:seed
SKIES=(
  "S0_point1:1:0:1:42"
  "S1_points_dr10:8:0:10:43"
  "S2_mixed_dr100:6:4:100:44"
  "S3_gauss_dr1000:4:8:1000:45"
)
NOISE_SEEDS="${NOISE_SEEDS:-101 102 103}"

mkdir -p "$OUT"
cp "${BASH_SOURCE[0]}" "$OUT/study_f64_baseline.sh"

cd "$CODE_ROOT"
jobs=0
for spec in "${SKIES[@]}"; do
  IFS=: read -r name np ng dr seed <<< "$spec"
  for nseed in $NOISE_SEEDS; do
    ds="$DATASET_ROOT/${name}_n${nseed}"
    if [ ! -f "$ds/obs.npz" ]; then
      "$PY" experiments/make_dataset.py --name "${name}_n${nseed}" \
        --n-point "$np" --n-gauss "$ng" --dr "$dr" --peak-flux 1.0 \
        --sigma 1e-4 --seed "$seed" --noise-seed "$nseed" \
        --uvcov "$STAGE_ROOT/data/uvcov/meerkat_1h_4ch_1p4ghz.npz" \
        --out "$ds" > "$OUT/make_${name}_n${nseed}.log" 2>&1
    fi
    run="$OUT/${name}_n${nseed}"
    if [ ! -f "$run/results.npz" ]; then
      "$PY" experiments/run_clean.py --dataset "$ds" --out "$run" \
        --gamma 0.1 --pf 1e-4 --maxiter 10000 --loop-max 10 \
        --nthreads 2 --q-vis float64 --q-dirty float64 --q-psf float64 --q-model float64 \
        > "$run.log" 2>&1 &
      # portable job pool (macOS bash 3.2 has no `wait -n`)
      while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 2; done
    fi
  done
done
wait
echo "study complete: $OUT"
