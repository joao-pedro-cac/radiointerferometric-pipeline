#!/usr/bin/env bash
# Extended v3 dataset ladder (range/DR/noise stress for the quantization study).
# One dataset per (sky spec, noise seed); skips existing ones.
#
#   NOISE_SEEDS="101" bash scripts/make_ladder_v3.sh
#
# Sky spec format: name:n_point:n_gauss:dr:peak_flux_jy:sigma_jy:seed
# S5/S6 stress intra-field DR (model image range); S7 stresses absolute range
# (peak 1e6 Jy -> |vis|max ~ 1e6 > f16 max 65504: overflow regime for raw cast,
# subnormal noise for rescaled cast). S8/S9 sweep noise at fixed sky.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "$HERE")"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
STAGE_ROOT="${STAGE_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/tmp/joao-venv}"
PY="$VENV_PATH/bin/python"
DATASET_ROOT="${DATASET_PATH:-$STAGE_ROOT/data/simulated/v2}"
UVCOV="${UVCOV:-$STAGE_ROOT/data/uvcov/meerkat_1h_4ch_1p4ghz.npz}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"

# name:n_point:n_gauss:dr:peak_flux:sigma:seed   (space-separated string)
SKIES="${SKIES:-S5_points_dr1e6:24:0:1e6:1.0:1e-4:46 S6_gauss_dr1e5:4:8:1e5:1.0:1e-4:47 S7_point_peak1e6:1:0:1:1e6:1e-4:48 S2_sigma1e-2:6:4:100:1.0:1e-2:44 S2_sigma1e-6:6:4:100:1.0:1e-6:44}"

cd "$CODE_ROOT"
for spec in $SKIES; do
  IFS=: read -r name np ng dr peak sigma seed <<< "$spec"
  for nseed in $NOISE_SEEDS; do
    ds="$DATASET_ROOT/${name}_n${nseed}"
    if [ -f "$ds/obs.npz" ]; then
      echo "exists: $ds"
      continue
    fi
    "$PY" experiments/make_dataset.py --name "${name}_n${nseed}" \
      --n-point "$np" --n-gauss "$ng" --dr "$dr" --peak-flux "$peak" \
      --sigma "$sigma" --seed "$seed" --noise-seed "$nseed" \
      --uvcov "$UVCOV" --out "$ds"
  done
done
echo "ladder v3 ready under $DATASET_ROOT"
