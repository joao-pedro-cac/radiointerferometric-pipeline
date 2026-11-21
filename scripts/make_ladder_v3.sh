#!/usr/bin/env bash
# Extended dataset ladder (range/DR/noise/seed stress for the quantization study).
# One dataset per (sky spec, noise seed); skips existing ones.
#
#   NOISE_SEEDS="101" SKIES="name:np:ng:nd:dr:peak:sigma:seed ..." \
#     bash scripts/make_ladder_v3.sh
#
# Sky spec format (space-separated string):
#   name : n_point : n_gauss : n_diffuse : dr : peak_flux_jy : sigma_jy : sky_seed
# Dataset dirs are "<name>_n<noise_seed>" under $DATASET_ROOT.
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

# name:n_point:n_gauss:n_diffuse:dr:peak_flux:sigma:seed   (space-separated)
SKIES="${SKIES:-"
S5_points_dr1e6:24:0:0:1e6:1.0:1e-4:46
S6_gauss_dr1e5:4:8:0:1e5:1.0:1e-4:47
S7_point_peak1e6:1:0:0:1:1e6:1e-4:48
S2_sigma1e-2:6:4:0:100:1.0:1e-2:44
S2_sigma1e-6:6:4:0:100:1.0:1e-6:44
b2_S6_gauss_dr1e5_s42:0:12:0:1e5:1.0:1e-4:42
b2_S8mix1000_dr1e2_s42:700:250:50:1e2:1.0:1e-4:42
b2_S8mix1000_dr1e3_s42:700:250:50:1e3:1.0:1e-4:42
b2_S8mix1000_dr1e4_s42:700:250:50:1e4:1.0:1e-4:42
b2_S8mix1000_dr1e5_s42:700:250:50:1e5:1.0:1e-4:42
b2_S8mix1000_dr1e6_s42:700:250:50:1e6:1.0:1e-4:42
b2_S8mix1000_dr1e4_s43:700:250:50:1e4:1.0:1e-4:43
b2_S8mix1000_dr1e4_s44:700:250:50:1e4:1.0:1e-4:44
b2_S8mix1000_dr1e4_s42_sig1e-2:700:250:50:1e4:1.0:1e-2:42
b2_S8mix1000_dr1e4_s42_sig1e-3:700:250:50:1e4:1.0:1e-3:42
b2_S8mix1000_dr1e4_s42_sig1e-6:700:250:50:1e4:1.0:1e-6:42
"}"

cd "$CODE_ROOT"
for spec in $SKIES; do
  IFS=: read -r name np ng nd dr peak sigma seed <<< "$spec"
  for nseed in $NOISE_SEEDS; do
    ds="$DATASET_ROOT/${name}_n${nseed}"
    if [ -f "$ds/obs.npz" ]; then
      echo "exists: $ds"
      continue
    fi
    "$PY" experiments/make_dataset.py --name "${name}_n${nseed}" \
      --n-point "$np" --n-gauss "$ng" --n-diffuse "$nd" --dr "$dr" \
      --peak-flux "$peak" --sigma "$sigma" --seed "$seed" --noise-seed "$nseed" \
      --uvcov "$UVCOV" --out "$ds"
  done
done
echo "ladder ready under $DATASET_ROOT"
