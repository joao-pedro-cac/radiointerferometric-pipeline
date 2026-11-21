#!/usr/bin/env bash
# batch2 / exp_seeds — statistical significance on the reference complex field:
# 3 sky-model seeds (s42/s43/s44) at sigma=1e-4, noise seed 103 (101/102 come
# from exp_dr) -> 3x2 seed spread in total for the headline field.
# Runs land in results/batch2/exp_seeds_<ts>/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "$HERE")"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
STAGE_ROOT="${STAGE_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
source "$HERE/batch2_common.sh"

SPECS="b2_S8mix1000_dr1e4_s42:700:250:50:1e4:1.0:1e-4:42 b2_S8mix1000_dr1e4_s43:700:250:50:1e4:1.0:1e-4:43 b2_S8mix1000_dr1e4_s44:700:250:50:1e4:1.0:1e-4:44"
NOISE_SEEDS="${NOISE_SEEDS:-103}"
STUDY="batch2/exp_seeds_$(date +%Y%m%d_%H%M%S)"

SKIES="$SPECS" NOISE_SEEDS="$NOISE_SEEDS" bash "$HERE/make_ladder_v3.sh" > /dev/null
SKIES="$(b2_prefixes "$SPECS")" NOISE_SEEDS="$NOISE_SEEDS" \
  SCHEMES="$B2_SCHEMES" STUDY="$STUDY" \
  bash "$HERE/study_sweep_v3.sh"
