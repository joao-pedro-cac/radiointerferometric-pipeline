#!/usr/bin/env bash
# batch2 / exp_noise — noise ladder on the reference complex field
# (S8 mixed 1000 sources, DR 1e4, sky seed 42) at sigma = 1e-2, 1e-3, 1e-6
# (1e-4 rows come from exp_dr), 2 noise seeds.
# Runs land in results/batch2/exp_noise_<ts>/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"
source "$HERE/batch2_common.sh"

SPECS="b2_S8mix1000_dr1e4_s42_sig1e-2:700:250:50:1e4:1.0:1e-2:42 b2_S8mix1000_dr1e4_s42_sig1e-3:700:250:50:1e4:1.0:1e-3:42 b2_S8mix1000_dr1e4_s42_sig1e-6:700:250:50:1e4:1.0:1e-6:42"
NOISE_SEEDS="${NOISE_SEEDS:-101 102}"
STUDY="batch2/exp_noise_$(date +%Y%m%d_%H%M%S)"

SKIES="$SPECS" NOISE_SEEDS="$NOISE_SEEDS" bash "$HERE/make_ladder_v3.sh" > /dev/null
SKIES="$(b2_prefixes "$SPECS")" NOISE_SEEDS="$NOISE_SEEDS" \
  SCHEMES="$B2_SCHEMES" STUDY="$STUDY" \
  bash "$HERE/study_sweep_v3.sh"
