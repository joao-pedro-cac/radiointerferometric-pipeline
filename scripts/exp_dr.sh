#!/usr/bin/env bash
# batch2 / exp_dr — field-DR ladder on complex skies at sigma=1e-4, 2 noise seeds.
# Fields: S6 gaussians DR 1e5 + S8 mixed 1000 sources (700 pt + 250 gauss +
# 50 diffuse), DR log-ladder 1e2..1e6, sky seed 42.
# Runs land in results/batch2/exp_dr_<ts>/<dataset>_<scheme>/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "$HERE")"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
STAGE_ROOT="${STAGE_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
source "$HERE/batch2_common.sh"

SPECS="b2_S6_gauss_dr1e5_s42:0:12:0:1e5:1.0:1e-4:42 b2_S8mix1000_dr1e2_s42:700:250:50:1e2:1.0:1e-4:42 b2_S8mix1000_dr1e3_s42:700:250:50:1e3:1.0:1e-4:42 b2_S8mix1000_dr1e4_s42:700:250:50:1e4:1.0:1e-4:42 b2_S8mix1000_dr1e5_s42:700:250:50:1e5:1.0:1e-4:42 b2_S8mix1000_dr1e6_s42:700:250:50:1e6:1.0:1e-4:42"
NOISE_SEEDS="${NOISE_SEEDS:-101 102}"
STUDY="batch2/exp_dr_$(date +%Y%m%d_%H%M%S)"

SKIES="$SPECS" NOISE_SEEDS="$NOISE_SEEDS" bash "$HERE/make_ladder_v3.sh" > /dev/null
SKIES="$(b2_prefixes "$SPECS")" NOISE_SEEDS="$NOISE_SEEDS" \
  SCHEMES="$B2_SCHEMES" STUDY="$STUDY" \
  bash "$HERE/study_sweep_v3.sh"
