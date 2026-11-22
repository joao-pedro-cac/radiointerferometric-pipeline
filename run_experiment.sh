#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_experiment.sh — launch the batch-2 CLEAN experiments.
#
# Pipeline position:  bash setup_env.sh  ->  bash simulate_data.sh  ->  bash run_experiment.sh
#
# Each experiment runs in its own named screen session (b2_<exp>); logs land in
# results/batch2/logs/<exp>.log. Falls back to nohup when screen is missing.
#
#   bash run_experiment.sh                 # all batch-2 experiments
#   bash run_experiment.sh exp_dr exp_noise # subset
#   bash run_experiment.sh --dry           # print what would launch
#
# Tunables via environment (see scripts/env.sh.example):
#   MAXJOBS      max concurrent CLEAN runs inside each experiment (default 8)
#   RESULTS_PATH results root (default <archive>/code/results)
#   DATASET_PATH dataset root (default <archive>/data/simulated/v2)
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/scripts/run_experiments.sh" "$@"
