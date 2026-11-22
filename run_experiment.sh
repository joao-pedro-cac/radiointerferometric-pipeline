#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_experiment.sh — launch the batch-2 CLEAN experiments.
#
# Pipeline position:  bash setup_env.sh  ->  bash simulate_data.sh  ->  bash run_experiment.sh
#
# Each experiment = ONE screen session (b2_<exp>) running a python process
# (experiments/run_batch.py) with an internal worker pool — no bash sleep loops.
# A b2_wait screen session (experiments/wait_batch.py) shows a live table.
# At most MAX_SCREENS sessions run concurrently (xargs -P); inside each session
# MAXJOBS runs in parallel, each with NTHREADS threads (MAXJOBS x NTHREADS ~ cores).
#
#   bash run_experiment.sh                  # all batch-2 experiments + wait session
#   bash run_experiment.sh exp_dr           # subset
#   bash run_experiment.sh --dry            # print what would launch
#   bash run_experiment.sh --no-wait        # skip the b2_wait monitor session
#   screen -r b2_exp_dr                     # live output of one experiment
#   screen -r b2_wait                       # live table of all experiments
#   tail -f results/batch2/logs/exp_dr_*.log
#
# Tunables via environment (see scripts/env.sh.example):
#   MAX_SCREENS cap on concurrent screen sessions   (default: unlimited)
#   MAXJOBS     concurrent CLEAN runs per experiment (default 8)
#   NTHREADS    threads per run                      (default 4)
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/scripts/run_experiments.sh" "$@"
