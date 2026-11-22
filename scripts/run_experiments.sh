#!/usr/bin/env bash
# Launch batch-2 experiments, each under its own named screen session
# (nohup fallback when screen is unavailable). Semantic session names b2_<exp>.
#
#   bash scripts/run_experiments.sh                 # all: exp_dr exp_noise exp_seeds
#   bash scripts/run_experiments.sh --dry exp_dr    # print what would run
#   screen -ls                                      # list sessions
#   tail -f code/results/batch2/logs/exp_dr.log     # follow a log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"
RESULTS="$RESULTS_PATH"

DRY=0
EXPS=()
for a in "$@"; do
  if [ "$a" = "--dry" ]; then DRY=1; else EXPS+=("$a"); fi
done
[ ${#EXPS[@]} -eq 0 ] && EXPS=(exp_dr exp_noise exp_seeds)

mkdir -p "$RESULTS/batch2/logs"
for e in "${EXPS[@]}"; do
  script="$HERE/$e.sh"
  if [ ! -f "$script" ]; then echo "missing experiment script: $script" >&2; exit 1; fi
  log="$RESULTS/batch2/logs/$e.log"
  if [ "$DRY" -eq 1 ]; then
    echo "DRY: screen -dmS b2_$e bash $script   (log: $log)"
    continue
  fi
  if command -v screen >/dev/null 2>&1; then
    screen -dmS "b2_$e" bash -c "bash '$script' > '$log' 2>&1"
    echo "launched screen session b2_$e  (log: $log)"
  else
    nohup bash "$script" > "$log" 2>&1 &
    echo "launched nohup for $e (pid $!)  (log: $log)"
  fi
done
if [ "$DRY" -eq 0 ] && command -v screen >/dev/null 2>&1; then
  echo "--- sessions:"; screen -ls || true
fi
