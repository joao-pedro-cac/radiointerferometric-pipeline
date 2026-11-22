#!/usr/bin/env bash
# Launch batch-2 experiments: one detached session per experiment via
# scripts/launch_one.sh, at most MAX_SCREENS at a time (xargs -P), plus a
# b2_wait monitor session. No sleep loops. Root wrapper: bash run_experiment.sh
#
#   bash run_experiment.sh [--dry] [--no-wait] [exp ...]      # from repo root
#   MAX_SCREENS=1 MAXJOBS=8 NTHREADS=4 bash scripts/run_experiments.sh
#   bash scripts/status.sh        # one-glance monitoring (sessions + progress + results)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"

MAX_SCREENS="${MAX_SCREENS:-99}"
BATCH_ROOT="${BATCH_ROOT:-$RESULTS_PATH/batch2}"

DRY=0; NOWAIT=0; EXPS=()
for a in "$@"; do
  case "$a" in
    --dry) DRY=1;; --no-wait) NOWAIT=1;; *) EXPS+=("$a");;
  esac
done
if [ ${#EXPS[@]} -eq 0 ]; then
  while IFS= read -r e; do EXPS+=("$e"); done < <("$PY" "$CODE_ROOT/experiments/run_batch.py" --list-exps | tr ' ' '\n')
fi

if [ "$DRY" -eq 1 ]; then
  for e in "${EXPS[@]}"; do
    echo "DRY: bash $HERE/launch_one.sh $e   -> screen -dmS b2_$e  (log: $BATCH_ROOT/logs/${e}_<ts>.log)"
  done
  [ "$NOWAIT" -eq 0 ] && echo "DRY: screen -dmS b2_wait $PY -u $CODE_ROOT/experiments/wait_batch.py --root $BATCH_ROOT"
  echo "DRY: MAX_SCREENS=$MAX_SCREENS caps concurrent sessions"
  exit 0
fi

printf '%s\n' "${EXPS[@]}" | xargs -P "$MAX_SCREENS" -n1 bash "$HERE/launch_one.sh"

if [ "$NOWAIT" -eq 0 ] && command -v screen >/dev/null 2>&1; then
  screen -dmS b2_wait "$PY" -u "$CODE_ROOT/experiments/wait_batch.py" --root "$BATCH_ROOT"
  echo "screen b2_wait started (live table: screen -r b2_wait)"
fi
echo "--- sessions:"; screen -ls 2>/dev/null || true
echo "monitor anytime: bash scripts/status.sh"
