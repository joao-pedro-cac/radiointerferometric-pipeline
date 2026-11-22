#!/usr/bin/env bash
# Launch batch-2 experiments: ONE screen session per experiment running a
# python process (experiments/run_batch.py) with an internal worker pool —
# no bash sleep/poll loops. At most MAX_SCREENS sessions run at once
# (xargs -P); inside each, MAXJOBS runs in parallel x NTHREADS threads each.
# A b2_wait monitor session (experiments/wait_batch.py) shows a live table.
#
#   bash run_experiment.sh [--dry] [--no-wait] [exp ...]     (root-level wrapper)
#   MAX_SCREENS=2 MAXJOBS=8 NTHREADS=4 bash scripts/run_experiments.sh
#   screen -r b2_exp_dr        # live output of one experiment's python process
#   screen -r b2_wait          # live table of all experiments
#   tail -f results/batch2/logs/<exp>_<ts>.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"

MAX_SCREENS="${MAX_SCREENS:-99}"     # cap on concurrent screen sessions
MAXJOBS="${MAXJOBS:-8}"              # concurrent runs inside each session
NTHREADS="${NTHREADS:-4}"            # threads per run (MAXJOBS x NTHREADS ~ cores)
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

mkdir -p "$BATCH_ROOT/logs"

launch_one() {
  local e="$1"
  local log="$BATCH_ROOT/logs/${e}_$(date +%Y%m%d_%H%M%S).log"
  screen -dmS "b2_$e" bash -c \
    "\"$PY\" \"$CODE_ROOT/experiments/run_batch.py\" --exp \"$e\" --batch-root \"$BATCH_ROOT\" --maxjobs '$MAXJOBS' --nthreads '$NTHREADS' 2>&1 | tee \"$log\""
  echo "launched screen b2_$e  (log: $log)"
}

if [ "$DRY" -eq 1 ]; then
  for e in "${EXPS[@]}"; do
    echo "DRY: screen -dmS b2_$e $PY $CODE_ROOT/experiments/run_batch.py --exp $e --batch-root $BATCH_ROOT --maxjobs $MAXJOBS --nthreads $NTHREADS"
  done
  [ "$NOWAIT" -eq 0 ] && echo "DRY: screen -dmS b2_wait $PY $CODE_ROOT/experiments/wait_batch.py --root $BATCH_ROOT"
  echo "DRY: MAX_SCREENS=$MAX_SCREENS caps concurrent sessions"
  exit 0
fi

if command -v screen >/dev/null 2>&1; then
  printf '%s\n' "${EXPS[@]}" | xargs -P "$MAX_SCREENS" -I{} bash -c "$(declare -f launch_one); launch_one {}"
  if [ "$NOWAIT" -eq 0 ]; then
    screen -dmS b2_wait "$PY" "$CODE_ROOT/experiments/wait_batch.py" --root "$BATCH_ROOT"
    echo "launched screen b2_wait (live table; screen -r b2_wait)"
  fi
  echo "--- sessions:"; screen -ls || true
else
  echo "screen not found — nohup fallback (still one process per exp, MAX_SCREENS at a time)"
  nohup_wait() {
    local e="$1"
    local log="$BATCH_ROOT/logs/${e}_$(date +%Y%m%d_%H%M%S).log"
    "$PY" "$CODE_ROOT/experiments/run_batch.py" --exp "$e" --batch-root "$BATCH_ROOT" \
      --maxjobs "$MAXJOBS" --nthreads "$NTHREADS" > "$log" 2>&1
    echo "finished $e (log: $log)"
  }
  export -f nohup_wait
  export PY CODE_ROOT BATCH_ROOT MAXJOBS NTHREADS
  printf '%s\n' "${EXPS[@]}" | xargs -P "$MAX_SCREENS" -I{} bash -c 'nohup_wait "$@"' _ {}
  [ "$NOWAIT" -eq 0 ] && "$PY" "$CODE_ROOT/experiments/wait_batch.py" --root "$BATCH_ROOT" --once || true
fi
