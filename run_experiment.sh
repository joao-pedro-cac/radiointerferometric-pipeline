#!/usr/bin/env bash
# Launch batch experiments as plain nohup'd python processes — no screen, no
# sleep loops. One process per experiment (experiments/run_batch.py), pid saved
# to results/batch2/logs/<exp>.pid, log to results/batch2/logs/<exp>_<ts>.log
# (python -u => the banner is in the log within a second).
#
#   bash run_experiment.sh [--dry] [exp ...]        # default: all plans
#   MAXPROCS=2 bash run_experiment.sh exp3_mc       # cap concurrent processes
#   bash scripts/status.sh                          # monitor: pids, logs, progress, results
#   tail -f results/batch2/logs/exp_dr_*.log
#
# Fail-fast: each launch is checked after 1 s — if the process died, the log
# tail is printed and the exp is marked FAILED. A wait/monitor process
# (experiments/wait_batch.py) is started unless --no-wait.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/scripts/env.sh" ] && source "$HERE/scripts/env.sh"
source "$HERE/scripts/paths.sh"

MAXPROCS="${MAXPROCS:-99}"        # concurrent experiment processes
MAXJOBS="${MAXJOBS:-8}"           # concurrent runs inside each process
NTHREADS="${NTHREADS:-4}"         # threads per run (MAXJOBS x NTHREADS ~ cores)
BATCH_ROOT="${BATCH_ROOT:-$RESULTS_PATH/batch2}"
LOGS="$BATCH_ROOT/logs"

DRY=0; NOWAIT=0; EXPS=()
for a in "$@"; do
  case "$a" in
    --dry) DRY=1;; --no-wait) NOWAIT=1;; *) EXPS+=("$a");;
  esac
done
if [ ${#EXPS[@]} -eq 0 ]; then
  while IFS= read -r e; do EXPS+=("$e"); done < <("$PY" "$CODE_ROOT/experiments/run_batch.py" --list-exps | tr ' ' '\n')
fi

mkdir -p "$LOGS"

launch() {
  local e="$1" log pid
  log="$LOGS/${e}_$(date +%Y%m%d_%H%M%S).log"
  nohup "$PY" -u "$CODE_ROOT/experiments/run_batch.py" --exp "$e" \
      --batch-root "$BATCH_ROOT" --maxjobs "$MAXJOBS" --nthreads "$NTHREADS" \
      > "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$LOGS/${e}.pid"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FAILED  $e (pid $pid died) — log $log:" >&2
    tail -5 "$log" >&2
    return 1
  fi
  if grep -q "run_batch starting" "$log"; then
    echo "OK      $e  pid=$pid  log=$log  ($(grep -o 'to run=[0-9]*' "$log" | head -1))"
  else
    echo "START?  $e  pid=$pid  log=$log  (no banner yet)"
  fi
}

if [ "$DRY" -eq 1 ]; then
  for e in "${EXPS[@]}"; do
    echo "DRY: nohup $PY -u $CODE_ROOT/experiments/run_batch.py --exp $e --batch-root $BATCH_ROOT --maxjobs $MAXJOBS --nthreads $NTHREADS > $LOGS/${e}_<ts>.log 2>&1 &"
  done
  [ "$NOWAIT" -eq 0 ] && echo "DRY: nohup $PY -u $CODE_ROOT/experiments/wait_batch.py --root $BATCH_ROOT > $LOGS/wait.log 2>&1 &"
  echo "DRY: MAXPROCS=$MAXPROCS caps concurrent experiment processes"
  exit 0
fi

rc=0
for e in "${EXPS[@]}"; do
  # cap concurrent experiment processes (only waits when at the cap)
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXPROCS" ]; do sleep 0.5; done
  if ! launch "$e"; then rc=1; fi
done

if [ "$NOWAIT" -eq 0 ]; then
  nohup "$PY" -u "$CODE_ROOT/experiments/wait_batch.py" --root "$BATCH_ROOT" \
      > "$LOGS/wait_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  echo $! > "$LOGS/wait.pid"
  echo "OK      wait-monitor pid=$(cat "$LOGS/wait.pid")  log=$LOGS/wait_*.log"
fi
echo "monitor anytime: bash scripts/status.sh"
exit $rc
