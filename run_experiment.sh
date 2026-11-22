#!/usr/bin/env bash
# Start batch experiments: ONE background python process per experiment.
# This file is the ONLY place where python processes are started.
#
#   bash run_experiment.sh [--dry] [--no-wait] [exp ...]   (default: all plans)
#   MAXPROCS=2 NTHREADS=4 bash run_experiment.sh exp3_mc exp3_range
#
# Per experiment:  log = results/batch2/logs/<exp>_<ts>.log   (python -u, live)
#                  pid = results/batch2/logs/<exp>.pid
# After 1 s each process is fail-fast checked (alive + banner in log); on
# failure the log tail is printed and the launcher exits non-zero.
# Monitor anytime:  bash scripts/status.sh   or   tail -f results/batch2/logs/<exp>_*.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/scripts/env.sh" ] && source "$HERE/scripts/env.sh"   # optional overrides
source "$HERE/scripts/paths.sh"                                  # auto-detected paths

MAXPROCS="${MAXPROCS:-99}"   # max concurrent experiment processes
NTHREADS="${NTHREADS:-4}"    # threads per CLEAN run (ducc0/numba), inside each process
BATCH_ROOT="${BATCH_ROOT:-$RESULTS_PATH/batch2}"
LOGS="$BATCH_ROOT/logs"

# --- parse args: --dry / --no-wait are flags, everything else is an exp name
DRY=0; NOWAIT=0; EXPS=()
for a in "$@"; do
  case "$a" in
    --dry) DRY=1 ;;
    --no-wait) NOWAIT=1 ;;
    *) EXPS+=("$a") ;;
  esac
done
# default: all plans
if [ ${#EXPS[@]} -eq 0 ]; then
  while IFS= read -r e; do EXPS+=("$e"); done < <("$PY" "$CODE_ROOT/experiments/run_batch.py" --list-exps | tr ' ' '\n')
fi
mkdir -p "$LOGS"

# --- start one experiment process and fail-fast check it
launch() {
  local e="$1" log pid
  log="$LOGS/${e}_$(date +%Y%m%d_%H%M%S).log"
  nohup "$PY" -u "$CODE_ROOT/experiments/run_batch.py" --exp "$e" \
      --batch-root "$BATCH_ROOT" --nthreads "$NTHREADS" > "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$LOGS/${e}.pid"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FAILED  $e (pid $pid died) — log $log:" >&2; tail -5 "$log" >&2; return 1
  fi
  grep -q "run_batch starting" "$log" \
    && echo "OK      $e  pid=$pid  log=$log  ($(grep -o 'to run=[0-9]*' "$log" | head -1))" \
    || echo "START?  $e  pid=$pid  log=$log  (no banner yet)"
}

# --- dry run: just print the launch lines
if [ "$DRY" -eq 1 ]; then
  for e in "${EXPS[@]}"; do
    echo "DRY: nohup $PY -u $CODE_ROOT/experiments/run_batch.py --exp $e --batch-root $BATCH_ROOT --nthreads $NTHREADS > $LOGS/${e}_<ts>.log 2>&1 &"
  done
  exit 0
fi

# --- launch all experiments (wait only if at the MAXPROCS cap)
rc=0
for e in "${EXPS[@]}"; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXPROCS" ]; do sleep 0.5; done
  launch "$e" || rc=1
done

# --- wait/monitor process (logs progress table until everything is done)
if [ "$NOWAIT" -eq 0 ]; then
  nohup "$PY" -u "$CODE_ROOT/experiments/wait_batch.py" --root "$BATCH_ROOT" \
      > "$LOGS/wait_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  echo $! > "$LOGS/wait.pid"
  echo "OK      wait-monitor pid=$(cat "$LOGS/wait.pid")  log=$LOGS/wait_*.log"
fi
echo "monitor anytime: bash scripts/status.sh"
exit $rc
