#!/usr/bin/env bash
# launch_one.sh <exp> — start ONE detached batch-2 experiment session.
#   screen available: screen session b2_<exp> running python directly
#   otherwise       : nohup process (same command, log-redirected)
# Called by run_experiments.sh via xargs -P (MAX_SCREENS cap). Readable on
# purpose: this is the only place where a session process is spawned.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"

MAXJOBS="${MAXJOBS:-8}"
NTHREADS="${NTHREADS:-4}"
BATCH_ROOT="${BATCH_ROOT:-$RESULTS_PATH/batch2}"

e="${1:?usage: launch_one.sh <exp_name>}"
mkdir -p "$BATCH_ROOT/logs"
log="$BATCH_ROOT/logs/${e}_$(date +%Y%m%d_%H%M%S).log"
cmd=("$PY" -u "$CODE_ROOT/experiments/run_batch.py" --exp "$e"
     --batch-root "$BATCH_ROOT" --maxjobs "$MAXJOBS" --nthreads "$NTHREADS"
     --log-file "$log")

if command -v screen >/dev/null 2>&1; then
  screen -dmS "b2_$e" "${cmd[@]}"
  echo "screen b2_$e started (python run_batch --exp $e; log: $log)"
else
  nohup "${cmd[@]}" > "$log" 2>&1 &
  echo "nohup b2_$e pid=$! (log: $log)"
fi
