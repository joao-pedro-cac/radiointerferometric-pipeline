#!/usr/bin/env bash
# status.sh — one-glance monitoring of batch runs, no screen needed:
#   1. experiment processes: pid alive/exited + last log line (from logs/<exp>.pid)
#   2. per-experiment progress table (from progress.json, via wait_batch --once)
#   3. last finished runs per experiment (summary.csv tail)
# Usage: bash scripts/status.sh   (from anywhere; paths auto-detected)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"
BATCH_ROOT="${BATCH_ROOT:-$RESULTS_PATH/batch2}"
LOGS="$BATCH_ROOT/logs"

echo "== experiment processes (logs/*.pid)"
shopt -s nullglob
found=0
for pidf in "$LOGS"/*.pid; do
  found=1
  e="$(basename "$pidf" .pid)"
  pid="$(cat "$pidf")"
  if kill -0 "$pid" 2>/dev/null; then st="alive"; else st="EXITED"; fi
  log="$(ls -t "$LOGS/${e}"_*.log 2>/dev/null | head -1)"
  last="$(tail -1 "$log" 2>/dev/null | cut -c1-110)"
  echo "  $st  $e  pid=$pid  log=${log##*/}"
  echo "        $last"
done
[ "$found" -eq 0 ] && echo "  (no processes — launch with: bash run_experiment.sh)"

echo
echo "== experiment progress (root: $BATCH_ROOT)"
"$PY" "$CODE_ROOT/experiments/wait_batch.py" --root "$BATCH_ROOT" --once || true

echo
echo "== last runs per experiment (summary.csv tail)"
found=0
for d in "$BATCH_ROOT"/exp_*; do
  [ -f "$d/summary.csv" ] || continue
  found=1
  echo "-- ${d##*/}"
  head -1 "$d/summary.csv"
  tail -n +2 "$d/summary.csv" | tail -6
done
[ "$found" -eq 0 ] && echo "(no summary.csv yet — live exp logs: $LOGS/<exp>_<ts>.log)"
