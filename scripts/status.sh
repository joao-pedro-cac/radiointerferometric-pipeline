#!/usr/bin/env bash
# status.sh — one-glance monitoring of batch-2 runs and results:
#   1. screen sessions (the per-experiment python processes + b2_wait)
#   2. per-experiment progress table (from progress.json, via wait_batch --once)
#   3. last finished runs per experiment (from summary.csv, with metrics)
# Usage: bash scripts/status.sh   (from anywhere; paths auto-detected)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
source "$HERE/paths.sh"
BATCH_ROOT="${BATCH_ROOT:-$RESULTS_PATH/batch2}"

echo "== screen sessions"
if command -v screen >/dev/null 2>&1; then screen -ls 2>/dev/null || echo "(none)"; else echo "(screen not installed; nohup processes)"; fi

echo
echo "== experiment progress (root: $BATCH_ROOT)"
"$PY" "$CODE_ROOT/experiments/wait_batch.py" --root "$BATCH_ROOT" --once || true

echo
echo "== last runs per experiment (summary.csv tail)"
shopt -s nullglob
found=0
for d in "$BATCH_ROOT"/exp_*; do
  [ -f "$d/summary.csv" ] || continue
  found=1
  echo "-- $d"
  head -1 "$d/summary.csv"
  tail -n +2 "$d/summary.csv" | tail -8
done
[ "$found" -eq 0 ] && echo "(no summary.csv yet — runs still in flight; per-run logs: $BATCH_ROOT/logs/)"
