#!/usr/bin/env python3
"""Monitor for batch-2 runs: live table over every exp's progress.json.
Typically run inside its own screen session by run_experiments.sh:

    screen -dmS b2_wait python experiments/wait_batch.py --root results/batch2
    screen -r b2_wait          # live table
    python experiments/wait_batch.py --root results/batch2 --once   # one-shot, for scripts

Exits 0 when every experiment found is complete (done+failed == total).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]


def default_root():
    results = os.environ.get("RESULTS_PATH") or str(CODE_ROOT / "results")
    return str(Path(results) / "batch2")


def collect(root):
    """One row per exp dir (exp_* containing progress.json or summary.csv)."""
    rows = []
    for d in sorted(Path(root).glob("exp*")):  # exp_dr..., exp3_mc...
        if not d.is_dir():
            continue
        pj = d / "progress.json"
        if pj.exists():
            p = json.loads(pj.read_text())
            total, done, failed, running = p["total"], len(p["done"]), len(p["failed"]), p["running"]
            started = p.get("started", "")
        else:
            total = done = failed = 0
            running = []
            started = ""
        rows.append(dict(exp=d.name, total=total, done=done, failed=failed,
                         running=len(running), started=started))
    return rows


def render(rows, root=""):
    lines = [f"batch monitor — {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
             f"  (root: {root})", ""]
    hdr = f"{'exp':<28}{'total':>6}{'done':>6}{'run':>5}{'fail':>5}  started"
    lines.append(hdr); lines.append("-" * len(hdr))
    all_done = True
    for r in rows:
        complete = r["total"] > 0 and (r["done"] + r["failed"]) >= r["total"]
        all_done = all_done and complete
        lines.append(f"{r['exp']:<28}{r['total']:>6}{r['done']:>6}{r['running']:>5}"
                     f"{r['failed']:>5}  {r['started'][:19]}"
                     f"{'  COMPLETE' if complete else ''}")
    if not rows:
        lines.append("(no experiments yet)")
        all_done = False
    lines.append("")
    lines.append("live log: tail -f logs/<exp>_*.log   overview: bash scripts/status.sh   per-run logs: <exp_root>/logs/<run>.log")
    return "\n".join(lines), all_done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="print one table and exit")
    args = ap.parse_args(argv)
    while True:
        rows = collect(args.root)
        table, all_done = render(rows, args.root)
        if args.once:
            print(table)
            return 0 if all_done else 2
        print("\033[2J\033[H" + table, flush=True)
        if all_done:
            print("all experiments complete.")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
