#!/usr/bin/env python3
"""Batch-2 runner: ONE python process per experiment, run inside a screen
session, with an internal worker pool. No bash sleep/poll loops.

    python experiments/run_batch.py --list-exps
    python experiments/run_batch.py --print-skies        # used by simulate_data.sh
    python experiments/run_batch.py --exp exp_dr --dry    # print the plan only
    python experiments/run_batch.py --exp exp_dr          # run (resumes: skips done runs)

Each job is a subprocess of run_clean.py with its own thread allocation
(--nthreads NTHREADS plus OMP/NUMBA env), so size the machine as
MAXJOBS x NTHREADS ~= cores. Per-run log: <exp_root>/logs/<run>.log.
Live progress: <exp_root>/progress.json (atomic writes) — watch it with
`screen -r b2_wait` (experiments/wait_batch.py) or `tail -f` the logs.
Final: <exp_root>/summary.csv aggregated from each run's metrics.json.

Plans (skies x noise seeds x schemes) are defined below — single source of
truth shared with simulate_data.sh via --print-skies.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]

# name:n_point:n_gauss:n_diffuse:dr:peak_flux:sigma:sky_seed   (space-separated when printed)
SKIES = [
    "b2_S6_gauss_dr1e5_s42:0:12:0:1e5:1.0:1e-4:42",
    "b2_S8mix1000_dr1e2_s42:700:250:50:1e2:1.0:1e-4:42",
    "b2_S8mix1000_dr1e3_s42:700:250:50:1e3:1.0:1e-4:42",
    "b2_S8mix1000_dr1e4_s42:700:250:50:1e4:1.0:1e-4:42",
    "b2_S8mix1000_dr1e5_s42:700:250:50:1e5:1.0:1e-4:42",
    "b2_S8mix1000_dr1e6_s42:700:250:50:1e6:1.0:1e-4:42",
    "b2_S8mix1000_dr1e4_s43:700:250:50:1e4:1.0:1e-4:43",
    "b2_S8mix1000_dr1e4_s44:700:250:50:1e4:1.0:1e-4:44",
    "b2_S8mix1000_dr1e4_s42_sig1e-2:700:250:50:1e4:1.0:1e-2:42",
    "b2_S8mix1000_dr1e4_s42_sig1e-3:700:250:50:1e4:1.0:1e-3:42",
    "b2_S8mix1000_dr1e4_s42_sig1e-6:700:250:50:1e4:1.0:1e-6:42",
]

# name -> (q_vis, q_dirty, q_psf, q_model, rescale_max)
SCHEMES = {
    "f64":      ("float64",  "float64",  "float64",  "float64",  65503),
    "allf16":   ("float16",  "float16",  "float16",  "float16",  65503),
    "allbf16":  ("bfloat16", "bfloat16", "bfloat16", "bfloat16", 65503),
    "modelf16": ("float64",  "float64",  "float64",  "float16",  65503),
    "visf16":   ("float16",  "float64",  "float64",  "float64",  65503),
    "dirtyf16": ("float64",  "float16",  "float64",  "float64",  65503),
}

# exp name -> (sky-name prefixes, noise seeds)
PLANS = {
    "exp_dr": (["b2_S6_gauss_dr1e5_s42"] + [f"b2_S8mix1000_dr1e{d}_s42" for d in (2, 3, 4, 5, 6)],
               ["101", "102"]),
    "exp_noise": (["b2_S8mix1000_dr1e4_s42_sig1e-2", "b2_S8mix1000_dr1e4_s42_sig1e-3",
                   "b2_S8mix1000_dr1e4_s42_sig1e-6"], ["101", "102"]),
    "exp_seeds": (["b2_S8mix1000_dr1e4_s42", "b2_S8mix1000_dr1e4_s43",
                   "b2_S8mix1000_dr1e4_s44"], ["103"]),
}

CLEAN_DEFAULTS = dict(gamma="0.1", pf="1e-4", maxiter="10000", loop_max="10")


def spec_name(spec):
    return spec.split(":", 1)[0]


def dataset_name(sky_prefix, noise_seed):
    return f"{sky_prefix}_n{noise_seed}"


def scheme_args(name):
    qv, qd, qp, qm, rmax = SCHEMES[name]
    return ["--q-vis", qv, "--q-dirty", qd, "--q-psf", qp, "--q-model", qm,
            "--rescale-max", str(rmax)]


def build_jobs(exp, dataset_root, exp_root, schemes=None):
    """Plan -> job dicts. Pure (no I/O except existence checks for resume)."""
    prefixes, seeds = PLANS[exp]
    schemes = list(schemes or SCHEMES)
    jobs = []
    for prefix in prefixes:
        for nseed in seeds:
            ds = Path(dataset_root) / dataset_name(prefix, nseed)
            for scheme in schemes:
                run = Path(exp_root) / f"{dataset_name(prefix, nseed)}_{scheme}"
                jobs.append(dict(exp=exp, dataset=str(ds), run=str(run),
                                 run_name=run.name, scheme=scheme,
                                 done=(run / "results.npz").exists()))
    return jobs


def _atomic_write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _default_roots():
    dataset_root = os.environ.get("DATASET_PATH")
    batch_root = os.environ.get("BATCH_ROOT")
    if batch_root is None:
        results = os.environ.get("RESULTS_PATH") or str(CODE_ROOT / "results")
        batch_root = str(Path(results) / "batch2")
    if dataset_root is None:
        d = CODE_ROOT
        while d != d.parent and not (d / "data" / "uvcov").is_dir() and not (d / "lib").is_dir():
            d = d.parent
        dataset_root = str((d if (d / "data" / "uvcov").is_dir() or (d / "lib").is_dir()
                           else CODE_ROOT) / "data" / "simulated" / "v2")
    return dataset_root, batch_root


def run_one(job, nthreads, log_dir, progress, lock):
    """Run a single CLEAN job as a subprocess with its own thread allocation."""
    name = job["run_name"]
    run_dir = Path(job["run"]); run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{name}.log"
    cmd = [sys.executable, str(CODE_ROOT / "experiments" / "run_clean.py"),
           "--dataset", job["dataset"], "--out", str(run_dir),
           "--gamma", CLEAN_DEFAULTS["gamma"], "--pf", CLEAN_DEFAULTS["pf"],
           "--maxiter", CLEAN_DEFAULTS["maxiter"], "--loop-max", CLEAN_DEFAULTS["loop_max"],
           "--nthreads", str(nthreads)] + scheme_args(job["scheme"])
    env = dict(os.environ, OMP_NUM_THREADS=str(nthreads),
               NUMBA_NUM_THREADS=str(nthreads), MKL_NUM_THREADS=str(nthreads))
    t0 = time.time()
    with lock:
        progress["running"].append(name)
        _atomic_write_json(Path(job["run"]).parent / "progress.json", progress)
    with open(log_path, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    wall = time.time() - t0
    metrics = {}
    mp = run_dir / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    with lock:
        progress["running"].remove(name)
        if rc != 0:
            progress["failed"].append(name)
        else:
            progress["done"].append(name)
        progress["walls"][name] = wall
        progress["metrics"][name] = metrics
        progress["updated"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(Path(job["run"]).parent / "progress.json", progress)
    return rc, wall, metrics


def write_summary(exp_root, jobs, progress):
    rows = []
    for job in jobs:
        m = progress["metrics"].get(job["run_name"], {})
        rows.append(dict(run=job["run_name"], scheme=job["scheme"],
                         dataset=Path(job["dataset"]).name,
                         status="done" if job["run_name"] in progress["done"] else "failed",
                         wall_s=round(progress["walls"].get(job["run_name"], 0.0), 1),
                         converged=m.get("converged_reason", ""),
                         n_components=m.get("n_components", ""),
                         achieved_dr=m.get("achieved_dr", ""),
                         residual_mad=m.get("residual_mad", ""),
                         residual_rms=m.get("residual_rms", "")))
    out = Path(exp_root) / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        w.writeheader(); w.writerows(rows)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=list(PLANS) + ["all"], default="all")
    ap.add_argument("--list-exps", action="store_true")
    ap.add_argument("--print-skies", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--maxjobs", type=int, default=int(os.environ.get("MAXJOBS", "8")))
    ap.add_argument("--nthreads", type=int, default=int(os.environ.get("NTHREADS", "4")))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--batch-root", default=None)
    args = ap.parse_args(argv)

    if args.list_exps:
        print(" ".join(PLANS)); return 0
    if args.print_skies:
        print(" ".join(SKIES)); return 0

    dataset_root, batch_root = _default_roots()
    dataset_root = args.dataset_root or dataset_root
    batch_root = args.batch_root or batch_root
    exps = list(PLANS) if args.exp == "all" else [args.exp]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp in exps:
        exp_root = Path(batch_root) / f"{exp}_{ts}"
        log_dir = exp_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
        jobs = build_jobs(exp, dataset_root, exp_root)
        missing = [j for j in jobs if not (Path(j["dataset"]) / "obs.npz").exists()]
        if missing:
            print(f"[{exp}] missing datasets: {sorted({Path(j['dataset']).name for j in missing})}"
                  f" — run simulate_data.sh first", file=sys.stderr)
            return 1
        todo = [j for j in jobs if not j["done"]]
        print(f"[{exp}] root={exp_root}  jobs={len(jobs)}  already done={len(jobs)-len(todo)}  to run={len(todo)}"
              f"  maxjobs={args.maxjobs} nthreads={args.nthreads}")
        if args.dry:
            for j in todo:
                print(f"  DRY {j['run_name']}  dataset={Path(j['dataset']).name}  {' '.join(scheme_args(j['scheme']))}")
            continue
        progress = dict(exp=exp, ts=ts, total=len(jobs), done=[j["run_name"] for j in jobs if j["done"]],
                        running=[], failed=[], walls={}, metrics={},
                        started=datetime.now(timezone.utc).isoformat(), updated="")
        _atomic_write_json(exp_root / "progress.json", progress)
        lock = threading.Lock()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.maxjobs) as pool:
            futs = {pool.submit(run_one, j, args.nthreads, log_dir, progress, lock): j for j in todo}
            for fut in futs:
                rc, wall, m = fut.result()
                name = futs[fut]["run_name"]
                print(f"[{exp}] ({len(progress['done'])+len(progress['failed'])}/{len(jobs)}) "
                      f"{'ok ' if rc == 0 else 'FAIL'} {name} ({wall:.0f}s) "
                      f"resMAD={m.get('residual_mad', float('nan')):.3g} achDR={m.get('achieved_dr', float('nan')):.3g}",
                      flush=True)
        summary = write_summary(exp_root, jobs, progress)
        print(f"[{exp}] complete in {time.time()-t0:.0f}s — summary: {summary} "
              f"(failed: {len(progress['failed'])})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
