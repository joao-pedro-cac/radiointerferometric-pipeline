#!/usr/bin/env python3
"""Batch experiment runner — deliberately simple: one foreground process per
experiment, jobs run SEQUENTIALLY in-process (run_clean.run). Process launching
is done ONLY by the bash launcher (run_experiment.sh) — python never spawns
python.

    python experiments/run_batch.py --list-exps
    python experiments/run_batch.py --print-skies [--exps exp3_mc ...]
    python experiments/run_batch.py --exp exp_dr --dry     # print the plan
    python experiments/run_batch.py --exp exp_dr           # run (resume-safe)

Plans (skies x noise seeds x schemes) are defined below — single source of
truth shared with simulate_data.sh via --print-skies. Per-run progress:
<exp_root>/progress.json (atomic); final: <exp_root>/summary.csv. Everything
the process prints lands in the experiment log (nohup redirect), live.
"""
import argparse
import csv
import json
import os
import sys
import time
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
    # batch 3 — Monte Carlo on the field with a batch-2 quantization signal
    "b3_s6gauss_dr1e5_s51:0:12:0:1e5:1.0:1e-4:51",
    "b3_s6gauss_dr1e5_s52:0:12:0:1e5:1.0:1e-4:52",
    "b3_s6gauss_dr1e5_s53:0:12:0:1e5:1.0:1e-4:53",
    # batch 3 — peak-flux range limit (where the f16 range bites)
    "b3_point_peak1e5:1:0:0:1:1e5:1e-4:71",
    "b3_point_peak1e6:1:0:0:1:1e6:1e-4:71",
    # batch 3 — deep-noise fields that reach the noise floor
    "b3_points_dr10_s81:8:0:0:10:1.0:1e-4:81",
    "b3_points_dr10_s81_sig1e-6:8:0:0:10:1.0:1e-6:81",
    "b3_gauss_dr1e3_s81:0:12:0:1e3:1.0:1e-4:81",
    "b3_gauss_dr1e3_s81_sig1e-6:0:12:0:1e3:1.0:1e-6:81",
]

# name -> (q_vis, q_dirty, q_psf, q_model, rescale_max)
SCHEMES = {
    "f64":      ("float64",  "float64",  "float64",  "float64",  65503),
    "allf32":   ("float32",  "float32",  "float32",  "float32",  65503),
    "allf16":   ("float16",  "float16",  "float16",  "float16",  65503),
    "allbf16":  ("bfloat16", "bfloat16", "bfloat16", "bfloat16", 65503),
    "modelf16": ("float64",  "float64",  "float64",  "float16",  65503),
    "visf16":   ("float16",  "float64",  "float64",  "float64",  65503),
    "dirtyf16": ("float64",  "float16",  "float64",  "float64",  65503),
}

# batch-2 scheme set is pinned (allf32 only enters via batch-3 plans)
B2_SCHEMES = ["f64", "allf16", "allbf16", "modelf16", "visf16", "dirtyf16"]

# exp name -> (sky-name prefixes, noise seeds, schemes or None = all)
PLANS = {
    "exp_dr": (["b2_S6_gauss_dr1e5_s42"] + [f"b2_S8mix1000_dr1e{d}_s42" for d in (2, 3, 4, 5, 6)],
               ["101", "102"], B2_SCHEMES),
    "exp_noise": (["b2_S8mix1000_dr1e4_s42_sig1e-2", "b2_S8mix1000_dr1e4_s42_sig1e-3",
                   "b2_S8mix1000_dr1e4_s42_sig1e-6"], ["101", "102"], B2_SCHEMES),
    "exp_seeds": (["b2_S8mix1000_dr1e4_s42", "b2_S8mix1000_dr1e4_s43",
                   "b2_S8mix1000_dr1e4_s44"], ["103"], B2_SCHEMES),
    # batch 3 — Monte Carlo (sky seed x noise seed) on the S6 field that showed
    # the all-bf16 signal; box plots over seeds in the report
    "exp3_mc": (["b3_s6gauss_dr1e5_s51", "b3_s6gauss_dr1e5_s52", "b3_s6gauss_dr1e5_s53"],
                ["201", "202"], ["f64", "allf16", "allbf16", "modelf16"]),
    # batch 3 — peak-flux range limit: where f16 range (ulp at pinned 65503) bites
    "exp3_range": (["b3_point_peak1e5", "b3_point_peak1e6"],
                   ["301"], ["f64", "allf16", "allbf16", "modelf16"]),
    # batch 3 — deep noise / low DR: fields that reach the noise floor;
    # f32 visibility + f16/bf16 floor vs k
    "exp3_floor": (["b3_points_dr10_s81", "b3_points_dr10_s81_sig1e-6",
                    "b3_gauss_dr1e3_s81", "b3_gauss_dr1e3_s81_sig1e-6"],
                   ["401"], ["f64", "allf32", "allf16", "allbf16", "modelf16"]),
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
    prefixes, seeds, plan_schemes = PLANS[exp]
    schemes = list(schemes or plan_schemes or SCHEMES)
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


def run_one(job, nthreads, progress):
    """Run one job IN-PROCESS via run_clean.run (python never spawns python —
    process launching is the bash launcher's job). A crash is caught and
    recorded as a failed run with its traceback in the experiment log."""
    import traceback
    sys.path.insert(0, str(CODE_ROOT / "experiments"))
    import run_clean
    name = job["run_name"]
    qv, qd, qp, qm, rmax = SCHEMES[job["scheme"]]
    t0 = time.time()
    progress["running"].append(name)
    _atomic_write_json(Path(job["run"]).parent / "progress.json", progress)
    print(f"[{job['exp']}] start {name}", flush=True)
    err = None
    try:
        run_clean.run(dataset=job["dataset"], out=job["run"],
                      gamma=float(CLEAN_DEFAULTS["gamma"]), pf=float(CLEAN_DEFAULTS["pf"]),
                      maxiter=int(CLEAN_DEFAULTS["maxiter"]), loop_max=int(CLEAN_DEFAULTS["loop_max"]),
                      nthreads=nthreads, q_vis=qv, q_dirty=qd, q_psf=qp, q_model=qm,
                      rescale_max=float(rmax))
    except Exception:  # noqa: BLE001 - one bad run must not kill the batch
        err = traceback.format_exc()
        print(err, flush=True)
    wall = time.time() - t0
    metrics = {}
    mp = Path(job["run"]) / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    progress["running"].remove(name)
    progress["failed" if err else "done"].append(name)
    progress["walls"][name] = wall
    progress["metrics"][name] = metrics
    progress["errors"][name] = err or ""
    progress["updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(Path(job["run"]).parent / "progress.json", progress)
    return (1 if err else 0), wall, metrics


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
    ap.add_argument("--exp", nargs="+", choices=list(PLANS) + ["all"], default=["all"])
    ap.add_argument("--list-exps", action="store_true")
    ap.add_argument("--print-skies", action="store_true")
    ap.add_argument("--exps", nargs="*", default=None,
                    help="with --print-skies: only skies of these plans")
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--nthreads", type=int, default=int(os.environ.get("NTHREADS", "4")))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--batch-root", default=None)
    args = ap.parse_args(argv)

    if args.list_exps:
        print(" ".join(PLANS)); return 0
    if args.print_skies:
        if args.exps:
            wanted = {p for e in args.exps for p in PLANS[e][0]}
            print(" ".join(s for s in SKIES if spec_name(s) in wanted))
        else:
            print(" ".join(SKIES))
        return 0

    exps = list(PLANS) if args.exp == ["all"] else args.exp
    dataset_root, batch_root = _default_roots()
    dataset_root = args.dataset_root or dataset_root
    batch_root = args.batch_root or batch_root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for exp in exps:
        exp_root = Path(batch_root) / f"{exp}_{ts}"
        exp_root.mkdir(parents=True, exist_ok=True)
        # banner first — the log must show life immediately, even if the run
        # then fails (e.g. missing datasets)
        print(f"[{exp}] run_batch starting {ts}  python={sys.executable}", flush=True)
        print(f"[{exp}] exp_root={exp_root}  dataset_root={dataset_root}", flush=True)
        jobs = build_jobs(exp, dataset_root, exp_root)
        missing = [j for j in jobs if not (Path(j["dataset"]) / "obs.npz").exists()]
        if missing:
            print(f"[{exp}] missing datasets: {sorted({Path(j['dataset']).name for j in missing})}"
                  f" — run simulate_data.sh first", file=sys.stderr, flush=True)
            return 1
        todo = [j for j in jobs if not j["done"]]
        print(f"[{exp}] jobs={len(jobs)}  already done={len(jobs)-len(todo)}  to run={len(todo)}  "
              f"sequential, nthreads={args.nthreads}", flush=True)
        if args.dry:
            for j in todo:
                print(f"  DRY {j['run_name']}  dataset={Path(j['dataset']).name}  {' '.join(scheme_args(j['scheme']))}",
                      flush=True)
            continue
        progress = dict(exp=exp, ts=ts, total=len(jobs), done=[j["run_name"] for j in jobs if j["done"]],
                        running=[], failed=[], walls={}, metrics={}, errors={},
                        started=datetime.now(timezone.utc).isoformat(), updated="")
        _atomic_write_json(exp_root / "progress.json", progress)
        t0 = time.time()
        for j in todo:   # sequential on purpose — see run_one docstring
            rc, wall, m = run_one(j, args.nthreads, progress)
            name = j["run_name"]
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
