#!/usr/bin/env python3
"""Run one imaging experiment (CLEAN with optional per-stage quantization)
and write a self-describing result.

Usage:
    python experiments/run_clean.py \
        --dataset $STAGE_ROOT/data/simulated/v2/S0_point1 \
        --out results/f64_baseline/S0_point1_n101 \
        --q-vis float64 --q-dirty float64 --q-psf float64 --q-model float64

Writes <out>/results.npz (arrays + all params + metrics as attrs) and
<out>/info.yaml. All parameters come from the CLI — nothing hardcoded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import operators, pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset dir with obs.npz + truth.fits")
    ap.add_argument("--out", required=True, help="output run dir")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--pf", type=float, default=1e-4)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--loop-max", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--q-vis", default="float64", choices=["float64", "float32", "float16", "bfloat16"])
    ap.add_argument("--q-dirty", default="float64", choices=["float64", "float32", "float16", "bfloat16"])
    ap.add_argument("--q-psf", default="float64", choices=["float64", "float32", "float16", "bfloat16"])
    ap.add_argument("--q-model", default="float64", choices=["float64", "float32", "float16", "bfloat16"])
    ap.add_argument("--rescale-max", type=float, default=65503.0,
                    help="f16 rescale ceiling; 0 disables rescaling (raw cast, overflow -> inf)")
    ap.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True,
                    help="record per-iteration log2 range probes to probes.npz")
    args = ap.parse_args()

    d = Path(args.dataset)
    obs = np.load(d / "obs.npz")
    truth = np.squeeze(fits.getdata(d / "truth.fits")).astype(np.float64)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    h = fits.getheader(d / "truth.fits")
    npix = int(h["NAXIS1"])
    cell = abs(float(h["CDELT1"]))
    geom = operators.Geometry(npix, npix, np.radians(cell), np.radians(cell))

    uvw, freq, vis, wgt = obs["uvw"], obs["freq"], obs["vis"], obs["weight"]

    t0 = time.time()
    out = pipeline.run_pipeline(
        vis, uvw, freq, wgt, geom,
        gamma=args.gamma, pf=args.pf, maxiter=args.maxiter, loop_max=args.loop_max,
        eps=args.eps, nthreads=args.nthreads,
        q_vis=args.q_vis, q_dirty=args.q_dirty, q_psf=args.q_psf, q_model=args.q_model,
        rescale_max=args.rescale_max if args.rescale_max > 0 else None,
        probe=args.probe)
    wall = time.time() - t0

    m = pipeline.compute_metrics(out["model"], out["residual"], out["psf"], truth)

    params = dict(dataset=str(d), dataset_name=d.name, gamma=args.gamma, pf=args.pf,
                  maxiter=args.maxiter, loop_max=args.loop_max, eps=args.eps,
                  nthreads=args.nthreads,
                  quantization=dict(vis=args.q_vis, dirty=args.q_dirty,
                                    psf=args.q_psf, model=args.q_model),
                  rescale_max=args.rescale_max, probe=bool(args.probe), npix=npix, cell_deg=cell,
                  nvis=int(vis.size), sigma_jy=meta.get("sigma_jy"),
                  field_dr=meta.get("field", {}).get("realized_dr"),
                  dataset_seed=meta.get("seed"), noise_seed=meta.get("noise_seed"))
    metrics = dict(m, n_major_cycles=out["n_major_cycles"],
                   converged_reason=out["converged_reason"],
                   n_components=out["n_components"],
                   psf_peak=out["psf_peak"],
                   scales={k: float(v) for k, v in out["scales"].items()},
                   wall_time_s=wall, **{f"time_{k}": v for k, v in out["timing"].items()})

    # restored image for display (display frame)
    bmaj, bmin, pa = operators.synth_beam_from_psf(out["psf"])
    kern = operators.make_beam_kernel(bmaj, bmin, pa)
    restored = operators.restore(operators.to_display(out["model"]), kern)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "results.npz",
        truth=truth.astype(np.float32),
        dirty=(operators.to_display(out["dirty"]) / out["psf_peak"]).astype(np.float32),   # Jy/beam
        model=operators.to_display(out["model"]).astype(np.float32),                       # Jy
        restored=restored.astype(np.float32),                                               # Jy/beam
        residual=operators.to_display(out["residual"]).astype(np.float32),                 # Jy/beam
        psf_lobe=(_psf_lobe(out["psf"]) / out["psf_peak"]).astype(np.float32))             # peak 1
    if args.probe and out.get("probes") is not None:
        np.savez_compressed(outdir / "probes.npz", **out["probes"].to_arrays())
    # params + metrics as sidecar files: the run dir is fully self-describing
    (outdir / "params.json").write_text(json.dumps(params, indent=2))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "info.yaml").write_text(yaml.safe_dump({"params": params, "metrics": metrics}))
    print(json.dumps(dict(run=str(outdir), **{k: metrics[k] for k in
          ("cross_correlation", "flux_conservation", "residual_mad", "achieved_dr",
           "n_major_cycles", "converged_reason", "wall_time_s")})), flush=True)


def _psf_lobe(psf, half=96):
    iy, ix = np.unravel_index(np.argmax(psf), psf.shape)
    return psf[max(iy - half, 0):iy + half + 1, max(ix - half, 0):ix + half + 1]


if __name__ == "__main__":
    main()
