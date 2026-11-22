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
    ap.add_argument("--probe-bins", type=int, default=None,
                    help="log2 histogram bin count (default 144 = 1 bin/octave)")
    ap.add_argument("--probe-snap", type=int, default=16,
                    help="block-mean downsample factor for per-iteration image "
                         "snapshots in probes.npz (0 disables)")
    args = ap.parse_args()
    run(dataset=args.dataset, out=args.out, gamma=args.gamma, pf=args.pf,
        maxiter=args.maxiter, loop_max=args.loop_max, eps=args.eps,
        nthreads=args.nthreads, q_vis=args.q_vis, q_dirty=args.q_dirty,
        q_psf=args.q_psf, q_model=args.q_model, rescale_max=args.rescale_max,
        probe=args.probe, probe_bins=args.probe_bins, probe_snap=args.probe_snap)


def run(dataset, out, gamma=0.1, pf=1e-4, maxiter=10000, loop_max=10, eps=1e-5,
        nthreads=8, q_vis="float64", q_dirty="float64", q_psf="float64",
        q_model="float64", rescale_max=65503.0, probe=True, probe_bins=None,
        probe_snap=16):
    """Run the full pipeline on a dataset dir and write a self-describing run
    dir (results.npz + probes.npz + params.json + metrics.json + info.yaml).
    Callable in-process (used by run_batch.py) or via the CLI."""
    d = Path(dataset)
    obs = np.load(d / "obs.npz")
    truth = np.squeeze(fits.getdata(d / "truth.fits")).astype(np.float64)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    h = fits.getheader(d / "truth.fits")
    npix = int(h["NAXIS1"])
    cell = abs(float(h["CDELT1"]))
    geom = operators.Geometry(npix, npix, np.radians(cell), np.radians(cell))

    uvw, freq, vis, wgt = obs["uvw"], obs["freq"], obs["vis"], obs["weight"]

    t0 = time.time()
    res = pipeline.run_pipeline(
        vis, uvw, freq, wgt, geom,
        gamma=gamma, pf=pf, maxiter=maxiter, loop_max=loop_max,
        eps=eps, nthreads=nthreads,
        q_vis=q_vis, q_dirty=q_dirty, q_psf=q_psf, q_model=q_model,
        rescale_max=rescale_max if rescale_max > 0 else None,
        probe=probe, probe_bins=probe_bins, probe_snap=probe_snap)
    wall = time.time() - t0

    # A raw-cast (rescale_max=0) f16 run can overflow to inf/NaN; the pipeline
    # itself survives that, but the beam/metrics path may not. Save the run
    # anyway with NaN metrics + an error note instead of losing it (the legacy
    # archive lost exactly these runs).
    err = None
    try:
        m = pipeline.compute_metrics(res["model"], res["residual"], res["psf"], truth)
    except Exception as e:  # noqa: BLE001 - any failure here must not lose the run
        err = f"{type(e).__name__}: {e}"
        m = dict(cross_correlation=float("nan"), flux_conservation=float("nan"),
                 residual_rms=float("nan"), residual_mad=float("nan"),
                 achieved_dr=float("nan"), snr_db=float("nan"),
                 beam_bmaj_px=float("nan"), beam_bmin_px=float("nan"),
                 beam_area_px=float("nan"), restored_peak=float("nan"))

    params = dict(dataset=str(d), dataset_name=d.name, gamma=gamma, pf=pf,
                  maxiter=maxiter, loop_max=loop_max, eps=eps,
                  nthreads=nthreads,
                  quantization=dict(vis=q_vis, dirty=q_dirty,
                                    psf=q_psf, model=q_model),
                  rescale_max=rescale_max, probe=bool(probe),
                  probe_bins=probe_bins, probe_snap=probe_snap,
                  npix=npix, cell_deg=cell,
                  nvis=int(vis.size), sigma_jy=meta.get("sigma_jy"),
                  field_dr=meta.get("field", {}).get("realized_dr"),
                  dataset_seed=meta.get("seed"), noise_seed=meta.get("noise_seed"))
    metrics = dict(m, n_major_cycles=res["n_major_cycles"],
                   converged_reason=res["converged_reason"],
                   n_components=res["n_components"],
                   psf_peak=res["psf_peak"],
                   scales={k: float(v) for k, v in res["scales"].items()},
                   wall_time_s=wall, **{f"time_{k}": v for k, v in res["timing"].items()})
    if err is not None:
        metrics["error"] = err

    # restored image for display (display frame)
    try:
        bmaj, bmin, pa = operators.synth_beam_from_psf(res["psf"])
        kern = operators.make_beam_kernel(bmaj, bmin, pa)
        restored = operators.restore(operators.to_display(res["model"]), kern)
    except Exception:  # noqa: BLE001
        restored = np.full(res["model"].shape, np.nan, dtype=np.float32)

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "results.npz",
        truth=truth.astype(np.float32),
        dirty=(operators.to_display(res["dirty"]) / res["psf_peak"]).astype(np.float32),   # Jy/beam
        model=operators.to_display(res["model"]).astype(np.float32),                       # Jy
        restored=restored.astype(np.float32),                                               # Jy/beam
        residual=operators.to_display(res["residual"]).astype(np.float32),                 # Jy/beam
        psf_lobe=(_psf_lobe(res["psf"]) / res["psf_peak"]).astype(np.float32))             # peak 1
    if probe and res.get("probes") is not None:
        np.savez_compressed(outdir / "probes.npz", **res["probes"].to_arrays())
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
