#!/usr/bin/env python3
"""Scheme-comparison image grids for a v3 sweep sky: rows = truth/dirty/model/
residual, columns = schemes (f64, allf16, allbf16, modelf16, allf16raw).

Usage: python analysis/make_v3_detail.py results/sweep_v3b_* --sky S7_point_peak1e6
       python analysis/make_v3_detail.py results/sweep_v3b_* --all
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.plot as P
from analysis.load import load_results

SCHEME_ORDER = ["f64", "allf16", "allbf16", "modelf16", "allf16raw"]
ROWS = [("truth", "truth (Jy/pix)", "log", P.SKY_CMAP),
        ("dirty", "dirty (Jy/beam)", "asinh", P.SKY_CMAP),
        ("model", "CLEAN model (Jy)", "log", P.SKY_CMAP),
        ("residual", "residual (Jy/beam)", "symlog", P.RESID_CMAP)]


def scheme_of(r):
    dn = r["params"]["dataset_name"]
    return Path(r["run_dir"]).name[len(dn) + 1:]


def grid(root, sky, outdir):
    runs = [r for r in load_results(root)
            if r["params"]["dataset_name"].startswith(sky + "_n") and "model" in r]
    runs.sort(key=lambda r: SCHEME_ORDER.index(scheme_of(r)) if scheme_of(r) in SCHEME_ORDER
              else 99)
    if not runs:
        print(f"no runs for {sky} in {root}")
        return None
    import matplotlib.pyplot as plt
    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(outdir)
    fig, axes = plt.subplots(len(ROWS), len(runs),
                             figsize=(3.4 * len(runs), 3.1 * len(ROWS)), squeeze=False)
    for j, r in enumerate(runs):
        for i, (key, title, scale, cmap) in enumerate(ROWS):
            a = np.asarray(r[key], np.float64)
            a[~np.isfinite(a)] = np.nan          # overflowed runs render as masked
            P.panel(axes[i, j], a, title if j == 0 else "", cmap=cmap, scale=scale,
                    colorbar=True)
            if i == 0:
                axes[i, j].set_title(scheme_of(r), fontsize=11)
    fig.suptitle(f"{sky}: images by scheme (v3 sweep)", fontsize=13)
    fig.tight_layout()
    return P._save(fig, f"detail_{sky}_schemes.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--sky", default=None)
    ap.add_argument("--all", action="store_true", help="grid for every sky found")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    args = ap.parse_args()
    for root in args.roots:
        if args.all:
            skies = sorted({r["params"]["dataset_name"].rsplit("_n", 1)[0]
                            for r in load_results(root)
                            if "model" in r and "params" in r})
            for sky in skies:
                grid(root, sky, args.out)
        elif args.sky:
            grid(root, args.sky, args.out)


if __name__ == "__main__":
    main()
