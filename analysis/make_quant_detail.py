#!/usr/bin/env python3
"""Detail figures for the deck: all-f16 / all-bf16 vs f64 on a given sky
(model / restored / residual panels), and per-stage (vis/dirty/psf/model)
dtype bars across the sky ladder.

Usage: python analysis/make_quant_detail.py [results_root ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.plot as P
from analysis.load import load_results, to_dataframe
from analysis.compare_schemes import scheme_label

STAGE_KEY = {"v": "vis", "d": "dirty", "p": "psf", "m": "model"}


def main():
    roots = sys.argv[1:] or [str(Path(__file__).resolve().parents[1] / "results")]
    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
    results = []
    for r in roots:
        results += load_results(r)
    df = to_dataframe(results)
    df["scheme"] = df["quantization"].map(scheme_label)
    df["sky"] = df["dataset_name"].str.replace(r"_n[0-9]+$", "", regex=True)

    def find(sky, seed, scheme):
        sub = df[(df.sky == sky) & (df.noise_seed == seed) & (df.scheme == scheme)]
        return sub.iloc[0]["run_dir"] if len(sub) else None

    # ---- detail figure: f64 vs all-f16 vs all-bf16 on the complex field ----
    for sky in ["S3_gauss_dr1000", "S2_mixed_dr100"]:
        runs = {lab: find(sky, 101, sch) for lab, sch in
                [("f64", "f64-baseline"),
                 ("all-f16", "vf16-df16-pf16-mf16"),
                 ("all-bf16", "vbf16-dbf16-pbf16-mbf16")]}
        if any(v is None for v in runs.values()):
            print(f"skip {sky}: missing runs {runs}")
            continue
        arrs = {lab: np.load(rd + "/results.npz") for lab, rd in runs.items()}
        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        for j, (lab, a) in enumerate(arrs.items()):
            P.panel(axes[0, j], a["model"], f"{lab} — model (Jy)", scale="log")
            beam = None
            P.panel(axes[1, j], a["restored"], f"{lab} — restored (Jy/beam)",
                    scale="asinh", contours=a["truth"])
            P.panel(axes[2, j], a["residual"], f"{lab} — residual (Jy/beam)",
                    cmap=P.RESID_CMAP, scale="symlog")
        fig.suptitle(f"{sky}: f64 baseline vs all-f16 vs all-bf16", fontsize=13)
        fig.tight_layout()
        P._save(fig, f"detail_{sky}_lowprec.png")
        # per-scheme 2-column detail (f64 vs scheme)
        for lab, sch in [("all-f16", "vf16-df16-pf16-mf16"),
                         ("all-bf16", "vbf16-dbf16-pbf16-mbf16")]:
            if runs.get(lab) is None:
                continue
            a2 = np.load(runs[lab] + "/results.npz")
            f2, ax2 = plt.subplots(3, 2, figsize=(9, 12))
            for j, (lab2, a) in enumerate([("f64", arrs["f64"]), (lab, a2)]):
                P.panel(ax2[0, j], a["model"], f"{lab2} — model (Jy)", scale="log")
                P.panel(ax2[1, j], a["restored"], f"{lab2} — restored (Jy/beam)",
                        scale="asinh", contours=a["truth"])
                P.panel(ax2[2, j], a["residual"], f"{lab2} — residual (Jy/beam)",
                        cmap=P.RESID_CMAP, scale="symlog")
            f2.suptitle(f"{sky}: f64 vs {lab}", fontsize=13)
            f2.tight_layout()
            P._save(f2, f"detail_{sky}_{lab}.png")

    # ---- per-stage frames: stage in {v,d,p,m} x dtype in {f32,f16,bf16} ----
    import pandas as pd
    qcsv = P.OUTPUT_DIR / "quant_summary.csv"
    q = pd.read_csv(qcsv)
    q = q[q.scheme != "f64-baseline"].drop_duplicates(["sky", "scheme"])
    # scheme label "vf16-df64-pf64-mf64": the differing stage is the one != f64
    def parse(label):
        diff = [(part[0], part[1:]) for part in label.split("-") if part[1:] != "f64"]
        return diff[0] if len(diff) == 1 else (None, None)  # single-stage schemes only
    parsed = q["scheme"].map(parse)
    q["stage"] = parsed.map(lambda t: t[0])
    q["dtype"] = parsed.map(lambda t: t[1])
    skies = ["S0_point1", "S1_points_dr10", "S2_mixed_dr100", "S3_gauss_dr1000"]
    for stage in ["v", "d", "p", "m"]:
        sub = q[q.stage == stage]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, col, lab, log in [(axes[0], "ratio_residual_mad", "resMAD / f64", True),
                                  (axes[1], "ratio_achieved_dr", "achDR / f64", True)]:
            for dt, color in [("f32", "tab:blue"), ("f16", "tab:orange"), ("bf16", "tab:red")]:
                xs, ys = [], []
                for i, sky in enumerate(skies):
                    r = sub[(sub.sky == sky) & (sub.dtype == dt)]
                    if len(r):
                        xs.append(i)
                        ys.append(float(r[col].iloc[0]))
                ax.bar([x + ({"f32": -0.25, "f16": 0.0, "bf16": 0.25}[dt]) for x in xs],
                       ys, width=0.25, color=color, label=dt)
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_xticks(range(len(skies)))
            ax.set_xticklabels([s.replace("_point", " pt").replace("_points", " pts")
                                .replace("_mixed", " mix").replace("_gauss", " gauss")
                                for s in skies], fontsize=8, rotation=15, ha="right")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3, axis="y")
            ax.legend()
        axes[0].set_title(f"stage = {STAGE_KEY[stage]} ({stage}...)", fontsize=11)
        fig.suptitle(f"Single-stage quantization of the {STAGE_KEY[stage]} — "
                     f"f32 / f16 / bf16 across the sky ladder", fontsize=12)
        fig.tight_layout()
        P._save(fig, f"stage_{stage}.png")
    print("detail figures ->", P.OUTPUT_DIR)


import matplotlib.pyplot as plt  # noqa: E402  (used inside main)

if __name__ == "__main__":
    main()
