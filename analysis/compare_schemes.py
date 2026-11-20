#!/usr/bin/env python3
"""Interpretative layer for the quantization studies: scheme vs f64 baseline.

Loads one (or more) quant study dirs + the f64 baseline dir, joins on
(dataset, noise_seed), computes deltas of each metric against the f64
reference, and writes:
  figures/quant_summary.csv     one row per (sky, seed, scheme) with deltas
  figures/quant_<metric>_<sky>.png   bar charts of delta vs scheme

Works incrementally: point it at a study dir while runs are still landing.

Usage: python analysis/compare_schemes.py results/quant_schemes_* results/f64_baseline_*
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.plot as P
from analysis.load import load_results, to_dataframe


def scheme_label(q):
    """dict(vis=...,dirty=...,psf=...,model=...) -> short label."""
    short = {"float64": "f64", "float32": "f32", "float16": "f16", "bfloat16": "bf16"}
    stages = [f"{k[0]}{short[v]}" for k, v in
              sorted(q.items(), key=lambda kv: ["vis", "dirty", "psf", "model"].index(kv[0]))]
    return "-".join(stages) if any(v != "float64" for v in q.values()) else "f64-baseline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="results roots (quant studies + f64 baseline)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    args = ap.parse_args()

    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(args.out)

    results = []
    for r in args.roots:
        results += load_results(r)
    df = to_dataframe(results)
    if df.empty:
        print("no results")
        return
    df["scheme"] = df["quantization"].map(scheme_label)
    df["sky"] = df["dataset_name"].str.replace(r"_n[0-9]+$", "", regex=True)

    # f64 reference per (sky, noise_seed): prefer the explicit baseline run in
    # the quant study; fall back to the f64_baseline study rows
    ref = (df[df["scheme"] == "f64-baseline"]
           .drop_duplicates(["sky", "noise_seed"])
           .set_index(["sky", "noise_seed"]))
    if ref.empty:
        print("no f64-baseline rows found in the given roots")
        return

    metrics = ["cross_correlation", "flux_conservation", "residual_mad", "achieved_dr"]
    rows = []
    for _, r in df.iterrows():
        key = (r["sky"], r["noise_seed"])
        if key not in ref.index:
            continue
        b = ref.loc[key]
        row = dict(sky=r["sky"], noise_seed=r["noise_seed"], scheme=r["scheme"],
                   dataset_name=r["dataset_name"])
        for m in metrics:
            row[m] = r[m]
            row[f"d_{m}"] = r[m] - b[m]
            row[f"ratio_{m}"] = r[m] / b[m] if b[m] else np.nan
        rows.append(row)
    out = P.OUTPUT_DIR
    import pandas as pd
    qdf = pd.DataFrame(rows).sort_values(["sky", "scheme"])
    qdf.to_csv(out / "quant_summary.csv", index=False)
    print(f"{len(qdf)} rows -> {out/'quant_summary.csv'}")

    # bar charts: delta vs scheme, one figure per (metric, sky)
    for sky in sorted(qdf["sky"].unique()):
        sub = qdf[qdf.sky == sky]
        for m, lab, log in [("cross_correlation", "delta cc", False),
                            ("flux_conservation", "delta flux", False),
                            ("residual_mad", "residual MAD ratio (scheme/f64)", True),
                            ("achieved_dr", "achieved DR ratio (scheme/f64)", True)]:
            fig, ax = P.plt.subplots(figsize=(10, 4.5))
            x = np.arange(len(sub))
            vals = (sub[f"ratio_{m}"] if m in ("residual_mad", "achieved_dr") else sub[f"d_{m}"])
            ax.bar(x, vals, color="steelblue")
            if m in ("residual_mad", "achieved_dr"):
                ax.axhline(1.0, color="red", lw=1, ls="--")
            else:
                ax.axhline(0.0, color="red", lw=1, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["scheme"], rotation=45, ha="right", fontsize=7)
            if log:
                ax.set_yscale("log")
            ax.set_ylabel(lab)
            ax.set_title(f"{sky}: {lab} vs f64 baseline")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            P._save(fig, f"quant_{m}_{sky}.png")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
