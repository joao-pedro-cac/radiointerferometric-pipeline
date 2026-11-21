#!/usr/bin/env python3
"""Batch-2 sweep figures: metric vs noise sigma / vs field DR / vs major cycle,
per scheme, with seed error bars. Ratios are taken against the f64 run of the
same (sky, noise_seed).

Usage:
    python analysis/make_sweep.py results/batch2 --out analysis/figures/batch2
    python analysis/make_sweep.py results/batch2 --ref-sky b2_S8mix1000_dr1e4_s42
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.plot as P
from analysis.load import load_results, to_dataframe
from analysis.compare_schemes import scheme_label
from src.probe import ProbeCollector

METRICS = [("cross_correlation", "cc", False),
           ("flux_conservation", "flux", False),
           ("residual_mad", "resMAD", True),
           ("achieved_dr", "achDR", True),
           ("n_components", "n components", False)]


def add_ratios(df):
    """ratio/delta of each metric vs the f64 row with same sky+noise_seed."""
    df = df.copy()
    df["scheme"] = df["quantization"].map(scheme_label)
    df["sky"] = df["dataset_name"].str.replace(r"_n[0-9]+$", "", regex=True)
    df["fam"] = df["sky"].str.replace(r"_s[0-9]+$", "", regex=True)
    ref = (df[df["scheme"] == "f64-baseline"]
           .set_index(["sky", "noise_seed"]))
    rows = []
    for _, r in df.iterrows():
        key = (r["sky"], r["noise_seed"])
        if key not in ref.index:
            continue
        b = ref.loc[key]
        row = dict(r)
        for m, _, _ in METRICS:
            if m in ("cross_correlation", "flux_conservation"):
                row[f"d_{m}"] = r[m] - b[m]
            else:
                row[f"ratio_{m}"] = r[m] / b[m] if b[m] else np.nan
        row["resmad_over_noise"] = (r["residual_mad"] * np.sqrt(r["nvis"]) / r["sigma_jy"]
                                    if r.get("sigma_jy") else np.nan)
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)


def sweep_vs(df, x, name_prefix, outdir, xlabel):
    """One figure per metric: y vs x, hue=scheme, mean+-std over seeds."""
    if x not in df.columns or df[x].dropna().empty:
        return
    plots = ([("d_cross_correlation", "Δcc vs f64", False),
              ("d_flux_conservation", "Δflux vs f64", False),
              ("ratio_residual_mad", "resMAD / f64", True),
              ("ratio_achieved_dr", "achDR / f64", True),
              ("ratio_n_components", "n comps / f64", True),
              ("resmad_over_noise", "resMAD / (σ/√Nvis)", True)])
    for y, lab, logy in plots:
        if y not in df.columns:
            continue
        sub = df.dropna(subset=[x, y])
        if sub.empty:
            continue
        P.metric_vs_axis(sub, x, y, "scheme", f"{name_prefix}_{y}.png",
                         xlabel=xlabel, ylabel=lab, logy=logy,
                         ref_line=1.0 if y.startswith(("ratio", "resmad")) else 0.0,
                         title=f"{name_prefix}: {lab}")


def sweep_vs_k(root, ref_sky, outdir):
    """Per-major-cycle probe scalars (res_peak, res_mad) vs k, hue=scheme."""
    runs = [r for r in load_results(root)
            if r["params"]["dataset_name"].startswith(ref_sky + "_n")
            and "probes" in r]
    if not runs:
        print(f"no probed runs for ref sky {ref_sky}")
        return
    import pandas as pd
    rows = []
    for r in runs:
        pc = ProbeCollector.from_arrays(r["probes"])
        scheme = scheme_label(r["params"]["quantization"])
        for s in pc.snaps:
            if s["label"].startswith("k") and "res_peak" in s["scalars"]:
                rows.append(dict(scheme=scheme, k=s["scalars"]["k"],
                                 res_peak=s["scalars"]["res_peak"],
                                 res_mad=s["scalars"]["res_mad"],
                                 n_components=s["scalars"].get("n_components", np.nan)))
    if not rows:
        return
    df = pd.DataFrame(rows)
    for y, lab in [("res_peak", "residual peak (Jy/beam)"),
                   ("res_mad", "residual MAD (Jy/beam)"),
                   ("n_components", "n components")]:
        P.metric_vs_axis(df, "k", y, "scheme", f"sweep_k_{ref_sky}_{y}.png",
                         xlabel="major cycle k", ylabel=lab, logx=False,
                         logy=y != "n_components", title=f"{ref_sky}: {lab} vs k")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures" / "batch2"))
    ap.add_argument("--ref-sky", default=None,
                    help="sky prefix for the vs-iteration curves (default: first b2 sky found)")
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
    df = add_ratios(df)
    print(f"{len(df)} rows with ratios")

    # vs noise sigma (all skies pooled per scheme; batch2 noise ladder lives on S8)
    sweep_vs(df, "sigma_jy", "sweep_sigma", args.out, xlabel="noise σ (Jy/vis)")
    # vs field DR, one figure set per sky family
    for fam in sorted(df["fam"].dropna().unique()):
        sub = df[df.fam == fam]
        if sub["field_dr"].nunique() < 2:
            continue
        sweep_vs(sub, "field_dr", f"sweep_dr_{fam}", args.out,
                 xlabel="field DR (peak/faintest)")
    # vs iteration on the reference sky
    ref = args.ref_sky
    if ref is None:
        b2 = sorted({r["params"]["dataset_name"].rsplit("_n", 1)[0]
                     for r in results if r["params"]["dataset_name"].startswith("b2_")})
        ref = next((s for s in b2 if "dr1e4" in s), b2[0] if b2 else None)
    if ref:
        for root in args.roots:
            sweep_vs_k(root, ref, args.out)
    print(f"sweep figures -> {args.out}")


if __name__ == "__main__":
    main()
