#!/usr/bin/env python3
"""Batch-2 report from the metrics/params only (light scan, no array loading),
plus figures and printed interpretation tables. Image panels and probe curves
are built for a small representative subset.

    python analysis/batch2_report.py results/batch2 --out analysis/figures/batch2

Figures:
  noise_<y>.png        exp_noise: metric (ratio vs f64) vs noise sigma, hue=scheme
  seeds_<y>.png        exp_seeds: metric (ratio vs f64) vs sky seed, hue=scheme
  k_<y>_refs ky.png    per-major-cycle probe scalars vs k for the ref sky
  panel_<run>.png      canonical 6-panel for the subset runs
Prints: groupby tables with mean+/-std over seeds (the interpretation layer).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.plot as P
from analysis.compare_schemes import scheme_label
from src.probe import ProbeCollector

CODE_ROOT = Path(__file__).resolve().parents[1]


def scan(root):
    """One row per run dir (params.json + metrics.json only — no npz)."""
    import pandas as pd
    rows = []
    for mj in sorted(Path(root).glob("**/metrics.json")):
        d = mj.parent
        pj = d / "params.json"
        if not pj.exists():
            continue
        p = json.loads(pj.read_text())
        m = json.loads(mj.read_text())
        row = dict(p)
        row.update(m)
        row["run"] = d.name
        row["exp"] = d.relative_to(root).parts[0].split("_2026")[0]
        row["run_dir"] = str(d)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["scheme"] = df["quantization"].map(scheme_label)
    df["sky"] = df["dataset_name"].str.replace(r"_n[0-9]+$", "", regex=True)
    df["sigma_jy"] = df["sigma_jy"].astype(float)
    return df


def add_ratios(df):
    df = df.copy()
    # one f64 reference row per (sky, noise_seed): drop duplicates so ref.loc[key]
    # is always a Series even when several studies contain the same sky
    ref = (df[df.scheme == "f64-baseline"]
           .drop_duplicates(subset=["sky", "noise_seed"])
           .set_index(["sky", "noise_seed"]).sort_index())
    for col in ("residual_mad", "achieved_dr", "n_components", "residual_rms"):
        df[f"ratio_{col}"] = np.nan
    for col in ("cross_correlation", "flux_conservation", "snr_db"):
        df[f"d_{col}"] = np.nan
    df["resmad_over_noise"] = np.nan
    for i, r in df.iterrows():
        key = (r["sky"], r["noise_seed"])
        if key not in ref.index:
            continue
        b = ref.loc[key]
        for col in ("residual_mad", "achieved_dr", "n_components", "residual_rms"):
            df.loc[i, f"ratio_{col}"] = r[col] / b[col] if b[col] else np.nan
        for col in ("cross_correlation", "flux_conservation", "snr_db"):
            df.loc[i, f"d_{col}"] = r[col] - b[col]
        df.loc[i, "resmad_over_noise"] = r["residual_mad"] * np.sqrt(r["nvis"]) / r["sigma_jy"]
    return df


PLOTS = [("ratio_residual_mad", "resMAD / f64", True, 1.0),
         ("ratio_achieved_dr", "achDR / f64", True, 1.0),
         ("ratio_n_components", "n components / f64", True, 1.0),
         ("d_cross_correlation", "Δ cross-corr vs f64", False, 0.0),
         ("d_flux_conservation", "Δ flux conservation vs f64", False, 0.0),
         ("resmad_over_noise", "resMAD / (σ/√Nvis)", True, None)]


def sweep_noise(df, outdir):
    sub = df[df.exp.str.startswith("exp_noise")]
    if sub.empty:
        return
    for y, lab, logy, ref in PLOTS:
        if y in sub.columns and sub[y].notna().any():
            P.metric_vs_axis(sub, "sigma_jy", y, "scheme", f"noise_{y}.png",
                             xlabel="noise σ (Jy/vis)", ylabel=lab, logx=True,
                             logy=logy, ref_line=ref,
                             title=f"exp_noise (S8 1000-src, DR 1e4): {lab} vs σ")


def sweep_seeds(df, outdir):
    sub = df[df.exp.str.startswith("exp_seeds")]
    if sub.empty:
        return
    for y, lab, logy, ref in PLOTS:
        if y in sub.columns and sub[y].notna().any():
            P.metric_vs_axis(sub, "dataset_seed", y, "scheme", f"seeds_{y}.png",
                             xlabel="sky seed", ylabel=lab, logx=False,
                             logy=logy, ref_line=ref,
                             title=f"exp_seeds (S8, DR 1e4, σ=1e-4, noise seed 103): {lab} vs sky seed")


def sweep_dr(df, outdir):
    """Metric vs field DR per sky family (needs >=2 DR values — lights up when
    exp_dr lands)."""
    df = df.copy()
    # family = sky name without the trailing _s<seed> and the _dr<value> token
    df["fam"] = (df["sky"].str.replace(r"_s[0-9]+$", "", regex=True)
                 .str.replace(r"_dr[0-9.e]+$", "", regex=True))
    for fam, sub in df.groupby("fam"):
        if sub["field_dr"].nunique() < 2:
            continue
        for y, lab, logy, ref in PLOTS:
            if y in sub.columns and sub[y].notna().any():
                P.metric_vs_axis(sub, "field_dr", y, "scheme", f"dr_{fam}_{y}.png",
                                 xlabel="field DR (peak/faintest)", ylabel=lab, logx=True,
                                 logy=logy, ref_line=ref,
                                 title=f"{fam}: {lab} vs field DR")


def probe_k(df, outdir, ref_sky="b2_S8mix1000_dr1e4_s42"):
    import pandas as pd
    runs = df[df.sky == ref_sky]
    rows = []
    for rd in runs["run_dir"]:
        pz = Path(rd) / "probes.npz"
        if not pz.exists():
            continue
        pc = ProbeCollector.from_arrays(dict(np.load(pz)))
        scheme = runs[runs.run_dir == rd]["scheme"].iloc[0]
        for s in pc.snaps:
            if s["label"].startswith("k") and "res_peak" in s["scalars"]:
                rows.append(dict(scheme=scheme, k=s["scalars"]["k"],
                                 res_peak=s["scalars"]["res_peak"],
                                 res_mad=s["scalars"]["res_mad"],
                                 n_components=s["scalars"].get("n_components", np.nan)))
    if not rows:
        print(f"no probes for ref sky {ref_sky}")
        return
    pdf = pd.DataFrame(rows)
    for y, lab, logy in [("res_peak", "residual peak (Jy/beam)", True),
                         ("res_mad", "residual MAD (Jy/beam)", True),
                         ("n_components", "n components", False)]:
        P.metric_vs_axis(pdf, "k", y, "scheme", f"k_{ref_sky}_{y}.png",
                         xlabel="major cycle k", ylabel=lab, logx=False,
                         logy=logy, title=f"{ref_sky}: {lab} vs k")


def panels(df, outdir, ref_sky="b2_S8mix1000_dr1e4_s42", max_runs=12):
    runs = df[df.sky == ref_sky].sort_values("scheme")
    for _, r in runs.head(max_runs).iterrows():
        z = np.load(r["run_dir"] + "/results.npz")
        bm = r["beam_bmaj_px"]
        beam = (bm, r["beam_bmin_px"], 0.0) if np.isfinite(bm) else None
        P.six_panel(z["truth"], z["dirty"], z["psf_lobe"], z["model"],
                    z["restored"], z["residual"], beam=beam,
                    name=f"panel_b2_{r['sky']}_{r['noise_seed']}_{r['scheme']}.png",
                    suptitle=f"{r['run']}  cc={r['cross_correlation']:.4f} "
                              f"flux={r['flux_conservation']:.4f} achDR={r['achieved_dr']:.3g}")
        z.close()


# resMAD within this factor of sigma/sqrt(Nvis) counts as "noise floor reached"
NOISE_FLOOR_FACTOR = 3.0


def comparability(df):
    """Flag runs whose metrics are trustworthy for quantization comparisons.

    A run is `comparable` when its residual actually reached the noise floor
    (resMAD ~ sigma/sqrt(Nvis)) — there, quantization error would show if it
    mattered. Stall / max_major runs on complex fields sit on a CLEAN floor
    (resMAD >> sigma/sqrt(Nvis)): their metrics look quantization-safe but are
    floor-dominated — the "illusion of safe quantization".
    """
    df = df.copy()
    df["noise_floor_ratio"] = df["resmad_over_noise"]
    df["reached_noise_floor"] = df["noise_floor_ratio"] < NOISE_FLOOR_FACTOR
    df["stalled"] = df["converged_reason"].isin(["stall", "max_major"])
    df["comparable"] = (df["reached_noise_floor"] & df["ratio_residual_mad"].notna()
                        & np.isfinite(df["residual_mad"]))
    return df


def residual_peak_vs_k(df, outdir, skies=None, max_skies=10):
    """res_peak (and res_mad) vs major cycle k per scheme, one figure per sky,
    with the noise floor sigma/sqrt(Nvis) as a red dashed line. This is where
    f16/bf16/f32 divergence from f64 becomes visible on fields that reach the
    noise floor."""
    import pandas as pd
    out = []
    skies = skies or sorted(df["sky"].unique())
    for sky in skies:
        sub = df[df.sky == sky]
        if sub.empty:
            continue
        rows = []
        for _, r in sub.iterrows():
            pz = Path(r["run_dir"]) / "probes.npz"
            if not pz.exists():
                continue
            pc = ProbeCollector.from_arrays(dict(np.load(pz)))
            for s in pc.snaps:
                if s["label"].startswith("k") and "res_peak" in s["scalars"]:
                    rows.append(dict(scheme=r["scheme"], k=s["scalars"]["k"],
                                     res_peak=s["scalars"]["res_peak"],
                                     res_mad=s["scalars"]["res_mad"]))
        if not rows:
            continue
        pdf = pd.DataFrame(rows)
        floor = float(sub["sigma_jy"].iloc[0]) / np.sqrt(float(sub["nvis"].iloc[0]))
        for y, lab in [("res_peak", "residual peak (Jy/beam)"),
                       ("res_mad", "residual MAD (Jy/beam)")]:
            P.metric_vs_axis(pdf, "k", y, "scheme", f"respeak_k_{sky}_{y}.png",
                             xlabel="major cycle k", ylabel=lab, logx=False,
                             logy=True, ref_line=floor,
                             title=f"{sky}: {lab} vs k (red = noise floor σ/√N)")
        out.append(sky)
        if len(out) >= max_skies:
            break
    return out


def report_tables(df):
    import pandas as pd
    pd.set_option("display.width", 200)
    print("\n== comparability audit (runs whose residual reached the noise floor = comparable)")
    comp = comparability(df)
    g = (comp.groupby(["exp", "scheme"])
         .agg(n=("run", "count"), comparable=("comparable", "sum"),
              stalled=("stalled", "sum"),
              floor_ratio_med=("noise_floor_ratio", "median"))
         .round(2))
    print(g.to_string())
    noise = df[df.exp.str.startswith("exp_noise")]
    seeds = df[df.exp.str.startswith("exp_seeds")]
    print("\n== exp_noise: mean±std over noise seeds (101,102), per scheme x σ")
    if not noise.empty:
        g = (noise.groupby(["scheme", "sigma_jy"])
             .agg(resMAD_r=("ratio_residual_mad", "mean"), resMAD_s=("ratio_residual_mad", "std"),
                  achDR_r=("ratio_achieved_dr", "mean"), ncomp_r=("ratio_n_components", "mean"),
                  dcc=("d_cross_correlation", "mean"), dflux=("d_flux_conservation", "mean"),
                  resmad_over_noise=("resmad_over_noise", "mean"))
             .round(4))
        print(g.to_string())
    print("\n== exp_seeds: mean±std over sky seeds (42,43,44), per scheme (σ=1e-4, noise 103)")
    if not seeds.empty:
        g = (seeds.groupby("scheme")
             .agg(resMAD_r=("ratio_residual_mad", "mean"), resMAD_s=("ratio_residual_mad", "std"),
                  achDR_r=("ratio_achieved_dr", "mean"), achDR_s=("ratio_achieved_dr", "std"),
                  ncomp_r=("ratio_n_components", "mean"),
                  dcc=("d_cross_correlation", "mean"), dflux=("d_flux_conservation", "mean"),
                  f64_resMAD=("residual_mad", "mean"), f64_achDR=("achieved_dr", "mean"))
             .round(4))
        print(g.to_string())
    print("\n== f64 baseline variability across sky seeds (exp_seeds):")
    if not seeds.empty:
        f = seeds[seeds.scheme == "f64-baseline"]
        print(f[["sky", "residual_mad", "achieved_dr", "n_components",
                 "cross_correlation", "flux_conservation", "converged_reason"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", default=[str(CODE_ROOT / "results" / "batch2")])
    ap.add_argument("--out", default=str(CODE_ROOT / "analysis" / "figures" / "batch2"))
    ap.add_argument("--ref-sky", default="b2_S8mix1000_dr1e4_s42")
    ap.add_argument("--skies", nargs="*", default=None,
                    help="substring filter for the per-sky residual-peak-vs-k figures")
    args = ap.parse_args()
    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(args.out)

    import pandas as pd
    dfs = [scan(r) for r in args.roots]
    df = add_ratios(pd.concat(dfs, ignore_index=True))
    df = comparability(df)
    print(f"{len(df)} runs: {sorted(df.exp.unique())}")
    df.to_csv(P.OUTPUT_DIR / "batch2_metrics.csv", index=False)
    sweep_noise(df, P.OUTPUT_DIR)
    sweep_seeds(df, P.OUTPUT_DIR)
    sweep_dr(df, P.OUTPUT_DIR)
    probe_k(df, P.OUTPUT_DIR, args.ref_sky)
    skies = None
    if args.skies:
        skies = [s for s in sorted(df.sky.unique())
                 if any(f in s for f in args.skies)]
    did = residual_peak_vs_k(df, P.OUTPUT_DIR, skies=skies)
    print("residual-peak-vs-k figures for skies:", did)
    panels(df, P.OUTPUT_DIR, args.ref_sky)
    report_tables(df)


if __name__ == "__main__":
    main()
