#!/usr/bin/env python3
"""Generate LaTeX summary tables from a results root into deck/tables/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.load import load_results, to_dataframe


def tex_table(df, cols, header, caption_rows=None):
    lines = ["\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}",
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = [str(r[c]).replace("_", "\\_") for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", nargs="?",
                    default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    outdir = Path(__file__).resolve().parent / "tables"
    outdir.mkdir(exist_ok=True)
    df = to_dataframe(load_results(args.results_root))
    if df.empty:
        print("no results yet")
        return
    df = df.copy()
    # baseline summary table: f64 rows only (quant rows go to quant_*.tex)
    if "quantization" in df.columns:
        is_f64 = df["quantization"].map(
            lambda q: all(v == "float64" for v in q.values()))
        df = df[is_f64]
    df["achDR"] = df["achieved_dr"].map(lambda v: f"{v:.3g}")
    df["resMAD"] = df["residual_mad"].map(lambda v: f"{v:.2e}")
    df["cc"] = df["cross_correlation"].map(lambda v: f"{v:.5f}")
    df["flux"] = df["flux_conservation"].map(lambda v: f"{v:.5f}")
    cols = ["dataset_name", "noise_seed", "cc", "flux", "resMAD", "achDR"]
    header = ["dataset", "seed", "cc", "flux", "res. MAD", "ach. DR"]
    all_tbl = tex_table(df.sort_values(["dataset_name", "noise_seed"])[cols].reset_index(drop=True),
                        cols, header)
    (outdir / "summary_all.tex").write_text(all_tbl)
    for sky in sorted(df["dataset_name"].unique()):
        sub = df[df.dataset_name == sky]
        (outdir / f"summary_{sky.split('_')[0].lower()}.tex").write_text(
            tex_table(sub.sort_values("noise_seed")[cols].reset_index(drop=True), cols, header))

    # quantization pilot table (from compare_schemes.py output)
    qcsv = Path(__file__).resolve().parents[1] / "analysis/figures/quant_summary.csv"
    if qcsv.exists():
        import pandas as pd
        q = pd.read_csv(qcsv)
        q = q[q.scheme != "f64-baseline"].drop_duplicates(["sky", "scheme"])
        for sky in sorted(q["sky"].unique()):
            sub = q[q.sky == sky].copy()
            sub["dcc"] = sub["d_cross_correlation"].map(lambda v: f"{v:+.1e}")
            sub["dflux"] = sub["d_flux_conservation"].map(lambda v: f"{v:+.1e}")
            sub["rMAD"] = sub["ratio_residual_mad"].map(lambda v: f"{v:.2f}")
            sub["rDR"] = sub["ratio_achieved_dr"].map(lambda v: f"{v:.3f}")
            cols2 = ["scheme", "dcc", "dflux", "rMAD", "rDR"]
            hdr2 = ["scheme (v-d-p-m)", "$\\Delta$cc", "$\\Delta$flux", "resMAD/f64", "achDR/f64"]
            (outdir / f"quant_{sky.split('_')[0].lower()}.tex").write_text(
                tex_table(sub[cols2].reset_index(drop=True), cols2, hdr2))
            # low-precision detail rows (all-f16 / all-bf16) for the detail frames
            lp = sub[sub["scheme"].isin(["vf16-df16-pf16-mf16",
                                         "vbf16-dbf16-pbf16-mbf16"])].copy()
            lp["scheme"] = lp["scheme"].map(
                {"vf16-df16-pf16-mf16": "all-f16", "vbf16-dbf16-pbf16-mbf16": "all-bf16"})
            if len(lp):
                (outdir / f"quant_lowprec_{sky.split('_')[0].lower()}.tex").write_text(
                    tex_table(lp[cols2].reset_index(drop=True), cols2, hdr2))
    # v3 sweep pilot tables: one per sky, rows = schemes (incl. raw-cast),
    # ratios vs the f64 row of the same study, plus loop/probe facts
    results_root = Path(args.results_root)
    sweep_runs = [r for r in load_results(results_root)
                  if Path(r["run_dir"]).parent.name.startswith("sweep_v3")]
    if sweep_runs:
        sdf = to_dataframe(sweep_runs).copy()
        sdf["sky"] = sdf["dataset_name"].str.replace(r"_n[0-9]+$", "", regex=True)
        # scheme = run-dir suffix after "<dataset_name>_"
        sdf["scheme"] = [Path(r["run_dir"]).name[len(r["dataset_name"]) + 1:]
                         for _, r in sdf.iterrows()]
        sdf = sdf.drop_duplicates(["sky", "scheme"], keep="last")  # prefer newest study
        sdf["s_vis"] = sdf["scales"].map(lambda s: f"{s['vis']:.3g}" if isinstance(s, dict) else "-")
        sdf["s_model"] = sdf["scales"].map(lambda s: f"{s['model']:.3g}" if isinstance(s, dict) else "-")
        for sky in sorted(sdf["sky"].unique()):
            sub = sdf[sdf.sky == sky].copy()
            ref = sub[sub["scheme"] == "f64"]
            if ref.empty:
                continue
            ref = ref.iloc[0]
            sub["rMAD"] = sub.apply(
                lambda r: f"{r['residual_mad'] / ref['residual_mad']:.2f}", axis=1)
            sub["rDR"] = sub.apply(
                lambda r: (f"{r['achieved_dr'] / ref['achieved_dr']:.2e}"
                           if ref['achieved_dr'] and r['achieved_dr'] / ref['achieved_dr'] < 0.01
                           else f"{r['achieved_dr'] / ref['achieved_dr']:.3f}"), axis=1)
            sub["cc"] = sub["cross_correlation"].map(lambda v: f"{v:.6f}")
            sub["flux"] = sub["flux_conservation"].map(lambda v: f"{v:.5f}")
            sub["resMAD"] = sub["residual_mad"].map(lambda v: f"{v:.2e}")
            sub["k"] = sub["n_major_cycles"].astype(int)
            sub["conv"] = sub["converged_reason"]
            cols3 = ["scheme", "cc", "flux", "resMAD", "rMAD", "rDR", "k", "conv", "s_vis", "s_model"]
            hdr3 = ["scheme", "cc", "flux", "resMAD", "resMAD/f64", "achDR/f64", "k",
                    "stop", "s$_{vis}$", "s$_{model}$"]
            (outdir / f"pilot_v3_{sky}.tex").write_text(
                tex_table(sub.sort_values("scheme")[cols3].reset_index(drop=True), cols3, hdr3))
    print(f"tables -> {outdir} ({len(df)} runs)")


if __name__ == "__main__":
    main()
