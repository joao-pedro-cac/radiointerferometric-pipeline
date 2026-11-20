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
    print(f"tables -> {outdir} ({len(df)} runs)")


if __name__ == "__main__":
    main()
