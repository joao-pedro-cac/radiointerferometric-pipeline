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
        lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")
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
    print(f"tables -> {outdir} ({len(df)} runs)")


if __name__ == "__main__":
    main()
