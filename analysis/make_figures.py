#!/usr/bin/env python3
"""Regenerate all canonical figures from a results root into analysis/figures/.

Usage: python analysis/make_figures.py [results_root] [--gallery]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.plot as P
from analysis.load import load_results, to_dataframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", nargs="?",
                    default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--gallery", action="store_true", help="also build gallery/index.html")
    args = ap.parse_args()

    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

    results = load_results(args.results_root)
    print(f"loaded {len(results)} runs from {args.results_root}")
    if not results:
        return
    df = to_dataframe(results)

    # one 6-panel figure per run
    for r in results:
        name = Path(r["run_dir"]).name
        bm = r["metrics"]["beam_bmaj_px"]
        beam = (bm, r["metrics"]["beam_bmin_px"], 0.0) if np.isfinite(bm) else None
        pk = float(r["metrics"].get("psf_peak", 1.0))
        dirty = r["dirty"] / pk if np.nanmax(np.abs(r["dirty"])) > 1e6 else r["dirty"]
        psf_lobe = r["psf_lobe"] / pk if np.nanmax(np.abs(r["psf_lobe"])) > 1e6 else r["psf_lobe"]
        P.six_panel(r["truth"], dirty, psf_lobe, r["model"],
                    r["restored"], r["residual"], beam=beam,
                    name=f"panel_{name}.png",
                    suptitle=f"{name}  cc={r['metrics']['cross_correlation']:.4f} "
                              f"flux={r['metrics']['flux_conservation']:.4f} "
                              f"achDR={r['metrics']['achieved_dr']:.3g}")
        # zoom on the brightest truth pixel
        iy, ix = np.unravel_index(np.argmax(r["truth"]), r["truth"].shape)
        P.zoom_compare(r["truth"], r["restored"], r["residual"], (iy, ix),
                       half=200, beam=beam, name=f"zoom_{name}.png",
                       suptitle=name)

    # ladder summaries
    P.metric_vs_dr(df, "cross_correlation", "cross-correlation (restored vs truth)",
                   "ladder_cc.png")
    P.metric_vs_dr(df, "flux_conservation", "flux conservation sum(model)/sum(truth)",
                   "ladder_flux.png")
    P.metric_vs_dr(df, "achieved_dr", "achieved DR (peak / residual MAD)",
                   "ladder_achdr.png", logy=True)
    df2 = df.copy()
    df2["residual_mad_over_sigma"] = df2["residual_mad"] / df2["sigma_jy"]
    P.metric_vs_dr(df2, "residual_mad_over_sigma", "residual MAD / sigma_vis",
                   "ladder_resmad.png", logy=True)

    # grouped bars per dataset
    for metric, ylabel, fname in [
            ("cross_correlation", "cc", "bars_cc.png"),
            ("flux_conservation", "flux conservation", "bars_flux.png"),
            ("achieved_dr", "achieved DR", "bars_achdr.png")]:
        P.metric_bars(df, metric, ylabel, fname)

    # summary table
    cols = ["dataset_name", "noise_seed", "cross_correlation", "flux_conservation",
            "residual_mad", "achieved_dr", "n_major_cycles", "converged_reason"]
    have = [c for c in cols if c in df.columns]
    out_csv = P.OUTPUT_DIR / "summary.csv"
    df[have].sort_values(["dataset_name", "noise_seed"]).to_csv(out_csv, index=False)
    print(f"figures -> {P.OUTPUT_DIR}  ({len(results)} panels, summary.csv)")

    if args.gallery:
        build_gallery(results, P.OUTPUT_DIR)


def build_gallery(results, figdir):
    import html
    gal = figdir / "gallery"
    gal.mkdir(exist_ok=True)
    for f in figdir.glob("*.png"):
        (gal / f.name).write_bytes(f.read_bytes())
    items = []
    for r in sorted(results, key=lambda r: r["run_dir"]):
        name = Path(r["run_dir"]).name
        m = r["metrics"]
        items.append(
            f"<h3>{html.escape(name)}</h3>"
            f"<p>cc={m['cross_correlation']:.4f} flux={m['flux_conservation']:.4f} "
            f"resMAD={m['residual_mad']:.2e} achDR={m['achieved_dr']:.3g} "
            f"nmaj={m['n_major_cycles']} {html.escape(m['converged_reason'])}</p>"
            f"<img src='panel_{name}.png' style='width:49%'>"
            f"<img src='zoom_{name}.png' style='width:49%'>")
    for f in ["ladder_cc.png", "ladder_flux.png", "ladder_achdr.png",
              "ladder_resmad.png", "bars_cc.png", "bars_flux.png", "bars_achdr.png"]:
        if (gal / f).exists():
            items.append(f"<h3>{f}</h3><img src='{f}' style='width:70%'>")
    (gal / "index.html").write_text(
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;"
        "max-width:1200px;margin:auto}img{margin:4px}</style></head><body>"
        + "\n".join(items) + "</body></html>")
    print(f"gallery -> {gal/'index.html'}")


if __name__ == "__main__":
    main()
