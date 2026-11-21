#!/usr/bin/env python3
"""Probe figures from runs' probes.npz: log2-histogram waterfalls per variable
per major cycle, plus per-cycle scalar curves, plus cross-scheme comparison
grids for the deck.

Usage:
    python analysis/make_probe_figures.py results/sweep_v3_*            # all runs
    python analysis/make_probe_figures.py results/sweep_v3_* --runs S7   # filter
    python analysis/make_probe_figures.py results/sweep_v3_* --compare-sky S7_point_peak1e6
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.plot as P
from analysis.load import load_results
from src.probe import ProbeCollector, F16_MAX, F16_MIN_SUB, BF16_MAX, BF16_MIN_SUB

VARS_ORDER = ["model", "residual", "vis_resid", "vis", "dirty", "psf"]
VLINE_SPECS = [(np.log2(F16_MAX), "f16 max"), (np.log2(F16_MIN_SUB), "f16 sub"),
               (np.log2(BF16_MAX), "bf16 max"), (np.log2(BF16_MIN_SUB), "bf16 sub")]


def _vars_in(pc):
    """Variables present in the most snaps, base names first, then _cast."""
    from collections import Counter
    cnt = Counter()
    for s in pc.snaps:
        for k in s["hists"]:
            cnt[k] += 1
    base = [v for v in VARS_ORDER if v in cnt]
    base += [v for v in cnt if v not in VARS_ORDER and not v.endswith("_cast")]
    cast = sorted(k for k in cnt if k.endswith("_cast"))
    return base, cast


def _waterfall_fig(pc, var, name, title):
    rows, labels = [], []
    for s in pc.snaps:
        if var in s["hists"]:
            rows.append(s["hists"][var])
            labels.append(s["label"])
    if not rows:
        return None
    return P.log2_waterfall(pc.edges, np.array(rows), labels, name,
                            title=title, vlines=VLINE_SPECS)


def run_figures(run_dir, outdir):
    d = Path(run_dir)
    pz = d / "probes.npz"
    if not pz.exists():
        return []
    pc = ProbeCollector.from_arrays(dict(np.load(pz)))
    name = d.name
    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(outdir)
    base, cast = _vars_in(pc)
    made = []
    # one figure with up to 4 base-var panels is done per var; group 2 per fig
    for i in range(0, min(len(base), 4), 2):
        fig_vars = base[i:i + 2]
        if len(fig_vars) == 1:
            f = _waterfall_fig(pc, fig_vars[0], f"probe_{name}_wf_{fig_vars[0]}.png",
                               f"{name} — {fig_vars[0]}")
        else:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
            for ax, var in zip(axes, fig_vars):
                rows, labels = [], []
                for s in pc.snaps:
                    if var in s["hists"]:
                        rows.append(s["hists"][var]); labels.append(s["label"])
                c = np.log10(np.asarray(rows, float) + 1.0)
                im = ax.imshow(c, aspect="auto", origin="lower", cmap="magma",
                               extent=[pc.edges[0], pc.edges[-1], -0.5, len(labels) - 0.5])
                ax.set_xlim(pc.edges[0], pc.edges[-1])
                ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
                ax.set_xlabel("log2 |value|"); ax.set_title(f"{name} — {var}", fontsize=10)
                for x, lab in VLINE_SPECS:
                    if pc.edges[0] <= x <= pc.edges[-1]:
                        ax.axvline(x, color="cyan", lw=1.0, ls="--")
                plt.colorbar(im, ax=ax, label="log10(count+1)")
            fig.tight_layout()
            f = P._save(fig, f"probe_{name}_wf_{fig_vars[0]}_{fig_vars[1]}.png")
        if f is not None:
            made.append(f)
    for var in cast[:4]:
        f = _waterfall_fig(pc, var, f"probe_{name}_wf_{var}.png", f"{name} — {var} (as cast)")
        if f is not None:
            made.append(f)
    f = P.probe_curves(pc.snaps, f"probe_{name}_curves.png", title=name)
    if f is not None:
        made.append(f)
    return made


def compare_grid(root, sky, outdir, max_schemes=6):
    """One figure per sky: rows = schemes, cols = [model, residual, vis/vis_resid]."""
    import matplotlib.pyplot as plt
    results = load_results(root)
    runs = [r for r in results if r["params"]["dataset_name"].startswith(sky + "_n")
            and "probes" in r]
    if not runs:
        print(f"no probed runs for {sky}")
        return None
    def scheme_of(r):
        dn = r["params"]["dataset_name"]
        return Path(r["run_dir"]).name[len(dn) + 1:]
    runs.sort(key=lambda r: scheme_of(r))
    runs = runs[:max_schemes]
    vars_ = ["model", "residual", "vis_resid"]
    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(outdir)
    fig, axes = plt.subplots(len(runs), len(vars_),
                             figsize=(5.2 * len(vars_), 2.6 * len(runs)), squeeze=False)
    for i, r in enumerate(runs):
        pc = ProbeCollector.from_arrays(r["probes"])
        for j, var in enumerate(vars_):
            ax = axes[i, j]
            rows, labels = [], []
            for s in pc.snaps:
                v = var if var in s["hists"] else ("vis" if var == "vis_resid" and "vis" in s["hists"] else None)
                if v:
                    rows.append(s["hists"][v]); labels.append(s["label"])
            if not rows:
                ax.axis("off"); continue
            c = np.log10(np.asarray(rows, float) + 1.0)
            im = ax.imshow(c, aspect="auto", origin="lower", cmap="magma",
                           extent=[pc.edges[0], pc.edges[-1], -0.5, len(labels) - 0.5])
            ax.set_xlim(pc.edges[0], pc.edges[-1])
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
            for x, lab in VLINE_SPECS:
                if pc.edges[0] <= x <= pc.edges[-1]:
                    ax.axvline(x, color="cyan", lw=0.8, ls="--")
            ax.set_title(f"{scheme_of(r)} — {var}", fontsize=9)
            ax.set_xlabel("log2 |value|", fontsize=8)
    fig.suptitle(f"log2 range probes per major cycle — {sky}", fontsize=12)
    fig.tight_layout()
    P._save(fig, f"probe_cmp_{sky}.png")
    # per-iteration image montages (downsampled snaps): model / residual at
    # k=1, mid, final — shows convergence (or its stall) as images
    if any(s.get("images") for r in runs for s in ProbeCollector.from_arrays(r["probes"]).snaps):
        import matplotlib.pyplot as plt
        picks = ["k1", None, "final"]   # None -> middle k
        cols = []
        for var in ("model", "residual"):
            for p in picks:
                cols.append((var, p))
        fig2, axes2 = plt.subplots(len(runs), len(cols),
                                   figsize=(2.6 * len(cols), 2.4 * len(runs)),
                                   squeeze=False)
        for i, r in enumerate(runs):
            pc = ProbeCollector.from_arrays(r["probes"])
            ks = [s["label"] for s in pc.snaps if s["label"].startswith("k")]
            mid = ks[len(ks) // 2] if ks else None
            resolved = [p if p is not None else mid for (_, p) in cols]
            for j, (var, p) in enumerate(zip([c[0] for c in cols], resolved)):
                ax = axes2[i, j]
                snap = next((s for s in pc.snaps if s["label"] == p), None)
                if snap is None or var not in snap.get("images", {}):
                    ax.axis("off"); continue
                a = np.asarray(snap["images"][var], np.float64)
                a[~np.isfinite(a)] = np.nan
                scale = "log" if var == "model" else "symlog"
                norm = P._stretch(a, scale)
                cmap = P.SKY_CMAP if var == "model" else P.RESID_CMAP
                im = ax.imshow(a, origin="lower", cmap=cmap, norm=norm)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{var} {p}", fontsize=8)
            axes2[i, 0].set_ylabel(scheme_of(r), fontsize=9)
        fig2.suptitle(f"{sky}: per-iteration model / residual snapshots by scheme",
                      fontsize=12)
        fig2.tight_layout()
        P._save(fig2, f"probe_img_{sky}.png")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="results roots to scan")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    ap.add_argument("--runs", default=None, help="substring filter on run dir name")
    ap.add_argument("--compare-sky", default=None,
                    help="build a scheme x variable waterfall grid for this sky (prefix of dataset_name)")
    args = ap.parse_args()
    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = Path(args.out)

    if args.compare_sky:
        for root in args.roots:
            compare_grid(root, args.compare_sky, args.out)
        return

    for root in args.roots:
        for r in load_results(root):
            if args.runs and args.runs not in r["run_dir"]:
                continue
            if "probes" not in r:
                continue
            made = run_figures(r["run_dir"], args.out)
            print(r["run_dir"], "->", len(made), "figures")


if __name__ == "__main__":
    main()
