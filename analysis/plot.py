"""Visualisation layer — pure functions: numpy arrays in, matplotlib Figure out.

No file paths inside plot functions: a module-level SAVE_FIGURES flag and
OUTPUT_DIR control saving; the same function works batch or interactively.

Conventions:
- sky-like images (truth, dirty, model, restored, PSF): perceptual map
  ("magma"), asinh stretch by default so faint structure stays visible.
- residuals: diverging map ("coolwarm"), symmetric around 0.
- every image panel gets its own colorbar, a beam ellipse (bottom-left), and
  optional truth contours.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import AsinhNorm, SymLogNorm

SAVE_FIGURES = False
OUTPUT_DIR = Path("figures")

SKY_CMAP = "magma"
RESID_CMAP = "coolwarm"
PSF_CMAP = "viridis"


def _save(fig, name):
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    return fig


def _stretch(img, scale):
    a = np.asarray(img, np.float64)
    if scale == "asinh":
        peak = np.nanmax(np.abs(a))
        return AsinhNorm(linear_width=peak / 100.0 if peak > 0 else 1.0)
    if scale == "log":
        pos = a[a > 0]
        hi = np.nanmax(a) if a.size else 1.0
        lo = max(np.nanpercentile(pos, 1) if pos.size else 1e-12, 1e-4 * hi)
        return SymLogNorm(linthresh=lo, vmin=lo, vmax=hi)
    if scale == "symlog":
        peak = np.nanmax(np.abs(a)) or 1.0
        return SymLogNorm(linthresh=peak / 1000.0, vmin=-peak, vmax=peak)
    return None  # linear


def panel(ax, img, title="", cmap=SKY_CMAP, scale="asinh", colorbar=True,
          beam=None, contours=None, contour_color="cyan", zoom=None):
    """One image panel. beam=(bmaj_px,bmin_px,pa_deg) draws the restoring beam."""
    a = np.asarray(img, np.float64)
    if zoom is not None:
        y0, y1, x0, x1 = zoom
        a = a[y0:y1, x0:x1]
    norm = _stretch(a, scale)
    im = ax.imshow(a, origin="lower", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if contours is not None:
        c = np.asarray(contours, np.float64)
        if zoom is not None:
            y0, y1, x0, x1 = zoom
            c = c[y0:y1, x0:x1]
        levels = np.geomspace(np.nanmax(c) * 1e-3, np.nanmax(c), 5)
        ax.contour(c, levels=levels, colors=contour_color, linewidths=0.6)
    if beam is not None:
        bmaj, bmin, pa = beam
        ny, nx = a.shape
        el = Ellipse((nx * 0.08, ny * 0.08), width=bmin, height=bmaj, angle=pa,
                     fill=False, edgecolor="white", lw=1.2)
        ax.add_patch(el)
    return ax


def six_panel(truth, dirty, psf_lobe, model, restored, residual, beam=None,
              name="six_panel.png", suptitle=""):
    """Canonical figure: truth | dirty | PSF | clean model | restored | residual."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    panel(axes[0, 0], truth, "truth (Jy/pix)", scale="log")
    panel(axes[0, 1], dirty, "dirty image (Jy/beam)", scale="asinh")
    panel(axes[0, 2], psf_lobe, "PSF (lobe)", cmap=PSF_CMAP, scale="log")
    panel(axes[1, 0], model, "CLEAN model (Jy)", scale="log")
    panel(axes[1, 1], restored, "restored (Jy/beam)", beam=beam,
          contours=truth, scale="asinh")
    panel(axes[1, 2], residual, "residual (Jy/beam)", cmap=RESID_CMAP, scale="symlog")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def zoom_compare(truth, restored, residual, center, half=150, beam=None,
                 name="zoom.png", suptitle=""):
    """Zoomed cutout around `center`=(y,x): truth | restored+contours | residual."""
    y, x = center
    zoom = (max(y - half, 0), min(y + half, truth.shape[0]),
            max(x - half, 0), min(x + half, truth.shape[1]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel(axes[0], truth, "truth (zoom)", zoom=zoom)
    panel(axes[1], restored, "restored (zoom, truth contours)", beam=beam,
          contours=truth, zoom=zoom)
    panel(axes[2], residual, "residual (zoom)", cmap=RESID_CMAP, scale="symlog", zoom=zoom)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def metric_vs_dr(df, metric, ylabel, name, logy=False, ax=None):
    """Ladder summary: metric vs sky DR, one point per run, grouped by sky."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    df = df.copy()
    df["sky"] = df["dataset_name"].str.replace(r"_n\d+$", "", regex=True)
    for sky, g in df.groupby("sky"):
        x = g["field_dr"].astype(float)
        ax.scatter(x, g[metric], label=sky, s=28, zorder=3)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("sky dynamic range (peak / faintest)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, name)


def log2_waterfall(edges, counts, labels, name, title="", vlines=None):
    """log2-histogram waterfall: x = log2 bin, y = snapshot (iteration),
    color = log10(count+1). vlines = [(x, label), ...] e.g. dtype limits."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    c = np.log10(np.asarray(counts, np.float64) + 1.0)
    im = ax.imshow(c, aspect="auto", origin="lower", cmap="magma",
                   extent=[edges[0], edges[-1], -0.5, len(labels) - 0.5])
    ax.set_xlim(edges[0], edges[-1])     # keep out-of-range dtype lines from stretching
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("log2 |value|")
    for x, lab in (vlines or []):
        if not (edges[0] <= x <= edges[-1]):
            continue
        ax.axvline(x, color="cyan", lw=1.0, ls="--")
        ax.text(x + 0.3, len(labels) - 0.6, lab, color="cyan", fontsize=7,
                rotation=90, va="top")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="log10(count+1)")
    fig.tight_layout()
    return _save(fig, name)


def probe_curves(snaps, name, title=""):
    """Per-major-cycle probe scalars vs cycle: res_peak, res_mad,
    n_components, s_model."""
    ks = [s for s in snaps if s["label"].startswith("k")]
    if not ks:
        return None
    k = np.array([s["scalars"]["k"] for s in ks])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, key, log in [(axes[0, 0], "res_peak", True), (axes[0, 1], "res_mad", True),
                         (axes[1, 0], "n_components", False), (axes[1, 1], "s_model", True)]:
        v = np.array([s["scalars"].get(key, np.nan) for s in ks])
        ax.plot(k, v, "o-", ms=4)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("major cycle k")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def metric_bars(df, metric, ylabel, name, hue="quantization.model"):
    """Grouped bars: mean +/- std of `metric` per dataset (and hue if present)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = sorted(df["dataset_name"].unique())
    hues = sorted(df[hue].unique()) if hue in df.columns else [None]
    w = 0.8 / max(len(hues), 1)
    for i, h in enumerate(hues):
        sub = df if h is None else df[df[hue] == h]
        means = [sub[sub.dataset_name == k][metric].mean() for k in keys]
        stds = [sub[sub.dataset_name == k][metric].std() for k in keys]
        xs = np.arange(len(keys)) + i * w
        ax.bar(xs, means, w, yerr=stds, label=str(h), capsize=3)
    ax.set_xticks(np.arange(len(keys)) + w * (len(hues) - 1) / 2)
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    if hues != [None]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)
