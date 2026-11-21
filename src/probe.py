"""Per-iteration range probes: log2 histograms + representability stats.

Pure functions: arrays in, dicts out. No I/O, no file paths.

The log2 histogram of |x| over nonzero entries (fixed edges shared by all
snapshots) is the community-standard, scale-free way to see an array's actual
dynamic range; unlike range-only stats it shows where the mass sits and what a
low-precision cast flushes to zero. `array_stats` adds limit-coverage fractions
against the IEEE-754 representable ranges:

  float16  : max normal 65504 (~2^16), min subnormal 2^-24
  bfloat16 : max ~3.39e38 (~2^128),    min subnormal 2^-133

ProbeCollector gathers one snapshot per pipeline stage (init + every major
cycle + final) and serializes to/from a flat npz-ready dict: histograms are
tiny (nbins uint64 per array per snapshot) so probing costs ~100 ms per cycle
against ~40-90 s of CLEAN compute, and stores KBs instead of GBs of arrays.

Snapshot convention: a snap labelled "k<k>" holds the residual *after* major
cycle k together with the model *before* cycle k's component addition (the
residual is computed from that model); the "final" snap holds the returned
model/residual. "_cast" arrays (present only when the stage scale s != 1) are
the values actually stored in the low-precision dtype, i.e. eff * s.
"""
import numpy as np

LOG2_LO = -64.0
LOG2_HI = 80.0
NBINS = 144         # 1 bin per octave over 144 decades (raw dirty/PSF reach 2^48,
                    # 2^68 at 1e6 Jy peak; Jy/beam frames live near 2^0..2^-30)

F16_MAX = 65504.0
F16_MIN_SUB = 2.0 ** -24
BF16_MAX = np.finfo(np.float32).max       # ~3.39e38
BF16_MIN_SUB = 2.0 ** -133


def log2_hist(x, edges=None, lo=LOG2_LO, hi=LOG2_HI, nbins=NBINS):
    """Histogram of log2|x| over the nonzero entries of x (complex -> |x|).

    Returns (counts uint64, edges float64). Values outside [lo, hi] are
    clipped into the edge bins, so counts always sum to the number of nonzero
    entries (nothing silently dropped). All-zero or empty input -> zero
    counts.
    """
    if edges is None:
        edges = np.linspace(lo, hi, nbins + 1)
    a = np.abs(np.asarray(x))
    nz = a[a > 0]
    counts = np.zeros(edges.size - 1, dtype=np.uint64)
    if nz.size:
        lv = np.clip(np.log2(nz), edges[0], edges[-1])
        counts, _ = np.histogram(lv, bins=edges)
    return counts.astype(np.uint64), edges


def array_stats(x):
    """Scale-free summary + representability fractions of x (complex -> |x|)."""
    a = np.abs(np.asarray(x))
    nz = a[a > 0]
    if nz.size == 0:
        return dict(n=int(a.size), n_zero=int(a.size), max=0.0, min_nonzero=0.0,
                    median=0.0, mad=0.0, log2_min=0.0, log2_max=0.0,
                    frac_below_f16_sub=0.0, frac_above_f16_max=0.0,
                    frac_below_bf16_sub=0.0, frac_above_bf16_max=0.0)
    med = float(np.median(a))
    mad = 1.4826 * float(np.median(np.abs(a - med)))
    return dict(
        n=int(a.size), n_zero=int((a == 0).sum()),
        max=float(a.max()), min_nonzero=float(nz.min()),
        median=med, mad=mad,
        log2_min=float(np.log2(nz.min())), log2_max=float(np.log2(a.max())),
        frac_below_f16_sub=float(np.mean(nz < F16_MIN_SUB)),
        frac_above_f16_max=float(np.mean(a > F16_MAX)),
        frac_below_bf16_sub=float(np.mean(nz < BF16_MIN_SUB)),
        frac_above_bf16_max=float(np.mean(a > BF16_MAX)),
    )


def downsample(a, factor):
    """Block-mean downsample of a 2D array by an integer factor (edges trimmed).
    Used for per-iteration image snapshots: displayable montages at ~1/256 of
    the storage."""
    a = np.asarray(a)
    f = int(factor)
    if f <= 1 or a.ndim != 2:
        return a
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:ny, :nx]
    return b.reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


class ProbeCollector:
    """Collects labelled snapshots {label: {hists, stats, scalars, images}} and
    serializes to/from a flat npz-ready dict (keys "hist<i>__<var>",
    "stat<i>__<var>__<field>", "scalar<i>__<name>", "img<i>__<var>", plus
    "probe_edges" and "probe_labels"). Heterogeneous keys across snapshots are
    fine. `images` are downsampled 2D snapshots (display montages); hists/stats
    always run on the full-resolution arrays."""

    def __init__(self, lo=LOG2_LO, hi=LOG2_HI, nbins=NBINS, snap_factor=0):
        self.lo, self.hi, self.nbins = float(lo), float(hi), int(nbins)
        self.edges = np.linspace(self.lo, self.hi, self.nbins + 1)
        self.snap_factor = int(snap_factor)   # 0 = no image snapshots
        self.snaps = []

    def snap(self, label, arrays=None, scalars=None, image_vars=()):
        s = dict(label=str(label), hists={}, stats={}, images={},
                 scalars={k: float(v) for k, v in (scalars or {}).items()})
        for k, v in (arrays or {}).items():
            s["hists"][k], _ = log2_hist(v, edges=self.edges)
            s["stats"][k] = array_stats(v)
            if self.snap_factor > 1 and k in image_vars:
                s["images"][k] = downsample(np.asarray(v, np.float64),
                                            self.snap_factor).astype(np.float32)
        self.snaps.append(s)
        return s

    def to_arrays(self):
        out = {"probe_edges": self.edges,
               "probe_labels": np.array([s["label"] for s in self.snaps]),
               "probe_snap_factor": np.int64(self.snap_factor)}
        for i, s in enumerate(self.snaps):
            for k, c in s["hists"].items():
                out[f"hist{i}__{k}"] = c
            for k, st in s["stats"].items():
                for kk, vv in st.items():
                    out[f"stat{i}__{k}__{kk}"] = np.float64(vv)
            for kk, vv in s["scalars"].items():
                out[f"scalar{i}__{kk}"] = np.float64(vv)
            for k, im in s.get("images", {}).items():
                out[f"img{i}__{k}"] = im
        return out

    @classmethod
    def from_arrays(cls, d):
        edges = np.asarray(d["probe_edges"], dtype=np.float64)
        c = cls(lo=edges[0], hi=edges[-1], nbins=edges.size - 1,
                snap_factor=int(d["probe_snap_factor"]) if "probe_snap_factor" in d else 0)
        c.edges = edges
        labels = [str(s) for s in d["probe_labels"]]
        for i, lab in enumerate(labels):
            s = dict(label=lab, hists={}, stats={}, scalars={}, images={})
            for k in d.keys():
                if k.startswith(f"hist{i}__"):
                    s["hists"][k[len(f"hist{i}__"):]] = np.asarray(d[k])
                elif k.startswith(f"img{i}__"):
                    s["images"][k[len(f"img{i}__"):]] = np.asarray(d[k])
                elif k.startswith(f"stat{i}__"):
                    rest = k[len(f"stat{i}__"):]
                    var, field = rest.split("__", 1)
                    s["stats"].setdefault(var, {})[field] = float(d[k])
                elif k.startswith(f"scalar{i}__"):
                    s["scalars"][k[len(f"scalar{i}__"):]] = float(d[k])
            c.snaps.append(s)
        return c
