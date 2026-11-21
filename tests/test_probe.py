"""Unit tests for src/probe.py — log2 histograms, limit stats, collector round-trip."""
import numpy as np
import pytest

from src import operators, pipeline, probe


# ---------- log2_hist ----------

def test_log2_hist_known_values():
    # |x| nonzero = [1, 2, 4, 0.5, 8] -> log2 = [0, 1, 2, -1, 3]; the 0 is dropped
    x = np.array([1.0, 2.0, 4.0, 0.5, 0.0, -8.0])
    counts, edges = probe.log2_hist(x, lo=-4, hi=4, nbins=8)   # 1 bin per octave
    assert edges.tolist() == [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    c = counts.tolist()
    assert c[3] == 1 and c[4] == 1 and c[5] == 1 and c[6] == 1 and c[7] == 1
    assert sum(c) == 5                      # zero excluded, nothing dropped
    assert counts.dtype == np.uint64


def test_log2_hist_complex_uses_modulus():
    counts, _ = probe.log2_hist(np.array([1 + 1j, 3 - 4j]), lo=-4, hi=4, nbins=8)
    # |1+1j| = sqrt(2) -> log2 = 0.5 -> bin [0,1); |3-4j| = 5 -> log2 ~ 2.32 -> bin [2,3)
    assert counts[4] == 1 and counts[6] == 1 and counts.sum() == 2


def test_log2_hist_zero_and_empty():
    c0, e0 = probe.log2_hist(np.zeros(10))
    assert c0.sum() == 0 and e0.size == probe.NBINS + 1
    c1, _ = probe.log2_hist(np.array([]))
    assert c1.sum() == 0


def test_log2_hist_clips_out_of_range_into_edge_bins():
    x = np.array([2.0 ** 50, 2.0 ** -50, 1.0])   # outside default-ish window
    counts, edges = probe.log2_hist(x, lo=-4, hi=4, nbins=8)
    assert counts.sum() == 3                   # clipped, not dropped
    assert counts[0] == 1                      # 2^-50 -> first bin
    assert counts[-1] == 1                     # 2^50 -> last bin
    assert counts[4] == 1                      # 1.0 -> log2 0 -> bin [0,1)


# ---------- array_stats ----------

def test_array_stats_values():
    x = np.array([0.0, 1.0, -2.0, 4.0])
    st = probe.array_stats(x)
    assert st["n"] == 4 and st["n_zero"] == 1
    assert st["max"] == 4.0 and st["min_nonzero"] == 1.0
    assert st["median"] == pytest.approx(1.5)                 # median |x| of [0,1,2,4]
    assert st["mad"] == pytest.approx(1.4826 * 1.0)            # |a-1.5| = [1.5,.5,.5,2.5]
    assert st["log2_min"] == 0.0 and st["log2_max"] == 2.0


def test_array_stats_f16_limit_fractions():
    x = np.array([1e6, 1e-9, 1.0])     # above f16 max, below f16 subnormal, normal
    st = probe.array_stats(x)
    assert st["frac_above_f16_max"] == pytest.approx(1 / 3)
    assert st["frac_below_f16_sub"] == pytest.approx(1 / 3)   # 2^-24 ~ 6e-8 > 1e-9
    assert st["frac_above_bf16_max"] == 0.0 and st["frac_below_bf16_sub"] == 0.0


def test_array_stats_all_zero():
    st = probe.array_stats(np.zeros(5))
    assert st["n_zero"] == 5 and st["max"] == 0.0 and st["frac_below_f16_sub"] == 0.0


def test_limit_constants_match_ieee754():
    assert probe.F16_MAX == 65504.0
    assert probe.F16_MIN_SUB == 2.0 ** -24
    assert probe.BF16_MIN_SUB == 2.0 ** -133
    assert probe.BF16_MAX == np.finfo(np.float32).max


# ---------- ProbeCollector ----------

def test_collector_roundtrip():
    pc = probe.ProbeCollector(lo=-8, hi=8, nbins=16)
    pc.snap("init", arrays={"dirty": np.array([1.0, 2.0, 0.0])},
            scalars={"psf_peak": 3.5})
    pc.snap("k1", arrays={"model": np.array([0.5, 4.0]), "residual": np.ones(4)},
            scalars={"res_peak": 0.25, "k": 1})
    d = pc.to_arrays()
    pc2 = probe.ProbeCollector.from_arrays(d)
    assert [s["label"] for s in pc2.snaps] == ["init", "k1"]
    assert np.array_equal(pc2.edges, pc.edges)
    assert np.array_equal(pc2.snaps[0]["hists"]["dirty"], pc.snaps[0]["hists"]["dirty"])
    assert pc2.snaps[1]["stats"]["model"]["max"] == 4.0
    assert pc2.snaps[0]["scalars"]["psf_peak"] == 3.5
    assert pc2.snaps[1]["scalars"]["res_peak"] == 0.25
    # heterogeneous keys across snaps survive
    assert "residual" in pc2.snaps[1]["hists"] and "residual" not in pc2.snaps[0]["hists"]


def test_collector_hist_counts_sum_to_nonzero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 100)) * 1e3
    pc = probe.ProbeCollector()
    s = pc.snap("init", arrays={"a": x})
    nz = int((np.abs(x) > 0).sum())
    assert int(s["hists"]["a"].sum()) == nz


# ---------- pipeline integration ----------

def _geom(n=128, cell_deg=1.0 / 128):
    return operators.Geometry(n, n, np.radians(cell_deg), np.radians(cell_deg))


def _ring_uv(n=64, radius_m=400.0, freq=1.4e9):
    ang = np.arange(n) * 2 * np.pi / n
    uvw = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1) * radius_m
    return uvw, np.array([freq])


def _tiny_vis(geom, uvw, freq, scale=1.0):
    truth = np.zeros((128, 128)); truth[60, 70] = 1.0
    vis = operators.dirty2vis(truth, uvw, freq, np.ones((uvw.shape[0], 1)), geom)
    return vis * scale


def test_pipeline_probe_snaps_and_scales():
    geom = _geom(128)
    uvw, freq = _ring_uv(64)
    wgt = np.ones((64, 1))
    vis = _tiny_vis(geom, uvw, freq)
    out = pipeline.run_pipeline(vis, uvw, freq, wgt, geom, loop_max=3,
                                nthreads=1, probe=True)
    pc = out["probes"]
    labels = [s["label"] for s in pc.snaps]
    assert labels[0] == "init" and labels[-1] == "final"
    assert any(l.startswith("k") for l in labels)
    init = pc.snaps[0]
    for var in ("vis", "dirty", "psf"):
        assert var in init["hists"]
    # histogram counts equal the nonzero counts of the returned arrays
    assert int(init["hists"]["dirty"].sum()) == int((np.abs(out["dirty"]) > 0).sum())
    assert int(init["hists"]["psf"].sum()) == int((np.abs(out["psf"]) > 0).sum())
    # scalars carry the per-cycle loop state
    ks = [s for s in pc.snaps if s["label"].startswith("k")]
    assert all("res_peak" in s["scalars"] and "res_mad" in s["scalars"] for s in ks)
    assert "s_vis" in init["scalars"]
    assert all("s_vis_resid" in s["scalars"] and "s_model" in s["scalars"] for s in ks)
    # scales are returned for auditing even without probing
    assert set(out["scales"]) == {"vis", "wgt", "dirty", "psf", "model"}


def test_pipeline_raw_cast_overflow_survives_and_is_probed():
    # rescale_max=None + |vis| ~ 1e6 > 65504: raw f16 cast overflows to inf,
    # the pipeline must still return (arrays may be non-finite) with probes.
    geom = _geom(128)
    uvw, freq = _ring_uv(64)
    wgt = np.ones((64, 1))
    vis = _tiny_vis(geom, uvw, freq, scale=1e6)
    with np.errstate(over="ignore", invalid="ignore"):
        out = pipeline.run_pipeline(vis, uvw, freq, wgt, geom, loop_max=2,
                                    nthreads=1, probe=True, q_vis="float16",
                                    rescale_max=None)
    assert "probes" in out
    assert [s["label"] for s in out["probes"].snaps][0] == "init"
    assert not np.isfinite(out["dirty"]).all()          # overflow did happen
    init = out["probes"].snaps[0]
    assert init["stats"]["vis"]["frac_above_f16_max"] > 0.0


def test_pipeline_probe_cast_arrays_appear_only_when_scaled():
    geom = _geom(128)
    uvw, freq = _ring_uv(64)
    wgt = np.ones((64, 1))
    vis = _tiny_vis(geom, uvw, freq, scale=1e6)   # |vis|max ~ 1e6 > 65503 -> rescale
    out = pipeline.run_pipeline(vis, uvw, freq, wgt, geom, loop_max=2,
                                nthreads=1, probe=True, q_vis="float16")
    init = out["probes"].snaps[0]
    assert init["scalars"]["s_vis"] < 1.0                       # rescale triggered
    assert "vis_cast" in init["hists"]
    # cast values stay finite and within the complex f16 envelope: real/imag are
    # each cast with |component| <= 65503, so the modulus can reach 65503*sqrt(2)
    st = init["stats"]["vis_cast"]
    assert st["max"] <= probe.F16_MAX * np.sqrt(2.0) + 1e-2
    assert st["frac_above_bf16_max"] == 0.0
    # unscaled stages have no duplicate _cast keys
    assert "dirty_cast" not in init["hists"]
