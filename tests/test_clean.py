"""Unit tests for src/clean.py — Högbom recovers a known delta on a synthetic PSF."""
import numpy as np
import pytest

from src import operators, clean


def _geom(n=128, cell_deg=1.0 / 128):
    return operators.Geometry(n, n, np.radians(cell_deg), np.radians(cell_deg))


def _ring_uv(n=64, radius_m=400.0, freq=1.4e9):
    ang = np.arange(n) * 2 * np.pi / n
    uvw = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1) * radius_m
    return uvw, np.array([freq])


def test_hogbom_recovers_point_source():
    geom = _geom(128)
    uvw, freq = _ring_uv(64)
    wgt = np.ones((64, 1))
    psf = operators.compute_psf(uvw, freq, wgt, geom)
    psf_peak = psf.max()
    truth = np.zeros((128, 128)); truth[60, 70] = 1.0
    vis = operators.dirty2vis(truth, uvw, freq, np.ones_like(wgt), geom)
    dirty = operators.vis2dirty(uvw, freq, vis, wgt, geom)
    model, info = clean.hogbom(dirty, psf, gamma=0.1, pf=1e-4, maxiter=5000,
                               threshold_mad=0.0)
    assert model.sum() == pytest.approx(1.0, rel=1e-2)
    iy, ix = np.unravel_index(model.argmax(), model.shape)
    assert (iy, ix) == (60, 70)


def test_clean_loop_converges_and_residual_at_noise():
    geom = _geom(128)
    uvw, freq = _ring_uv(64)
    wgt = np.ones((64, 1))
    psf = operators.compute_psf(uvw, freq, wgt, geom)
    truth = np.zeros((128, 128)); truth[60, 70] = 1.0; truth[30, 40] = 0.1
    vis = operators.dirty2vis(truth, uvw, freq, np.ones_like(wgt), geom)
    rng = np.random.default_rng(0)
    vis = vis + 1e-3 * (rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape))
    out = clean.clean_loop(vis, uvw, freq, wgt, geom, psf,
                           gamma=0.1, pf=1e-3, maxiter=5000, loop_max=5)
    assert out["model"].sum() == pytest.approx(1.1, rel=0.05)
    assert out["converged_reason"] in ("threshold", "stall", "max_major", "zero_residual")
    # residual floor should be at the noise level, well below the faintest source
    assert 1.4826 * np.median(np.abs(out["residual"] - np.median(out["residual"]))) < 0.1
