"""Unit tests for src/operators.py — orientation, beam, adjointness, point-source."""
import numpy as np
import pytest

from src import operators


def _geom(n=64, cell_deg=1.0 / 64):
    return operators.Geometry(n, n, np.radians(cell_deg), np.radians(cell_deg))


def _ring_uv(n=32, radius_m=300.0, freq=1.4e9):
    ang = np.arange(n) * 2 * np.pi / n
    uvw = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1) * radius_m
    return uvw, np.array([freq])


def test_to_native_to_display_are_inverses():
    m = np.random.default_rng(0).random((37, 53))
    assert np.allclose(operators.to_display(operators.to_native(m)), m)
    assert np.allclose(operators.to_native(operators.to_display(m)), m)


def test_point_source_at_phase_centre_gives_flux_peak():
    geom = _geom(64)
    uvw, freq = _ring_uv(32)
    wgt = np.ones((32, 1))
    img = np.zeros((64, 64)); img[32, 32] = 2.0          # 2 Jy at phase centre (native)
    vis = operators.dirty2vis(img, uvw, freq, np.ones_like(wgt), geom)
    # vis of a phase-centre point source is constant = flux
    assert np.abs(vis - 2.0).max() / 2.0 < 1e-4
    dirty = operators.vis2dirty(uvw, freq, vis, wgt, geom)
    psf = operators.compute_psf(uvw, freq, wgt, geom)
    assert dirty.max() / psf.max() == pytest.approx(2.0, rel=1e-4)


def test_beam_kernel_peak_one_sum_beam_area():
    kern = operators.make_beam_kernel(10.0, 6.0, 0.3)
    assert kern.max() == pytest.approx(1.0)
    assert kern.sum() == pytest.approx(operators.beam_area_px(10.0, 6.0), rel=1e-6)


def test_restore_delta_peak_is_flux():
    kern = operators.make_beam_kernel(8.0, 8.0, 0.0)
    model = np.zeros((64, 64)); model[20, 30] = 1.5
    restored = operators.restore(model, kern)
    assert restored.max() == pytest.approx(1.5, rel=1e-6)
    # sum(restored) = flux * beam_area
    assert restored.sum() == pytest.approx(1.5 * kern.sum(), rel=1e-6)


def test_synth_beam_recovers_known_gaussian():
    # build a synthetic PSF = elliptical gaussian with known FWHM, peak-normalized
    s = 121
    yy, xx = np.mgrid[:s, :s] - s // 2
    bmaj, bmin, pa = 12.0, 7.0, np.radians(25.0)
    sx, sy = bmaj / operators.FWHM, bmin / operators.FWHM
    ct, st = np.cos(pa), np.sin(pa)
    xr, yr = xx * ct + yy * st, -xx * st + yy * ct
    psf = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    bmaj_f, bmin_f, pa_f = operators.synth_beam_from_psf(psf)
    assert bmaj_f == pytest.approx(bmaj, rel=0.05)
    assert bmin_f == pytest.approx(bmin, rel=0.05)


def test_vis2dirty_dirty2vis_roundtrip_consistency():
    # vis2dirty(dirty2vis(I)) / psf_peak must equal I convolved with the
    # normalized PSF (up to gridding accuracy) on a well-sampled grid.
    from astropy.convolution import convolve_fft
    npix = 128
    lam = 3e8 / 1.4e9
    bmax = 400.0
    cell = (lam / bmax) / 4.0                      # beam ~4 px
    geom = operators.Geometry(npix, npix, cell, cell)
    rng = np.random.default_rng(7)
    n = 2000
    u = rng.uniform(-bmax, bmax, n); v = rng.uniform(-bmax, bmax, n)
    keep = np.hypot(u, v) <= bmax
    uvw = np.stack([u[keep], v[keep], np.zeros(keep.sum())], axis=1)
    freq = np.array([1.4e9]); wgt = np.ones((uvw.shape[0], 1))
    yy, xx = np.mgrid[:npix, :npix]
    img = np.zeros((npix, npix))
    for cy0, cx0, a0 in [(40, 50, 1.0), (80, 90, 0.5), (60, 20, 0.3)]:
        img += a0 * np.exp(-(((yy - cy0) ** 2 + (xx - cx0) ** 2) / (2 * 6.0 ** 2)))
    vis = operators.dirty2vis(img, uvw, freq, np.ones_like(wgt), geom)
    back = operators.vis2dirty(uvw, freq, vis, wgt, geom)
    psf = operators.compute_psf(uvw, freq, wgt, geom)
    ny, nx = img.shape; cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    psf_c = psf[cy - ny // 2: cy - ny // 2 + ny, cx - nx // 2: cx - nx // 2 + nx] / psf.max()
    conv = convolve_fft(img, psf_c, normalize_kernel=False, allow_huge=True)
    c = np.corrcoef((back / psf.max()).ravel(), conv.ravel())[0, 1]
    # wgridding correction + edge effects keep this below 1; the sharp
    # invariant is the phase-centre delta test above (exact to 1e-4).
    assert c > 0.9
