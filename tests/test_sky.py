"""Unit tests for src/sky.py — rendering and field sampling against analytics."""
import numpy as np
import pytest

from src import sky


def test_render_point_source():
    s = [sky.Source(x=10.4, y=20.6, flux=3.0, kind="point")]
    img = sky.render(s, 64)
    assert img.sum() == pytest.approx(3.0)
    assert img[21, 10] == pytest.approx(3.0)   # nearest pixel


def test_render_gauss_flux_normalized():
    s = [sky.Source(x=30.0, y=30.0, flux=2.0, kind="gauss",
                    bmaj=6.0, bmin=4.0, pa=0.4)]
    img = sky.render(s, 64)
    assert img.sum() == pytest.approx(2.0, rel=1e-6)
    sx, sy = 6.0 * sky.FWHM2SIG, 4.0 * sky.FWHM2SIG
    assert img.max() == pytest.approx(2.0 / (2 * np.pi * sx * sy), rel=1e-3)


def test_sample_field_realized_dr_exact():
    rng = np.random.default_rng(42)
    f = sky.sample_field(rng, n_point=5, n_gauss=3, dr=100.0, peak_flux=1.0,
                         ra0=-120.0, dec0=-60.0, fov_deg=1.0)
    assert len(f["ra"]) == 8
    assert f["meta"]["realized_dr"] == pytest.approx(100.0, rel=1e-9)
    assert f["flux"].max() == pytest.approx(1.0)
    assert f["flux"].min() == pytest.approx(0.01)


def test_sample_field_positions_inside_fov():
    rng = np.random.default_rng(1)
    f = sky.sample_field(rng, n_point=20, n_gauss=0, dr=10.0, peak_flux=1.0,
                         ra0=-120.0, dec0=-60.0, fov_deg=1.0)
    cosd = np.cos(np.radians(-60.0))
    dra = (f["ra"] + 120.0) * cosd
    ddec = f["dec"] + 60.0
    assert np.abs(dra).max() <= 1.0 / 2 + 1e-12
    assert np.abs(ddec).max() <= 1.0 / 2 + 1e-12


def test_render_diffuse_is_large_gauss():
    s = [sky.Source(x=100.0, y=100.0, flux=5.0, kind="diffuse",
                    bmaj=40.0, bmin=20.0, pa=0.0)]
    img = sky.render(s, 256)
    assert img.sum() == pytest.approx(5.0, rel=1e-5)
    sx, sy = 40.0 * sky.FWHM2SIG, 20.0 * sky.FWHM2SIG
    assert img.max() == pytest.approx(5.0 / (2 * np.pi * sx * sy), rel=1e-3)


def test_render_gauss_bounded_box_truncates_beyond_5sigma():
    s = [sky.Source(x=128.0, y=128.0, flux=1.0, kind="gauss", bmaj=20.0, bmin=20.0)]
    img = sky.render(s, 256, gauss_box_sigma=5.0)
    sx = 20.0 * sky.FWHM2SIG
    half = int(np.ceil(5.0 * sx))
    # outside the box the image is exactly zero (no full-grid evaluation)
    assert img[:128 - half, :].sum() == 0.0 and img[128 + half + 1:, :].sum() == 0.0
    # and the truncated flux is < 1e-5 of the total
    assert img.sum() == pytest.approx(1.0, rel=1e-5)


def test_render_1000_sources_is_fast():
    import time
    rng = np.random.default_rng(7)
    srcs = ([sky.Source(x=float(rng.uniform(0, 2520)), y=float(rng.uniform(0, 2520)),
                        flux=1e-3, kind="point") for _ in range(700)]
            + [sky.Source(x=float(rng.uniform(0, 2520)), y=float(rng.uniform(0, 2520)),
                          flux=1e-2, kind="gauss", bmaj=float(rng.uniform(7, 42)),
                          bmin=float(rng.uniform(2, 42)), pa=float(rng.uniform(0, 3)))
               for _ in range(250)]
            + [sky.Source(x=float(rng.uniform(0, 2520)), y=float(rng.uniform(0, 2520)),
                          flux=1.0, kind="diffuse", bmaj=float(rng.uniform(42, 280)),
                          bmin=float(rng.uniform(15, 280)), pa=float(rng.uniform(0, 3)))
               for _ in range(50)])
    t0 = time.time()
    img = sky.render(srcs, 2520)
    dt = time.time() - t0
    assert img.sum() > 0
    assert dt < 30.0          # bounded render: seconds, not hours


def test_sample_field_with_diffuse():
    rng = np.random.default_rng(11)
    f = sky.sample_field(rng, n_point=10, n_gauss=5, n_diffuse=3, dr=1e3,
                         peak_flux=1.0, ra0=-120.0, dec0=-60.0, fov_deg=1.0)
    assert len(f["ra"]) == 18
    assert f["meta"]["n_diffuse"] == 3
    assert list(f["kind"][-3:]) == ["diffuse"] * 3
    assert f["meta"]["realized_dr"] == pytest.approx(1e3, rel=1e-9)
    assert f["maj"][-3:].min() >= 60.0     # diffuse sizes in the diffuse range


def test_wcs_roundtrip():
    wcs = sky.make_wcs(2520, 1.0 / 2520, -120.0, -60.0)
    ra = np.array([-120.1, -119.9, -120.0])
    dec = np.array([-60.1, -59.9, -60.0])
    x, y = sky.radec_to_pix(ra, dec, wcs)
    ra2, dec2 = sky.pix_to_radec(x, y, wcs)
    assert np.allclose(ra, ra2, atol=1e-10)
    assert np.allclose(dec, dec2, atol=1e-10)
