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


def test_wcs_roundtrip():
    wcs = sky.make_wcs(2520, 1.0 / 2520, -120.0, -60.0)
    ra = np.array([-120.1, -119.9, -120.0])
    dec = np.array([-60.1, -59.9, -60.0])
    x, y = sky.radec_to_pix(ra, dec, wcs)
    ra2, dec2 = sky.pix_to_radec(x, y, wcs)
    assert np.allclose(ra, ra2, atol=1e-10)
    assert np.allclose(dec, dec2, atol=1e-10)
