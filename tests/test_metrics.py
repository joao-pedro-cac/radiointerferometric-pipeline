"""Unit tests for src/metrics.py — every metric against analytic ground truth.

See tests/REPORT.md for the metric -> test -> expected mapping.
"""
import numpy as np
import pytest

from src import metrics


def test_cross_correlation_identical_is_one():
    x = np.random.default_rng(0).standard_normal((64, 64))
    assert metrics.cross_correlation(x, x) == pytest.approx(1.0, abs=1e-12)


def test_cross_correlation_scale_offset_invariant():
    x = np.random.default_rng(1).standard_normal((64, 64))
    assert metrics.cross_correlation(x, 3.7 * x + 11.0) == pytest.approx(1.0, abs=1e-12)


def test_cross_correlation_anticorrelated():
    x = np.random.default_rng(2).standard_normal((64, 64))
    assert metrics.cross_correlation(x, -x) == pytest.approx(-1.0, abs=1e-12)


def test_flux_conservation_exact():
    truth = np.zeros((32, 32)); truth[10, 12] = 2.5; truth[20, 5] = 0.5
    assert metrics.flux_conservation(truth.copy(), truth) == pytest.approx(1.0)
    assert metrics.flux_conservation(0.5 * truth, truth) == pytest.approx(0.5)


def test_residual_rms_constant_and_noise():
    c = np.full((100, 100), 3.0)
    assert metrics.residual_rms(c) == pytest.approx(3.0)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((2000, 2000)) * 0.01
    assert metrics.residual_rms(noise) == pytest.approx(0.01, rel=0.02)


def test_residual_mad_gaussian_noise():
    rng = np.random.default_rng(4)
    noise = rng.standard_normal((2000, 2000)) * 2.5
    # for Gaussian noise, 1.4826*MAD == sigma
    assert metrics.residual_mad(noise) == pytest.approx(2.5, rel=0.02)


def test_achieved_dr_definition():
    rng = np.random.default_rng(5)
    resid = rng.standard_normal((1000, 1000)) * 1e-4
    peak = 2.0
    expect = peak / metrics.residual_mad(resid)
    assert metrics.achieved_dr(resid, peak) == pytest.approx(expect, rel=1e-6)


def test_snr_db_identical_is_huge_and_monotone():
    rng = np.random.default_rng(6)
    t = rng.random((64, 64)) + 0.1
    s_high = metrics.snr_db(t, t)                      # mse = 0 -> eps-limited, very large
    s_low = metrics.snr_db(t + 0.5, t)
    assert s_high > 100.0
    assert s_low < s_high
