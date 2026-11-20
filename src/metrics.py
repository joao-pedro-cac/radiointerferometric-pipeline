"""Image metrics — small, physical, unit-testable set.

All functions are pure (arrays in, floats out) and MUST have analytic unit
tests in tests/ (see tests/REPORT.md). Conventions:

- model   : CLEAN delta model, Jy/pixel on the truth grid
- truth   : sky image, Jy/pixel
- restored: model (or truth) convolved with the peak-normalized clean beam,
            Jy/beam
- residual: residual dirty image, Jy/beam (PSF-peak normalized)
"""
import numpy as np


def cross_correlation(a, b):
    """Zero-lag Pearson correlation in [-1, 1]. Scale/offset invariant."""
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float("nan") if denom == 0.0 else float(np.dot(x, y) / denom)


def flux_conservation(model, truth, eps=1e-12):
    """sum(model)/sum(truth) — total deconvolved flux over true flux [Jy].

    Equal to sum(restored)/beam_area_px / sum(truth) because the restoring
    kernel is peak-normalized with sum = beam_area_px.
    """
    m = float(np.sum(np.asarray(model, np.float64)))
    t = float(np.sum(np.asarray(truth, np.float64)))
    return m / (t + eps)


def residual_rms(residual):
    """RMS of the residual image (Jy/beam)."""
    r = np.asarray(residual, np.float64)
    return float(np.sqrt(np.mean(r * r)))


def residual_mad(residual):
    """Robust residual scale: 1.4826 * MAD (Jy/beam)."""
    r = np.asarray(residual, np.float64)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def achieved_dr(residual, restored_peak, eps=1e-12):
    """Achieved dynamic range = restored peak / robust residual floor (linear)."""
    return float(restored_peak / (residual_mad(residual) + eps))


def snr_db(restored, truth_restored, eps=1e-12):
    """SNR vs truth (dB): truth power over MSE."""
    a = np.asarray(restored, np.float64)
    b = np.asarray(truth_restored, np.float64)
    mse = np.mean((a - b) ** 2)
    return float(10.0 * np.log10(np.mean(b * b) / (mse + eps)))
