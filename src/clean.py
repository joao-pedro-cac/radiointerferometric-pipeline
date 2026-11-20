"""Högbom CLEAN with the major-cycle (feedback) loop.

Pure functions: arrays in, arrays out. No I/O.

Units: vis2dirty (do_wgridding) maps a unit point source to peak ~1, so the
dirty image is in Jy and the PSF peak is dimensionless-ish; we normalize both
by psf_peak so CLEAN works in Jy/beam. The returned model is a delta image in
Jy (numerically equal to the normalized CLEAN model because the normalized
PSF has peak 1).

The major loop's stall test compares the residual peak against the *previous*
iteration's peak, so `prev_peak` MUST start at +inf — starting at 0 made the
test vacuously true at k=1 and every historical run stopped after one major
cycle (carnet-de-bord F1).
"""
import numpy as np

import pfb_imaging.deconv.hogbom as pfb_hogbom

from . import operators


def hogbom(dirty, psf, gamma=0.1, pf=1e-4, maxiter=10000, threshold_mad=3.0):
    """One Högbom deconvolution. dirty in Jy (npix grid), psf on the 2x grid.

    Returns (model [Jy delta image], info dict).
    """
    dirty = np.asarray(dirty, np.float64)
    psf = np.asarray(psf, np.float64)
    psf_peak = float(np.max(psf))
    dirty_n = dirty / psf_peak
    psf_n = psf / psf_peak
    mad = 1.4826 * np.median(np.abs(dirty_n - np.median(dirty_n)))
    model_cube, status = pfb_hogbom.hogbom(
        dirty_n[np.newaxis, :, :],
        psf_n[np.newaxis, :, :],
        threshold=threshold_mad * mad,   # pf remains the effective fractional stop
        gamma=gamma, pf=pf, maxit=int(maxiter), verbosity=0,
    )
    return np.squeeze(model_cube), dict(status=int(status), psf_peak=psf_peak,
                                        mad=float(mad))


def clean_loop(vis, uvw, freq, wgt, geom, psf,
               gamma=0.1, pf=1e-4, maxiter=10000, loop_max=10, eps=1e-5):
    """Högbom + major-cycle residual re-imaging through the visibilities.

    vis : data visibilities in Jy (nrow, nchan); wgt : imaging weights.
    The model is kept in Jy (delta image); the residual of each major cycle is
    re-imaged from (vis - dirty2vis(model, unit weights)) — this is the step
    where visibility quantization enters the pipeline.

    Returns dict(model [Jy], residual [Jy/beam, PSF-peak normalized],
                 psf_peak, n_major_cycles, converged_reason, n_components).
    """
    psf_peak = float(np.max(psf))
    dirty = operators.vis2dirty(uvw, freq, vis, wgt, geom, eps)
    model, info = hogbom(dirty, psf, gamma, pf, maxiter)   # beam units == Jy delta

    ones = np.ones_like(wgt)
    residual = dirty.copy()
    k = 0
    prev_peak = np.inf                      # F1: must start at +inf, not 0
    converged = "max_major"
    while k < loop_max:
        k += 1
        model_vis = operators.dirty2vis(model, uvw, freq, ones, geom, eps)
        residual = operators.vis2dirty(uvw, freq, vis - model_vis, wgt, geom, eps)
        res_peak = float(np.max(np.abs(residual))) / psf_peak   # Jy/beam
        res_mad = 1.4826 * np.median(np.abs(residual / psf_peak
                                             - np.median(residual / psf_peak)))
        if not np.isfinite(res_peak) or res_peak == 0.0:
            converged = "zero_residual"
            break
        if res_peak < 3.0 * res_mad:
            converged = "threshold"
            break
        if k > 1 and (prev_peak - res_peak) < 0.02 * prev_peak:
            converged = "stall"
            break
        prev_peak = res_peak
        comp, _ = hogbom(residual, psf, gamma, pf, maxiter)     # beam units
        model = model + comp

    return dict(model=model, residual=residual / psf_peak, psf_peak=psf_peak,
                n_major_cycles=int(k), converged_reason=converged,
                n_components=int(np.count_nonzero(model)))
