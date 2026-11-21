"""Full imaging pipeline with per-stage quantization.

Pure orchestration over src modules: arrays in, dict of arrays+metrics out.
No I/O. Quantization semantics: a stage quantized to dtype d stores
cast(x * s) and the pipeline continues with the de-scaled value
cast(x * s) / s — i.e. quantization acts as pure value rounding, with no
scale bookkeeping in the math. `s` is chosen from the f64 max (f16 range
protection) unless given.

Stages: visibilities (and weights), dirty image, PSF, clean model, and the
re-imaged residual visibilities inside the major loop.
"""
import time

import numpy as np

from . import operators, clean, quantize, metrics, probe as probe_mod


def _q_vis(vis, dtype, rescale_max, scale=None):
    q, s = quantize.quantize_visibilities(vis, dtype, rescale_max, scale=scale)
    return np.asarray(q / s, dtype=np.complex128), s


def _q_img(img, dtype, rescale_max, scale=None):
    q, s = quantize.quantize_array(img, dtype, rescale_max, scale=scale)
    return np.asarray(q / s, dtype=np.float64), s


def run_pipeline(vis, uvw, freq, wgt, geom,
                 gamma=0.1, pf=1e-4, maxiter=10000, loop_max=10,
                 eps=1e-5, nthreads=8,
                 q_vis="float64", q_dirty="float64", q_psf="float64",
                 q_model="float64", rescale_max=65503.0, probe=False):
    """Run dirty image + PSF + CLEAN loop with per-stage quantization.

    Returns dict with arrays (dirty, psf, model, residual — native frame) and
    metrics computed in the display frame, plus timing and loop info.
    If probe=True, also returns "probes": a probe.ProbeCollector with a log2
    histogram + stats snapshot of every stage array at init, after each major
    cycle, and at the end (see src/probe.py for the snapshot convention).
    """
    t = {}
    pc = probe_mod.ProbeCollector() if probe else None

    def _casts(d):
        # {name_cast: raw * scale} for stages whose scale != 1 (otherwise the
        # effective array already equals the cast array)
        return {name + "_cast": arr * s for name, (arr, s) in d.items() if s != 1.0}

    def _snap(label, arrays, casts=None, scalars=None):
        # arrays: physical units (vis/model in Jy, images in Jy/beam, wgt raw).
        # casts: {name_cast: raw_array * scale} — values as stored in the
        # low-precision dtype.
        if pc is None:
            return
        arrays = dict(arrays)
        arrays.update(casts or {})
        pc.snap(label, arrays=arrays, scalars=scalars or {})

    t0 = time.time()
    vis_eff, s_vis = _q_vis(vis, q_vis, rescale_max)
    wgt_eff, s_w = quantize.quantize_weights(wgt, q_vis, rescale_max)
    wgt_eff = np.asarray(wgt_eff / s_w, dtype=np.float64)

    dirty = operators.vis2dirty(uvw, freq, vis_eff, wgt_eff, geom, eps, nthreads)
    dirty_eff, s_dirty = _q_img(dirty, q_dirty, rescale_max)

    psf = operators.compute_psf(uvw, freq, wgt_eff, geom, eps, nthreads)
    psf_peak = float(np.max(psf))                    # f64 reference normalization
    psf_eff, s_psf = _q_img(psf, q_psf, rescale_max)
    t["imaging"] = time.time() - t0
    _snap("init",
          dict(vis=vis_eff, wgt=wgt_eff, dirty=dirty_eff / psf_peak,
               psf=psf_eff / psf_peak),
          casts=_casts(dict(vis=(vis_eff, s_vis), wgt=(wgt_eff, s_w),
                            dirty=(dirty_eff, s_dirty), psf=(psf_eff, s_psf))),
          scalars=dict(psf_peak=psf_peak, s_vis=s_vis, s_wgt=s_w,
                       s_dirty=s_dirty, s_psf=s_psf))

    t0 = time.time()
    model, info = clean.hogbom(dirty_eff, psf_eff, gamma, pf, maxiter)
    model, s_model = _q_img(model, q_model, rescale_max)

    ones = np.ones_like(wgt_eff)
    residual = dirty_eff.copy()
    k = 0
    prev_peak = np.inf
    converged = "max_major"
    while k < loop_max:
        k += 1
        model_vis = operators.dirty2vis(model, uvw, freq, ones, geom, eps, nthreads)
        resid_vis, _ = _q_vis(vis_eff - model_vis, q_vis, rescale_max, scale=s_vis)
        residual = operators.vis2dirty(uvw, freq, resid_vis, wgt_eff, geom, eps, nthreads)
        residual_eff, _ = _q_img(residual, q_dirty, rescale_max, scale=s_dirty)
        res_peak = float(np.max(np.abs(residual_eff))) / psf_peak
        res_mad = 1.4826 * np.median(np.abs(residual_eff / psf_peak
                                             - np.median(residual_eff / psf_peak)))
        _snap(f"k{k}",
              dict(vis_resid=resid_vis, residual=residual_eff / psf_peak, model=model),
              casts=_casts(dict(vis_resid=(resid_vis, s_vis),
                                residual=(residual_eff, s_dirty),
                                model=(model, s_model))),
              scalars=dict(k=k, res_peak=res_peak, res_mad=res_mad,
                           n_components=int(np.count_nonzero(model)),
                           s_vis_resid=s_vis, s_residual=s_dirty, s_model=s_model))
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
        comp, _ = clean.hogbom(residual_eff, psf_eff, gamma, pf, maxiter)
        model, s_model = _q_img(model + comp, q_model, rescale_max)
    t["clean"] = time.time() - t0
    _snap("final",
          dict(model=model, residual=residual_eff / psf_peak,
               dirty=dirty_eff / psf_peak, psf=psf_eff / psf_peak),
          casts=_casts(dict(model=(model, s_model), residual=(residual_eff, s_dirty),
                            dirty=(dirty_eff, s_dirty), psf=(psf_eff, s_psf))),
          scalars=dict(n_major_cycles=k, psf_peak=psf_peak))

    residual_beam = residual_eff / psf_peak
    out = dict(dirty=dirty_eff, psf=psf_eff, psf_peak=psf_peak,
               model=model, residual=residual_beam,
               n_major_cycles=int(k), converged_reason=converged,
               n_components=int(np.count_nonzero(model)),
               scales=dict(vis=s_vis, wgt=s_w, dirty=s_dirty, psf=s_psf,
                           model=s_model),
               timing=t)
    if pc is not None:
        out["probes"] = pc
    return out


def compute_metrics(model, residual, psf, truth):
    """Metrics in the display frame. model/residual/psf/truth: model & residual
    native, psf 2x native, truth display. Returns dict of floats + beam."""
    bmaj, bmin, pa = operators.synth_beam_from_psf(psf)
    kern = operators.make_beam_kernel(bmaj, bmin, pa)
    model_d = operators.to_display(model)
    resid_d = operators.to_display(residual)
    restored = operators.restore(model_d, kern)
    true_restored = operators.restore(np.asarray(truth, np.float64), kern)
    return dict(
        cross_correlation=metrics.cross_correlation(restored, true_restored),
        flux_conservation=metrics.flux_conservation(model_d, truth),
        residual_rms=metrics.residual_rms(resid_d),
        residual_mad=metrics.residual_mad(resid_d),
        achieved_dr=metrics.achieved_dr(resid_d, float(restored.max())),
        snr_db=metrics.snr_db(restored, true_restored),
        beam_bmaj_px=float(bmaj), beam_bmin_px=float(bmin),
        beam_area_px=float(operators.beam_area_px(bmaj, bmin)),
        restored_peak=float(restored.max()),
    )
