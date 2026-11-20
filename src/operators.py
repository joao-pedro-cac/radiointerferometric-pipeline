"""Gridding / degridding operators around ducc0.wgridder, plus beam utilities.

Pure functions: arrays + a Geometry in, arrays out. No I/O.

Orientation convention: ducc0 works in a "native" grid; the FITS/display
frame used for truth images is related by the involution
    display = transpose(native)[::-1, :]
(verified: applying it twice returns the original array).
"""
from dataclasses import dataclass

import numpy as np
import ducc0.wgridder as dg
from astropy.convolution import convolve_fft
from astropy.modeling import models, fitting

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class Geometry:
    npix_x: int
    npix_y: int
    pixsize_x: float   # rad
    pixsize_y: float   # rad


def to_display(native_img):
    """ducc0 native grid -> FITS display frame (legacy convention)."""
    return np.transpose(native_img)[::-1, :]


def to_native(display_img):
    """FITS display frame -> ducc0 native grid (exact inverse of to_display)."""
    return np.transpose(display_img[::-1, :])


def _prep(uvw, freq, wgt):
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    freq = np.ascontiguousarray(freq, dtype=np.float64)
    wgt = np.ascontiguousarray(wgt, dtype=np.float64)
    return uvw, freq, wgt


def vis2dirty(uvw, freq, vis, wgt, geom, eps=1e-5):
    """Visibilities -> dirty image on geom's grid (native orientation)."""
    uvw, freq, wgt = _prep(uvw, freq, wgt)
    vis = np.ascontiguousarray(vis, dtype=np.complex128)
    return dg.vis2dirty(uvw=uvw, freq=freq, vis=vis, wgt=wgt,
                        npix_x=geom.npix_x, npix_y=geom.npix_y,
                        pixsize_x=geom.pixsize_x, pixsize_y=geom.pixsize_y,
                        epsilon=eps, do_wgridding=True, nthreads=1,
                        double_precision_accumulation=True, verbosity=False)


def dirty2vis(img, uvw, freq, wgt, geom, eps=1e-5):
    """Image (native orientation) -> visibilities. Adjoint of vis2dirty."""
    uvw, freq, wgt = _prep(uvw, freq, wgt)
    img = np.ascontiguousarray(img, dtype=np.float64)
    return dg.dirty2vis(uvw=uvw, freq=freq, dirty=img, wgt=wgt,
                        pixsize_x=geom.pixsize_x, pixsize_y=geom.pixsize_y,
                        center_x=0, center_y=0,
                        epsilon=eps, do_wgridding=True, nthreads=1,
                        verbosity=False)


def compute_psf(uvw, freq, wgt, geom, eps=1e-5):
    """PSF = dirty image of unit visibilities on a 2x padded grid (native)."""
    uvw, freq, wgt = _prep(uvw, freq, wgt)
    nvis = uvw.shape[0]
    nchan = freq.shape[0]
    ones = np.ones((nvis, nchan), dtype=np.complex128)
    return dg.vis2dirty(uvw=uvw, freq=freq, vis=ones, wgt=wgt,
                        npix_x=2 * geom.npix_x, npix_y=2 * geom.npix_y,
                        pixsize_x=geom.pixsize_x, pixsize_y=geom.pixsize_y,
                        epsilon=eps, do_wgridding=True, nthreads=1,
                        double_precision_accumulation=True, verbosity=False)


# ---------------------------------------------------------------------------
# clean/restoring beam: fit the PSF main lobe, restore model to that resolution
# ---------------------------------------------------------------------------

def synth_beam_from_psf(psf, fit_halfwidth_px=32):
    """Fit a 2-D Gaussian to the PSF main lobe -> (bmaj_px, bmin_px, pa_rad) FWHM."""
    psf = np.asarray(psf, dtype=np.float64)
    yx = np.unravel_index(np.argmax(psf), psf.shape)
    h = fit_halfwidth_px
    sl = (slice(max(yx[0] - h, 0), yx[0] + h + 1),
          slice(max(yx[1] - h, 0), yx[1] + h + 1))
    cut = psf[sl]
    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
    p0 = models.Gaussian2D(cut.max(), cut.shape[1] // 2, cut.shape[0] // 2, 2.0, 2.0)
    g = fitting.LevMarLSQFitter()(p0, xx, yy, cut)
    sx, sy = abs(g.x_stddev.value), abs(g.y_stddev.value)
    return FWHM * max(sx, sy), FWHM * min(sx, sy), float(g.theta.value)


def beam_area_px(bmaj, bmin):
    """Beam area in pixels for a peak-normalized Gaussian restoring kernel."""
    return np.pi / (4.0 * np.log(2.0)) * bmaj * bmin


def make_beam_kernel(bmaj_px, bmin_px, pa_rad, size=None):
    """Peak-normalized elliptical Gaussian kernel (sum = beam_area_px)."""
    s = size or int(2 * np.ceil(2 * bmaj_px) + 1)
    yy, xx = np.mgrid[:s, :s] - s // 2
    sx, sy = bmaj_px / FWHM, bmin_px / FWHM
    ct, st = np.cos(pa_rad), np.sin(pa_rad)
    xr, yr = xx * ct + yy * st, -xx * st + yy * ct
    return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def restore(model, kernel):
    """Convolve a (delta) model with a peak-normalized beam -> Jy/beam image."""
    return convolve_fft(np.asarray(model, np.float64), kernel,
                        normalize_kernel=False, allow_huge=True)
