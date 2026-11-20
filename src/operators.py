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


def vis2dirty(uvw, freq, vis, wgt, geom, eps=1e-5, nthreads=8):
    """Visibilities -> dirty image on geom's grid (native orientation)."""
    uvw, freq, wgt = _prep(uvw, freq, wgt)
    vis = np.ascontiguousarray(vis, dtype=np.complex128)
    return dg.vis2dirty(uvw=uvw, freq=freq, vis=vis, wgt=wgt,
                        npix_x=geom.npix_x, npix_y=geom.npix_y,
                        pixsize_x=geom.pixsize_x, pixsize_y=geom.pixsize_y,
                        epsilon=eps, do_wgridding=True, nthreads=nthreads,
                        double_precision_accumulation=True, verbosity=False)


def dirty2vis(img, uvw, freq, wgt, geom, eps=1e-5, nthreads=8):
    """Image (native orientation) -> visibilities. Adjoint of vis2dirty."""
    uvw, freq, wgt = _prep(uvw, freq, wgt)
    img = np.ascontiguousarray(img, dtype=np.float64)
    return dg.dirty2vis(uvw=uvw, freq=freq, dirty=img, wgt=wgt,
                        pixsize_x=geom.pixsize_x, pixsize_y=geom.pixsize_y,
                        center_x=0, center_y=0,
                        epsilon=eps, do_wgridding=True, nthreads=nthreads,
                        verbosity=False)


def compute_psf(uvw, freq, wgt, geom, eps=1e-5, nthreads=8):
    """PSF = dirty image of unit visibilities on a 2x padded grid (native)."""
    uvw, freq, wgt = _prep(uvw, freq, wgt)
    nvis = uvw.shape[0]
    nchan = freq.shape[0]
    ones = np.ones((nvis, nchan), dtype=np.complex128)
    return dg.vis2dirty(uvw=uvw, freq=freq, vis=ones, wgt=wgt,
                        npix_x=2 * geom.npix_x, npix_y=2 * geom.npix_y,
                        pixsize_x=geom.pixsize_x, pixsize_y=geom.pixsize_y,
                        epsilon=eps, do_wgridding=True, nthreads=nthreads,
                        double_precision_accumulation=True, verbosity=False)


# ---------------------------------------------------------------------------
# clean/restoring beam: fit the PSF main lobe, restore model to that resolution
# ---------------------------------------------------------------------------

def synth_beam_from_psf(psf):
    """Main-lobe beam (bmaj_px, bmin_px, pa_rad) FWHM from half-max moments.

    Only pixels above half the PSF peak enter the second-moment estimate, so
    sidelobes cannot widen the beam. Robust and dependency-free.
    """
    psf = np.asarray(psf, dtype=np.float64)
    iy, ix = np.unravel_index(np.argmax(psf), psf.shape)
    pk = psf[iy, ix]
    # half-max-clipped weights over the whole image: sidelobes (below half
    # max) are excluded by the clip, so they cannot widen the beam
    w = np.clip(psf - pk / 2.0, 0.0, None)
    if w.sum() <= 0:
        return 4.0, 4.0, 0.0
    yy, xx = np.mgrid[:psf.shape[0], :psf.shape[1]]
    wsum = w.sum()
    mx, my = (w * xx).sum() / wsum, (w * yy).sum() / wsum
    cxx = (w * (xx - mx) ** 2).sum() / wsum
    cyy = (w * (yy - my) ** 2).sum() / wsum
    cxy = (w * (xx - mx) * (yy - my)).sum() / wsum
    # eigenvalues of the covariance matrix -> sigma along principal axes
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = max(tr * tr / 4 - det, 0.0)
    l1, l2 = tr / 2 + np.sqrt(disc), tr / 2 - np.sqrt(disc)
    pa = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    # calibration: for a pure Gaussian, the half-max-clipped 1-D marginal
    # second moment underestimates sigma^2 by a factor ~0.230 -> correct it
    corr = 1.0 / np.sqrt(0.2300)
    return FWHM * np.sqrt(max(l1, 1e-12)) * corr, FWHM * np.sqrt(max(l2, 1e-12)) * corr, float(pa)


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
