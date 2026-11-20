"""Analytic sky models on the image grid.

Pure functions: arrays/components in, arrays out. No I/O, no file paths.
Coordinates are pixel coordinates in the *display* orientation (the one
written to truth FITS); convert with operators.to_native before gridding.
"""
from dataclasses import dataclass

import numpy as np
from astropy.wcs import WCS

FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class Source:
    """One sky component in pixel coordinates (display orientation)."""
    x: float            # pixel x (float, 0-indexed)
    y: float            # pixel y
    flux: float         # integrated flux, Jy
    kind: str           # "point" | "gauss"
    bmaj: float = 0.0   # gaussian FWHM, pixels
    bmin: float = 0.0
    pa: float = 0.0     # position angle, rad, E of N


def make_wcs(npix, cell_deg, ra0, dec0):
    """SIN projection WCS matching the OSKAR truth images (CARTA-compatible)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [npix / 2 + 1, npix / 2 + 1]
    w.wcs.cdelt = [-cell_deg, cell_deg]
    w.wcs.cunit = ["deg", "deg"]
    return w


def radec_to_pix(ra, dec, wcs):
    """RA/Dec (deg) -> float pixel coords (0-indexed, display orientation)."""
    x, y = wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
    return x, y


def pix_to_radec(x, y, wcs):
    return wcs.all_pix2world(np.asarray(x, float), np.asarray(y, float), 0)


def sample_field(rng, n_point, n_gauss, dr, peak_flux, ra0, dec0, fov_deg,
                 maj_arcsec_range=(10.0, 60.0), axis_ratio_range=(0.3, 1.0),
                 freq0=1.4e9, spix=-0.7):
    """Draw a sky field: n_point point sources + n_gauss gaussians.

    Fluxes are log-uniform in [peak_flux/dr, peak_flux] with the brightest
    forced to peak_flux, so realized intra-field DR == dr exactly.
    Positions are uniform in a square of side fov_deg centred on (ra0, dec0),
    with the RA offset scaled by 1/cos(dec0) (angularly square field).
    Returns (sources, meta) with sources in RA/Dec + osm columns.
    """
    n = n_point + n_gauss
    if n == 0:
        raise ValueError("empty sky field")
    if dr < 1:
        raise ValueError("dr must be >= 1")
    s_min = peak_flux / dr
    if n == 1:
        flux = np.array([peak_flux])
    else:
        # pin both endpoints so realized DR == dr exactly
        mid = 10.0 ** rng.uniform(np.log10(s_min), np.log10(peak_flux), n - 2)
        flux = np.concatenate(([peak_flux, s_min], mid))
    rng.shuffle(flux)

    cosd = np.cos(np.radians(dec0))
    dra = rng.uniform(-fov_deg / 2, fov_deg / 2, n) / cosd
    ddec = rng.uniform(-fov_deg / 2, fov_deg / 2, n)
    ra = ra0 + dra
    dec = dec0 + ddec

    maj = np.zeros(n)
    minr = np.zeros(n)
    pa = np.zeros(n)
    if n_gauss > 0:
        g = slice(n_point, n) if n_point > 0 else slice(0, n)
        # gaussians get sizes in arcsec, converted by caller via cell size
        maj[g] = rng.uniform(*maj_arcsec_range, n_gauss)
        minr[g] = maj[g] * rng.uniform(*axis_ratio_range, n_gauss)
        pa[g] = rng.uniform(0.0, 180.0, n_gauss)

    kind = np.array(["point"] * n_point + ["gauss"] * n_gauss)
    meta = dict(n_point=int(n_point), n_gauss=int(n_gauss), dr=float(dr),
                peak_flux=float(peak_flux), ra0=float(ra0), dec0=float(dec0),
                fov_deg=float(fov_deg), freq0=float(freq0), spix=float(spix),
                realized_dr=float(flux.max() / flux.min()))
    return dict(ra=ra, dec=dec, flux=flux, kind=kind, maj=maj, minr=minr,
                pa=pa, meta=meta)


def to_sources(field, wcs, cell_arcsec):
    """Convert an RA/Dec field dict to pixel-space Source list (display frame)."""
    x, y = radec_to_pix(field["ra"], field["dec"], wcs)
    sources = []
    for i in range(len(x)):
        if field["kind"][i] == "gauss":
            sources.append(Source(x[i], y[i], float(field["flux"][i]), "gauss",
                                  float(field["maj"][i]) / cell_arcsec,
                                  float(field["minr"][i]) / cell_arcsec,
                                  np.radians(float(field["pa"][i]))))
        else:
            sources.append(Source(x[i], y[i], float(field["flux"][i]), "point"))
    return sources


def render(sources, npix):
    """Rasterize sources to a (npix, npix) Jy/pixel image (display orientation).

    Point source -> flux deposited in the nearest pixel.
    Gaussian -> elliptical Gaussian normalized so sum(image) == flux (Jy).
    """
    img = np.zeros((npix, npix), dtype=np.float64)
    yy, xx = np.mgrid[0:npix, 0:npix]
    for s in sources:
        if s.kind == "point":
            xi, yi = int(round(s.x)), int(round(s.y))
            if 0 <= xi < npix and 0 <= yi < npix:
                img[yi, xi] += s.flux
        elif s.kind == "gauss":
            sx = s.bmaj * FWHM2SIG
            sy = s.bmin * FWHM2SIG
            ct, st = np.cos(s.pa), np.sin(s.pa)
            dx, dy = xx - s.x, yy - s.y
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            g = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
            img += (s.flux / (2.0 * np.pi * sx * sy)) * g
        else:
            raise ValueError(f"unknown source kind {s.kind!r}")
    return img


def field_dr(sources):
    f = np.array([s.flux for s in sources])
    return float(f.max() / f.min())
