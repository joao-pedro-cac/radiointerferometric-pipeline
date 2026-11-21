#!/usr/bin/env python3
"""Build a deterministic, OSKAR-compatible simulated dataset.

Sky model is drawn analytically (seeded), rasterized to the truth image, and
visibilities are forward-modelled with ducc0.dirty2vis on a real OSKAR
uv-coverage, plus seeded complex Gaussian noise. Output layout matches the
OSKAR dataset convention so canonical OSKAR re-simulation (sky.osm +
settings.ini via scripts + singularity) reproduces the same sky:

    <out>/
      obs.npz        uvw (nrow,3) m, freq (nchan,) Hz, vis (nrow,nchan), weight
      truth.fits     truth image, Jy/pixel, display orientation, SIN WCS
      sky.osm        OSKAR sky model (RA Dec I Q U V freq0 spix RM maj min pa)
      settings.ini   OSKAR settings referencing sky.osm (canonical route)
      meta.json      full provenance: seed, sigma, geometry, uv-coverage, field

Usage:
    python experiments/make_dataset.py --name S0_point1 \
        --n-point 1 --n-gauss 0 --dr 1 --peak-flux 1.0 --sigma 1e-4 --seed 42 \
        --uvcov $STAGE_ROOT/data/uvcov/meerkat_1h_4ch_1p4ghz.npz \
        --out $STAGE_ROOT/data/simulated/v2/S0_point1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import sky, operators


def write_osm(path, field, freq0, spix):
    g = lambda v: f"{v:.10g}"
    lines = [
        "# OSKAR sky model (v2 factory) | n_point=%d n_gauss=%d dr=%g seed-see-meta\n"
        % (field["meta"]["n_point"], field["meta"]["n_gauss"], field["meta"]["dr"]),
        "# RA Dec I Q U V freq0 spix RM maj min pa\n",
        "#  (deg) (deg) (Jy)  (Hz)            (\") (\") (deg)\n",
    ]
    for i in range(len(field["ra"])):
        lines.append(f"{g(field['ra'][i])} {g(field['dec'][i])} {g(field['flux'][i])} "
                     f"0 0 0 {g(freq0)} {g(spix)} 0 {g(field['maj'][i])} "
                     f"{g(field['minr'][i])} {g(field['pa'][i])}\n")
    Path(path).write_text("".join(lines))


def write_settings(path, osm_name, ms_name, telescope):
    Path(path).write_text(f"""[simulator]
use_gpus=false

[observation]
start_frequency_hz=1400000000.0
num_channels=4
frequency_inc_hz=1000000.0
start_time_utc=2000-01-01T00:00:00
length=3600.0
num_time_steps=360
phase_centre_ra_deg=-120.0
phase_centre_dec_deg=-60.0

[telescope]
input_directory={telescope}

[sky]
oskar_sky_model/file={osm_name}

[interferometer]
ms_filename={ms_name}
channel_bandwidth_hz=1000000.0
time_average_sec=10.0
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-point", type=int, default=1)
    ap.add_argument("--n-gauss", type=int, default=0)
    ap.add_argument("--n-diffuse", type=int, default=0,
                    help="number of diffuse (large-gaussian, 60-400 arcsec) components")
    ap.add_argument("--dr", type=float, default=1.0, help="intra-field DR = peak/faintest")
    ap.add_argument("--peak-flux", type=float, default=1.0, help="brightest source, Jy")
    ap.add_argument("--sigma", type=float, default=1e-4, help="per-visibility noise, Jy")
    ap.add_argument("--seed", type=int, default=42, help="sky-field seed (positions/fluxes)")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="noise seed (default: same as --seed; vary for seed-spread studies)")
    ap.add_argument("--uvcov", required=True, help="uv-coverage .npz from harvest_uvcov.py")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--npix", type=int, default=2520)
    ap.add_argument("--fov-deg", type=float, default=1.0)
    ap.add_argument("--ra0", type=float, default=-120.0)
    ap.add_argument("--dec0", type=float, default=-60.0)
    ap.add_argument("--telescope", default="meerkat.tm",
                    help="telescope layout file (for settings.ini reference)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cell_deg = args.fov_deg / args.npix
    cell_arcsec = cell_deg * 3600.0
    wcs = sky.make_wcs(args.npix, cell_deg, args.ra0, args.dec0)
    geom = operators.Geometry(npix_x=args.npix, npix_y=args.npix,
                              pixsize_x=np.radians(cell_deg),
                              pixsize_y=np.radians(cell_deg))

    rng = np.random.default_rng(args.seed)
    field = sky.sample_field(rng, args.n_point, args.n_gauss, args.dr,
                             args.peak_flux, args.ra0, args.dec0, args.fov_deg,
                             n_diffuse=args.n_diffuse)
    noise_rng = np.random.default_rng(args.noise_seed if args.noise_seed is not None else args.seed)
    sources = sky.to_sources(field, wcs, cell_arcsec)
    truth = sky.render(sources, args.npix)                    # display orientation, Jy/pixel

    cov = np.load(args.uvcov)
    uvw, freq, weight = cov["uvw"], cov["freq"], cov["weight"]

    # forward-model noiseless visibilities from the truth image (native frame).
    # dirty2vis(img, wgt) = wgt * FFT(img), so use UNIT weights -> vis in Jy (OSKAR convention)
    ones = np.ones_like(weight)
    vis_clean = operators.dirty2vis(operators.to_native(truth), uvw, freq, ones, geom)
    # seeded complex Gaussian noise, sigma per component (vis in Jy)
    noise = args.sigma * (noise_rng.standard_normal(vis_clean.shape)
                          + 1j * noise_rng.standard_normal(vis_clean.shape)) / np.sqrt(2.0)
    vis = vis_clean + noise

    # natural weights from sigma (w = 1/sigma^2); stored for the pipeline
    wgt = np.full_like(weight, 1.0 / args.sigma ** 2, dtype=np.float64)

    np.savez_compressed(out / "obs.npz", uvw=uvw, freq=freq, vis=vis, weight=wgt)

    hdr = wcs.to_header()
    hdr["BUNIT"] = "Jy/pixel"
    fits.PrimaryHDU(data=truth.astype(np.float32), header=hdr).writeto(
        out / "truth.fits", overwrite=True)

    write_osm(out / "sky.osm", field, field["meta"]["freq0"], field["meta"]["spix"])
    write_settings(out / "settings.ini", "sky.osm", "obs.ms", args.telescope)

    meta = dict(name=args.name, seed=args.seed,
                noise_seed=args.noise_seed if args.noise_seed is not None else args.seed,
                sigma_jy=args.sigma,
                npix=args.npix, fov_deg=args.fov_deg, cell_deg=cell_deg,
                ra0=args.ra0, dec0=args.dec0, uvcov=str(args.uvcov),
                producer="ducc0-forward-v2", oskar_compatible=True,
                field=field["meta"], n_sources=len(sources))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"dataset {args.name} -> {out}")
    print(f"  sources={len(sources)} (point={args.n_point}, gauss={args.n_gauss}, "
          f"diffuse={args.n_diffuse}) "
          f"dr={meta['field']['realized_dr']:.4g} peak={args.peak_flux} Jy sigma={args.sigma} Jy")
    print(f"  truth sum={truth.sum():.6g} Jy peak={truth.max():.6g} Jy/pix  "
          f"vis |max|={np.abs(vis).max():.4g}  nvis={vis.size}")


if __name__ == "__main__":
    main()
