#!/usr/bin/env python3
"""Harvest uv-coverage (UVW/FREQ/WEIGHT) from an OSKAR-produced obs_I.xds
into a compact .npz, reusable to forward-model new visibilities.

Usage:
    python experiments/harvest_uvcov.py \
        --xds $STAGE_ROOT/data/simulated/preliminary_experiments/multiple_point_sources/obs_I.xds \
        --out $STAGE_ROOT/data/uvcov/meerkat_1h_4ch_1p4ghz.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xds", required=True, help="path to obs_I.xds (zarr)")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    from daskms.fsspec_store import DaskMSStore
    from pfb_imaging.utils.naming import xds_from_url

    xds, _ = xds_from_url(DaskMSStore(args.xds).url)
    x = xds[0]
    uvw = np.asarray(x["UVW"].data)                  # (nrow, 3) metres
    freq = np.asarray(x["FREQ"].data).ravel()         # (nchan,) Hz
    weight = np.squeeze(np.asarray(x["WEIGHT"].data))  # (nrow, nchan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, uvw=uvw, freq=freq, weight=weight)
    meta = dict(source_xds=str(args.xds), nrow=int(uvw.shape[0]), nchan=int(freq.shape[0]),
                freq0=float(freq[0]), weight_unique=np.unique(weight).tolist())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}  nrow={meta['nrow']} nchan={meta['nchan']} "
          f"freq0={meta['freq0']:.4g} weights={meta['weight_unique']}")


if __name__ == "__main__":
    main()
