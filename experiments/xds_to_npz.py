#!/usr/bin/env python3
"""Convert an OSKAR-produced dataset (obs_I.xds) to the v2 obs.npz format,
so the whole v2 pipeline runs unchanged on canonical OSKAR output.

Usage:
    python experiments/xds_to_npz.py --xds <...>/obs_I.xds --out <dataset_dir>
Writes <dataset_dir>/obs.npz with uvw/freq/vis/weight.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xds", required=True)
    ap.add_argument("--out", required=True, help="dataset directory (obs.npz written inside)")
    args = ap.parse_args()

    from daskms.fsspec_store import DaskMSStore
    from pfb_imaging.utils.naming import xds_from_url

    xds, _ = xds_from_url(DaskMSStore(args.xds).url)
    x = xds[0]
    uvw = np.asarray(x["UVW"].data)
    freq = np.asarray(x["FREQ"].data).ravel()
    vis = np.squeeze(np.asarray(x["VIS"].data))
    weight = np.squeeze(np.asarray(x["WEIGHT"].data))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "obs.npz", uvw=uvw, freq=freq, vis=vis, weight=weight)
    (outdir / "meta.json").write_text(json.dumps(
        dict(source="oskar_xds", source_xds=str(args.xds),
             nrow=int(uvw.shape[0]), nchan=int(freq.shape[0])), indent=2))
    print(f"wrote {outdir/'obs.npz'}  nrow={uvw.shape[0]} nchan={freq.shape[0]}")


if __name__ == "__main__":
    main()
