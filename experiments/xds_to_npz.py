#!/usr/bin/env python3
"""Convert an OSKAR-produced dataset to the v2 obs.npz format, so the whole
v2 pipeline runs unchanged on canonical OSKAR output.

Usage:
    python experiments/xds_to_npz.py --xds <...>/obs_I.xds --out <dataset_dir>
    python experiments/xds_to_npz.py --ms  <...>/obs.ms     --out <dataset_dir>
Writes <dataset_dir>/obs.npz with uvw/freq/vis/weight.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def extract_from_xds(xds_path):
    from daskms.fsspec_store import DaskMSStore
    from pfb_imaging.utils.naming import xds_from_url

    xds, _ = xds_from_url(DaskMSStore(xds_path).url)
    x = xds[0]
    return (np.asarray(x["UVW"].data), np.asarray(x["FREQ"].data).ravel(),
            np.squeeze(np.asarray(x["VIS"].data)),
            np.squeeze(np.asarray(x["WEIGHT"].data)))


def extract_from_ms(ms_path):
    import daskms

    xds = daskms.xds_from_ms(ms_path, columns=["UVW", "FREQ", "DATA", "WEIGHT"])
    x = xds[0]
    return (np.asarray(x["UVW"].data), np.asarray(x["FREQ"].data).ravel(),
            np.squeeze(np.asarray(x["DATA"].data)),
            np.squeeze(np.asarray(x["WEIGHT"].data)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--xds", help="OSKAR obs_I.xds directory")
    src.add_argument("--ms", help="OSKAR Measurement Set (obs.ms)")
    ap.add_argument("--out", required=True, help="dataset directory (obs.npz written inside)")
    args = ap.parse_args()

    if args.xds:
        uvw, freq, vis, weight = extract_from_xds(args.xds)
        source = dict(source="oskar_xds", source_xds=str(args.xds))
    else:
        uvw, freq, vis, weight = extract_from_ms(args.ms)
        source = dict(source="oskar_ms", source_ms=str(args.ms))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "obs.npz", uvw=uvw, freq=freq, vis=vis, weight=weight)
    meta_path = outdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(source)
    meta.update(nrow=int(uvw.shape[0]), nchan=int(freq.shape[0]))
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {outdir/'obs.npz'}  nrow={uvw.shape[0]} nchan={freq.shape[0]}")


if __name__ == "__main__":
    main()
