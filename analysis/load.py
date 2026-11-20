"""Data access for v2 results — decoupled from experiments.

load_results(root) globs **/results.npz recursively and returns a list of
dicts: arrays under their keys, params from params.json, metrics from
metrics.json (fallback: info.yaml). Any directory can be the root.
"""
import json
from pathlib import Path

import numpy as np
import yaml


def load_results(root):
    root = Path(root)
    out = []
    for npz in sorted(root.glob("**/results.npz")):
        d = npz.parent
        entry = dict(np.load(npz))
        pj = d / "params.json"
        mj = d / "metrics.json"
        if pj.exists():
            entry["params"] = json.loads(pj.read_text())
        if mj.exists():
            entry["metrics"] = json.loads(mj.read_text())
        pz = d / "probes.npz"
        if pz.exists():
            entry["probes"] = dict(np.load(pz))
        entry["run_dir"] = str(d)
        out.append(entry)
    return out


def filter_results(results, **conditions):
    """Exact-match filtering on params/metrics, e.g. filter_results(rs, pf=1e-4)
    matches params['pf'] or metrics['pf']... conditions are looked up in
    params first, then metrics."""
    def get(r, k):
        if k in r.get("params", {}):
            return r["params"][k]
        q = r.get("params", {}).get("quantization", {})
        if k.startswith("q_") and k[2:] in q:
            return q[k[2:]]
        return r.get("metrics", {}).get(k)
    return [r for r in results
            if all(get(r, k) == v for k, v in conditions.items())]


def to_dataframe(results):
    import pandas as pd
    rows = []
    for r in results:
        row = dict(r["params"])
        row.update(r["metrics"])
        row["run_dir"] = r["run_dir"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_scene(result):
    """Load the dataset a result was computed on (truth + obs) using the
    dataset path stored in the result's params."""
    from astropy.io import fits
    d = Path(result["params"]["dataset"])
    obs = np.load(d / "obs.npz")
    truth = np.squeeze(fits.getdata(d / "truth.fits")).astype(np.float64)
    return dict(obs=obs, truth=truth, meta=json.loads((d / "meta.json").read_text()))
