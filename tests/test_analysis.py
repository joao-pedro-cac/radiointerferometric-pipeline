"""Unit tests for analysis layer — comparability audit + boxplot plotting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from analysis.batch2_report import comparability, NOISE_FLOOR_FACTOR  # noqa: E402
import analysis.plot as P  # noqa: E402


def _df(**kw):
    base = dict(run="r", exp="e", scheme="f64-baseline", sky="s", noise_seed=101,
                residual_mad=1e-3, ratio_residual_mad=1.0, resmad_over_noise=1.0,
                converged_reason="threshold")
    base.update(kw)
    return pd.DataFrame([base])


def test_comparability_noise_floor_reached_is_comparable():
    df = comparability(_df(resmad_over_noise=1.5, converged_reason="threshold"))
    assert df.loc[0, "reached_noise_floor"] and df.loc[0, "comparable"]
    assert not df.loc[0, "stalled"]


def test_comparability_floor_dominated_stall_is_not_comparable():
    # CLEAN-floor dominated (resMAD >> sigma/sqrt(N)) + stall: the "illusion of
    # safe quantization" case — must be flagged NOT comparable
    df = comparability(_df(resmad_over_noise=1100.0, converged_reason="stall"))
    assert not df.loc[0, "reached_noise_floor"]
    assert df.loc[0, "stalled"]
    assert not df.loc[0, "comparable"]
    df2 = comparability(_df(resmad_over_noise=500.0, converged_reason="max_major"))
    assert df2.loc[0, "stalled"] and not df2.loc[0, "comparable"]


def test_comparability_boundary_and_nan():
    df = comparability(_df(resmad_over_noise=NOISE_FLOOR_FACTOR))  # boundary: not <
    assert not df.loc[0, "reached_noise_floor"]
    df2 = comparability(_df(resmad_over_noise=2.9, ratio_residual_mad=np.nan))
    assert df2.loc[0, "reached_noise_floor"] and not df2.loc[0, "comparable"]  # no f64 ref


def test_metric_boxplots_writes_file(tmp_path):
    P.SAVE_FIGURES = True
    P.OUTPUT_DIR = tmp_path
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "seed": [1, 2, 3, 4] * 2,
        "scheme": ["f64"] * 4 + ["allf16"] * 4,
        "ratio_residual_mad": np.concatenate([1 + 0.01 * rng.standard_normal(4),
                                              1.5 + 0.1 * rng.standard_normal(4)]),
    })
    out = P.metric_boxplots(df, "seed", "ratio_residual_mad", "scheme",
                            "box_test.png", xlabel="seed", ylabel="ratio",
                            logy=True, ref_line=1.0, title="t")
    assert (tmp_path / "box_test.png").exists()
    P.SAVE_FIGURES = False
