"""Unit tests for experiments/run_batch.py — plan building, scheme mapping,
resume logic, CLI surface. No CLEAN runs are executed."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "experiments"))
import run_batch  # noqa: E402


def test_sky_specs_well_formed_and_unique():
    names = [run_batch.spec_name(s) for s in run_batch.SKIES]
    assert len(names) == len(set(names))
    for s in run_batch.SKIES:
        fields = s.split(":")
        assert len(fields) == 8, s
        int(fields[1]); int(fields[2]); int(fields[3]); int(fields[7])  # counts/seed
        float(fields[4]); float(fields[5]); float(fields[6])            # dr/peak/sigma


def test_print_skies_roundtrip():
    out = subprocess.run([sys.executable, str(CODE_ROOT / "experiments" / "run_batch.py"),
                          "--print-skies"], capture_output=True, text=True)
    assert out.returncode == 0
    assert out.stdout.split() == run_batch.SKIES


def test_list_exps():
    out = subprocess.run([sys.executable, str(CODE_ROOT / "experiments" / "run_batch.py"),
                          "--list-exps"], capture_output=True, text=True)
    assert out.returncode == 0
    assert out.stdout.split() == list(run_batch.PLANS)


def test_scheme_args_mapping():
    a = run_batch.scheme_args("f64")
    assert a == ["--q-vis", "float64", "--q-dirty", "float64", "--q-psf", "float64",
                 "--q-model", "float64", "--rescale-max", "65503"]
    a = run_batch.scheme_args("modelf16")
    assert a[7] == "float16" and a[1] == "float64"          # only model quantized
    a = run_batch.scheme_args("visf16")
    assert a[1] == "float16" and a[3] == "float64"
    a = run_batch.scheme_args("dirtyf16")
    assert a[3] == "float16" and a[1] == "float64"
    a = run_batch.scheme_args("allbf16")
    assert all(x == "bfloat16" for x in a[1:8:2])


def test_build_jobs_counts_and_names(tmp_path):
    dataset_root = tmp_path / "ds"; dataset_root.mkdir()
    exp_root = tmp_path / "out"
    jobs = run_batch.build_jobs("exp_dr", dataset_root, exp_root)
    # 6 skies x 2 noise seeds x 6 schemes
    assert len(jobs) == 6 * 2 * len(run_batch.SCHEMES)
    assert jobs[0]["run_name"].startswith("b2_S6_gauss_dr1e5_s42_n101_")
    assert jobs[0]["dataset"].endswith("b2_S6_gauss_dr1e5_s42_n101")
    assert all(not j["done"] for j in jobs)
    jobs_n = run_batch.build_jobs("exp_noise", dataset_root, exp_root)
    assert len(jobs_n) == 3 * 2 * len(run_batch.SCHEMES)
    jobs_s = run_batch.build_jobs("exp_seeds", dataset_root, exp_root)
    assert len(jobs_s) == 3 * 1 * len(run_batch.SCHEMES)
    assert {j["run_name"].split("_n")[1].split("_")[0] for j in jobs_s} == {"103"}


def test_build_jobs_resume_skip(tmp_path):
    dataset_root = tmp_path / "ds"; exp_root = tmp_path / "out"
    jobs = run_batch.build_jobs("exp_seeds", dataset_root, exp_root)
    done_run = Path(jobs[0]["run"]); done_run.mkdir(parents=True)
    (done_run / "results.npz").write_bytes(b"x")
    jobs2 = run_batch.build_jobs("exp_seeds", dataset_root, exp_root)
    assert sum(j["done"] for j in jobs2) == 1
    assert jobs2[0]["done"] and not jobs2[1]["done"]


def test_dry_run_prints_plan_and_runs_nothing(tmp_path, capsys):
    dataset_root = tmp_path / "ds"
    for prefixes, seeds in run_batch.PLANS.values():
        for prefix in prefixes:
            for nseed in seeds + ["101", "102", "103"]:
                d = dataset_root / run_batch.dataset_name(prefix, nseed)
                d.mkdir(parents=True, exist_ok=True); (d / "obs.npz").write_bytes(b"x")
    rc = run_batch.main(["--exp", "exp_seeds", "--dry",
                         "--dataset-root", str(dataset_root),
                         "--batch-root", str(tmp_path / "batch")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY b2_S8mix1000_dr1e4_s42_n103_f64" in out
    assert "to run=18" in out
    assert not (tmp_path / "batch").exists() or not list((tmp_path / "batch").rglob("results.npz"))


def test_missing_dataset_fails_clearly(tmp_path, capsys):
    rc = run_batch.main(["--exp", "exp_seeds", "--dry",
                         "--dataset-root", str(tmp_path / "empty"),
                         "--batch-root", str(tmp_path / "batch")])
    assert rc == 1
    assert "run simulate_data.sh first" in capsys.readouterr().err


def test_wait_batch_collect_and_render(tmp_path):
    sys.path.insert(0, str(CODE_ROOT / "experiments"))
    import wait_batch
    exp = tmp_path / "exp_dr_20261122_120000"; exp.mkdir()
    (exp / "progress.json").write_text(json.dumps(
        dict(exp="exp_dr", ts="x", total=10, done=["a", "b"], running=["c"],
             failed=[], walls={}, metrics={}, started="2026-11-22T12:00:00+00:00", updated="")))
    rows = wait_batch.collect(tmp_path)
    assert rows == [dict(exp="exp_dr_20261122_120000", total=10, done=2, failed=0,
                         running=1, started="2026-11-22T12:00:00+00:00")]
    table, all_done = wait_batch.render(rows, str(tmp_path))
    assert not all_done and "exp_dr_20261122_120000" in table
    rc = wait_batch.main(["--root", str(tmp_path), "--once"])
    assert rc == 2  # not complete
