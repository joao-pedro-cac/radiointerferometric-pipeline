# code/ — v2 project (branch `v2`)

Low-precision (float16/bfloat16) quantization of the Högbom CLEAN pipeline in
radio-interferometric imaging. Layer model per root `INSTRUCTIONS.md`:

```
src/          pure library — arrays in/out, no I/O, no hardcoded paths
              sky.py operators.py clean.py quantize.py metrics.py pipeline.py
experiments/  CLI runners (argparse, all params explicit)
              harvest_uvcov.py  xds_to_npz.py  make_dataset.py  run_clean.py
scripts/      study launchers (self-copying, timestamped results dirs)
              env.sh.example (gitignored env.sh holds local paths)
tests/        pytest gate — 30 tests, analytic ground truths, REPORT.md
analysis/     load.py (results -> dicts/dataframe)  plot.py (arrays -> figures)
              make_figures.py (+ gallery)   figures/
results/      gitignored, self-describing runs (results.npz + params/metrics/info)
deck/         beamer deck, tables auto-generated from results/ (build.sh)
attic/        legacy v1 code/configs/scripts (git history also has everything)
```

## Quickstart

```bash
source scripts/env.sh                 # from env.sh.example; sets STAGE_ROOT, VENV_PATH, ...
$VENV_PATH/bin/python -m pytest tests/ -q          # gate: must pass
$VENV_PATH/bin/python experiments/make_dataset.py --name S0_n101 --n-point 1 --n-gauss 0 \
    --dr 1 --seed 42 --noise-seed 101 \
    --uvcov $STAGE_ROOT/data/uvcov/meerkat_1h_4ch_1p4ghz.npz --out $STAGE_ROOT/data/simulated/v2/S0_n101
$VENV_PATH/bin/python experiments/run_clean.py --dataset $STAGE_ROOT/data/simulated/v2/S0_n101 \
    --out results/test/S0_n101 --gamma 0.1 --pf 1e-4 --loop-max 10 --nthreads 8 \
    --q-vis float64 --q-dirty float64 --q-psf float64 --q-model float64
MAXJOBS=4 bash scripts/study_f64_baseline.sh        # full f64 baseline (long; use background)
$VENV_PATH/bin/python analysis/make_figures.py results --gallery
bash deck/build.sh
```

Long runs (>120 s) must be launched through background process tools.

## Conventions

- Datasets are OSKAR-compatible: `obs.npz` (uvw/freq/vis/weight; convert an
  OSKAR `obs_I.xds` with `experiments/xds_to_npz.py`), `truth.fits`, `sky.osm`,
  `settings.ini`, `meta.json`. Canonical OSKAR re-simulation:
  `attic/scripts_v1/simulate.sh` + `lib/OSKAR-2.12.2-Python3.sif` (singularity;
  not available on this machine — the ducc0 forward model in `make_dataset.py`
  is the executable route and produces the same layout).
- Units: vis in Jy; weights `1/σ²`; images normalized to Jy/beam by psf_peak.
  Orientation: `operators.to_display/to_native` (exact inverses).
- Metrics are defined in `src/metrics.py` and unit-tested in `tests/` — do not
  add a metric without an analytic test.
- Quantization semantics (`src/quantize.py`): value rounding per stage, f16
  rescaled to 65503; pipeline continues on de-scaled values.
