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

## Quickstart — three commands (local or remote server)

Run from the repo root. The same commands work on the local tree
(`<archive>/code/`, shared `data/`+`lib/` one level up) and on the flattened
remote tree (everything at the repo root): `scripts/paths.sh` auto-detects
`CODE_ROOT`/`STAGE_ROOT`, and the venv is in-tree by default (`./venv`).

```bash
bash setup_env.sh                     # uv-based env -> ./venv (python3-venv fallback); runs pytest gate
bash simulate_data.sh                 # build datasets (SIM_MODE=ducc0 fast | oskar via singularity)
bash run_experiment.sh                # one nohup'd python process per exp + wait monitor (logs/*.pid)
bash run_experiment.sh --dry          # show what would launch
bash run_experiment.sh --dry          # show what would launch
# batch 3 (Monte Carlo / range limit / noise-floor fields):
#   EXPS="exp3_mc exp3_range exp3_floor" NOISE_SEEDS="201 202 301 401" bash simulate_data.sh
#   bash run_experiment.sh exp3_mc exp3_range exp3_floor
```

Optional overrides go in `scripts/env.sh` (copy `scripts/env.sh.example`):
`VENV_PATH`, `DATASET_PATH`, `RESULTS_PATH`, `NOISE_SEEDS`, `MAXJOBS`, and for
the OSKAR route `SIM_MODE=oskar` + `SINGULARITY`/`OSKAR_SIF`/`TELESCOPE_DIR`.

Equivalent manual route (single dataset + single CLEAN run):

```bash
source scripts/paths.sh               # path detection + defaults (VENV_PATH, DATASET_PATH, UVCOV, ...)
$VENV_PATH/bin/python -m pytest tests/ -q          # gate: must pass
bash simulate_data.sh --dry           # see the dataset list (SKIES x NOISE_SEEDS)
$VENV_PATH/bin/python experiments/make_dataset.py --name S0_n101 --n-point 1 --n-gauss 0 \
    --dr 1 --seed 42 --noise-seed 101 \
    --uvcov $UVCOV --out $DATASET_PATH/S0_n101
$VENV_PATH/bin/python experiments/run_clean.py --dataset $DATASET_PATH/S0_n101 \
    --out results/test/S0_n101 --gamma 0.1 --pf 1e-4 --loop-max 10 --nthreads 8 \
    --q-vis float64 --q-dirty float64 --q-psf float64 --q-model float64
$VENV_PATH/bin/python analysis/make_figures.py results --gallery
bash deck/build.sh
```

Long runs (>120 s) must be launched through background process tools.

## Conventions

- Datasets are OSKAR-compatible: `obs.npz` (uvw/freq/vis/weight), `truth.fits`,
  `sky.osm`, `settings.ini`, `meta.json`. Canonical OSKAR simulation:
  `SIM_MODE=oskar bash simulate_data.sh` runs `oskar_simulator settings.ini`
  inside each dataset dir via `singularity exec --bind <ds>,<TELESCOPE_DIR>
  $OSKAR_SIF` (SINGULARITY/OSKAR_SIF/TELESCOPE_DIR in `scripts/env.sh`), then
  `experiments/xds_to_npz.py --ms obs.ms --out .` replaces `obs.npz` with the
  OSKAR visibilities. `--xds obs_I.xds` is also accepted. The default
  `SIM_MODE=ducc0` route (ducc0 forward model on a harvested OSKAR uv-coverage
  + seeded noise) is the executable fallback and produces the same layout.
- Units: vis in Jy; weights `1/σ²`; images normalized to Jy/beam by psf_peak.
  Orientation: `operators.to_display/to_native` (exact inverses).
- Metrics are defined in `src/metrics.py` and unit-tested in `tests/` — do not
  add a metric without an analytic test.
- Quantization semantics (`src/quantize.py`): value rounding per stage, f16
  rescaled to 65503; pipeline continues on de-scaled values.
