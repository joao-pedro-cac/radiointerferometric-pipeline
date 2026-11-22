# Running studies — local or remote server, same commands

All launchers source `scripts/paths.sh`, which auto-detects the tree layout:

- local: `<archive>/code/{src,tests,scripts,...}` with shared `<archive>/{data,lib}`
- remote (flattened): repo root holds `{src,tests,scripts,data,lib,results,venv,...}`

The venv is in-tree by default (`<repo>/venv`). Override anything via
`scripts/env.sh` (gitignored; copy from `env.sh.example`) — env.sh is sourced
BEFORE paths.sh, so it only needs the variables you want to change.

On the server (flattened tree), from the repo root:

```bash
bash setup_env.sh                 # creates ./venv with uv (python3-venv fallback), runs pytest gate
bash simulate_data.sh             # build datasets (SIM_MODE=ducc0 default)
bash run_experiment.sh --dry      # preview
bash run_experiment.sh            # batch 2 in screen sessions b2_<exp>
screen -ls                        # sessions
tail -f results/batch2/logs/exp_dr.log
```

Canonical OSKAR re-simulation (optional, singularity mandatory — set in env.sh):
`SIM_MODE=oskar bash simulate_data.sh` with `SINGULARITY`, `OSKAR_SIF`
(default `<stage>/lib/OSKAR-2.12.2-Python3.sif`), `TELESCOPE_DIR` (default
`<stage>/data/oskar/telescopes`), `TELESCOPE=meerkat.tm`.

Batch 2 experiments (dispatcher `run_experiments.sh`, schemes in `batch2_common.sh`,
datasets via `make_ladder_v3.sh`):

- `exp_dr.sh` — S6 + S8(1000 src) DR ladder 1e2..1e6, σ=1e-4, noise seeds 101 102
  → `results/batch2/exp_dr_<ts>/`
- `exp_noise.sh` — S8 DR 1e4 at σ=1e-2/1e-3/1e-6, noise seeds 101 102
  → `results/batch2/exp_noise_<ts>/`
- `exp_seeds.sh` — S8 DR 1e4 sky seeds s42/s43/s44, noise seed 103
  → `results/batch2/exp_seeds_<ts>/`

Older single studies (still location-agnostic): `study_f64_baseline.sh`,
`study_quant_schemes.sh`, `study_sweep_v3.sh` — e.g.
`MAXJOBS=8 SKIES="S0_point1 S2_mixed_dr100" NOISE_SEEDS="101 102" bash scripts/study_quant_schemes.sh`
(self-copies into a timestamped results dir).

Then pull `results/` back and regenerate the render locally:

```bash
$VENV_PATH/bin/python analysis/make_figures.py results --gallery
$VENV_PATH/bin/python analysis/make_sweep.py results/batch2 --out analysis/figures/batch2
bash deck/build.sh
```

Datasets are reproducible from seed (`experiments/make_dataset.py`), so the
server only needs `data/uvcov/` + the code — or regenerate everything there
with `simulate_data.sh`.
