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

Batch 2 (DR x noise x seeds sweeps) — one python process per experiment:

```bash
bash run_experiment.sh --dry        # preview: one launch line per experiment
bash run_experiment.sh              # launch b2_exp_dr b2_exp_noise b2_exp_seeds + b2_wait
MAX_SCREENS=1 MAXJOBS=8 NTHREADS=4 bash run_experiment.sh   # cap concurrent sessions if needed

# monitoring — one command:
bash scripts/status.sh              # screen sessions + progress table + last runs (summary.csv)
screen -r b2_exp_dr                 # live per-run output of one experiment (banner + start/done lines)
screen -r b2_wait                   # live table (total/done/running/failed per exp)
tail -f results/batch2/logs/exp_dr_*.log   # session log (tee'd by run_batch --log-file)
```

Launch mechanics: `scripts/launch_one.sh <exp>` is the single place a session is
spawned — `screen -dmS b2_<exp> python -u experiments/run_batch.py ... --log-file <log>`
(nohup fallback when screen is missing). `run_experiments.sh` just feeds the exp
list through `xargs -P $MAX_SCREENS` and then starts `b2_wait`.

- `experiments/run_batch.py` is the single source of truth: sky specs (SKIES),
  quantization schemes (SCHEMES), and per-experiment plans (PLANS:
  exp_dr = 6 skies x noise seeds 101/102, exp_noise = sigma ladder x 101/102,
  exp_seeds = sky seeds s42/43/44 x noise seed 103; 6 schemes each, 126 runs).
  Resume-safe: rerunning skips runs whose results.npz exists.
  `--dry` prints the plan; `--print-skies` feeds simulate_data.sh.
- Each job is a subprocess of `run_clean.py` with `--nthreads NTHREADS`
  (OMP/NUMBA env set too), so MAXJOBS x NTHREADS should match the cores.
- Per-run logs: `<exp_root>/logs/<run>.log`; live progress:
  `<exp_root>/progress.json`; final table: `<exp_root>/summary.csv`.
- `experiments/wait_batch.py --root results/batch2 [--once]` is the wait/monitor
  (used by the b2_wait session; `--once` exits 2 while incomplete, 0 when done).

Then pull `results/` back and regenerate the render locally:

```bash
$VENV_PATH/bin/python analysis/make_figures.py results --gallery
$VENV_PATH/bin/python analysis/make_sweep.py results/batch2 --out analysis/figures/batch2
bash deck/build.sh
```

Datasets are reproducible from seed (`experiments/make_dataset.py`), so the
server only needs `data/uvcov/` + the code — or regenerate everything there
with `simulate_data.sh`.
