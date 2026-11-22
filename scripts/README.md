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
bash run_experiment.sh            # batch 2: one nohup'd python process per experiment
bash scripts/status.sh            # pids + progress + last results
tail -f results/batch2/logs/exp_dr_*.log
```

Canonical OSKAR re-simulation (optional, singularity mandatory — set in env.sh):
`SIM_MODE=oskar bash simulate_data.sh` with `SINGULARITY`, `OSKAR_SIF`
(default `<stage>/lib/OSKAR-2.12.2-Python3.sif`), `TELESCOPE_DIR` (default
`<stage>/data/oskar/telescopes`), `TELESCOPE=meerkat.tm`.

Batch 2 (DR x noise x seeds sweeps) — one python process per experiment:

```bash
bash run_experiment.sh --dry        # preview: one launch line per experiment
bash run_experiment.sh              # launch one nohup'd process per experiment + wait monitor
MAXPROCS=1 MAXJOBS=8 NTHREADS=4 bash run_experiment.sh   # cap concurrent processes if needed

# monitoring — one command:
bash scripts/status.sh              # pids (alive/EXITED) + last log line + progress + results
tail -f results/batch2/logs/exp_dr_*.log   # session log (tee'd by run_batch --log-file)
```

Launch mechanics: `scripts/launch_one.sh <exp>` is the single place a session is
spawned — `nohup python -u experiments/run_batch.py --exp <exp> ... > logs/<exp>_<ts>.log 2>&1 &`
with a pid file and a 1-second fail-fast check (alive + banner in log).

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
  (started as a nohup'd process by run_experiment.sh, log in logs/wait_*.log;
   `--once` exits 2 while incomplete, 0 when done).

Batch 3 (Monte Carlo + range limit + noise-floor fields) — plans live in
`experiments/run_batch.py` (`exp3_mc`, `exp3_range`, `exp3_floor`, 52 runs):

```bash
# datasets: only batch-3 skies, with the batch-3 noise seeds
EXPS="exp3_mc exp3_range exp3_floor" NOISE_SEEDS="201 202 301 401" bash simulate_data.sh
# runs: one nohup'd process per experiment
bash run_experiment.sh exp3_mc exp3_range exp3_floor
bash scripts/status.sh
```

`--print-skies --exps ...` filters which skies `simulate_data.sh` builds.

Then pull `results/` back and regenerate the render locally:

```bash
$VENV_PATH/bin/python analysis/make_figures.py results --gallery
$VENV_PATH/bin/python analysis/make_sweep.py results/batch2 --out analysis/figures/batch2
bash deck/build.sh
```

Datasets are reproducible from seed (`experiments/make_dataset.py`), so the
server only needs `data/uvcov/` + the code — or regenerate everything there
with `simulate_data.sh`.
