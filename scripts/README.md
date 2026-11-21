# Running studies on a remote server

The study scripts are location-agnostic: all paths come from
`scripts/env.sh` (gitignored; copy from `env.sh.example`).

On the server:

```bash
# 1. clone/copy the archive, create the venv once
bash code/setup_env.sh /path/to/venv        # or reuse requirements.txt
# 2. fill code/scripts/env.sh with SERVER paths:
#    STAGE_ROOT=/path/to/2026_stage-joao
#    VENV_PATH=/path/to/venv
#    DATASET_PATH=$STAGE_ROOT/data/simulated/v2
#    RESULTS_PATH=$STAGE_ROOT/code/results
# 3. OSKAR canonical re-simulation (optional, needs singularity):
#    OSKAR_SIF=$STAGE_ROOT/lib/OSKAR-2.12.2-Python3.sif
# 4. launch (nohup; the script self-copies into the results dir)
cd $STAGE_ROOT/code
nohup env MAXJOBS=8 SKIES="S0_point1 S1_points_dr10 S2_mixed_dr100 S3_gauss_dr1000" \
    NOISE_SEEDS="101 102 103" bash scripts/study_quant_schemes.sh \
    > /tmp/quant_schemes.log 2>&1 &
#    f64 baseline (if datasets not yet made):
#    nohup env MAXJOBS=8 bash scripts/study_f64_baseline.sh > /tmp/f64.log 2>&1 &
# 5. poll
tail -f /tmp/quant_schemes.log
find results -name results.npz | wc -l
```

Then pull `code/results/` back and regenerate the render locally:

```bash
$VENV_PATH/bin/python analysis/make_figures.py code/results --gallery
$VENV_PATH/bin/python analysis/compare_schemes.py code/results/quant_* code/results/f64_*
bash deck/build.sh
```

Datasets are reproducible from seed (`experiments/make_dataset.py`), so the
server only needs `data/uvcov/` + the code — or regenerate everything there.

## Batch 2 (noise × DR × seeds sweeps)

Individual experiment scripts with semantic results dirs, plus a dispatcher
that runs each under a named screen session:

```bash
cd $STAGE_ROOT/code
bash scripts/run_experiments.sh --dry                 # preview
bash scripts/run_experiments.sh                        # all three, in screen
bash scripts/run_experiments.sh exp_dr                 # just one
screen -ls                                             # sessions b2_exp_*
tail -f results/batch2/logs/exp_dr.log
```

- `scripts/exp_dr.sh` — S6 + S8(1000 src) DR ladder 1e2..1e6, σ=1e-4, noise seeds 101 102
  → `results/batch2/exp_dr_<ts>/`
- `scripts/exp_noise.sh` — S8 DR 1e4 at σ=1e-2/1e-3/1e-6, noise seeds 101 102
  → `results/batch2/exp_noise_<ts>/`
- `scripts/exp_seeds.sh` — S8 DR 1e4 sky seeds s42/s43/s44, noise seed 103
  → `results/batch2/exp_seeds_<ts>/`
- shared scheme list: `scripts/batch2_common.sh`; datasets: `make_ladder_v3.sh`.
- after pulling results back:
  `python analysis/make_sweep.py code/results/batch2 --out analysis/figures/batch2`
  plus `make_probe_figures.py` / `make_figures.py` / `deck/build.sh` as usual.

