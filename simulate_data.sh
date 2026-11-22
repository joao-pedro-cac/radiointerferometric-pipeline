#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# simulate_data.sh — build all simulated datasets for the v2 pipeline.
#
# Pipeline position:  bash setup_env.sh  ->  bash simulate_data.sh  ->  bash run_experiment.sh
#
# Run from the repo root (works on both the local <archive>/code/ tree and the
# flattened remote tree — paths are auto-detected by scripts/paths.sh):
#   bash setup_env.sh && bash simulate_data.sh && bash run_experiment.sh
#
# Configuration is via environment variables (export them before calling, or
# copy scripts/env.sh.example to scripts/env.sh and edit):
#   VENV_PATH       venv created by setup_env.sh (default: <repo>/venv, in-tree)
#   DATASET_PATH    output root for datasets (default <stage>/data/simulated/v2)
#   UVCOV           harvested uv-coverage npz (default <stage>/data/uvcov/meerkat_1h_4ch_1p4ghz.npz)
#   SIM_MODE        "ducc0" (fast, default) or "oskar" (canonical OSKAR simulation)
#   SKIES           space-separated sky specs "name:np:ng:nd:dr:peak:sigma:skyseed"
#   NOISE_SEEDS     space-separated visibility-noise seeds (dataset suffix _n<seed>)
#   MAXJOBS         datasets built in parallel (default 1)
#
# OSKAR mode (SIM_MODE=oskar) — singularity support is mandatory:
#   SINGULARITY     singularity binary (default "singularity"; must be on PATH)
#   OSKAR_SIF       OSKAR singularity image (default $STAGE_ROOT/lib/OSKAR-2.12.2-Python3.sif)
#   TELESCOPE_DIR   directory of OSKAR .tm layouts (default $STAGE_ROOT/data/oskar/telescopes)
#   TELESCOPE       layout file inside TELESCOPE_DIR (default meerkat.tm)
#   For each dataset: make_dataset.py writes sky.osm + settings.ini + truth.fits,
#   then `singularity exec --bind <ds>,<TELESCOPE_DIR> $OSKAR_SIF oskar_simulator settings.ini`
#   runs inside the dataset dir (produces obs.ms), and xds_to_npz.py --ms converts
#   obs.ms -> obs.npz (OSKAR visibilities replace the ducc0 forward model).
#
#   bash simulate_data.sh --dry   prints the planned work without running it.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # = CODE_ROOT in both trees
[ -f "$HERE/scripts/env.sh" ] && source "$HERE/scripts/env.sh"
source "$HERE/scripts/paths.sh"

SIM_MODE="${SIM_MODE:-ducc0}"
SINGULARITY="${SINGULARITY:-singularity}"
TELESCOPE="${TELESCOPE:-meerkat.tm}"
MAXJOBS="${MAXJOBS:-1}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"

# name:n_point:n_gauss:n_diffuse:dr:peak_flux:sigma:sky_seed   (space-separated)
# Default: the plan sky specs, single source of truth in run_batch.py.
# EXPS="exp3_mc exp3_range exp3_floor" restricts to those plans' skies (batch 3).
SKIES="${SKIES:-$("$PY" "$HERE/experiments/run_batch.py" --print-skies ${EXPS:+--exps $EXPS})}"

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

case "$SIM_MODE" in
  ducc0) ;;
  oskar)
    command -v "$SINGULARITY" >/dev/null 2>&1 || {
      echo "ERROR: SIM_MODE=oskar requires singularity ('$SINGULARITY' not found on PATH)." >&2
      echo "       Set SINGULARITY=/path/to/singularity or install apptainer/singularity." >&2
      exit 1; }
    [ -f "$OSKAR_SIF" ] || {
      echo "ERROR: OSKAR image not found: $OSKAR_SIF" >&2
      echo "       Set OSKAR_SIF=/path/to/OSKAR-2.12.2-Python3.sif" >&2
      exit 1; }
    [ -f "$TELESCOPE_DIR/$TELESCOPE" ] || {
      echo "ERROR: telescope layout not found: $TELESCOPE_DIR/$TELESCOPE" >&2
      exit 1; }
    ;;
  *) echo "ERROR: SIM_MODE must be 'ducc0' or 'oskar' (got '$SIM_MODE')" >&2; exit 1 ;;
esac

mkdir -p "$DATASET_PATH"

build_one() {  # $1=spec  $2=noise_seed
  local spec="$1" nseed="$2"
  IFS=: read -r name np ng nd dr peak sigma seed <<< "$spec"
  local ds="$DATASET_PATH/${name}_n${nseed}"
  if [ -f "$ds/obs.npz" ] && [ "$SIM_MODE" != "oskar" -o -f "$ds/.oskar_done" ]; then
    echo "exists: $ds"
    return 0
  fi
  rm -rf "$ds"
  "$PY" "$HERE/experiments/make_dataset.py" --name "${name}_n${nseed}" \
    --n-point "$np" --n-gauss "$ng" --n-diffuse "$nd" --dr "$dr" \
    --peak-flux "$peak" --sigma "$sigma" --seed "$seed" --noise-seed "$nseed" \
    --uvcov "$UVCOV" --telescope "$TELESCOPE_DIR/$TELESCOPE" --out "$ds"
  if [ "$SIM_MODE" = "oskar" ]; then
    echo "oskar: simulating $ds with $SINGULARITY ($OSKAR_SIF)"
    ( cd "$ds" && "$SINGULARITY" exec --bind "$ds,$TELESCOPE_DIR" "$OSKAR_SIF" \
        oskar_simulator settings.ini )
    "$PY" "$HERE/experiments/xds_to_npz.py" --ms "$ds/obs.ms" --out "$ds"
    "$PY" - "$ds" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "meta.json"
m = json.loads(p.read_text()); m["producer"] = "oskar-2.12.2-singularity"
p.write_text(json.dumps(m, indent=2))
EOF
    touch "$ds/.oskar_done"
  fi
  echo "done: $ds"
}

if [ "$DRY" -eq 1 ]; then
  echo "DRY: SIM_MODE=$SIM_MODE DATASET_PATH=$DATASET_PATH MAXJOBS=$MAXJOBS"
  [ "$SIM_MODE" = "oskar" ] && echo "DRY: singularity=$SINGULARITY sif=$OSKAR_SIF telescope=$TELESCOPE_DIR/$TELESCOPE"
  for spec in $SKIES; do
    for nseed in $NOISE_SEEDS; do
      IFS=: read -r name np ng nd dr peak sigma seed <<< "$spec"
      echo "DRY: dataset ${name}_n${nseed}  (np=$np ng=$ng nd=$nd dr=$dr peak=$peak sigma=$sigma skyseed=$seed)"
    done
  done
  exit 0
fi

# job-control loop: at most MAXJOBS datasets built concurrently
pids=""; fails=0
for spec in $SKIES; do
  for nseed in $NOISE_SEEDS; do
    build_one "$spec" "$nseed" &
    pids="$pids $!"
    while [ "$(echo $pids | wc -w)" -ge "$MAXJOBS" ]; do
      for p in $pids; do
        if ! wait "$p"; then fails=$((fails+1)); fi
      done
      pids=""
    done
  done
done
for p in $pids; do
  if ! wait "$p"; then fails=$((fails+1)); fi
done
[ "$fails" -eq 0 ] || { echo "ERROR: $fails dataset build(s) failed" >&2; exit 1; }
echo "all datasets ready under $DATASET_PATH (SIM_MODE=$SIM_MODE)"
echo "next: bash run_experiment.sh"
