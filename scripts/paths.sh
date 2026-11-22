# scripts/paths.sh — layout detection + shared path defaults. SOURCE, don't run.
#
# Two supported trees, same scripts:
#   local : <archive>/code/{src,tests,scripts,experiments,...}
#           with shared <archive>/{data,lib} next to code/
#   remote: <repo>/{src,tests,scripts,experiments,data,lib,results,venv,...}
#           (flattened — everything at the repo root)
#
# Source scripts/env.sh FIRST (user overrides), then this file: every variable
# below only fills in what is still unset.

# CODE_ROOT = directory holding src/ and tests/ (parent of this scripts/ dir)
_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-$_CODE_ROOT}"

# STAGE_ROOT = nearest ancestor of CODE_ROOT holding the shared data/ and lib/
# (locally the archive root; on the flattened remote tree, CODE_ROOT itself)
if [ -z "${STAGE_ROOT:-}" ]; then
  _d="$_CODE_ROOT"
  while [ "$_d" != "/" ] && [ ! -d "$_d/data/uvcov" ] && [ ! -d "$_d/lib" ]; do
    _d="$(dirname "$_d")"
  done
  if [ -d "$_d/data/uvcov" ] || [ -d "$_d/lib" ]; then STAGE_ROOT="$_d"; else STAGE_ROOT="$_CODE_ROOT"; fi
  unset _d
fi

VENV_PATH="${VENV_PATH:-$CODE_ROOT/venv}"          # in-tree venv by default
PY="$VENV_PATH/bin/python"
DATASET_PATH="${DATASET_PATH:-$STAGE_ROOT/data/simulated/v2}"
RESULTS_PATH="${RESULTS_PATH:-$CODE_ROOT/results}"
UVCOV="${UVCOV:-$STAGE_ROOT/data/uvcov/meerkat_1h_4ch_1p4ghz.npz}"
OSKAR_SIF="${OSKAR_SIF:-$STAGE_ROOT/lib/OSKAR-2.12.2-Python3.sif}"
TELESCOPE_DIR="${TELESCOPE_DIR:-$STAGE_ROOT/data/oskar/telescopes}"
PFB_DIR="${PFB_DIR:-$CODE_ROOT/lib/pfb-imaging}"
[ -d "$PFB_DIR" ] || PFB_DIR="$STAGE_ROOT/lib/pfb-imaging"
