#!/usr/bin/env bash
# Create the v2 pipeline environment with uv (uv assumed installed; falls back
# to python3 -m venv when uv is absent, e.g. the local macOS box).
#
#   bash setup_env.sh /path/to/venv        # then: export VENV_PATH=/path/to/venv
#
# Requires: uv (https://docs.astral.sh/uv/) or python3 >= 3.9.
set -euo pipefail
VENV="${1:-venv}"
ROOT="${STAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v uv >/dev/null 2>&1; then
  echo "== uv $(uv --version) — creating $VENV"
  uv venv "$VENV" --python 3.11 --seed 2>/dev/null || uv venv "$VENV" --python 3 --seed
  PIP="uv pip install --python $VENV/bin/python"
else
  echo "== uv not found — falling back to python3 -m venv (install uv for faster setup)"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade "pip<25"
  PIP="$VENV/bin/pip install"
fi

# pinned scientific stack (numpy 1.26.4: numpy 2.x breaks dask-ms 0.2.x)
$PIP -r "$HERE/requirements-v2.txt"
# dask-ms without deps (python-casacore needs system casacore libs);
# pfb-imaging from the pinned clone in ../lib (skip its pyproject deps/jax)
$PIP --no-deps "dask-ms==0.2.21"

$PIP --no-deps --no-refresh -e "$HERE/lib/pfb-imaging"


"$VENV/bin/python" -m pytest "$HERE/tests/" -q || echo "WARNING: test gate failed — inspect before running experiments"
echo "Environment ready at $VENV"
echo "Next: export VENV_PATH=$VENV STAGE_ROOT=$ROOT"
echo "      bash simulate_data.sh && bash run_experiment.sh"
