#!/usr/bin/env bash
# Create the v2 pipeline environment with uv (uv assumed installed; falls back
# to python3 -m venv when uv is absent, e.g. the local macOS box).
#
#   bash setup_env.sh            # creates ./venv in the repo tree (in-tree)
#   bash setup_env.sh /path/to/venv
#
# Works from both trees (local <archive>/code/ and the flattened remote repo
# root): paths are auto-detected via scripts/paths.sh.
# Env vars: PYTHON_VERSION (default 3.11 — pfb-imaging requires >=3.10);
#           overrides in scripts/env.sh (see env.sh.example).
# Requires: uv (https://docs.astral.sh/uv/) or python3 >= 3.9.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # = CODE_ROOT in both trees
[ -f "$HERE/scripts/env.sh" ] && source "$HERE/scripts/env.sh"
source "$HERE/scripts/paths.sh"
VENV="${1:-$VENV_PATH}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if command -v uv >/dev/null 2>&1; then
  echo "== uv $(uv --version) — creating $VENV (python $PYTHON_VERSION)"
  uv venv "$VENV" --python "$PYTHON_VERSION" --seed 2>/dev/null || uv venv "$VENV" --python 3 --seed
  PIP="uv pip install --python $VENV/bin/python"
else
  echo "== uv not found — falling back to python3 -m venv (install uv for faster setup)"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade "pip<25"
  PIP="$VENV/bin/pip install"
fi

# pinned scientific stack (numpy 1.26.4: numpy 2.x breaks dask-ms 0.2.x)
$PIP -r "$CODE_ROOT/requirements-v2.txt"
# dask-ms without deps (python-casacore needs system casacore libs) and
# pfb-imaging from the pinned clone (skip its pyproject deps/jax).
# Use the venv's own pip for these two: uv has no --ignore-requires-python.
"$VENV/bin/pip" install --no-deps "dask-ms==0.2.21"
"$VENV/bin/pip" install --no-deps --ignore-requires-python -e "$PFB_DIR"

# record the exact build in the log — version drift is the first thing to
# check when a test passes locally but fails on the remote
"$VENV/bin/python" - <<'EOF'
import sys
print("python", sys.version.split()[0], sys.executable)
for m in ("numpy", "numba", "llvmlite", "ducc0", "astropy", "dask", "xarray",
          "zarr", "daskms", "pfb_imaging", "pytest"):
    try:
        mod = __import__(m)
        print(f"  {m} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m} MISSING: {type(e).__name__}: {e}")
EOF

"$VENV/bin/python" -m pytest "$CODE_ROOT/tests/" -q || echo "WARNING: test gate failed — inspect before running experiments"
echo "Environment ready at $VENV (VENV_PATH=$VENV_PATH)"
echo "Next (from this directory): bash simulate_data.sh && bash run_experiment.sh"
