#!/usr/bin/env bash
# Create the v2 pipeline environment with uv (uv assumed installed; falls back
# to python3 -m venv when uv is absent, e.g. the local macOS box).
#
#   bash setup_env.sh /path/to/venv        # then: export VENV_PATH=/path/to/venv
#
# Env vars: PYTHON_VERSION (default 3.11 — pfb-imaging requires >=3.10),
#           PFB_DIR (default: code/lib/pfb-imaging if present, else <root>/lib/pfb-imaging)
# Requires: uv (https://docs.astral.sh/uv/) or python3 >= 3.9.
set -euo pipefail
VENV="${1:-venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ROOT="${STAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
$PIP -r "$HERE/requirements-v2.txt"
# dask-ms without deps (python-casacore needs system casacore libs) and
# pfb-imaging from the pinned clone (skip its pyproject deps/jax).
# Use the venv's own pip for these two: uv has no --ignore-requires-python.
"$VENV/bin/pip" install --no-deps "dask-ms==0.2.21"
PFB_DIR="${PFB_DIR:-$HERE/lib/pfb-imaging}"
[ -d "$PFB_DIR" ] || PFB_DIR="$ROOT/lib/pfb-imaging"
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

"$VENV/bin/python" -m pytest "$HERE/tests/" -q || echo "WARNING: test gate failed — inspect before running experiments"
echo "Environment ready at $VENV"
echo "Next: export VENV_PATH=$VENV STAGE_ROOT=$ROOT"
echo "      bash simulate_data.sh && bash run_experiment.sh"
