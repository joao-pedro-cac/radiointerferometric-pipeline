#!/usr/bin/env bash
# Create a working pipeline environment on Python 3.9 (macOS arm64 tested).
# Usage: bash setup_env.sh /path/to/venv
set -euo pipefail
VENV="${1:-/tmp/joao-venv}"
ROOT="${STAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade "pip<25"
# numpy pinned to 1.26.x: 2.x removes np.product, which dask-ms 0.2.x still uses
"$VENV/bin/pip" install "numpy==1.26.4" "numba==0.60.0" numexpr xarray "zarr==2.18.2" \
    "dask==2024.8.0" ducc0 astropy matplotlib psutil fsspec s3fs \
    toolz cloudpickle partd locket packaging importlib_metadata donfig rich "scipy==1.13.1"
# dask-ms without its casacore extra (python-casacore needs system casacore libs)
"$VENV/bin/pip" install --no-deps "dask-ms==0.2.21"
# pfb-imaging from the pinned clone in ../lib (skip its pyproject deps/jax)
"$VENV/bin/pip" install --no-deps --ignore-requires-python -e "$ROOT/../lib/pfb-imaging"

echo "Environment ready at $VENV"
echo "Run experiments with: VENV_PATH=$VENV bash scripts/run_minimal.sh"
