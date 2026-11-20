#!/usr/bin/env bash
# Build the beamer deck: tables from results, then latexmk.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "$HERE")"
VENV_PATH="${VENV_PATH:-/tmp/joao-venv}"
"$VENV_PATH/bin/python" "$HERE/gen_tables.py" "$CODE_ROOT/results"
cd "$HERE"
latexmk -pdf -interaction=nonstopmode main.tex > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built $HERE/main.pdf"
grep -c "Overfull" build.log || true
