#!/usr/bin/env bash
# Quick runner: execute only the fast-tier tests (developer-friendly)
# Usage: ./scripts/run_fast.sh [-v]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHON=${PYTHON:-python3}
if [ "$#" -gt 0 ] && [ "$1" = "-v" ]; then
  echo "Running fast tests (verbose)"
  exec "$PYTHON" -m pytest -m fast -q
else
  echo "Running fast tests"
  exec "$PYTHON" -m pytest -m fast -q
fi
