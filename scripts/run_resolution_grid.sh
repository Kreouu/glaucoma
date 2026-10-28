#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "$HOME/miniconda3/envs/glaucoma/bin/python" -u scripts/run_resolution_grid.py "$@"
