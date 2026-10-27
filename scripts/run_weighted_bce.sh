#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
for weight in 3 10; do
  bash scripts/run_baseline.sh 5 "$weight"
done
