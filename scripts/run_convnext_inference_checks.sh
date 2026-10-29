#!/usr/bin/env bash
# Follow the existing three-model job; do not compete for its GPU.
set -euo pipefail
cd /home/ng/workspace/glaucoma
trap 'result=$?; printf "%s\n" "$result" > outputs/convnext_inference_checks_v1.exit_code' EXIT
while [[ ! -f outputs/three_model_overnight_v1.exit_code ]]; do
  tmux has-session -t glaucoma-three-models
  date -Is
  echo 'Waiting for the live three-model job before TTA inference.'
  sleep 3600
done
[[ "$(cat outputs/three_model_overnight_v1.exit_code)" == 0 ]]
[[ -z "$(/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"$HOME/miniconda3/envs/glaucoma/bin/python" -u scripts/evaluate_convnext_tta.py
date -Is
