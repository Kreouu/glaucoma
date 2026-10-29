#!/usr/bin/env bash
set -euo pipefail
cd /home/ng/workspace/glaucoma
trap 'result=$?; printf "%s\n" "$result" > outputs/convnext_candidate_verification_v1.exit_code' EXIT
while [[ ! -f outputs/convnext_inference_checks_v1.exit_code ]]; do
  tmux has-session -t glaucoma-convnext-inference
  date -Is
  echo 'Waiting for queued validation TTA checks.'
  sleep 3600
done
[[ "$(cat outputs/convnext_inference_checks_v1.exit_code)" == 0 ]]
[[ -z "$(/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"$HOME/miniconda3/envs/glaucoma/bin/python" -u scripts/verify_convnext_inference_candidates.py
date -Is
