#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="$HOME/miniconda3/envs/glaucoma/bin/python"
epochs="${1:-1}"
run_dir="outputs/baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs

echo "Experiment: $run_dir"
echo "Log: ${run_dir}.log"
echo "Full train/validation data, pretrained EfficientNet-B0, epochs=$epochs."

"$python_bin" -u train.py \
  --data-root datasets/glaucoma_first \
  --output-dir "$run_dir" \
  --pretrained-path pretrain_model/efficientnet_b0_rwightman-7f5810bc.pth \
  --epochs "$epochs" \
  --jpeg-draft \
  --batch-size 16 \
  --workers 4 \
  2>&1 | tee "${run_dir}.log"

echo "Completed: $run_dir"
