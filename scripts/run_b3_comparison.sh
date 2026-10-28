#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="$HOME/miniconda3/envs/glaucoma/bin/python"
run_dir="outputs/b3_384_w3_seed43_v1"
mkdir -p outputs
test ! -e "$run_dir"
test ! -e "${run_dir}.log"
"$python_bin" scripts/verify_b3_weights.py

"$python_bin" -u train.py \
  --data-root datasets/glaucoma_first --output-dir "$run_dir" \
  --model efficientnet_b3 \
  --pretrained-path pretrain_model/efficientnet_b3_rwightman-b3899882.pth \
  --epochs 5 --batch-size 16 --size 384 --workers 4 \
  --jpeg-draft --lr 0.0001 --pos-weight 3 --seed 43 \
  2>&1 | tee "${run_dir}.log"
"$python_bin" -u validate.py --run-dir "$run_dir" \
  2>&1 | tee "${run_dir}.reload.log"

# Reuse validation-only samples; this does not generate test-set predictions.
"$python_bin" predict.py --checkpoint "$run_dir/best.pt" \
  --input-csv outputs/inference_validation_001/input.csv \
  --images-dir outputs/inference_validation_001/pictures_100 \
  --output "$run_dir/timing_reference.csv"
"$python_bin" cost.py "$run_dir/timing_reference.csv" \
  outputs/inference_validation_001/pictures_100 \
  outputs/inference_validation_001/pictures_5 \
  --checkpoint "$run_dir/best.pt" --output-dir "$run_dir/cost"
