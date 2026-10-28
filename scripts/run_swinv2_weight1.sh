#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="$HOME/miniconda3/envs/glaucoma/bin/python"
run_dir="outputs/swinv2_384_w1_seed43_v1"
test ! -e "$run_dir"
test ! -e "${run_dir}.log"
test -f outputs/inference_validation_001/input.csv
test -d outputs/inference_validation_001/pictures_100
test -d outputs/inference_validation_001/pictures_5
"$python_bin" scripts/prepare_swinv2_weights.py
"$python_bin" -u train.py --model swinv2_tiny \
  --data-root datasets/glaucoma_first --output-dir "$run_dir" \
  --pretrained-path pretrain_model/swinv2_tiny_window8_256_ms_in1k.bin \
  --epochs 5 --batch-size 16 --size 384 --workers 4 --jpeg-draft \
  --lr 0.00003 --lr-schedule constant \
  --pos-weight 1 --seed 43 2>&1 | tee "${run_dir}.log"
"$python_bin" -m pip freeze > "$run_dir/environment-freeze.txt"
"$python_bin" -u validate.py --run-dir "$run_dir" 2>&1 | tee "${run_dir}.reload.log"
"$python_bin" predict.py --checkpoint "$run_dir/best.pt" \
  --input-csv outputs/inference_validation_001/input.csv \
  --images-dir outputs/inference_validation_001/pictures_100 \
  --output "$run_dir/timing_reference.csv"
"$python_bin" cost.py "$run_dir/timing_reference.csv" \
  outputs/inference_validation_001/pictures_100 outputs/inference_validation_001/pictures_5 \
  --checkpoint "$run_dir/best.pt" --output-dir "$run_dir/cost"
