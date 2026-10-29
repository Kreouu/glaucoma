#!/usr/bin/env bash
# 640 vs frozen 512 baseline, three paired seeds; no test inference or upload.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="$HOME/miniconda3/envs/glaucoma/bin/python"
batch="outputs/convnext_640_w3_v1"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
  echo "Usage: bash scripts/run_convnext_640.sh [--dry-run]" >&2
  exit 2
fi
dry_run=false
[[ "${1:-}" == "--dry-run" ]] && dry_run=true
if ! "$dry_run"; then
  "$python_bin" - <<'PY'
import hashlib,json
from pathlib import Path
plan=json.loads(Path('docs/evidence/model_selection_20261029/plan.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ('train.py','dataset.py','model.py','model_convnext.py','amp_precision.py','lr_schedule.py','validate.py'):
    assert sha(name)==plan['sources'][name],f'Frozen source differs: {name}'
assert sha('pretrain_model/convnext_tiny_fb_in1k.bin')==plan['pretrained_sha256']['convnext_tiny']
for split,h in plan['csv_sha256'].items():
    assert sha(f'datasets/glaucoma_first/{split}.csv')==h,f'Data CSV differs: {split}'
print('Baseline source, original weights and data CSV guards PASS')
PY
  # Refuse to overwrite or ambiguously resume an existing batch.
  mkdir "$batch"
fi
smoke=("$python_bin" -u train.py --model convnext_tiny --precision fp16
  --data-root datasets/glaucoma_first --output-dir "$batch/smoke"
  --pretrained-path pretrain_model/convnext_tiny_fb_in1k.bin
  --epochs 1 --batch-size 16 --size 640 --workers 4 --jpeg-draft
  --lr 0.00003 --lr-schedule constant --pos-weight 3 --seed 42
  --train-limit 64 --valid-limit 64)
if "$dry_run"; then
  printf '%q ' "${smoke[@]}"; printf '\n'
  printf '%q ' "$python_bin" -u validate.py --run-dir "$batch/smoke"; printf '\n'
else
  "${smoke[@]}" 2>&1 | tee "$batch/smoke.train.log"
  "$python_bin" -u validate.py --run-dir "$batch/smoke" 2>&1 | tee "$batch/smoke.reload.log"
fi
for seed in 42 43 44; do
  run="$batch/seed${seed}"
  command=("$python_bin" -u train.py --model convnext_tiny --precision fp16
    --data-root datasets/glaucoma_first --output-dir "$run"
    --pretrained-path pretrain_model/convnext_tiny_fb_in1k.bin
    --epochs 5 --batch-size 16 --size 640 --workers 4 --jpeg-draft
    --lr 0.00003 --lr-schedule constant --pos-weight 3 --seed "$seed")
  if "$dry_run"; then
    printf '%q ' "${command[@]}"; printf '\n'
    printf '%q ' "$python_bin" -u validate.py --run-dir "$run"; printf '\n'
  else
    "${command[@]}" 2>&1 | tee "$batch/seed${seed}.train.log"
    "$python_bin" -u validate.py --run-dir "$run" 2>&1 | tee "$batch/seed${seed}.reload.log"
  fi
done
if ! "$dry_run"; then
  echo "Training/reload complete. Compare all three paired seeds with the frozen 512 constant-LR baseline."
fi
