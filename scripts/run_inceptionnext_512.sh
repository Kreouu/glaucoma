#!/usr/bin/env bash
# InceptionNeXt-Tiny vs frozen ConvNeXt-512 baseline, three paired seeds; no test inference or upload.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="$HOME/miniconda3/envs/glaucoma/bin/python"
batch="outputs/inceptionnext_512_w3_v1"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
  echo "Usage: bash scripts/run_convnext_640.sh [--dry-run]" >&2
  exit 2
fi
dry_run=false
[[ "${1:-}" == "--dry-run" ]] && dry_run=true
if ! "$dry_run"; then
  "$python_bin" scripts/prepare_inceptionnext_weights.py
  "$python_bin" - <<'PYGUARD'
import hashlib,json
from pathlib import Path
plan=json.loads(Path('docs/evidence/inceptionnext_512_w3_v1/plan.json').read_text())
for name,h in plan['sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
for split,h in plan['csv_sha256'].items():
    assert hashlib.sha256(Path(f'datasets/glaucoma_first/{split}.csv').read_bytes()).hexdigest()==h,split
print('Source and data guards PASS')
PYGUARD
  # Refuse to overwrite or ambiguously resume an existing batch.
  mkdir "$batch"
fi
smoke=("$python_bin" -u train.py --model inception_next_tiny --precision fp16
  --data-root datasets/glaucoma_first --output-dir "$batch/smoke"
  --pretrained-path pretrain_model/inception_next_tiny_sail_in1k.bin
  --epochs 1 --batch-size 16 --size 512 --workers 4 --jpeg-draft
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
  command=("$python_bin" -u train.py --model inception_next_tiny --precision fp16
    --data-root datasets/glaucoma_first --output-dir "$run"
    --pretrained-path pretrain_model/inception_next_tiny_sail_in1k.bin
    --epochs 5 --batch-size 16 --size 512 --workers 4 --jpeg-draft
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
