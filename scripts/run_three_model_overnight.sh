#!/usr/bin/env bash
# Wait for the existing InceptionNeXt job; never start a competing GPU trainer.
set -euo pipefail
cd /home/ng/workspace/glaucoma
bundle="$1"
target="$2"
python_bin="$HOME/miniconda3/envs/glaucoma/bin/python"
trap 'result=$?; printf "%s\n" "$result" > outputs/three_model_overnight_v1.exit_code' EXIT
while [[ ! -f outputs/inceptionnext_512_w3_v1.exit_code ]]; do
  if ! tmux has-session -t glaucoma-inceptionnext-512 2>/dev/null; then
    echo 'Original InceptionNeXt session missing without an exit marker; stop for inspection.' >&2
    exit 1
  fi
  date -Is
  echo 'Waiting for verified live InceptionNeXt tmux session.'
  tail -2 outputs/inceptionnext_512_w3_v1.supervisor.log
  sleep 1800
done
[[ "$(cat outputs/inceptionnext_512_w3_v1.exit_code)" == 0 ]]
[[ -z "$(/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
git diff --exit-code
git diff --cached --exit-code
git fetch "$bundle" HEAD
git merge --ff-only FETCH_HEAD
[[ "$(git rev-parse HEAD)" == "$target" ]]
date -Is
echo 'InceptionNeXt complete; starting FasterViT-1 smoke and three seeds.'
bash scripts/run_fastervit_512.sh
mkdir outputs/three_model_comparison_v1
OMP_NUM_THREADS=1 "$python_bin" -u scripts/audit_three_models.py --output outputs/three_model_comparison_v1/comparison.json
"$python_bin" -u scripts/benchmark_three_models.py
OMP_NUM_THREADS=1 "$python_bin" -u scripts/export_three_model_backups.py
"$python_bin" - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
roots=[Path('outputs/inceptionnext_512_w3_v1'),Path('outputs/fastervit_512_w3_v1'),Path('outputs/three_model_comparison_v1')]
files=sorted(p for root in roots for p in root.rglob('*') if p.is_file() and p.suffix!='.pt')
manifest=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
index=Path('outputs/three_model_comparison_v1/records_manifest.json');index.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open('outputs/three_model_records_v1.tar','w') as archive:
    for p in files+[index]:archive.add(p,arcname=str(p),recursive=False)
with tarfile.open('outputs/three_model_backups_v1.tar','w') as archive:
    for p in sorted(Path('outputs/three_model_comparison_v1/model_backups').iterdir()):
        archive.add(p,arcname=str(p),recursive=False)
print('Training, audit, runtime comparison, exports and archives complete.')
PY
date -Is
