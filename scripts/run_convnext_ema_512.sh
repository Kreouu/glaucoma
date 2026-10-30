#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="$HOME/miniconda3/envs/glaucoma/bin/python"
batch=outputs/convnext_512_ema999_v1
"$python_bin" - <<'PY'
import hashlib,json
from pathlib import Path
p=json.loads(Path('docs/evidence/convnext_512_ema999_v1/plan.json').read_text())
for name,h in p['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
for split,h in p['csv_sha256'].items():assert hashlib.sha256(Path(f'datasets/glaucoma_first/{split}.csv').read_bytes()).hexdigest()==h,split
assert hashlib.sha256(Path('pretrain_model/convnext_tiny_fb_in1k.bin').read_bytes()).hexdigest()==p['pretrained_sha256']
print('Frozen source, data and pretrained guards PASS')
PY
mkdir "$batch"
common=(--model convnext_tiny --precision fp16 --data-root datasets/glaucoma_first
 --pretrained-path pretrain_model/convnext_tiny_fb_in1k.bin --batch-size 16
 --size 512 --workers 4 --jpeg-draft --lr 0.00003 --lr-schedule constant --pos-weight 3)
for mode in plain ema; do
 extra=();[[ "$mode" == ema ]] && extra=(--ema-decay .999)
 "$python_bin" -u train.py "${common[@]}" --epochs 2 --seed 42 --train-limit 64 --valid-limit 64 \
  --output-dir "$batch/smoke_$mode" "${extra[@]}" 2>&1 | tee "$batch/smoke_$mode.log"
 "$python_bin" -u validate.py --run-dir "$batch/smoke_$mode"
done
"$python_bin" -u validate.py --run-dir "$batch/smoke_ema/ema"
"$python_bin" - <<'PY'
import json,torch
from pathlib import Path
root=Path('outputs/convnext_512_ema999_v1');a=root/'smoke_plain';b=root/'smoke_ema'
x=torch.load(a/'best.pt',map_location='cpu',weights_only=True);y=torch.load(b/'best.pt',map_location='cpu',weights_only=True)
assert x['model'].keys()==y['model'].keys() and all(torch.equal(v,y['model'][k]) for k,v in x['model'].items())
for filename in ('train_manifest.csv','valid_manifest.csv','best_valid_predictions.csv'):assert (a/filename).read_bytes()==(b/filename).read_bytes()
left=[json.loads(l) for l in (a/'metrics.jsonl').read_text().splitlines()];right=[json.loads(l) for l in (b/'metrics.jsonl').read_text().splitlines()]
for p,q in zip(left,right):
 for key in ('epoch','train_loss','valid_loss','val_auroc','val_auprc'):assert p[key]==q[key],key
ema=[json.loads(l) for l in (b/'ema/metrics.jsonl').read_text().splitlines()]
assert ema[-1]['ema_updates']==8-sum(q['amp_skipped_updates'] for q in right)
r=dict(status='PASS',online_parameters='exactly equal to EMA-disabled control',online_predictions='exactly equal',metrics='exactly equal except runtime/memory/logging',ema_updates=ema[-1]['ema_updates'],steps=8)
(root/'smoke_pair_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
PY
for seed in 42 43 44; do
 "$python_bin" -u train.py "${common[@]}" --epochs 5 --seed "$seed" --ema-decay .999 \
  --output-dir "$batch/seed$seed" 2>&1 | tee "$batch/seed$seed.log"
 "$python_bin" -u validate.py --run-dir "$batch/seed$seed"
 "$python_bin" -u validate.py --run-dir "$batch/seed$seed/ema"
done
"$python_bin" -u scripts/audit_convnext_ema.py
