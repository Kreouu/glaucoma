"""Screen LR1e-4 on seed42, then confirm43/44 only after a predefined gain."""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan_path=ROOT/'docs/evidence/convnext_512_lr1e4_v1/plan.json'
    plan=json.loads(plan_path.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest,split
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False)
    (out/'plan.json').write_bytes(plan_path.read_bytes())
    results=[]
    for seed in (42,43,44):
        run=out/f'seed{seed}'
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16',
            '--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],
            '--batch-size','16','--size','512','--jpeg-draft','--workers','4','--epochs','5',
            '--lr','0.0001','--lr-schedule','constant','--pos-weight','3','--ema-decay','.999',
            '--seed',str(seed),'--output-dir',str(run)]
        print('START',seed,flush=True)
        with run.with_suffix('.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            path=run/kind
            subprocess.run([sys.executable,'validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
            config=json.loads((path/'config.json').read_text())
            control=json.loads((ROOT/f'outputs/convnext_512_ema_lr_v1/lr6e-05_seed{seed}/ema/config.json').read_text())
            for key in ('model','size','batch_size','epochs','precision','pos_weight','ema_decay','lr_schedule','augmentation','preprocessing','pretrained_sha256','csv_sha256'):
                assert config[key]==control[key],key
            assert config['lr']==1e-4
            reload=json.loads((path/'reload_check.json').read_text())
            assert reload['status']=='PASS' and reload['images']==5849 and reload['positives']==191
            assert sha(path/'best.pt')==reload['checkpoint_sha256']==(path/'best.sha256').read_text().split()[0]
        rows=[json.loads(s) for s in (run/'ema/metrics.jsonl').read_text().splitlines()]
        assert len(rows)==5 and all(math.isfinite(m[k]) for m in rows for k in ('val_auroc','val_auprc','valid_loss'))
        base=json.loads((ROOT/f'outputs/convnext_512_ema_lr_v1/lr6e-05_seed{seed}/ema/reload_check.json').read_text())
        entry=dict(seed=seed,best_epoch=reload['checkpoint_epoch'],auroc=reload['val_auroc'],ap=reload['val_auprc'],
            delta_auroc=reload['val_auroc']-base['val_auroc'],checkpoint_sha256=reload['checkpoint_sha256'])
        results.append(entry);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
        print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']<plan['screen_min_delta_auroc']:break
    report=dict(status='PASS',results=results,confirmed=len(results)==3,
        next='Fixed three-model equal-logit no-TTA batch1 verification if all seeds completed; otherwise another single factor',test_used=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
