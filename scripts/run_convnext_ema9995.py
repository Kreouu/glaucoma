"""EMA .9995 screen with an exact online-trajectory control; confirm only if promising."""
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def metrics(path):
    return [json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]


def main():
    plan_path=ROOT/'docs/evidence/convnext_512_ema9995_v1/plan.json';plan=json.loads(plan_path.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    for split,h in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==h,split
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False);(out/'plan.json').write_bytes(plan_path.read_bytes())
    common=['--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first',
        '--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft',
        '--lr','0.00006','--lr-schedule','constant','--pos-weight','3','--ema-decay','.9995']
    def train(path,seed,epochs,extra):
        print('START',str(path),flush=True)
        with path.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable,'-u','train.py',*common,'--epochs',str(epochs),'--output-dir',str(path),
                '--seed',str(seed),*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    def control_check(run,control,epochs):
        a,b=metrics(run),metrics(control);assert len(a)==epochs and len(b)>=epochs
        keys=['epoch','train_loss','valid_loss','val_auroc','val_auprc','lr_last','amp_skipped_updates']
        assert all(x[k]==y[k] for x,y in zip(a,b[:epochs]) for k in keys)
        with (run/'learning_rate.csv').open() as stream:steps=list(csv.DictReader(stream))
        with (control/'learning_rate.csv').open() as stream:old=list(csv.DictReader(stream))
        assert steps==old[:len(steps)]
        ema=metrics(run/'ema');old_ema=metrics(control/'ema')
        assert all(a['ema_updates']==b['ema_updates'] for a,b in zip(ema,old_ema[:epochs]))
        result=dict(status='PASS',online_metrics_exact_epochs=epochs,lr_amp_trace_exact=True,ema_update_counts_exact=True,
                    scope='Only EMA decay intentionally differs; no claim that shadow EMA values should match')
        (run/'control_check.json').write_text(json.dumps(result,indent=2)+'\n');return result
    smoke=out/'smoke';train(smoke,42,2,['--workers','0','--train-limit','64','--valid-limit','64'])
    proof=control_check(smoke,ROOT/'outputs/convnext_512_latecosine_v1/smoke_control',2)
    subprocess.run([sys.executable,'validate.py','--run-dir',str(smoke/'ema')],cwd=ROOT,check=True)
    (out/'smoke_check.json').write_text(json.dumps(proof,indent=2)+'\n');print('SMOKE PASS',flush=True)
    results=[]
    for seed in (42,43,44):
        run=out/f'seed{seed}';control=ROOT/f'outputs/convnext_512_ema8_v1/seed{seed}'
        train(run,seed,8,['--workers','4']);control_check(run,control,8)
        assert (run/'best_valid_predictions.csv').read_bytes()==(control/'best_valid_predictions.csv').read_bytes()
        for kind in ('','ema'):
            path=run/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
            config=json.loads((path/'config.json').read_text());assert config['ema_decay']==.9995 and config['lr']==6e-5 and config['epochs']==8
            reload=json.loads((path/'reload_check.json').read_text());assert reload['status']=='PASS' and reload['images']==5849 and reload['positives']==191
            assert sha(path/'best.pt')==reload['checkpoint_sha256']==(path/'best.sha256').read_text().split()[0]
        observed=metrics(run/'ema');assert all(math.isfinite(m[k]) for m in observed for k in ('train_loss','valid_loss','val_auroc','val_auprc'))
        best=max(observed,key=lambda m:m['val_auroc']);base=max(metrics(control/'ema'),key=lambda m:m['val_auroc'])
        assert reload['val_auroc']==best['val_auroc'] and reload['val_auprc']==best['val_auprc']
        entry=dict(seed=seed,best_epoch=best['epoch'],auroc=best['val_auroc'],ap=best['val_auprc'],delta_auroc=best['val_auroc']-base['val_auroc'],checkpoint_sha256=reload['checkpoint_sha256'])
        results.append(entry);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']<plan['screen_min_delta_auroc']:break
    result=dict(status='PASS',results=results,confirmed=len(results)==3,mean_paired_auroc_delta=statistics.mean(r['delta_auroc'] for r in results),
        next='Fixed no-TTA three-model equal-logit batch1 verification if confirmed; otherwise continue optimization',scope='EMA decay only, exact full online-trajectory control, no test fitting')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
