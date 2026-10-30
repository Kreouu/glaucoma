"""Screen late cosine decay after an exact five-epoch prefix, then confirm paired seeds."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def metrics(path):
    return [json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]


def check_prefix(run,control,batches):
    for kind in ('','ema'):
        keys=['epoch','train_loss','valid_loss','val_auroc','val_auprc','amp_skipped_updates','lr_last']
        if kind:keys.append('ema_updates')
        a,b=metrics(run/kind),metrics(control/kind)
        assert len(a)==len(b)==8
        for x,y in zip(a[:5],b[:5]):
            for key in keys:assert x[key]==y[key],(kind,x['epoch'],key)
    with (run/'learning_rate.csv').open() as stream:actual=list(csv.DictReader(stream))
    with (control/'learning_rate.csv').open() as stream:old=list(csv.DictReader(stream))
    assert len(actual)==len(old)==8*batches and actual[:5*batches]==old[:5*batches]
    tail=[float(r['lr']) for r in actual[5*batches:]]
    assert tail[0]==6e-5 and tail[-1]==6e-6 and all(a>=b for a,b in zip(tail,tail[1:]))
    report=dict(status='PASS',prefix_epochs=5,online_and_ema_metrics_exact=True,lr_amp_trace_prefix_exact=True,
                cosine_steps=len(tail),first_tail_lr=tail[0],last_lr=tail[-1],tail_monotone=True)
    (run/'prefix_check.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    plan_path=ROOT/'docs/evidence/convnext_512_latecosine_v1/plan.json';plan=json.loads(plan_path.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in plan['control_source_sha256'].items():assert sha(ROOT/'outputs/convnext_512_ema8_v1/source_snapshot'/name)==h,name
    for split,h in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==h,split
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False);(out/'plan.json').write_bytes(plan_path.read_bytes())
    common=['--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first',
        '--pretrained-path',plan['pretrained_path'],'--epochs','8','--batch-size','16','--size','512',
        '--jpeg-draft','--lr','0.00006','--pos-weight','3','--ema-decay','.999']
    late=['--lr-schedule','late_cosine','--decay-start-epoch','5','--min-lr','0.000006']
    def train(script,path,seed,extra):
        print('START',str(path),flush=True)
        with path.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(script),*common,'--output-dir',str(path),'--seed',str(seed),*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    train(ROOT/'outputs/convnext_512_ema8_v1/source_snapshot/train.py',out/'smoke_control',42,
          ['--workers','0','--train-limit','64','--valid-limit','64','--lr-schedule','constant'])
    train(ROOT/'train.py',out/'smoke_late',42,['--workers','0','--train-limit','64','--valid-limit','64',*late])
    smoke=check_prefix(out/'smoke_late',out/'smoke_control',4)
    subprocess.run([sys.executable,'validate.py','--run-dir',str(out/'smoke_late/ema')],cwd=ROOT,check=True)
    (out/'smoke_check.json').write_text(json.dumps(smoke,indent=2)+'\n');print('SMOKE PASS',flush=True)
    results=[]
    for seed in (42,43,44):
        run=out/f'seed{seed}';control=ROOT/f'outputs/convnext_512_ema8_v1/seed{seed}'
        train(ROOT/'train.py',run,seed,['--workers','4',*late]);check_prefix(run,control,3213)
        for kind in ('','ema'):subprocess.run([sys.executable,'validate.py','--run-dir',str(run/kind)],cwd=ROOT,check=True)
        reload=json.loads((run/'ema/reload_check.json').read_text());assert reload['status']=='PASS' and reload['images']==5849 and reload['positives']==191
        assert sha(run/'ema/best.pt')==reload['checkpoint_sha256']==(run/'ema/best.sha256').read_text().split()[0]
        best=max(metrics(run/'ema'),key=lambda m:m['val_auroc']);base=max(metrics(control/'ema'),key=lambda m:m['val_auroc'])
        assert best['val_auroc']==reload['val_auroc'] and best['val_auprc']==reload['val_auprc']
        entry=dict(seed=seed,best_epoch=best['epoch'],auroc=best['val_auroc'],ap=best['val_auprc'],delta_auroc=best['val_auroc']-base['val_auroc'],checkpoint_sha256=reload['checkpoint_sha256'])
        results.append(entry);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']<plan['screen_min_delta_auroc']:break
    summary=dict(status='PASS',results=results,confirmed=len(results)==3,mean_paired_auroc_delta=statistics.mean(r['delta_auroc'] for r in results),
        next='Fixed no-TTA batch1 ensemble if confirmed; otherwise continue the active optimization goal',scope='Late cosine only; same eight-epoch budget and exact five-epoch numerical prefix; no test fitting')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
