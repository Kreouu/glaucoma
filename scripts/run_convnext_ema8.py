"""Eight-epoch paired study; reproduce the frozen five-epoch metrics and AMP trace."""
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/evidence/convnext_512_ema8_v1/plan.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan=json.loads(PLAN.read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest,split
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    output=ROOT/plan['output_dir'];output.mkdir(exist_ok=False)
    (output/'plan.json').write_bytes(PLAN.read_bytes());results=[]
    for seed in (42,43,44):
        run=output/f'seed{seed}';baseline=ROOT/f'outputs/convnext_512_ema_lr_v1/lr6e-05_seed{seed}'
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16',
            '--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],
            '--output-dir',str(run),'--epochs','8','--batch-size','16','--size','512','--workers','4',
            '--jpeg-draft','--lr','0.00006','--lr-schedule','constant','--pos-weight','3','--seed',str(seed),'--ema-decay','.999']
        print('START',seed,flush=True)
        with (output/f'seed{seed}.log').open('x') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        prefix=[];entry={'seed':seed}
        for kind in ('','ema'):
            path=run/kind;old=baseline/kind
            config=json.loads((path/'config.json').read_text())
            for key,value in dict(epochs=8,lr=6e-5,seed=seed,size=512,batch_size=16,workers=4,pos_weight=3,ema_decay=.999,train_count=51396,valid_count=5849).items():assert config[key]==value,key
            metrics=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
            control=[json.loads(s) for s in (old/'metrics.jsonl').read_text().splitlines()]
            assert [m['epoch'] for m in metrics]==list(range(1,9)) and len(control)==5
            keys=['epoch','train_loss','valid_loss','val_auroc','val_auprc','lr_last','amp_skipped_updates']
            if kind=='ema':keys.append('ema_updates')
            for actual,expected in zip(metrics[:5],control):
                for key in keys:assert actual[key]==expected[key],(seed,kind,actual['epoch'],key)
            assert all(math.isfinite(m[k]) for m in metrics for k in ('train_loss','valid_loss','val_auroc','val_auprc'))
            for split in ('train','valid'):assert (path/f'{split}_manifest.csv').read_bytes()==(old/f'{split}_manifest.csv').read_bytes()
            subprocess.run([sys.executable,'-u','validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
            reload=json.loads((path/'reload_check.json').read_text());best=max(metrics,key=lambda m:m['val_auroc'])
            assert reload['status']=='PASS' and reload['images']==5849 and reload['positives']==191
            assert reload['checkpoint_sha256']==sha(path/'best.pt')==(path/'best.sha256').read_text().split()[0]
            assert reload['val_auroc']==best['val_auroc'] and reload['val_auprc']==best['val_auprc']
            entry[kind or 'online']=dict(best_epoch=best['epoch'],auroc=best['val_auroc'],ap=best['val_auprc'],
                delta_auroc=best['val_auroc']-max(m['val_auroc'] for m in control),checkpoint_sha256=reload['checkpoint_sha256'])
            prefix.append(dict(weight_type=kind or 'online',first_five_metrics_exact=True,compared_keys=keys))
        assert (run/'learning_rate.csv').read_bytes().startswith((baseline/'learning_rate.csv').read_bytes())
        (run/'prefix_check.json').write_text(json.dumps(dict(status='PASS',checks=prefix,first_five_lr_amp_trace_exact=True,
            scope='Exact numerical metrics and AMP/LR trace; no claim of saved epoch-five parameter/RNG snapshots'),indent=2)+'\n')
        results.append(entry);(output/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
    summary=dict(status='PASS',seeds=results,mean_paired_ema_auroc_delta=statistics.mean(r['ema']['delta_auroc'] for r in results),
        positive_seeds=sum(r['ema']['delta_auroc']>0 for r in results),next='Always evaluate fixed no-TTA equal-logit ensemble at batch1 against the frozen current candidate',
        scope='Eight-epoch best-of-more-epochs selection on reused validation data; no test fitting or upload')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
