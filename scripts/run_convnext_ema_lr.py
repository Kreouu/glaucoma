"""Bounded EMA-512 learning-rate screen, then paired confirmation of one winner."""
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/evidence/convnext_512_ema_lr_v1/plan.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan=json.loads(PLAN.read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest,split
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    output=ROOT/plan['output_dir'];output.mkdir(exist_ok=False)
    (output/'plan.json').write_bytes(PLAN.read_bytes())
    baseline={}
    for seed in (42,43,44):
        path=ROOT/f'outputs/convnext_512_ema999_v1/seed{seed}/ema'
        metrics=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
        baseline[seed]=max(metrics,key=lambda r:r['val_auroc'])
        assert baseline[seed]['val_auroc']==plan['baseline_ema_auroc'][str(seed)]
    results=[]

    def run(lr,seed):
        name=f'lr{lr:.0e}_seed{seed}';folder=output/name
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16',
            '--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],
            '--output-dir',str(folder),'--epochs','5','--batch-size','16','--size','512','--workers','4',
            '--jpeg-draft','--lr',str(lr),'--lr-schedule','constant','--pos-weight','3','--seed',str(seed),'--ema-decay','.999']
        print('START',name,flush=True)
        with (output/f'{name}.log').open('x') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            subprocess.run([sys.executable,'-u','validate.py','--run-dir',str(folder/kind)],cwd=ROOT,check=True)
        config=json.loads((folder/'ema/config.json').read_text())
        for key,value in dict(lr=lr,seed=seed,size=512,epochs=5,batch_size=16,workers=4,pos_weight=3,ema_decay=.999,weight_type='ema',train_count=51396,valid_count=5849).items():assert config[key]==value,key
        for key,value in config['source_sha256'].items():assert sha(ROOT/key)==value,key
        metrics=[json.loads(s) for s in (folder/'ema/metrics.jsonl').read_text().splitlines()]
        assert [m['epoch'] for m in metrics]==[1,2,3,4,5]
        assert all(math.isfinite(m[k]) for m in metrics for k in ('val_auroc','val_auprc','train_loss','valid_loss'))
        reload=json.loads((folder/'ema/reload_check.json').read_text());assert reload['status']=='PASS' and reload['images']==5849
        assert reload['checkpoint_sha256']==sha(folder/'ema/best.pt')==(folder/'ema/best.sha256').read_text().split()[0]
        best=max(metrics,key=lambda m:m['val_auroc'])
        assert reload['val_auroc']==best['val_auroc'] and reload['val_auprc']==best['val_auprc']
        for split in ('train','valid'):assert (folder/f'{split}_manifest.csv').read_bytes()==(ROOT/f'outputs/convnext_512_ema999_v1/seed{seed}/{split}_manifest.csv').read_bytes()
        entry=dict(lr=lr,seed=seed,best_epoch=best['epoch'],auroc=best['val_auroc'],ap=best['val_auprc'],
            delta_auroc=best['val_auroc']-baseline[seed]['val_auroc'],checkpoint_sha256=reload['checkpoint_sha256'],reload='PASS')
        results.append(entry);(output/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        return entry

    screen=[run(lr,42) for lr in plan['screen_learning_rates']]
    winner=max(screen,key=lambda r:r['auroc'])
    if winner['delta_auroc']>=plan['screen_min_delta_auroc']:
        for seed in (43,44):run(winner['lr'],seed)
        confirmed=[r for r in results if r['lr']==winner['lr']]
        mean=statistics.mean(r['delta_auroc'] for r in confirmed)
        decision=dict(selected_lr=winner['lr'],paired_mean_delta_auroc=mean,
            positive_seeds=sum(r['delta_auroc']>0 for r in confirmed),
            proceed_to_batch_one=mean>=.0005 and sum(r['delta_auroc']>0 for r in confirmed)>=2)
    else:decision=dict(proceed_to_batch_one=False,reason='Neither seed42 screen exceeded baseline by 0.0005; no confirmation runs')
    (output/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,decision=decision,
        scope='Validation-only exploratory selection, no test fitting or automatic upload; AP recorded, not hard gate'),indent=2)+'\n')
    print('COMPLETE',json.dumps(decision),flush=True)


if __name__=='__main__':main()
