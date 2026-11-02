"""Small22K cosine plus crop adaptation screen on WSL."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_small22k_crop_v1/plan.json').read_text())
    previous=json.loads(Path(plan['terminal_proof']).read_text());assert previous['status']=='PASS'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    assert json.loads((ROOT/'docs/evidence/convnext_small22k_crop_v1/cpu_check.json').read_text())['status']=='PASS'
    out=ROOT/'outputs/convnext_small22k_crop_v1';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')

    def run(lr,seed,smoke=False):
        name='smoke' if smoke else f'lr{lr:g}_seed{seed}';path=out/name
        command=[sys.executable,'-u','train.py','--model','convnext_small','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','8','--size','512','--jpeg-draft','--workers','4','--lr',str(lr),'--lr-schedule','warmup_cosine','--warmup-epochs','1','--min-lr',str(lr/10),'--train-crop-min-area','.8','--pos-weight','3','--ema-decay','.999','--epochs','2' if smoke else '5','--seed',str(seed),'--output-dir',str(path)]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==(64 if smoke else 5849) and check['max_probability_difference']<=1e-6 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text())
        assert config['model']=='convnext_small' and config['batch_size']==8 and config['size']==512 and config['lr']==lr and config['train_crop_min_area']==.8 and config['pretrained_sha256']==plan['pretrained_sha256']
        assert all(config['source_sha256'][n]==d for n,d in plan['source_sha256'].items() if n in config['source_sha256'])
        metrics=[json.loads(x) for x in (path/'ema/metrics.jsonl').read_text().splitlines()];assert metrics[-1]['ema_updates']>0
        entry=dict(lr=lr,seed=seed,check=check);print('DONE',json.dumps(entry),flush=True);return entry

    run(plan['learning_rates'][0],42,True)
    results=[]
    for lr in plan['learning_rates']:
        results.append(run(lr,42));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda r:r['check']['val_auroc'])
    if best['check']['val_auroc']>=plan['confirmation_auroc']:
        for seed in (43,44):
            results.append(run(best['lr'],seed));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Small22K512/b8/crop.8; one epoch warmup cosine peak3e-5/1e-4 to peak/10; comparison is whole recipe including reduced batch. No test use; native inference separate.'),indent=2)+'\n')


if __name__=='__main__':main()
