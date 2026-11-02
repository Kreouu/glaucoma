"""Fixed Tiny weight-decay screen with unchanged default recipe."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_weight_decay_v1/plan.json').read_text())
    previous=json.loads(Path(plan['terminal_proof']).read_text());assert previous['status']=='PASS'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    out=ROOT/'outputs/convnext_weight_decay_v1';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')

    def run(wd,seed,smoke=False):
        name='smoke' if smoke else f'wd{wd:g}_seed{seed}';path=out/name
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--weight-decay',str(wd),'--pos-weight','3','--ema-decay','.999','--epochs','1' if smoke else '5','--seed',str(seed),'--output-dir',str(path)]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==(64 if smoke else 5849) and check['max_probability_difference']<=1e-6 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text())
        assert config['model']=='convnext_tiny' and config['batch_size']==16 and config['size']==512 and config['lr']==.0001 and config['weight_decay']==wd and config['pretrained_sha256']==plan['pretrained_sha256']
        assert all(config['source_sha256'][n]==d for n,d in plan['source_sha256'].items() if n in config['source_sha256'])
        metrics=[json.loads(x) for x in (path/'ema/metrics.jsonl').read_text().splitlines()];assert metrics[-1]['ema_updates']>0
        entry=dict(weight_decay=wd,seed=seed,check=check);print('DONE',json.dumps(entry),flush=True);return entry

    run(plan['weight_decays'][0],42,True)
    results=[]
    for wd in plan['weight_decays']:
        results.append(run(wd,42));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda r:r['check']['val_auroc'])
    if best['check']['val_auroc']>=plan['confirmation_auroc']:
        for seed in (43,44):
            results.append(run(best['weight_decay'],seed));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Tiny1K512/batch16/fixedLR1e-4/BCE3/EMA.999/5epochs; weight decay0 and0.1 vs original0.01. No test use; native inference separate.'),indent=2)+'\n')


if __name__=='__main__':main()
