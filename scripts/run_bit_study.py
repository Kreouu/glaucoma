"""BCE positive-weight adaptation of the new22K cosine crop recipe."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/resnetv2_bit_v1/plan.json').read_text())
    previous=json.loads(Path(plan['terminal_proof']).read_text());assert previous['status']=='PASS'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    out=ROOT/'outputs/resnetv2_bit_v1';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')

    def run(weight,seed,smoke=False):
        name='smoke' if smoke else f'lr{weight:g}_seed{seed}';path=out/name
        command=[sys.executable,'-u','train.py','--model','resnetv2_bit','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','448','--jpeg-draft','--workers','4','--lr',str(weight),'--lr-schedule','warmup_cosine','--warmup-epochs','1','--min-lr',str(weight/10),'--train-crop-min-area','.8','--pos-weight','3','--ema-decay','0','--epochs','2' if smoke else '5','--seed',str(seed),'--output-dir',str(path)]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('',):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==(64 if smoke else 5849) and check['max_probability_difference']<=1e-6 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'config.json').read_text())
        assert config['model']=='resnetv2_bit' and config['batch_size']==16 and config['size']==448 and config['lr']==weight and config['pos_weight']==3 and config['train_crop_min_area']==.8 and config['pretrained_sha256']==plan['pretrained_sha256']
        assert all(config['source_sha256'][n]==d for n,d in plan['source_sha256'].items() if n in config['source_sha256'])
        metrics=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]
        entry=dict(lr=weight,seed=seed,check=check);print('DONE',json.dumps(entry),flush=True);return entry

    run(plan['learning_rates'][0],42,True)
    results=[]
    for weight in plan['learning_rates']:
        results.append(run(weight,42));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda r:r['check']['val_auroc'])
    if best['check']['val_auroc']>=plan['confirmation_auroc']:
        for seed in (43,44):
            results.append(run(best['lr'],seed));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Tiny22K512/b16/crop.8/EMA.999/cosine1e-4->1e-5; BiTResNetV2-50x1 448/FP16/scale256/noEMA, peaks3e-5/1e-4, BCE3 fixed. Validation unweighted; no test use.'),indent=2)+'\n')


if __name__=='__main__':main()
