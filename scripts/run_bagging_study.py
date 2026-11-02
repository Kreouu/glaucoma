"""Five predeclared official-train subsets; no validation/test rows in fold construction."""
import csv,hashlib,json,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_bagging_v1/plan.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    previous=json.loads(Path(plan['terminal_proof']).read_text());assert previous['status']=='PASS'
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    data=ROOT/'datasets/glaucoma_first'
    for split,digest in plan['csv_sha256'].items():assert sha(data/f'{split}.csv')==digest
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    with (data/'train.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    assert len(rows)==51396 and sum(int(r['True_Label']) for r in rows)==1657
    names=[r['Image_name'] for r in rows];assert len(set(n.casefold() for n in names))==51396
    with (data/'valid.csv').open(encoding='utf-8-sig') as f:valid_names={r['Image_name'].casefold() for r in csv.DictReader(f)}
    assert not set(n.casefold() for n in names)&valid_names
    groups=[[i for i,r in enumerate(rows) if r['True_Label']==str(c)] for c in (0,1)];fold=[None]*len(rows);rng=random.Random(plan['split_seed'])
    for indices in groups:
        rng.shuffle(indices)
        for j,index in enumerate(indices):fold[index]=j%5
    assert set(fold)==set(range(5))
    out=ROOT/'outputs/convnext_bagging_v1';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');summaries=[]
    for k in range(5):
        folder=out/f'data/fold{k}';folder.mkdir(parents=True)
        kept=[r for i,r in enumerate(rows) if fold[i]!=k]
        assert all(row==rows[i] for i,row in zip((i for i in range(len(rows)) if fold[i]!=k),kept))
        with (folder/'train.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
        for name in ('train','valid','valid.csv'):(folder/name).symlink_to(data/name)
        summaries.append(dict(fold=k,train=len(kept),positive=sum(int(r['True_Label']) for r in kept),excluded=len(rows)-len(kept),train_csv_sha256=sha(folder/'train.csv')))
    assert sum(s['train'] for s in summaries)==4*51396 and sum(s['positive'] for s in summaries)==4*1657
    (out/'fold_summary.json').write_text(json.dumps(dict(status='PASS',split_seed=plan['split_seed'],folds=summaries,scope='Every official train row excluded once, used four times; official validation untouched; no patient-ID independence claim'),indent=2)+'\n')

    def run(fold_id,seed,smoke=False):
        folder=out/f'data/fold{fold_id}';name='smoke' if smoke else f'fold{fold_id}_seed{seed}';path=out/name
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root',str(folder),'--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','constant','--pos-weight','3','--ema-decay','.999','--epochs','1' if smoke else '6','--seed',str(seed),'--output-dir',str(path)]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==(64 if smoke else 5849) and check['max_probability_difference']==0 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text());assert config['csv_sha256']==dict(train=summaries[fold_id]['train_csv_sha256'],valid=plan['csv_sha256']['valid'])
        assert config['model']=='convnext_tiny' and config['size']==512 and config['batch_size']==16 and config['ema_decay']==.999 and config['lr']==1e-4 and config['seed']==seed and config['jpeg_draft']
        entry=dict(fold=fold_id,seed=seed,check=check);print('DONE',json.dumps(entry),flush=True);return entry

    run(0,42,True)
    results=[]
    for k in range(5):
        results.append(run(k,42+k));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='All five fixed folds retained; no pruning/picking folds by validation score; evaluate standalone five and platform-three plus all five, equal raw logits'),indent=2)+'\n')


if __name__=='__main__':main()
