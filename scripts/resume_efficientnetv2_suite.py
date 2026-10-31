"""Continue the verified partial study after a decimal-name log collision; never rerun completed42."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from run_efficientnetv2_suite import experiment_log_path


def main():
    out=ROOT/'outputs/efficientnetv2_suite_v1'
    original_plan=json.loads((out/'plan.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in original_plan['source_sha256'].items():
        source=out/'source_snapshot/run_efficientnetv2_suite.original.py' if name=='scripts/run_efficientnetv2_suite.py' else ROOT/name
        assert sha(source)==digest,name
    assert sha(ROOT/original_plan['pretrained_path'])==original_plan['pretrained_sha256']
    for split,digest in original_plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    results=json.loads((out/'progress.json').read_text());assert len(results)==1 and results[0]['name']=='lr0.0001_seed42'
    first=out/results[0]['name'];check=json.loads((first/'reload_check.json').read_text())
    assert check['status']=='PASS' and check==results[0]['check'] and sha(first/'best.pt')==check['checkpoint_sha256']
    plan=dict(original_plan_sha256=sha(out/'plan.json'),scope='Orchestration-only resume; original first run and raw log preserved',source_sha256={n:sha(ROOT/n) for n in ['scripts/resume_efficientnetv2_suite.py','scripts/run_efficientnetv2_suite.py','train.py','validate.py','model_efficientnetv2.py']})
    (out/'resume_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def run(lr,seed):
        name=f'lr{lr}_seed{seed}';path=out/name;assert not path.exists()
        command=[sys.executable,'-u','train.py','--model','efficientnet_v2_s','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',original_plan['pretrained_path'],'--batch-size','16','--size','384','--jpeg-draft','--workers','4','--lr',str(lr),'--lr-schedule','warmup_cosine','--warmup-epochs','1','--min-lr',str(lr/10),'--pos-weight','3','--epochs','8','--seed',str(seed),'--output-dir',str(path)]
        print('START',name,flush=True)
        with experiment_log_path(path).open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
        check=json.loads((path/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==5849 and check['positives']==191 and check['max_probability_difference']<=1e-6
        assert sha(path/'best.pt')==check['checkpoint_sha256']
        entry=dict(name=name,lr=lr,seed=seed,check=check);results.append(entry)
        (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
    run(3e-4,42)
    best=max(results,key=lambda r:r['check']['val_auroc'])
    if best['check']['val_auroc']>=.989:
        for seed in (43,44):run(best['lr'],seed)
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Resumed only incomplete stages; source provenance preserved'),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        Path('outputs/efficientnetv2_suite_v1.resume.exit').write_text('1\n');raise
    Path('outputs/efficientnetv2_suite_v1.resume.exit').write_text('0\n')
