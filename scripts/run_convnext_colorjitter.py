"""Brightness/contrast .1 screen against the completed same-host fixed-LR control."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_colorjitter_v1/plan.json').read_text())
    for n,h in plan['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    for split,h in plan['csv_sha256'].items():assert hashlib.sha256((ROOT/f'datasets/glaucoma_first/{split}.csv').read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/plan['pretrained_path']).read_bytes()).hexdigest()==plan['pretrained_sha256']
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    common=['--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','constant','--pos-weight','3','--ema-decay','.999']
    def train(path,seed,extra):
        print('START',path,flush=True)
        with path.with_suffix('.log').open('x') as f:subprocess.run([sys.executable,'-u','train.py',*common,'--seed',str(seed),'--output-dir',str(path),*extra],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    def read(path):return [json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
    smoke=out/'smoke_control';train(smoke,42,['--epochs','1','--train-limit','64','--valid-limit','64'])
    for kind in ('','ema'):
        old=read(ROOT/'outputs/convnext_4060_pair_v1/smoke'/kind);new=read(smoke/kind)
        assert all(old[0][k]==new[0][k] for k in ('train_loss','valid_loss','val_auroc','val_auprc','amp_skipped_updates'))
    aug=out/'smoke_aug';train(aug,42,['--epochs','1','--train-limit','64','--valid-limit','64','--color-jitter','.1'])
    subprocess.run([sys.executable,'validate.py','--run-dir',str(aug/'ema')],cwd=ROOT,check=True)
    (out/'smoke_check.json').write_text(json.dumps(dict(status='PASS',disabled_augmentation_matches_old=True,augmented_ema_reload=True))+'\n')
    results=[]
    for seed in (42,43,44):
        control=ROOT/f'outputs/convnext_4060_pair_v1/constant_seed{seed}'
        if not control.exists():train(control,seed,['--epochs','5'])
        run=out/f'seed{seed}';train(run,seed,['--epochs','5','--color-jitter','.1'])
        for path in (control/'ema',run,run/'ema'):
            subprocess.run([sys.executable,'validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
            check=json.loads((path/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==5849 and check['max_probability_difference']<=1e-6
            assert hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()==check['checkpoint_sha256']
        base=json.loads((control/'ema/reload_check.json').read_text())
        entry=dict(seed=seed,augmented=check,control=base,delta_auroc=check['val_auroc']-base['val_auroc'])
        results.append(entry);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']<.0003:break
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,confirmed=len(results)==3,test_used=False),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        Path('outputs/convnext_colorjitter_v1.exit').write_text('1\n');raise
    Path('outputs/convnext_colorjitter_v1.exit').write_text('0\n')
