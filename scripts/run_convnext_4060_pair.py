"""Isolated same-host constant versus early cosine study on the 4060 Ti."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_4060_pair_v1/plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for split,digest in plan['csv_sha256'].items():assert hashlib.sha256((ROOT/f'datasets/glaucoma_first/{split}.csv').read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/plan['pretrained_path']).read_bytes()).hexdigest()==plan['pretrained_sha256']
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (out/'environment-freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    common=['--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first',
        '--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft',
        '--workers','4','--lr','0.0001','--pos-weight','3','--ema-decay','.999']
    def train(path,seed,extra):
        print('START',str(path),flush=True)
        with path.with_suffix('.log').open('x') as f:
            subprocess.run([sys.executable,'-u','train.py',*common,'--seed',str(seed),'--output-dir',str(path),*extra],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    smoke=out/'smoke';train(smoke,42,['--epochs','1','--train-limit','64','--valid-limit','64'])
    for kind in ('','ema'):subprocess.run([sys.executable,'validate.py','--run-dir',str(smoke/kind)],cwd=ROOT,check=True)
    (out/'smoke_pass.json').write_text(json.dumps(dict(status='PASS',batch_size=16,size=512,precision='fp16'))+'\n')
    results=[]
    for seed in (42,43,44):
        pair={}
        for schedule in ('constant','late_cosine'):
            run=out/f'{schedule}_seed{seed}'
            train(run,seed,['--epochs','5','--lr-schedule',schedule,'--decay-start-epoch','2','--min-lr','0.00001'])
            for kind in ('','ema'):
                subprocess.run([sys.executable,'validate.py','--run-dir',str(run/kind)],cwd=ROOT,check=True)
                check=json.loads((run/kind/'reload_check.json').read_text())
                assert check['status']=='PASS' and check['images']==5849 and check['positives']==191 and check['max_probability_difference']<=1e-6
                assert hashlib.sha256((run/kind/'best.pt').read_bytes()).hexdigest()==check['checkpoint_sha256']
            pair[schedule]=check
        # Same host and seed, identical first two constant-LR epochs before decay.
        for kind in ('','ema'):
            rows=[]
            for schedule in ('constant','late_cosine'):
                rows.append([json.loads(s) for s in (out/f'{schedule}_seed{seed}'/kind/'metrics.jsonl').read_text().splitlines()])
            assert all(a[k]==b[k] for a,b in zip(rows[0][:2],rows[1][:2]) for k in ('train_loss','valid_loss','val_auroc','val_auprc','amp_skipped_updates'))
        entry=dict(seed=seed,constant=pair['constant'],cosine=pair['late_cosine'],delta_auroc=pair['late_cosine']['val_auroc']-pair['constant']['val_auroc'],prefix_exact_epochs=2)
        results.append(entry);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']<.0003:break
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,confirmed=len(results)==3,test_used=False),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        Path('outputs/convnext_4060_pair_v1.exit').write_text('1\n');raise
    Path('outputs/convnext_4060_pair_v1.exit').write_text('0\n')
