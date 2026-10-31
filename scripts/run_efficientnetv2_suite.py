"""After the live crop job, screen a distinct ImageNet-only EfficientNetV2-S family."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    pid=int(sys.argv[1]);proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert b'run_convnext_trim.py' in proc.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
    assert Path('/home/ng/workspace/glaucoma/outputs/convnext_trim_v1.exit').read_text().strip()=='0'
    plan=json.loads((ROOT/'docs/evidence/efficientnetv2_suite_v1/plan.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    for split,h in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==h
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def run(name,lr,seed,smoke=False):
        path=out/name
        cmd=[sys.executable,'-u','train.py','--model','efficientnet_v2_s','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','384','--jpeg-draft','--workers','4','--lr',str(lr),'--lr-schedule','warmup_cosine','--warmup-epochs','1','--min-lr',str(lr/10),'--pos-weight','3','--epochs','8' if not smoke else '2','--seed',str(seed),'--output-dir',str(path)]
        if smoke:cmd+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_suffix('.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
        check=json.loads((path/'reload_check.json').read_text());assert check['status']=='PASS' and check['max_probability_difference']<=1e-6
        if not smoke:assert check['images']==5849 and check['positives']==191
        assert sha(path/'best.pt')==check['checkpoint_sha256']
        return dict(name=name,lr=lr,seed=seed,check=check)
    run('smoke',1e-4,42,True)
    results=[]
    for lr in (1e-4,3e-4):
        results.append(run(f'lr{lr}_seed42',lr,42));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda r:r['check']['val_auroc'])
    if best['check']['val_auroc']>=.989:
        for seed in (43,44):results.append(run(f"lr{best['lr']}_seed{seed}",best['lr'],seed));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='ImageNet-1k only; no EMA for batch-normalized architecture; potential ensemble complement requires later batch1 checks'),indent=2)+'\n')


if __name__=='__main__':main()
