"""Same-source 22K optimizer-recipe screen after the complete fixed seven-member check."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIRST=Path('/home/ng/workspace/glaucoma_pretrain22k_20261101/outputs/convnext_pretrain22k_v1/pretrain22k_seed42')
PRIOR=Path('/home/ng/workspace/glaucoma_seven22k_20261101/outputs/seven_model_22k_native_v1')


def main():
    pid=11180;proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert b'run_seven_after_confirmation.py' in proc.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
    prior=json.loads((PRIOR/'verification.json').read_text());reload=json.loads((PRIOR/'full_reload.json').read_text())
    assert prior['status']==reload['status']=='PASS' and prior['full_reload_verified'] and reload['images']==5849 and reload['max_probability_difference']==0
    if prior['goal995_met']:print('SKIP: verified goal already met',flush=True);return
    plan=json.loads((ROOT/'docs/evidence/convnext22k_lr_v1/plan.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    reference=json.loads((FIRST/'ema/config.json').read_text());control=json.loads((FIRST/'ema/reload_check.json').read_text())
    assert control['status']=='PASS' and sha(FIRST/'ema/best.pt')==control['checkpoint_sha256']
    for name,digest in reference['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/plan['pretrained_path'])==reference['pretrained_sha256']
    for split,digest in reference['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    out=ROOT/'outputs/convnext22k_lr_v1';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')

    def run(case,seed,smoke=False):
        name='smoke' if smoke else f"{case['id']}_seed{seed}";path=out/name
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr',str(case['lr']),'--lr-schedule',case['schedule'],'--pos-weight','3','--ema-decay','.999','--epochs','2' if smoke else '5','--seed',str(seed),'--output-dir',str(path)]
        if case['schedule']=='warmup_cosine':command+=['--warmup-epochs','1','--min-lr',str(case['lr']/10)]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==(64 if smoke else 5849) and check['max_probability_difference']<=1e-6 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text())
        for key in ('model','size','batch_size','precision','pos_weight','ema_decay','augmentation','preprocessing','csv_sha256','pretrained_sha256','loss'):assert config[key]==reference[key],key
        for name,digest in reference['source_sha256'].items():assert config['source_sha256'][name]==digest,name
        assert config['lr']==case['lr'] and config['lr_schedule']==case['schedule']
        assert json.loads((path/'ema/metrics.jsonl').read_text().splitlines()[-1])['ema_updates']>0
        entry=dict(case=case,seed=seed,check=check,delta_vs_same_host_original42=check['val_auroc']-control['val_auroc']);print('DONE',json.dumps(entry),flush=True);return entry

    run(plan['cases'][1],42,True)
    results=[]
    for case in plan['cases']:
        results.append(run(case,42));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda r:r['check']['val_auroc'])
    if best['delta_vs_same_host_original42']>=.0003:
        for seed in (43,44):
            results.append(run(best['case'],seed));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Unchanged source and same host; learning rate or schedule changed, no subset fitting'),indent=2)+'\n')


if __name__=='__main__':main()
