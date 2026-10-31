"""Controlled ConvNeXt follow-up studies, isolated from existing live work."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    study=sys.argv[1];assert study in ('small','drop_path','mixup')
    plan=json.loads((ROOT/f'docs/evidence/convnext_{study}_v1/plan.json').read_text())
    pid=plan['wait_pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert plan['wait_command'].encode() in proc.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
    if study=='small':assert Path(plan['terminal_proof']).read_text().strip()=='0'
    elif study=='mixup':
        proof=json.loads(Path(plan['terminal_proof']).read_text());assert proof['status']=='PASS'
        if proof['goal995_met']:
            print('SKIP: prior train-only index candidate meets goal',flush=True);return
    else:
        proof=json.loads(Path(plan['terminal_proof']).read_text());assert proof['status']=='PASS' and proof['max_probability_difference']==0
        prior=json.loads(Path(plan['terminal_proof']).with_name('verification.json').read_text())
        if prior['auroc']>=.995 and prior['mean_ms']<100 and prior['model_bytes']<500000000:
            print('SKIP: verified prior candidate already meets goal',flush=True);return
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    assert json.loads((ROOT/plan['cpu_proof']).read_text())['status']=='PASS'
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def train(case,seed,path,smoke=False,force_zero=False):
        rate=0 if force_zero else case.get('drop_path',0)
        command=[sys.executable,'-u','train.py','--model',plan['model'],'--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','constant','--pos-weight','3','--ema-decay','.999','--epochs','1' if smoke else '5','--seed',str(seed),'--output-dir',str(path),'--drop-path',str(rate)]
        if study=='mixup':command+=['--mixup-alpha',str(0 if force_zero else case['mixup_alpha'])]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',path,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            report=json.loads((part/'reload_check.json').read_text());assert report['status']=='PASS' and report['images']==(64 if smoke else 5849) and report['max_probability_difference']<=1e-6
            assert sha(part/'best.pt')==report['checkpoint_sha256']
        last=json.loads((path/'ema/metrics.jsonl').read_text().splitlines()[-1]);assert last['ema_updates']>0
        return report
    cases=plan['cases'];smoke=out/'smoke';train(cases[0],42,smoke,True)
    if study in ('drop_path','mixup'):
        default=out/'smoke_default';train(cases[0],42,default,True,True)
        control=Path(plan['smoke_default_control'])
        for kind in ('','ema'):
            old=json.loads((control/kind/'metrics.jsonl').read_text().splitlines()[0]);new=json.loads((default/kind/'metrics.jsonl').read_text().splitlines()[0])
            assert all(old[k]==new[k] for k in ('train_loss','valid_loss','val_auroc','val_auprc','amp_skipped_updates'))
    (out/'smoke_check.json').write_text(json.dumps(dict(status='PASS',full_batch16=True,default_control_exact=study in ('drop_path','mixup')))+'\n')
    jobs=[(case,42) for case in cases];results=[]
    for case,seed in jobs:
        path=out/f"{case['id']}_seed{seed}";report=train(case,seed,path)
        control=Path(plan['control_template'].format(seed=seed));base=json.loads((control/'reload_check.json').read_text())
        assert base['status']=='PASS' and sha(control/'best.pt')==base['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text());base_config=json.loads((control/'config.json').read_text())
        for k in ('size','batch_size','epochs','lr','lr_schedule','pos_weight','ema_decay','precision','preprocessing','csv_sha256'):assert config[k]==base_config[k],k
        expected_augmentation=base_config['augmentation']
        if study=='mixup':
            assert config['mixup_alpha']==case['mixup_alpha']
            expected_augmentation+=f"; training batch Mixup alpha={case['mixup_alpha']}, soft BCE targets"
        assert config['augmentation']==expected_augmentation
        assert config['model']==plan['model'] and config['drop_path']==case.get('drop_path',0)
        entry=dict(case=case,seed=seed,check=report,control=base,delta_auroc=report['val_auroc']-base['val_auroc']);results.append(entry)
        (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']>=.0003:jobs.extend([(case,43),(case,44)])
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False),indent=2)+'\n')


if __name__=='__main__':main()
