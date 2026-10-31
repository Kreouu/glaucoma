"""Same-host positive-weight1/5 screen against the frozen weight3 LR1e-4 control."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_bce_4060_v1/plan.json').read_text())
    for n,h in plan['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    for split,h in plan['csv_sha256'].items():assert hashlib.sha256((ROOT/f'datasets/glaucoma_first/{split}.csv').read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/plan['pretrained_path']).read_bytes()).hexdigest()==plan['pretrained_sha256']
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    # Screen both weights before confirming either, without choosing mixed per-seed weights.
    jobs=[(1,42),(5,42)]
    for weight,seed in jobs:
        run=out/f'w{weight}_seed{seed}';print('START',run,flush=True)
        cmd=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','constant','--pos-weight',str(weight),'--ema-decay','.999','--epochs','5','--seed',str(seed),'--output-dir',str(run)]
        control=ROOT/f'outputs/convnext_4060_pair_v1/constant_seed{seed}'
        if not control.exists():
            control_cmd=cmd.copy();control_cmd[control_cmd.index('--pos-weight')+1]='3';control_cmd[-1]=str(control)
            print('START CONTROL',control,flush=True)
            with control.with_suffix('.log').open('x') as log:subprocess.run(control_cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'validate.py','--run-dir',str(control/'ema')],cwd=ROOT,check=True)
        with run.with_suffix('.log').open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            path=run/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
            check=json.loads((path/'reload_check.json').read_text())
            assert check['status']=='PASS' and check['images']==5849 and check['positives']==191 and check['max_probability_difference']<=1e-6
            assert hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()==check['checkpoint_sha256']
            config=json.loads((path/'config.json').read_text())
            base_config=json.loads((ROOT/f'outputs/convnext_4060_pair_v1/constant_seed{seed}/ema/config.json').read_text())
            for k in ('model','size','batch_size','epochs','precision','lr','lr_schedule','ema_decay','augmentation','preprocessing','csv_sha256','pretrained_sha256'):assert config[k]==base_config[k],k
            assert config['pos_weight']==weight and config.get('color_jitter',0)==0
        base=json.loads((ROOT/f'outputs/convnext_4060_pair_v1/constant_seed{seed}/ema/reload_check.json').read_text())
        entry=dict(weight=weight,seed=seed,ema=check,control=base,delta_auroc=check['val_auroc']-base['val_auroc'])
        results.append(entry);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']>=.0003:jobs.extend([(weight,43),(weight,44)])
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        Path('outputs/convnext_bce_4060_v1.exit').write_text('1\n');raise
    Path('outputs/convnext_bce_4060_v1.exit').write_text('0\n')
