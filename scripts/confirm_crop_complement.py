"""Confirm seeds only after native complementarity passes; preserve the original training source."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NATIVE=Path('/home/kreouu/Model_train/glaucoma_sixcrop_20261101/outputs/six_crop_native_v1')
FIRST=Path('/home/kreouu/Model_train/glaucoma_traincrop_20261101/outputs/convnext_train_crop_v1/area0p8_seed42')


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_crop_confirmation_v1/plan.json').read_text())
    pid=plan['wait_pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert b'screen_crop_tta.py' in proc.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    report=json.loads((NATIVE/'verification.json').read_text());reload=json.loads((NATIVE/'full_reload.json').read_text())
    assert report['status']==reload['status']=='PASS' and report['full_reload_verified']
    assert report['images']==reload['images']==5849 and reload['max_probability_difference']<=1e-6 and reload['label_changes']==0
    out=ROOT/'outputs/convnext_crop_confirmation_v1';out.mkdir(exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if report['goal995_met'] or report['delta_auroc']<.0003 or report['mean_ms']>=100 or report['model_bytes']>=500000000:
        (out/'summary.json').write_text(json.dumps(dict(status='SKIP',reason='Native complement not eligible, or verified goal already met',native=report),indent=2)+'\n');return
    reference=json.loads((FIRST/'ema/config.json').read_text())
    for name,digest in reference['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/plan['pretrained_path'])==reference['pretrained_sha256']
    for split,digest in reference['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    results=[]
    for seed in (43,44):
        path=out/f'seed{seed}';print('START',seed,flush=True)
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','constant','--pos-weight','3','--ema-decay','.999','--train-crop-min-area','.8','--epochs','5','--seed',str(seed),'--output-dir',str(path)]
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==5849 and check['max_probability_difference']<=1e-6 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text())
        for key in ('model','size','batch_size','epochs','precision','lr','lr_schedule','pos_weight','ema_decay','augmentation','train_crop_min_area','preprocessing','csv_sha256','pretrained_sha256','source_sha256','loss'):assert config[key]==reference[key],key
        results.append(dict(seed=seed,check=check));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(results[-1]),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Complement confirmation; unchanged source and full preset seed set'),indent=2)+'\n')


if __name__=='__main__':main()
