"""Fixed full-set crop complement screen after both replication seeds finish."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
SIX=Path('/home/kreouu/Model_train/glaucoma_sixcrop_20261101/outputs/six_crop_native_v1')
CONFIRM=ROOT/'outputs/convnext_crop_confirmation_v1'


def main():
    plan=json.loads((ROOT/'docs/evidence/crop_confirmation_screen_v1/plan.json').read_text())
    pid=plan['wait_pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert b'confirm_crop_complement.py' in proc.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\nselect.select([fd],[],[]);os.close(fd)',str(pid)],check=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(Path(__file__))==plan['script_sha256']
    proof=json.loads((CONFIRM/'summary.json').read_text());assert proof['status']=='PASS' and [r['seed'] for r in proof['results']]==[43,44]
    for name,digest in plan['baseline_sha256'].items():assert sha(SIX/name)==digest,name
    def read(path):
        with path.open() as f:return list(csv.DictReader(f))
    baseline=read(SIX/'valid_predictions.csv');keys=lambda data:[(r['Image_name'],int(r['True_Label'])) for r in data]
    labels=np.array([int(r['True_Label']) for r in baseline]);assert len(labels)==5849 and labels.sum()==191
    def logits(data):
        assert keys(data)==keys(baseline)
        p=np.array([float(r['Probability']) for r in data]);assert np.all((p>0)&(p<1));return np.log(p)-np.log1p(-p)
    original=logits(baseline);total=6*original;checks=[]
    for seed in (43,44):
        path=CONFIRM/f'seed{seed}'/'ema';check=json.loads((path/'reload_check.json').read_text())
        assert check['status']=='PASS' and check['images']==5849 and check['max_probability_difference']==0 and sha(path/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'config.json').read_text());assert config['seed']==seed and config['train_crop_min_area']==.8 and config['lr']==1e-4 and config['weight_type']=='ema'
        total+=logits(read(path/'best_valid_predictions.csv'));checks.append(check)
    metric=lambda values:dict(auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)))
    base=metric(original);candidate=metric(total/8);delta=candidate['auroc']-base['auroc']
    out=ROOT/'outputs/crop_confirmation_screen_v1';out.mkdir(exist_ok=False)
    report=dict(status='SCREEN_ONLY',images=5849,baseline=base,candidate=candidate,delta_auroc=delta,promising=delta>=.0003,checks=checks,plan=plan,scope='Old five plus all crop42/43/44, equal eight logits. Saved probability inverse-logit screen with mixed batch1/16; native full validation/reload/runtime required before promotion. No seed subset/weight search, no test rows read.')
    (out/'screen.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
