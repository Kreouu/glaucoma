"""Conservative border-trim screen; validation follows the same frozen transform."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_trim_v1/plan.json').read_text())
    for n,h in plan['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    for split,h in plan['csv_sha256'].items():assert hashlib.sha256((ROOT/f'datasets/glaucoma_first/{split}.csv').read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/plan['pretrained_path']).read_bytes()).hexdigest()==plan['pretrained_sha256']
    out=ROOT/plan['output_dir'];out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    common=['--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','constant','--pos-weight','3','--ema-decay','.999','--trim-black']
    def train(path,seed,extra):
        print('START',path,flush=True)
        with path.with_suffix('.log').open('x') as f:subprocess.run([sys.executable,'-u','train.py',*common,'--seed',str(seed),'--output-dir',str(path),*extra],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    smoke=out/'smoke';train(smoke,42,['--epochs','1','--train-limit','64','--valid-limit','64'])
    subprocess.run([sys.executable,'validate.py','--run-dir',str(smoke/'ema')],cwd=ROOT,check=True)
    # Explicit new single-image predictor must implement exactly the validation transform.
    sys.path.insert(0,str(ROOT))
    from crop_inference import CropPredictor
    from dataset import FundusDataset
    import torch
    predictor=CropPredictor(smoke/'ema/best.pt');data=FundusDataset(ROOT/'datasets/glaucoma_first','valid',512,64,42,True)
    from fundus_crop import TrimBlackBorder
    data.transform.transforms.insert(0,TrimBlackBorder())
    for i in range(5):assert torch.equal(predictor.preprocess(data.samples[i][1]),data[i][0])
    del predictor;torch.cuda.empty_cache()
    (out/'smoke_check.json').write_text(json.dumps(dict(status='PASS',crop_validation_reload=True,predictor_preprocess_equal_images=5))+'\n')
    results=[]
    for seed in (42,43,44):
        run=out/f'seed{seed}';train(run,seed,['--epochs','5'])
        for kind in ('','ema'):
            path=run/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(path)],cwd=ROOT,check=True)
            check=json.loads((path/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==5849 and check['max_probability_difference']<=1e-6
            assert hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()==check['checkpoint_sha256']
        base=json.loads((ROOT/f'outputs/convnext_512_lr1e4_v1/seed{seed}/ema/reload_check.json').read_text())
        entry=dict(seed=seed,cropped=check,baseline=base,delta_auroc=check['val_auroc']-base['val_auroc']);results.append(entry)
        (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',json.dumps(entry),flush=True)
        if seed==42 and entry['delta_auroc']<.0003:break
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        Path('outputs/convnext_trim_v1.exit').write_text('1\n');raise
    Path('outputs/convnext_trim_v1.exit').write_text('0\n')
