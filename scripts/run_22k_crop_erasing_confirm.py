"""Training-only small-area random erasing with fixed best learning-rate recipe."""
import hashlib,json,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext22k_crop_erasing_v1/plan.json').read_text())
    previous=json.loads(Path(plan['terminal_proof']).read_text());assert previous['status']=='PASS'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    out=ROOT/'outputs/convnext22k_crop_erasing_v1';assert out.is_dir()

    def run(weight,seed,smoke=False):
        name='smoke' if smoke else f'lr{weight:g}_seed{seed}';path=out/name
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','warmup_cosine','--warmup-epochs','1','--min-lr','0.00001','--train-crop-min-area','.8','--pos-weight','3','--ema-decay','.999','--epochs','2' if smoke else '5','--seed',str(seed),'--output-dir',str(path)]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,GLAUCOMA_ERASE_P=str(weight)))
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==(64 if smoke else 5849) and check['max_probability_difference']<=1e-6 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text())
        assert config['model']=='convnext_tiny' and config['batch_size']==16 and config['size']==512 and config['lr']==0.0001 and config['pos_weight']==3 and config['train_crop_min_area']==.8 and config['pretrained_sha256']==plan['pretrained_sha256']
        assert all(config['source_sha256'][n]==d for n,d in plan['source_sha256'].items() if n in config['source_sha256'])
        metrics=[json.loads(x) for x in (path/'ema/metrics.jsonl').read_text().splitlines()];assert metrics[-1]['ema_updates']>0
        entry=dict(lr=weight,seed=seed,check=check);print('DONE',json.dumps(entry),flush=True);return entry

    original=json.loads((out/'summary.json').read_text())
    assert original['status']=='PASS'
    results=[r for r in original['results'] if r['lr']==.1]
    assert len(results)==1
    for seed in (43,44):
        results.append(run(.1,seed))
        (out/'confirmation_progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'confirmation_summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Complete erasingp0.1 seeds42/43/44 confirmation triggered by fixed global6+seed42 equal7 screen gain; not original singlemodel threshold.'),indent=2)+'\n')


if __name__=='__main__':main()
