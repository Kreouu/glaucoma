"""Isolated Full JPEG decode fidelity screen, serialized after the active WSL recipe study."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_full_decode_recovery_v1/plan.json').read_text())
    pid=plan['wait_pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert b'run_batch32_study.py' in proc.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\nselect.select([fd],[],[]);os.close(fd)',str(pid)],check=True)
    previous=json.loads(Path(plan['terminal_proof']).read_text());assert previous['status']=='PASS'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
    cpu=json.loads((ROOT/'docs/evidence/convnext_full_decode_recovery_v1/cpu_check.json').read_text());assert cpu['status']=='CPU_PREPROCESS_ONLY' and cpu['results']['False']['mean_ms']<100
    assert sha(ROOT/'inputs/runtime_sample.json')==plan['runtime_sample_sha256']
    out=ROOT/'outputs/convnext_full_decode_recovery_v1';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')

    def run(lr,seed,smoke=False):
        name='smoke' if smoke else f'lr{lr:g}_seed{seed}';path=out/name
        command=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','512','--workers','4','--lr',str(lr),'--lr-schedule','constant','--pos-weight','3','--ema-decay','.999','--epochs','1' if smoke else '5','--seed',str(seed),'--output-dir',str(path)]
        if smoke:command+=['--train-limit','64','--valid-limit','64']
        print('START',name,flush=True)
        with path.with_name(path.name+'.log').open('x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            part=path/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
            check=json.loads((part/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==(64 if smoke else 5849) and check['max_probability_difference']<=1e-6 and sha(part/'best.pt')==check['checkpoint_sha256']
        config=json.loads((path/'ema/config.json').read_text())
        assert config['model']=='convnext_tiny' and config['batch_size']==16 and config['size']==512 and not config['jpeg_draft'] and config['lr']==lr and config['pretrained_sha256']==plan['pretrained_sha256']
        assert all(config['source_sha256'][n]==d for n,d in plan['source_sha256'].items() if n in config['source_sha256'])
        metrics=[json.loads(x) for x in (path/'ema/metrics.jsonl').read_text().splitlines()];assert metrics[-1]['ema_updates']>0
        entry=dict(lr=lr,seed=seed,check=check);print('DONE',json.dumps(entry),flush=True);return entry

    run(plan['learning_rates'][0],42,True)
    # Benchmark the actual no-draft path before committing to a long training job.
    import csv
    sample=json.loads((ROOT/'inputs/runtime_sample.json').read_text());runtime=out/'smoke_runtime';runtime.mkdir()
    assert len(sample['warmup'])==5 and len(sample['measured'])==100
    for split in ('warmup','measured'):
        folder=runtime/split;folder.mkdir()
        for row in sample[split]:(folder/row['actual_filename']).symlink_to(ROOT/'datasets/glaucoma_first/valid'/row['actual_filename'])
    names=runtime/'measured_names.csv'
    with names.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Image_name']);writer.writerows([r['Image_name']] for r in sample['measured'])
    reference=runtime/'reference.csv';checkpoint=out/'smoke/ema/best.pt'
    subprocess.run([sys.executable,'predict.py','--checkpoint',str(checkpoint),'--input-csv',str(names),'--images-dir',str(runtime/'measured'),'--output',str(reference)],check=True,cwd=ROOT)
    subprocess.run([sys.executable,'cost.py',str(reference),str(runtime/'measured'),str(runtime/'warmup'),'--checkpoint',str(checkpoint),'--output-dir',str(runtime/'cost')],check=True,cwd=ROOT)
    cost=json.loads((runtime/'cost/report.json').read_text());assert cost['consistency_pass'] and cost['mean_ms']<100
    print('PRETRAIN RUNTIME PASS',cost['mean_ms'],flush=True)

    results=[]
    for lr in plan['learning_rates']:
        results.append(run(lr,42));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda r:r['check']['val_auroc'])
    if best['check']['val_auroc']>=plan['confirmation_auroc']:
        for seed in (43,44):
            results.append(run(best['lr'],seed));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='Unchanged Tiny512 batch16/LR1e-4/BCE3/EMA.999/5epochs; only JPEG draft disabled. Smoke native runtime gate precedes full train; no test use.'),indent=2)+'\n')


if __name__=='__main__':main()
