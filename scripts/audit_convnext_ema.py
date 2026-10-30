"""Check paired online/EMA training, successful update counts, reloads and equal-ensemble validation."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_three_models import sha


def inspect_run(run,kind,seed):
    config=json.loads((run/'config.json').read_text())
    for k,v in dict(model='convnext_tiny',size=512,epochs=5,batch_size=16,workers=4,lr=3e-5,pos_weight=3,
                    seed=seed,precision='fp16',lr_schedule='constant',ema_decay=.999,weight_type=kind,
                    train_count=51396,valid_count=5849,train_limit=None,valid_limit=None).items():assert config[k]==v,(run,k)
    for n,h in config['source_sha256'].items():assert sha(ROOT/n)==h,n
    for split,h in config['csv_sha256'].items():assert sha(ROOT/config['data_root']/f'{split}.csv')==h
    assert sha(ROOT/config['pretrained_path'])==config['pretrained_sha256']
    metrics=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in metrics]==[1,2,3,4,5]
    assert all(np.isfinite(r[k]) for r in metrics for k in ('train_loss','valid_loss','val_auroc','val_auprc'))
    best=max(metrics,key=lambda r:r['val_auroc']);digest=sha(run/'best.pt')
    assert digest==(run/'best.sha256').read_text().split()[0]
    saved=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    assert saved['config']==config and saved['metrics']==best and all(torch.isfinite(t).all() for t in saved['model'].values())
    reload=json.loads((run/'reload_check.json').read_text())
    assert reload['status']=='PASS' and reload['images']==5849 and reload['positives']==191
    assert reload['checkpoint_sha256']==digest and reload['max_probability_difference']<=1e-6
    with (run/'best_valid_predictions.csv').open() as stream:rows=list(csv.DictReader(stream))
    names=[r['Image_name'] for r in rows];labels=[int(r['True_Label']) for r in rows];values=np.array([float(r['Probability']) for r in rows])
    assert len(names)==len(set(names))==5849 and sum(labels)==191 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    with (run/'valid_manifest.csv').open() as stream:manifest=list(csv.DictReader(stream))
    assert [(r['Image_name'],int(r['True_Label'])) for r in manifest]==list(zip(names,labels))
    assert roc_auc_score(labels,values)==best['val_auroc']==reload['val_auroc']
    assert average_precision_score(labels,values)==best['val_auprc']==reload['val_auprc']
    return saved,dict(seed=seed,best_epoch=best['epoch'],auroc=best['val_auroc'],ap=best['val_auprc'],checkpoint_sha256=digest,metrics=metrics,reload='PASS'),names,labels,values


def main():
    root=ROOT/'outputs/convnext_512_ema999_v1';rows=[];old_values=[];ema_values=[];reference_names=None
    for seed in (42,43,44):
        run=root/f'seed{seed}';online,o,names,labels,values=inspect_run(run,'online',seed)
        ema,e,ema_names,ema_labels,averaged=inspect_run(run/'ema','ema',seed)
        assert names==ema_names and labels==ema_labels
        if reference_names is None:reference_names=names;reference_labels=labels
        assert names==reference_names and labels==reference_labels
        baseline=ROOT/'outputs/convnext_512_w3_v1'/f'seed{seed}'
        assert sha(baseline/'best.pt')==(baseline/'best.sha256').read_text().split()[0]
        original=torch.load(baseline/'best.pt',map_location='cpu',weights_only=True)
        original_metrics=[json.loads(l) for l in (baseline/'metrics.jsonl').read_text().splitlines()]
        metric_equal=all(a[k]==b[k] for a,b in zip(o['metrics'],original_metrics) for k in ('epoch','train_loss','valid_loss','val_auroc','val_auprc'))
        parameters_equal=online['model'].keys()==original['model'].keys() and all(torch.equal(t,original['model'][k]) for k,t in online['model'].items())
        predictions_equal=(run/'best_valid_predictions.csv').read_bytes()==(baseline/'best_valid_predictions.csv').read_bytes()
        with (baseline/'best_valid_predictions.csv').open() as stream:old=list(csv.DictReader(stream))
        assert [(r['Image_name'],r['True_Label']) for r in old]==[(n,str(l)) for n,l in zip(names,labels)]
        old_values.append([float(r['Probability']) for r in old]);ema_values.append(averaged)
        with (run/'learning_rate.csv').open() as stream:steps=list(csv.DictReader(stream))
        assert len(steps)==16065 and all(float(r['lr'])==3e-5 for r in steps)
        successful=[1-int(r['amp_skipped']) for r in steps]
        for metric in e['metrics']:assert metric['ema_updates']==sum(successful[:metric['epoch']*3213])
        assert sum(1-x for x in successful)==sum(r['amp_skipped_updates'] for r in o['metrics'])
        for name in ('train_manifest.csv','valid_manifest.csv'):assert (run/name).read_bytes()==(baseline/name).read_bytes()==(run/'ema'/name).read_bytes()
        rows.append(dict(seed=seed,online=o,ema=e,online_control_exact=metric_equal and parameters_equal and predictions_equal,
                         online_metrics_exact=metric_equal,online_parameters_exact=parameters_equal,online_predictions_exact=predictions_equal,
                         paired_auroc_delta=e['auroc']-o['auroc'],paired_ap_delta=e['ap']-o['ap'],successful_steps=sum(successful),amp_skipped=16065-sum(successful)))
        del original,online,ema
        print(seed,rows[-1]['online_control_exact'],e['auroc'],e['ap'],flush=True)
    base=np.mean(old_values,axis=0);candidate=np.mean(ema_values,axis=0)
    base_auc=float(roc_auc_score(reference_labels,base));auc=float(roc_auc_score(reference_labels,candidate))
    base_ap=float(average_precision_score(reference_labels,base));ap=float(average_precision_score(reference_labels,candidate))
    result=dict(status='PASS' if all(r['online_control_exact'] for r in rows) else 'CONTROL_MISMATCH',seeds=rows,
                mean_paired_auroc_delta=statistics.mean(r['paired_auroc_delta'] for r in rows),
                baseline_ensemble=dict(auroc=base_auc,ap=base_ap),ema_ensemble=dict(auroc=auc,ap=ap),
                ensemble_auroc_delta=auc-base_auc,ensemble_ap_delta=ap-base_ap,
                promising=auc-base_auc>=.0005 and ap>=base_ap,
                scope='Fixed decay .999, same five-epoch recipe; validation only, not a claim of official score improvement')
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'ensemble_valid_predictions.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(reference_names,reference_labels,candidate.tolist()))
    print(json.dumps({k:v for k,v in result.items() if k!='seeds'},indent=2),flush=True)
    assert result['status']=='PASS','Online control differs; inspect before attributing differences to EMA'


if __name__=='__main__':main()
