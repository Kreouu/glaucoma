"""Export exact weights, verify full batch-one validation and time four fixed inference recipes."""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ensemble_inference import ConvNeXtProbabilityPredictor
from inference import file_sha256, write_prediction


def main():
    output=ROOT/'outputs/convnext_512_inference_candidates_v1'
    output.mkdir(parents=True,exist_ok=False)
    sources={n:file_sha256(ROOT/n) for n in ('ensemble_inference.py','inference.py','dataset.py',
                                            'model.py','model_convnext.py','amp_precision.py')}
    members=[]
    for seed in (42,43,44):
        original=ROOT/'outputs/convnext_512_w3_v1'/f'seed{seed}/best.pt'
        assert file_sha256(original)==(original.parent/'best.sha256').read_text().split()[0]
        saved=torch.load(original,map_location='cpu',weights_only=True)
        compact={k:saved[k] for k in ('model','config','metrics')}
        path=output/f'seed{seed}.pt';torch.save(compact,path)
        restored=torch.load(path,map_location='cpu',weights_only=True)
        assert restored['config']==saved['config'] and restored['metrics']==saved['metrics']
        assert all(torch.equal(t,restored['model'][k]) for k,t in saved['model'].items())
        members.append(dict(seed=seed,file=path.name,sha256=file_sha256(path),bytes=path.stat().st_size,
                            original_sha256=file_sha256(original),tensor_equality='PASS'))
        del saved,compact,restored
    manifests={}
    for name,selected,flip in [('baseline',[members[1]],False),('single_tta',[members[1]],True),
                              ('ensemble',members,False),('ensemble_tta',members,True)]:
        manifest=dict(method='equal_probability_mean',horizontal_flip=flip,members=selected,source_sha256=sources,
                      precision='FP16 autocast, FP32 sigmoid and mean',test_inference=False)
        path=output/f'{name}.json';path.write_text(json.dumps(manifest,indent=2)+'\n');manifests[name]=path
    predictor=ConvNeXtProbabilityPredictor(manifests['ensemble'])
    with (ROOT/'outputs/convnext_512_w3_v1/seed43/valid_manifest.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    labels=[int(r['True_Label']) for r in rows]
    assert len(rows)==5849 and sum(labels)==191
    names=[r['Image_name'] for r in rows]
    image_root=ROOT/predictor.predictors[0].config['data_root']/'valid'
    paths=[image_root/r['actual_filename'] for r in rows]
    assert all(p.is_file() for p in paths)
    results={name:[] for name in manifests}
    with torch.inference_mode():
        for step,path in enumerate(paths,1):
            image=predictor.predictors[0].preprocess(path).unsqueeze(0).cuda()
            normal=predictor.probabilities(image)
            flipped=predictor.probabilities(image.flip(-1))
            values=dict(baseline=normal[1].item(),single_tta=((normal[1]+flipped[1])/2).item(),
                        ensemble=normal.mean().item(),ensemble_tta=((normal+flipped)/2).mean().item())
            for name,value in values.items():results[name].append(value)
            if step%500==0:print(f'Full batch-one candidate validation {step}/{len(paths)}',flush=True)
    del predictor
    torch.cuda.empty_cache()
    summary=dict(status='PASS',images=len(rows),positives=sum(labels),batch_size=1,
                 device=torch.cuda.get_device_name(),scope='Validation only; local GPU timing is not official T4 verification',
                 candidates={})
    sample=json.loads((ROOT/'outputs/three_model_comparison_v1/runtime/sample.json').read_text())
    with (ROOT/'outputs/convnext_512_w3_v1/seed43/best_valid_predictions.csv').open() as stream:
        old=list(csv.DictReader(stream))
    assert names==[r['Image_name'] for r in old]
    expected_single=[float(r['Probability']) for r in old]
    old_ensemble=[]
    with (ROOT/'outputs/convnext_512_equal_ensemble_v1/valid_predictions.csv').open() as stream:
        old_ensemble=list(csv.DictReader(stream))
    assert names==[r['Image_name'] for r in old_ensemble]
    for name,path in manifests.items():
        assert np.isfinite(results[name]).all()
        with (output/f'{name}_valid_predictions.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['Image_name','True_Label','Probability'])
            writer.writerows(zip(names,labels,results[name]))
        predictor=ConvNeXtProbabilityPredictor(path)
        # New instance verifies exported model loading, not just the already-loaded validation instance.
        for row in sample['warmup']:predictor.predict_one(image_root/row['actual_filename'])
        timed=[];max_error=0.;label_changes=0
        lookup=dict(zip(names,results[name]))
        for repeat in range(1,4):
            durations=[]
            with (output/f'{name}_runtime_round{repeat}.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
                for row in sample['measured']:
                    torch.cuda.synchronize();started=time.perf_counter()
                    probability,label=predictor.predict_one(image_root/row['actual_filename'])
                    write_prediction(writer,stream,row['Image_name'],probability,label)
                    torch.cuda.synchronize();durations.append((time.perf_counter()-started)*1000)
                    expected=lookup[row['Image_name']]
                    max_error=max(max_error,abs(probability-expected))
                    label_changes+=int(label!=int(expected>=.5))
            timed.append(durations)
        assert max_error<=1e-6 and label_changes==0,(name,max_error,label_changes)
        entry=dict(auroc=float(roc_auc_score(labels,results[name])),ap=float(average_precision_score(labels,results[name])),
                   manifest_sha256=file_sha256(path),model_bytes=sum(r['bytes'] for r in predictor.manifest['members']),
                   mean_ms=statistics.mean(v for a in timed for v in a),round_mean_ms=[statistics.mean(a) for a in timed],
                   reload_max_difference=max_error,reload_label_changes=label_changes)
        if name=='baseline':entry['max_difference_vs_batch16']=float(np.max(np.abs(np.array(results[name])-expected_single)))
        if name=='ensemble':entry['max_difference_vs_batch16']=float(np.max(np.abs(np.array(results[name])-[float(r['Probability']) for r in old_ensemble])))
        summary['candidates'][name]=entry
        del predictor
        torch.cuda.empty_cache()
        print(name,json.dumps(entry),flush=True)
    for name,entry in summary['candidates'].items():
        entry['delta_auroc_vs_baseline']=entry['auroc']-summary['candidates']['baseline']['auroc']
        entry['delta_ap_vs_baseline']=entry['ap']-summary['candidates']['baseline']['ap']
    (output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
