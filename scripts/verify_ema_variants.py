"""Native-logit, full batch-one validation and same-image runtime for four fixed EMA recipes."""
import csv
import gc
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ema_variant_inference import EMAVariantPredictor
from inference import file_sha256,write_prediction


def main():
    experiment=ROOT/'outputs/convnext_512_ema999_v1'
    audit=json.loads((experiment/'comparison.json').read_text());assert audit['status']=='PASS'
    output=ROOT/'outputs/convnext_512_ema999_inference_v1';output.mkdir(parents=True,exist_ok=False)
    weights=json.loads((ROOT/'outputs/convnext_512_inference_candidates_v1/ensemble.json').read_text())
    for entry in weights['members']:
        original=experiment/f"seed{entry['seed']}/ema/best.pt"
        path=output/entry['file'];shutil.copy(original,path)
        assert file_sha256(path)==(original.parent/'best.sha256').read_text().split()[0]
        entry.update(sha256=file_sha256(path),original_sha256=file_sha256(original),bytes=path.stat().st_size,weight_type='ema')
    (output/'weights.json').write_text(json.dumps(weights,indent=2)+'\n')
    recipes={}
    for aggregation in ('probability','logit'):
        for flip in (False,True):
            name=aggregation+('_tta' if flip else '')
            path=output/f'{name}.json';recipe=dict(aggregation=aggregation,horizontal_flip=flip,
                weights_manifest='weights.json',weights_manifest_sha256=file_sha256(output/'weights.json'),
                source_sha256=file_sha256(ROOT/'ema_variant_inference.py'))
            path.write_text(json.dumps(recipe,indent=2)+'\n');recipes[name]=path
    with (ROOT/'outputs/convnext_512_w3_v1/seed43/valid_manifest.csv').open() as stream:manifest=list(csv.DictReader(stream))
    names=[r['Image_name'] for r in manifest];labels=[int(r['True_Label']) for r in manifest]
    assert len(names)==5849 and sum(labels)==191
    predictor=EMAVariantPredictor(recipes['probability'])
    image_root=ROOT/predictor.loader.predictors[0].config['data_root']/'valid'
    values={name:[] for name in recipes}
    with torch.inference_mode():
        for step,row in enumerate(manifest,1):
            image=predictor.loader.predictors[0].preprocess(image_root/row['actual_filename']).unsqueeze(0).cuda()
            original=predictor.logits(image);both=torch.cat([original,predictor.logits(image.flip(-1))],dim=0)
            for name in recipes:
                value=predictor.aggregate(both if name.endswith('_tta') else original,name.split('_')[0]).item()
                assert 0<=value<=1;values[name].append(value)
            if step%1000==0:print(f'EMA native variants {step}/5849',flush=True)
    del predictor;gc.collect();torch.cuda.empty_cache()
    with (ROOT/'outputs/convnext_512_inference_candidates_v1/ensemble_valid_predictions.csv').open() as stream:base=list(csv.DictReader(stream))
    assert [(r['Image_name'],int(r['True_Label'])) for r in base]==list(zip(names,labels))
    base_values=[float(r['Probability']) for r in base]
    baseline=dict(auroc=float(roc_auc_score(labels,base_values)),ap=float(average_precision_score(labels,base_values)))
    sample=json.loads((ROOT/'outputs/three_model_comparison_v1/runtime/sample.json').read_text())
    result=dict(status='PASS',images=5849,positives=191,batch_size=1,baseline=baseline,device=torch.cuda.get_device_name(),
        scope='Native logits, fixed equal weights; validation exploration, not official score gain or T4 verification',candidates={})
    for name,path in recipes.items():
        with (output/f'{name}_valid_predictions.csv').open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(names,labels,values[name]))
        predictor=EMAVariantPredictor(path)
        for row in sample['warmup']:predictor.predict_one(image_root/row['actual_filename'])
        lookup=dict(zip(names,values[name]));timings=[];difference=0.;label_changes=0
        for repeat in range(1,4):
            elapsed=[]
            with (output/f'{name}_round{repeat}.csv').open('w',newline='') as stream:
                w=csv.writer(stream);w.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
                for row in sample['measured']:
                    torch.cuda.synchronize();start=time.perf_counter()
                    probability,label=predictor.predict_one(image_root/row['actual_filename'])
                    write_prediction(w,stream,row['Image_name'],probability,label)
                    torch.cuda.synchronize();elapsed.append((time.perf_counter()-start)*1000)
                    reference=lookup[row['Image_name']];difference=max(difference,abs(probability-reference));label_changes+=int(label!=int(reference>=.5))
            timings.append(elapsed)
        assert difference<=1e-6 and label_changes==0
        auc=float(roc_auc_score(labels,values[name]));ap=float(average_precision_score(labels,values[name]));latency=statistics.mean(v for r in timings for v in r)
        entry=dict(auroc=auc,ap=ap,delta_auroc=auc-baseline['auroc'],delta_ap=ap-baseline['ap'],mean_ms=latency,
            model_bytes=sum(r['bytes'] for r in weights['members']),reload_difference=difference,reload_label_changes=label_changes,
            promising=auc-baseline['auroc']>=.0005 and ap>=baseline['ap'] and latency<100)
        result['candidates'][name]=entry;print(name,json.dumps(entry),flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
