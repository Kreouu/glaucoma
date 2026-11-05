"""Independent six-member full reload and local end-to-end timing; no test use."""
import csv,json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from inference import Predictor,file_sha256,write_prediction,resolve_images
from sklearn.metrics import roc_auc_score,average_precision_score
OUT=ROOT/'outputs/erasing_native_recovery_v1'

import importlib.util
spec=importlib.util.spec_from_file_location('erasing_frozen_inference',Path('/home/kreouu/Model_train/glaucoma_22k_crop_erasing_20261105/inference.py'))
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)

def main():
    evidence=json.loads((OUT/'native_logits.json').read_text())
    models=[]
    for m in evidence['members']:
        p=OUT/m['file'];assert file_sha256(p)==m['sha256'];models.append((Predictor if m['seed']<45 else frozen.Predictor)(p))
    assert len(models)==6
    @torch.inference_mode()
    def predict(path):
        tensor=models[0].preprocess(path).unsqueeze(0).cuda()
        with torch.autocast('cuda',dtype=models[0].autocast_dtype):
            logits=torch.stack([p.model(tensor).flatten() for p in models]).float()
        value=logits.mean(0).sigmoid().item();return value,int(value>=.5)
    rows=list(csv.DictReader((OUT/'six_valid_predictions.csv').open()))
    images=ROOT/'datasets/glaucoma_first/valid';values=[];difference=0.;changes=0
    mapping=dict(zip([r['Image_name'] for r in rows],resolve_images([r['Image_name'] for r in rows],images)))
    for i,r in enumerate(rows):
        value,label=predict(mapping[r['Image_name']]);values.append(value)
        difference=max(difference,abs(value-float(r['Probability'])));changes+=label!=int(float(r['Probability'])>=.5)
        if (i+1)%1000==0:print('RELOAD',i+1,flush=True)
    assert len(values)==5849 and difference<=1e-6 and changes==0
    sample=json.loads((ROOT/'inputs/runtime_sample.json').read_text())
    assert len(sample['warmup'])==5 and len(sample['measured'])==100
    for r in sample['warmup']:predict(mapping[r['Image_name']])
    timings=[]
    for repeat in range(1,4):
        with (OUT/f'round{repeat}.csv').open('x') as f:
            w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
            for r in sample['measured']:
                torch.cuda.synchronize();start=time.perf_counter();value,label=predict(mapping[r['Image_name']]);write_prediction(w,f,r['Image_name'],value,label);torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
    y=[int(r['True_Label']) for r in rows]
    report=dict(status='PASS',images=5849,auroc=float(roc_auc_score(y,values)),ap=float(average_precision_score(y,values)),max_probability_difference=difference,label_changes=int(changes),mean_ms=statistics.mean(timings),model_bytes=evidence['model_bytes'],official_T4_verified=False,test_used=False)
    (OUT/'full_reload_runtime.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
