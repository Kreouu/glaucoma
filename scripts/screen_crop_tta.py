"""One fixed center-view screen; no test data or scale/weight search."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from inference import Predictor,file_sha256
from crop_tta import preprocess_center


def main():
    base=ROOT/'outputs/six_crop_native_v1'
    train=Path('/home/kreouu/Model_train/glaucoma_traincrop_20261101/outputs/convnext_train_crop_v1/area0p8_seed42')
    out=ROOT/'outputs/crop_center_tta_screen_v1';out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'docs/evidence/crop_center_tta_screen_v1/plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert file_sha256(ROOT/name)==digest
    for name,digest in plan['input_sha256'].items():assert file_sha256(base/name)==digest
    assert file_sha256(ROOT/'datasets/glaucoma_first/valid.csv')==plan['valid_csv_sha256']
    def rows(path):
        with path.open() as f:return list(csv.DictReader(f))
    reference=rows(base/'valid_predictions.csv');original=rows(train/'ema/best_valid_predictions.csv');samples=rows(train/'valid_manifest.csv')
    keys=lambda data:[(r['Image_name'],int(r['True_Label'])) for r in data]
    assert keys(reference)==keys(original)==keys(samples)
    labels=np.array([int(r['True_Label']) for r in reference]);assert len(labels)==5849 and labels.sum()==191
    def logits(data):
        p=np.array([float(r['Probability']) for r in data]);assert np.all((p>0)&(p<1));return np.log(p)-np.log1p(-p)
    baseline=logits(reference);full=logits(original)
    predictor=Predictor(base/'traincrop42.pt');center=[]
    with torch.inference_mode():
        for i,row in enumerate(samples,1):
            image=preprocess_center(ROOT/'datasets/glaucoma_first/valid'/row['actual_filename'],predictor).unsqueeze(0).cuda()
            with torch.autocast('cuda',dtype=predictor.autocast_dtype):value=predictor.model(image).flatten()[0]
            center.append(value.float().item())
            if i%1000==0:print('CENTER',i,flush=True)
    center=np.array(center);assert np.isfinite(center).all()
    candidate=baseline+(center-full)/12
    metric=lambda values:dict(auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)))
    b=metric(baseline);c=metric(candidate);delta=c['auroc']-b['auroc']
    np.savez(out/'screen_logits.npz',baseline=baseline,full=full,center=center,labels=labels)
    result=dict(status='SCREEN_ONLY',images=5849,baseline=b,candidate=c,delta_auroc=delta,promising=delta>=.0003,scope='Saved probabilities inverted to logits; full crop branch uses training validation batch16, center branch batch1. Native full ensemble, reload and timing required before promotion; no test rows read.',plan=plan)
    (out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
