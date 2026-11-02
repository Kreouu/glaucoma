"""Fixed hflip logit averaging screen of the complete crop22K seed set."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from three_22k_crop_inference import Three22KCropPredictor
from inference import file_sha256
BASE=ROOT/'outputs/three22k_crop_native_v1';OUT=ROOT/'outputs/three22k_crop_hflip_screen_v1'
OUT.mkdir(exist_ok=False)
plan=dict(base_manifest_sha256=file_sha256(BASE/'manifest.json'),script_sha256=file_sha256(Path(__file__)),method='All3 seeds, equal original and horizontal-flipped logits; no subset or weight search',promotion_delta=.0003,test_used=False)
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with (BASE/'valid_predictions.csv').open() as f:reference=list(csv.DictReader(f))
with (Path('/home/ng/workspace/glaucoma_22k_cosine_crop_20261102/outputs/convnext22k_cosine_crop_v1/seed42/valid_manifest.csv')).open() as f:manifest=list(csv.DictReader(f))
assert [(r['Image_name'],r['True_Label']) for r in reference]==[(r['Image_name'],r['True_Label']) for r in manifest]
y=[int(r['True_Label']) for r in reference];assert len(y)==5849 and sum(y)==191
predictor=Three22KCropPredictor(BASE/'manifest.json');values=[];base_difference=0.
with torch.inference_mode():
 for i,(row,item) in enumerate(zip(reference,manifest),1):
  x=predictor.predictors[0].preprocess(ROOT/'datasets/glaucoma_first/valid'/item['actual_filename']).unsqueeze(0).cuda()
  original=predictor.logits(x);flipped=predictor.logits(x.flip(-1));p=original.mean(0).sigmoid().item();base_difference=max(base_difference,abs(p-float(row['Probability'])));assert base_difference==0
  values.append(((original+flipped)/2).mean(0).sigmoid().item())
  if i%1000==0:print('HFLIP',i,flush=True)
with (OUT/'valid_predictions.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['Image_name','True_Label','Probability']);w.writerows((r['Image_name'],r['True_Label'],p) for r,p in zip(reference,values))
base=roc_auc_score(y,[float(r['Probability']) for r in reference]);auc=roc_auc_score(y,values)
report=dict(status='NATIVE_SCREEN_ONLY',auroc=auc,ap=average_precision_score(y,values),baseline_auroc=base,delta_auroc=auc-base,promising=auc-base>=plan['promotion_delta'],base_probability_difference=base_difference,independent_reload_verified=False,runtime_verified=False,test_used=False)
(OUT/'screen.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
