"""Fixed old four + crop42 + selected fixed-LR Base42 member, screen only."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/base_complement_screen_v1';EVIDENCE=ROOT/'docs/evidence/base_complement_screen_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={'old_four':ROOT/'outputs/four_model_native_v1/valid_predictions.csv','base42':OUT/'inputs/base42.csv','crop42':OUT/'inputs/crop42.csv','baseline_six':ROOT/'outputs/six_crop_native_v1/valid_predictions.csv'}
plan=dict(aggregation='Old four mean weighted4 plus crop42 plus best fixedLR Base42, divided6. Replace old22k42 with Base; no combination/weight search.',promotion_delta_auroc=.0003,inputs={k:sha(v) for k,v in paths.items()},script_sha256=sha(Path(__file__)),scope='Saved probability inverse logits; mixed batch/GPU paths; no test rows read')
(EVIDENCE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
values={};reference=None
for name,path in paths.items():
 with path.open() as f:rows=list(csv.DictReader(f))
 keys=[(r['Image_name'],int(r['True_Label'])) for r in rows]
 if reference is None:reference=keys
 assert keys==reference
 p=np.array([float(r['Probability']) for r in rows]);assert np.all((p>0)&(p<1));values[name]=np.log(p)-np.log1p(-p)
y=np.array([r[1] for r in reference]);assert len(y)==5849 and y.sum()==191

def metric(scores):
 order=np.argsort(-scores,kind='stable');ends=np.r_[np.flatnonzero(np.diff(scores[order])),len(scores)-1];tp=np.cumsum(y[order])[ends];fp=ends+1-tp;t=np.r_[0,tp]/sum(y);f=np.r_[0,fp]/sum(1-y)
 return dict(auroc=float(np.sum((t[1:]+t[:-1])*np.diff(f)/2)),ap=float(np.sum(np.diff(np.r_[0,tp])/sum(y)*tp/(ends+1))))
new=values['base42'];candidate=metric((4*values['old_four']+values['crop42']+new)/6);base=metric(values['baseline_six']);delta=candidate['auroc']-base['auroc']
r=dict(status='SCREEN_ONLY',images=5849,candidate=candidate,baseline=base,delta_auroc=delta,promising=delta>=.0003,base_diagnostic=metric(new),plan=plan,native_verified=False,test_used=False)
(EVIDENCE/'screen.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
