"""Frozen six plus the bright-region member, screen only."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/bright_region_complement_screen_v1';EVIDENCE=ROOT/'docs/evidence/bright_region_complement_screen_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={'bright42':OUT/'inputs/bright42.csv','baseline_six':ROOT/'outputs/six_crop_native_v1/valid_predictions.csv'}
plan=dict(aggregation='Frozen six plus bright42, equal seven logits. No combination/weight search.',promotion_delta_auroc=.0003,inputs={k:sha(v) for k,v in paths.items()},script_sha256=sha(Path(__file__)),scope='Saved probability inverse logits; mixed batch/GPU paths; no test rows read')
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
new=values['bright42'];candidate=metric((6*values['baseline_six']+new)/7);base=metric(values['baseline_six']);delta=candidate['auroc']-base['auroc']
r=dict(status='SCREEN_ONLY',images=5849,candidate=candidate,baseline=base,delta_auroc=delta,promising=delta>=.0003,bright_diagnostic=metric(new),plan=plan,native_verified=False,test_used=False)
(EVIDENCE/'screen.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
