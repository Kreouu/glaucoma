import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'outputs/cam_local_screen_v1/inputs/lr3e-5.csv',ROOT/'outputs/cam_local_screen_v1/inputs/lr1e-4.csv',ROOT/'outputs/three22k_crop_native_v1/valid_predictions.csv',ROOT/'outputs/extra_seeds_native_v1/six_valid_predictions.csv']
values=[];reference=None
for path in paths:
 rows=list(csv.DictReader(path.open()));keys=[(r['Image_name'],int(r['True_Label'])) for r in rows]
 if reference is None:reference=keys
 assert keys==reference
 p=np.array([float(r['Probability']) for r in rows]);assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
 values.append(np.log(p)-np.log1p(-p))
y=np.array([r[1] for r in reference]);assert len(y)==5849 and y.sum()==191
def metric(scores):
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    tp = np.cumsum(y[order])[ends]
    fp = ends + 1 - tp
    t, f = np.r_[0, tp] / sum(y), np.r_[0, fp] / sum(1 - y)
    return dict(auroc=float(np.sum((t[1:] + t[:-1]) * np.diff(f) / 2)),
                ap=float(np.sum(np.diff(np.r_[0, tp]) / sum(y) * tp / (ends + 1))))

result=dict(status='SCREEN_ONLY',test_used=False,method='Each of two localrecipes plus complete global3/6 equal member logits; no subset or weight fitting',limitations='Saved probability inverse logits; dynamic region native inference not verified',global_three=metric(values[2]),global_six=metric(values[3]),candidates=[])
for name,local in zip(('lr3e-5','lr1e-4'),values[:2]):result['candidates'].append(dict(recipe=name,local=metric(local),four=metric((3*values[2]+local)/4),seven=metric((6*values[3]+local)/7)))
result['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
out=ROOT/'docs/evidence/cam_local_screen_v1';out.mkdir(exist_ok=False);(out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
