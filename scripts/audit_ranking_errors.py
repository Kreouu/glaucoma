import csv,json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
paths=[root/'outputs/extra_seeds_native_v1/six_valid_predictions.csv',root/'outputs/erasing_native_recovery_v1/six_valid_predictions.csv']
rows=[list(csv.DictReader(p.open())) for p in paths]
keys=[(r['Image_name'],int(r['True_Label'])) for r in rows[0]]
assert [(r['Image_name'],int(r['True_Label'])) for r in rows[1]]==keys
scores=[np.array([float(r['Probability']) for r in rs]) for rs in rows];y=np.array([k[1] for k in keys]);assert len(y)==5849 and y.sum()==191
pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
def auc(s,p,n):
 a=np.sort(s[n]);b=s[p]
 return float(np.mean((np.searchsorted(a,b,side='left')+np.searchsorted(a,b,side='right'))/2)/len(n))
rng=np.random.default_rng(20261105);diff=[]
for _ in range(2000):
 p=rng.choice(pos,len(pos));n=rng.choice(neg,len(neg));diff.append(auc(scores[1],p,n)-auc(scores[0],p,n))
negative=np.sort(scores[0][neg]);counts=len(negative)-(np.searchsorted(negative,scores[0][pos],side='left')+np.searchsorted(negative,scores[0][pos],side='right'))/2
order=np.argsort(-counts);hard=[dict(name=keys[pos[i]][0],global_probability=float(scores[0][pos[i]]),erasing_probability=float(scores[1][pos[i]]),ranking_error_pairs=float(counts[i])) for i in order[:20]]
report=dict(status='DIAGNOSTIC_ONLY',test_used=False,images=5849,positives=191,method='Paired stratified bootstrap 2000, fixed seed20261105; neither labels nor predictions modified; CI does not correct repeated model selection',global_auroc=auc(scores[0],pos,neg),erasing_auroc=auc(scores[1],pos,neg),difference=float(auc(scores[1],pos,neg)-auc(scores[0],pos,neg)),paired_bootstrap_difference_ci95=[float(v) for v in np.quantile(diff,[.025,.975])],bootstrap_fraction_positive=float(np.mean(np.array(diff)>0)),ranking_error_pairs=float(counts.sum()),worst5_positive_share=float(counts[order[:5]].sum()/counts.sum()),worst20_positive_share=float(counts[order[:20]].sum()/counts.sum()),hardest_positive_images=hard,sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
out=root/'docs/evidence/ranking_error_audit_v1';out.mkdir(exist_ok=False);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('hardest_positive_images','sources')},indent=2));print(hard[:5])
