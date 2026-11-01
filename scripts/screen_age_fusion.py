import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/evidence/age_fusion_screen_v1';DATA=Path(sys.argv[1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=json.loads((ROOT/'docs/evidence/convnext_pretrain22k_v1/plan.json').read_text())['csv_sha256']
for split,digest in expected.items():assert sha(DATA/f'{split}.csv')==digest
rows={s:list(csv.DictReader((DATA/f'{s}.csv').open(encoding='utf-8-sig'))) for s in ('train','valid')}
assert len(rows['train'])==51396 and len(rows['valid'])==5849
age={s:np.array([float(r['Age']) if r['Age'].strip() else np.nan for r in data]) for s,data in rows.items()}
finite=np.isfinite(age['train']);mean=float(age['train'][finite].mean());std=float(age['train'][finite].std());assert std>0
X={s:np.column_stack([np.ones(len(a)),np.where(np.isfinite(a),(a-mean)/std,0),(~np.isfinite(a)).astype(float)]) for s,a in age.items()}
y={s:np.array([int(r['True_Label']) for r in data]) for s,data in rows.items()};assert y['train'].sum()==1657 and y['valid'].sum()==191
w=np.where(y['train']==1,3.,1.);prior=float(y['train'].mean());weighted_prior=3*prior/(1+2*prior);prior_logit=float(np.log(weighted_prior/(1-weighted_prior)))
beta=np.array([prior_logit,0.,0.]);ridge=np.diag([0.,.001,.001]);x=X['train']
for _ in range(20):
 z=x@beta;prob=1/(1+np.exp(-np.clip(z,-60,60)));grad=x.T@(w*(prob-y['train']))+ridge@beta;hess=x.T@((w*prob*(1-prob))[:,None]*x)+ridge;beta-=np.linalg.solve(hess,grad)
assert np.isfinite(beta).all()
ref=ROOT/'outputs/five_model_22k_native_v1/valid_predictions.csv';pred=list(csv.DictReader(ref.open()))
assert [(r['Image_name'],int(r['True_Label'])) for r in pred]==[(r['Image_name'],int(r['True_Label'])) for r in rows['valid']]
p=np.array([float(r['Probability']) for r in pred]);assert np.all((p>0)&(p<1));image=np.log(p)-np.log1p(-p);residual=X['valid']@beta-prior_logit

def metric(scores):
 labels=y['valid'];order=np.argsort(-scores,kind='stable');ends=np.r_[np.flatnonzero(np.diff(scores[order])),len(scores)-1];tp=np.cumsum(labels[order])[ends];fp=ends+1-tp;t=np.r_[0,tp]/sum(labels);f=np.r_[0,fp]/sum(1-labels)
 return {'auroc':float(np.sum((t[1:]+t[:-1])*np.diff(f)/2)),'ap':float(np.sum(np.diff(np.r_[0,tp])/sum(labels)*tp/(ends+1)))}
base=metric(image);results={}
for name,strength in [('full_prior_residual',1.),('one_expert_relative_to_five',.2)]:
 m=metric(image+strength*residual);m.update(residual_strength=strength,delta_auroc=m['auroc']-base['auroc'],promising=m['auroc']-base['auroc']>=.0003);results[name]=m
r={'status':'SCREEN_ONLY','images':5849,'train_age_mean':mean,'train_age_std':std,'train_missing':int((~finite).sum()),'valid_missing':int((~np.isfinite(age['valid'])).sum()),'coefficients':beta.tolist(),'training_pos_weight':3,'training_prior':prior,'ridge_diagonal':[0,.001,.001],'newton_steps':20,'baseline':base,'results':results,'source_sha256':{'train.csv':sha(DATA/'train.csv'),'valid.csv':sha(DATA/'valid.csv'),'native_five_valid_predictions.csv':sha(ref),'script':sha(Path(__file__))},'scope':'Age model and normalization fitted only on train; two fixed residual strengths; no test rows read; saved probability inverse-logit screen, not native fusion or official score','inference_interface':'Official page supports age auxiliary training; timing metadata delivery not explicitly documented; current formal candidate remains image-only'}
(OUT/'screen.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
