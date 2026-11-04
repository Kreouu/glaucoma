"""Fixed seeds42..47 native screening; no test use or subset selection."""
import csv,json,gc,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from inference import Predictor,file_sha256
from pack_convnext_fp16 import pack
OUT=ROOT/'outputs/extra_seeds_native_v1'
TRAIN=Path('/home/ng/workspace/glaucoma_22k_crop_extra_seeds_20261104/outputs/convnext22k_crop_extra_seeds_v1')
OLD=Path('/home/ng/workspace/glaucoma_three22k_crop_native_20261102/outputs/three22k_crop_native_v1')

def main():
    OUT.mkdir(exist_ok=False)
    rows=list(csv.DictReader((TRAIN/'w3_seed45/ema/valid_manifest.csv').open()))
    names=[r['Image_name'] for r in rows];y=[int(r['True_Label']) for r in rows]
    assert len(names)==5849 and sum(y)==191
    images=ROOT/'datasets/glaucoma_first/valid';members=[];logits=[]
    for seed in range(42,48):
        if seed<45: source=OLD/f'seed{seed}.pt'
        else:
            source=TRAIN/f'w3_seed{seed}/ema/best.pt'
            check=json.loads((source.parent/'reload_check.json').read_text())
            assert check['status']=='PASS' and check['images']==5849 and check['max_probability_difference']==0
            assert file_sha256(source)==check['checkpoint_sha256']
        packed=OUT/f'seed{seed}.pt'
        if seed<45:
            import shutil
            shutil.copyfile(source,packed)
        else:pack(source,packed)
        predictor=Predictor(packed);original=Predictor(source);values=[];difference=0.
        c=predictor.config
        assert c['seed']==seed and c['train_crop_min_area']==.8 and c['pos_weight']==3 and c['lr']==.0001
        assert c['pretrained_sha256']=='8cb65dc650a7fb25eaadbff3eb69559d02f83cdc5afdfb3785d1889cdc3e3518'
        with torch.inference_mode():
            for row in rows:
                tensor=predictor.preprocess(images/row['actual_filename']).unsqueeze(0).cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    value=predictor.model(tensor).float().flatten();expected=original.model(tensor).float().flatten()
                difference=max(difference,float((value-expected).abs().max()));values.append(value.item())
        assert difference==0
        logits.append(values);members.append(dict(seed=seed,file=packed.name,sha256=file_sha256(packed),bytes=packed.stat().st_size,packing_max_logit_difference=difference))
        np.save(OUT/f'seed{seed}_logits.npy',np.array(values))
        print('MEMBER',seed,flush=True);del predictor,original;gc.collect();torch.cuda.empty_cache()
    values=torch.tensor(np.array(logits),dtype=torch.float32)
    result={}
    for name,part in [('old_three',values[:3]),('new_three',values[3:]),('six',values)]:
        p=part.mean(0).sigmoid().numpy();result[name]=dict(auroc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))
        with (OUT/f'{name}_valid_predictions.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(names,y,p))
    report=dict(status='NATIVE_LOGITS_ONLY',images=5849,members=members,metrics=result,model_bytes=sum(m['bytes'] for m in members),runtime_verified=False,independent_reload_verified=False,test_used=False)
    assert report['model_bytes']<500000000
    (OUT/'native_logits.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
