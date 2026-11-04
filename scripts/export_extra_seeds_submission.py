"""Independent six-member full reload and local end-to-end timing; no test use."""
import csv,json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from inference import Predictor,file_sha256,write_prediction,resolve_images
from sklearn.metrics import roc_auc_score,average_precision_score
OUT=ROOT/'outputs/extra_seeds_native_v1'

def main():
    evidence=json.loads((OUT/'native_logits.json').read_text())
    models=[]
    for m in evidence['members']:
        p=OUT/m['file'];assert file_sha256(p)==m['sha256'];models.append(Predictor(p))
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
    test_csv=ROOT/'datasets/glaucoma_first/test.csv'
    assert file_sha256(test_csv)=='3eba62c4e23201f85ce2a25e2290d195221159cf78b7fe8c49ecfb0f30a17c54'
    test=list(csv.DictReader(test_csv.open()));names=[r['Image_name'] for r in test]
    assert len(names)==5855 and len(set(names))==5855
    paths=resolve_images(names,ROOT/'datasets/glaucoma_first/test')
    output=OUT/'submission.csv'
    with output.open('x') as f:
        w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
        for i,(name,path) in enumerate(zip(names,paths),1):
            value,label=predict(path);assert 0<=value<=1
            write_prediction(w,f,name,value,label)
            if i%1000==0:print('TEST',i,flush=True)
    manifest=dict(status='PASS',test_images=5855,submission_sha256=file_sha256(output),validation_probability_difference=difference,validation_label_changes=int(changes),test_used_for_training=False,platform_uploaded=False,members=evidence['members'])
    (OUT/'submission.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
