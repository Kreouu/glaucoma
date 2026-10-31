"""Native full validation, independent reload, same-image latency and conditional test CSV."""
import csv,gc,json,shutil,subprocess,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from inference import file_sha256,write_prediction,resolve_images
from four_model_inference import FourModelPredictor
MAIN=Path('/home/ng/workspace/glaucoma')
OUT=ROOT/'outputs/four_model_native_v1'


def main():
    with (MAIN/'outputs/convnext_512_w3_v1/seed43/valid_manifest.csv').open() as f:samples=list(csv.DictReader(f))
    labels=[int(r['True_Label']) for r in samples];names=[r['Image_name'] for r in samples];assert len(names)==5849 and sum(labels)==191
    images=ROOT/'datasets/glaucoma_first/valid'
    if '--reload' in sys.argv:
        with (OUT/'valid_predictions.csv').open() as f:reference=list(csv.DictReader(f))
        assert [(r['Image_name'],int(r['True_Label'])) for r in reference]==list(zip(names,labels))
        predictor=FourModelPredictor(OUT/'manifest.json');values=[];difference=0.;changes=0
        for row,sample in zip(reference,samples):
            value,label=predictor.predict_one(images/sample['actual_filename']);expected=float(row['Probability'])
            difference=max(difference,abs(value-expected));changes+=int(label!=int(expected>=.5));values.append(value)
        assert difference<=1e-6 and changes==0
        report=dict(status='PASS',images=5849,batch_size=1,auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)),max_probability_difference=difference,label_changes=changes)
        (OUT/'full_reload.json').write_text(json.dumps(report,indent=2)+'\n');print('FULL RELOAD',json.dumps(report),flush=True);return
    OUT.mkdir(exist_ok=False)
    source_names=['four_model_inference.py','inference.py','dataset.py','model.py','model_convnext.py','amp_precision.py']
    manifest=dict(aggregation='equal_raw_fp32_logit_mean',horizontal_flip=False,source_sha256={n:file_sha256(ROOT/n) for n in source_names},members=[])
    for host,seed in [('wsl',42),('wsl',43),('wsl',44),('ubuntu',42)]:
        identifier=f'{host}{seed}'
        source=MAIN/f'outputs/convnext_512_lr1e4_v1/seed{seed}/ema/best.pt' if host=='wsl' else ROOT/'inputs/ubuntu42.pt'
        expected=(source.parent/'best.sha256').read_text().split()[0] if host=='wsl' else 'bf62d51936fbe6203e23f36af0c637371b54ef0e13410cc73e6eed5fced73915'
        assert file_sha256(source)==expected
        target=OUT/f'{identifier}.pt';shutil.copy2(source,target)
        manifest['members'].append(dict(id=identifier,seed=seed,file=target.name,sha256=expected,bytes=target.stat().st_size))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    predictor=FourModelPredictor(OUT/'manifest.json');values=[];three=[]
    with torch.inference_mode():
        for step,row in enumerate(samples,1):
            image=predictor.predictors[0].preprocess(images/row['actual_filename']).unsqueeze(0).cuda();logits=predictor.logits(image)
            values.append(logits.mean(0).sigmoid().item());three.append(logits[:3].mean(0).sigmoid().item())
            if step%1000==0:print('NATIVE',step,flush=True)
    with (MAIN/'outputs/convnext_512_lr1e4_inference_v1/logit_valid_predictions.csv').open() as f:frozen=list(csv.DictReader(f))
    assert [(r['Image_name'],int(r['True_Label'])) for r in frozen]==list(zip(names,labels))
    baseline_difference=max(abs(value-float(row['Probability'])) for value,row in zip(three,frozen))
    assert baseline_difference<=1e-6
    with (OUT/'valid_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(names,labels,values))
    del predictor;gc.collect();torch.cuda.empty_cache()
    predictor=FourModelPredictor(OUT/'manifest.json')
    sample=json.loads((MAIN/'outputs/three_model_comparison_v1/runtime/sample.json').read_text())
    for row in sample['warmup']:predictor.predict_one(images/row['actual_filename'])
    lookup=dict(zip(names,values));timings=[];difference=0.
    for repeat in (1,2,3):
        with (OUT/f'round{repeat}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
            for row in sample['measured']:
                torch.cuda.synchronize();start=time.perf_counter();value,label=predictor.predict_one(images/row['actual_filename']);write_prediction(w,f,row['Image_name'],value,label);torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
                difference=max(difference,abs(value-lookup[row['Image_name']]))
    assert difference<=1e-6
    auc=float(roc_auc_score(labels,values));ap=float(average_precision_score(labels,values));base_auc=float(roc_auc_score(labels,three));base_ap=float(average_precision_score(labels,three))
    report=dict(status='PASS',images=5849,positives=191,batch_size=1,auroc=auc,ap=ap,baseline_same_pass_auroc=base_auc,baseline_same_pass_ap=base_ap,delta_auroc=auc-base_auc,delta_ap=ap-base_ap,mean_ms=statistics.mean(timings),model_bytes=sum(m['bytes'] for m in manifest['members']),same_image_difference=difference,frozen_baseline_probability_difference=baseline_difference,official_T4_verified=False,goal995_met=auc>=.995)
    assert report['model_bytes']<500000000
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',json.dumps(report),flush=True)
    del predictor;gc.collect();torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--reload'],cwd=ROOT,check=True)
    if auc-base_auc>=.0003 and report['mean_ms']<100:
        predictor=FourModelPredictor(OUT/'manifest.json');csv_path=ROOT/'datasets/glaucoma_first/test.csv'
        with csv_path.open() as f:test_names=[r['Image_name'] for r in csv.DictReader(f)]
        assert len(test_names)==5855 and len(set(test_names))==5855
        paths=resolve_images(test_names,ROOT/'datasets/glaucoma_first/test');target=OUT/'submission.csv'
        with target.open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
            for name,path in zip(test_names,paths):value,label=predictor.predict_one(path);write_prediction(w,f,name,value,label)
        (OUT/'submission.manifest.json').write_text(json.dumps(dict(rows=5855,output_sha256=file_sha256(target),recipe_sha256=file_sha256(OUT/'manifest.json'),names_csv_sha256=file_sha256(csv_path),platform_uploaded=False),indent=2)+'\n')


if __name__=='__main__':main()
