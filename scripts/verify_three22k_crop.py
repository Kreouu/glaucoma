"""Full native audit of the new complete recipe versus the platform-best old recipe."""
import csv,gc,json,subprocess,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from inference import Predictor,file_sha256,write_prediction,resolve_images
from three_22k_crop_inference import Three22KCropPredictor
from pack_convnext_fp16 import pack
FIRST=Path('/home/ng/workspace/glaucoma_22k_cosine_crop_20261102/outputs/convnext22k_cosine_crop_v1')
OLD=Path('/home/ng/workspace/glaucoma_v2_20261101/outputs/four_model_native_v1')
OUT=ROOT/'outputs/three22k_crop_native_v1'


def main():
    plan=json.loads((ROOT/'docs/evidence/three22k_crop_native_v1/plan.json').read_text())
    for split,digest in plan['csv_sha256'].items():assert file_sha256(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    for name,digest in plan['source_sha256'].items():assert file_sha256(ROOT/name)==digest,name
    assert file_sha256(ROOT/'inputs/runtime_sample.json')==plan['runtime_sample_sha256']
    with (FIRST/'seed42/valid_manifest.csv').open() as f:samples=list(csv.DictReader(f))
    names=[r['Image_name'] for r in samples];labels=[int(r['True_Label']) for r in samples];images=ROOT/'datasets/glaucoma_first/valid'
    assert len(labels)==5849 and sum(labels)==191
    if '--reload' in sys.argv:
        with (OUT/'valid_predictions.csv').open() as f:reference=list(csv.DictReader(f))
        assert [(r['Image_name'],int(r['True_Label'])) for r in reference]==list(zip(names,labels))
        predictor=Three22KCropPredictor(OUT/'manifest.json');difference=0.;changes=0;values=[]
        for row,sample in zip(reference,samples):
            p,label=predictor.predict_one(images/sample['actual_filename']);difference=max(difference,abs(p-float(row['Probability'])));changes+=int(label!=int(float(row['Probability'])>=.5));values.append(p)
        assert difference==0 and changes==0
        report=dict(status='PASS',images=5849,max_probability_difference=difference,label_changes=changes,auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)))
        (OUT/'full_reload.json').write_text(json.dumps(report,indent=2)+'\n');print('FULL RELOAD',json.dumps(report),flush=True);return
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    source_names=['three_22k_crop_inference.py','four_model_inference.py','inference.py','dataset.py','model.py','model_convnext.py','amp_precision.py']
    manifest=dict(aggregation='equal_raw_fp32_logit_mean',horizontal_flip=False,source_sha256={n:file_sha256(ROOT/n) for n in source_names},members=[])
    original=json.loads(json.dumps(manifest));packing=[]
    for seed in (42,43,44):
        path=FIRST/f'seed{seed}'/'ema';check=json.loads((path/'reload_check.json').read_text());assert check['status']=='PASS' and check['images']==5849 and check['max_probability_difference']==0
        source=path/'best.pt';assert file_sha256(source)==plan['checkpoints'][str(seed)]==check['checkpoint_sha256']
        target=OUT/f'seed{seed}.pt';result=pack(source,target);packing.append(result)
        original['members'].append(dict(seed=seed,file=str(source),sha256=result['original_sha256'],bytes=source.stat().st_size))
        manifest['members'].append(dict(seed=seed,file=target.name,sha256=result['packed_sha256'],bytes=target.stat().st_size,original_sha256=result['original_sha256']))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(OUT/'original_manifest.json').write_text(json.dumps(original,indent=2)+'\n');(OUT/'packing_report.json').write_text(json.dumps(packing,indent=2)+'\n')
    predictor=Three22KCropPredictor(OUT/'manifest.json');raw=Three22KCropPredictor(OUT/'original_manifest.json');old=[Predictor(OLD/f'wsl{s}.pt') for s in (42,43,44)];values=[];base=[];member_values=[[],[],[]];max_difference=0.
    with torch.inference_mode():
        for i,row in enumerate(samples,1):
            x=predictor.predictors[0].preprocess(images/row['actual_filename']).unsqueeze(0).cuda();a=predictor.logits(x);b=raw.logits(x);max_difference=max(max_difference,float((a-b).abs().max()));assert max_difference==0
            with torch.autocast('cuda',dtype=torch.float16):old_logits=torch.stack([p.model(x).flatten() for p in old])
            
            for j in range(3):member_values[j].append(a[j].sigmoid().item())
            values.append(a.mean(0).sigmoid().item());base.append(old_logits.float().mean(0).sigmoid().item())
            if i%1000==0:print('NATIVE',i,flush=True)
    baseline_auc=float(roc_auc_score(labels,base));assert abs(baseline_auc-.9900900175630484)<1e-6
    with (OUT/'valid_predictions.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(names,labels,values))
    for seed,probs in zip((42,43,44),member_values):
        with (OUT/f'seed{seed}_valid_predictions.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(names,labels,probs))
    del predictor,raw,old;gc.collect();torch.cuda.empty_cache();predictor=Three22KCropPredictor(OUT/'manifest.json')
    sample=json.loads((ROOT/'inputs/runtime_sample.json').read_text());assert len(sample['warmup'])==5 and len(sample['measured'])==100
    actual={r['Image_name']:r['actual_filename'] for r in samples};lookup=dict(zip(names,values))
    for row in sample['warmup']:predictor.predict_one(images/actual[row['Image_name']])
    timings=[];same=0.
    for repeat in (1,2,3):
        with (OUT/f'round{repeat}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
            for row in sample['measured']:
                torch.cuda.synchronize();start=time.perf_counter();p,label=predictor.predict_one(images/actual[row['Image_name']]);write_prediction(w,f,row['Image_name'],p,label);torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000);same=max(same,abs(p-lookup[row['Image_name']]))
    assert same==0
    report=dict(status='PASS',images=5849,auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)),baseline_same_pass_auroc=baseline_auc,delta_vs_platform_recipe=float(roc_auc_score(labels,values))-baseline_auc,delta_vs_local_best_six=float(roc_auc_score(labels,values))-.991386888601415,packing_max_logit_difference=max_difference,same_image_difference=same,mean_ms=statistics.mean(timings),model_bytes=sum(m['bytes'] for m in manifest['members']),official_T4_verified=False,full_reload_verified=False,member_metrics=[dict(seed=s,auroc=float(roc_auc_score(labels,p)),ap=float(average_precision_score(labels,p))) for s,p in zip((42,43,44),member_values)],scope='Fixed complete22K cosine+crop three-seed recipe; no TTA or subset/weight search; GPU idle during audit')
    assert report['mean_ms']<100 and report['model_bytes']<500000000
    del predictor;gc.collect();torch.cuda.empty_cache();subprocess.run([sys.executable,'-u',str(Path(__file__)),'--reload'],check=True,cwd=ROOT);report['full_reload_verified']=True
    if report['delta_vs_local_best_six']<.0003:
        print('No CSV: gain below fixed promotion margin',flush=True);return
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',json.dumps(report),flush=True)
    predictor=Three22KCropPredictor(OUT/'manifest.json');csv_path=ROOT/'datasets/glaucoma_first/test.csv'
    with csv_path.open() as f:test_names=[r['Image_name'] for r in csv.DictReader(f)]
    assert len(test_names)==5855 and len(set(test_names))==5855
    paths=resolve_images(test_names,ROOT/'datasets/glaucoma_first/test');target=OUT/'submission.csv'
    with target.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
        for name,path in zip(test_names,paths):p,label=predictor.predict_one(path);write_prediction(w,f,name,p,label)
    (OUT/'submission.manifest.json').write_text(json.dumps(dict(rows=5855,output_sha256=file_sha256(target),recipe_sha256=file_sha256(OUT/'manifest.json'),names_csv_sha256=file_sha256(csv_path),platform_uploaded=False),indent=2)+'\n');print('CSV READY',file_sha256(target),flush=True)


if __name__=='__main__':main()
