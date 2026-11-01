"""Six fixed members: raw-versus-packed exact logits, full reload and runtime checks."""
import csv,gc,json,subprocess,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from inference import file_sha256,write_prediction,resolve_images
from six_crop_model_inference import SixCropModelPredictor
from pack_convnext_fp16 import pack
FROZEN=ROOT/'inputs/frozen_five'
OLD=Path('/home/kreouu/Model_train/glaucoma_ensemble_20261101/outputs/six_model_native_v1')
NEW=Path('/home/kreouu/Model_train/glaucoma_traincrop_20261101/outputs/convnext_train_crop_v1/area0p8_seed42')
OUT=ROOT/'outputs/six_crop_native_v1'



def main():
    expected_csv={'train':'ec81a23ba7758eccdd09dba027ce1a50998d81c18ed30ab6d2d608b3397f67d7','valid':'edda257a801a1036ea2b1a9ffe983a1f9fea73c3f89fd6ac0f6ab6a4da355f2a'}
    for split,digest in expected_csv.items():assert file_sha256(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    with (NEW/'valid_manifest.csv').open() as f:samples=list(csv.DictReader(f))
    labels=[int(r['True_Label']) for r in samples];names=[r['Image_name'] for r in samples];assert len(names)==5849 and sum(labels)==191
    images=ROOT/'datasets/glaucoma_first/valid'
    if '--reload' in sys.argv:
        with (OUT/'valid_predictions.csv').open() as f:reference=list(csv.DictReader(f))
        assert [(r['Image_name'],int(r['True_Label'])) for r in reference]==list(zip(names,labels))
        predictor=SixCropModelPredictor(OUT/'manifest.json');values=[];difference=0.;changes=0
        for row,sample in zip(reference,samples):
            value,label=predictor.predict_one(images/sample['actual_filename']);expected=float(row['Probability'])
            difference=max(difference,abs(value-expected));changes+=int(label!=int(expected>=.5));values.append(value)
        assert difference<=1e-6 and changes==0
        report=dict(status='PASS',images=5849,batch_size=1,auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)),max_probability_difference=difference,label_changes=changes)
        (OUT/'full_reload.json').write_text(json.dumps(report,indent=2)+'\n');print('FULL RELOAD',json.dumps(report),flush=True);return
    plan=json.loads((ROOT/'docs/evidence/six_crop_native_v1/plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert file_sha256(ROOT/name)==digest,name
    assert file_sha256(FROZEN/'manifest.json')==plan['frozen_five_manifest_sha256']
    assert file_sha256(NEW/'ema/best.pt')==plan['crop_checkpoint_sha256']
    assert file_sha256(ROOT/'inputs/pretrain22k42.pt')==plan['pretrain22k_checkpoint_sha256']
    assert file_sha256(OLD/'original_manifest.json')==plan['old_six_original_manifest_sha256']
    assert file_sha256(FROZEN/'valid_predictions.csv')==plan['frozen_five_predictions_sha256']
    assert file_sha256(ROOT/'inputs/runtime_sample.json')==plan['runtime_sample_sha256']
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    source_names=['six_crop_model_inference.py','four_model_inference.py','inference.py','dataset.py','model.py','model_convnext.py','amp_precision.py']
    original_manifest=dict(aggregation='equal_raw_fp32_logit_mean',horizontal_flip=False,source_sha256={n:file_sha256(ROOT/n) for n in source_names},members=[])
    packing=[]
    frozen=json.loads((OLD/'original_manifest.json').read_text())
    candidates=[(m['id'],m['seed'],Path(m['file']),m['sha256']) for m in frozen['members'][:4]]
    candidates.append(('pretrain22k42',42,ROOT/'inputs/pretrain22k42.pt',plan['pretrain22k_checkpoint_sha256']))
    check=json.loads((NEW/'ema/reload_check.json').read_text())
    assert check['status']=='PASS' and check['images']==5849 and check['max_probability_difference']==0
    candidates.append(('traincrop42',42,NEW/'ema/best.pt',check['checkpoint_sha256']))
    for identifier,seed,source,expected in candidates:
        assert file_sha256(source)==expected
        original_manifest['members'].append(dict(id=identifier,seed=seed,file=str(source),sha256=expected,bytes=source.stat().st_size))
        result=pack(source,OUT/f'{identifier}.pt');result['id']=identifier;packing.append(result)
    manifest=json.loads(json.dumps(original_manifest))
    for entry,result in zip(manifest['members'],packing):entry.update(file=entry['id']+'.pt',sha256=result['packed_sha256'],bytes=result['packed_bytes'],original_sha256=result['original_sha256'])
    manifest['storage_policy']='Selective FP16 Conv/Linear storage; other tensors preserved; restore into FP32 model, existing FP16 autocast'
    (OUT/'original_manifest.json').write_text(json.dumps(original_manifest,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'packing_report.json').write_text(json.dumps(packing,indent=2)+'\n')
    predictor=SixCropModelPredictor(OUT/'manifest.json');original=SixCropModelPredictor(OUT/'original_manifest.json');values=[];three=[];packing_difference=0.
    with torch.inference_mode():
        for step,row in enumerate(samples,1):
            image=predictor.predictors[0].preprocess(images/row['actual_filename']).unsqueeze(0).cuda();logits=predictor.logits(image)
            expected=original.logits(image);packing_difference=max(packing_difference,float((logits-expected).abs().max()))
            assert packing_difference==0
            values.append(logits.mean(0).sigmoid().item());three.append(expected[:5].mean(0).sigmoid().item())
            if step%1000==0:print('NATIVE',step,flush=True)
    with (FROZEN/'valid_predictions.csv').open() as f:frozen=list(csv.DictReader(f))
    assert [(r['Image_name'],int(r['True_Label'])) for r in frozen]==list(zip(names,labels))
    baseline_difference=max(abs(value-float(row['Probability'])) for value,row in zip(three,frozen))
    # Different GPU/software: recompute frozen five on this host and gate against both baselines.
    frozen_auc=float(roc_auc_score(labels,[float(r['Probability']) for r in frozen]))
    with (OUT/'valid_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(names,labels,values))
    del predictor,original;gc.collect();torch.cuda.empty_cache()
    predictor=SixCropModelPredictor(OUT/'manifest.json')
    sample=json.loads((ROOT/'inputs/runtime_sample.json').read_text())
    assert len(sample['warmup'])==5 and len(sample['measured'])==100
    actual_names={r['Image_name']:r['actual_filename'] for r in samples}
    for row in sample['warmup']+sample['measured']:row['actual_filename']=actual_names[row['Image_name']]
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
    report=dict(status='PASS',images=5849,positives=191,batch_size=1,auroc=auc,ap=ap,baseline_same_pass_auroc=base_auc,baseline_same_pass_ap=base_ap,delta_auroc=auc-base_auc,delta_ap=ap-base_ap,mean_ms=statistics.mean(timings),model_bytes=sum(m['bytes'] for m in manifest['members']),same_image_difference=difference,frozen_baseline_probability_difference=baseline_difference,frozen_five_auroc=frozen_auc,packing_max_logit_difference=packing_difference,official_T4_verified=False,prior_native_best_auroc=.9910496003434881,delta_vs_prior_best=auc-.9910496003434881,goal995_met=False)
    assert report['model_bytes']<500000000
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',json.dumps(report),flush=True)
    del predictor;gc.collect();torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--reload'],cwd=ROOT,check=True)
    reload=json.loads((OUT/'full_reload.json').read_text());assert reload['status']=='PASS'
    report['goal995_met']=auc>=.995 and report['mean_ms']<100 and report['model_bytes']<500000000
    report['full_reload_verified']=True
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    if auc-max(base_auc,frozen_auc)>=.0003 and report['mean_ms']<100:
        predictor=SixCropModelPredictor(OUT/'manifest.json');csv_path=ROOT/'datasets/glaucoma_first/test.csv'
        with csv_path.open() as f:test_names=[r['Image_name'] for r in csv.DictReader(f)]
        assert len(test_names)==5855 and len(set(test_names))==5855
        paths=resolve_images(test_names,ROOT/'datasets/glaucoma_first/test');target=OUT/'submission.csv'
        with target.open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
            for name,path in zip(test_names,paths):value,label=predictor.predict_one(path);write_prediction(w,f,name,value,label)
        (OUT/'submission.manifest.json').write_text(json.dumps(dict(rows=5855,output_sha256=file_sha256(target),recipe_sha256=file_sha256(OUT/'manifest.json'),names_csv_sha256=file_sha256(csv_path),platform_uploaded=False),indent=2)+'\n')


if __name__=='__main__':main()
