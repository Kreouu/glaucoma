"""Train-only deterministic feature index, two fixed classifiers, native full validation."""
import csv,gc,json,subprocess,sys,time,statistics,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score
from four_model_inference import FourModelPredictor
from knn_inference import KNNPredictor
from dataset import FundusDataset
from inference import file_sha256,write_prediction
MAIN=Path('/home/kreouu/Model_train/glaucoma_codex_20261031')
SIX=Path('/home/kreouu/Model_train/glaucoma_ensemble_20261101/outputs/six_model_native_v1')
OUT=ROOT/'outputs/train_knn_v1'


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    samples=rows(MAIN/'outputs/convnext_4060_pair_v1/constant_seed42/valid_manifest.csv');labels=[int(r['True_Label']) for r in samples];names=[r['Image_name'] for r in samples]
    assert len(names)==5849 and sum(labels)==191
    image_root=ROOT/'datasets/glaucoma_first/valid'
    if '--reload' in sys.argv:
        mode=sys.argv[-1];predictor=KNNPredictor(OUT/f'{mode}.json');reference=rows(OUT/f'{mode}_valid_predictions.csv');values=[];diff=0.;changes=0
        assert [(r['Image_name'],int(r['True_Label'])) for r in reference]==list(zip(names,labels))
        for row,sample in zip(reference,samples):
            probability,label=predictor.predict_one(image_root/sample['actual_filename']);expected=float(row['Probability']);diff=max(diff,abs(probability-expected));changes+=int(label!=int(expected>=.5));values.append(probability)
        assert diff<=1e-6 and changes==0
        report=dict(status='PASS',images=5849,batch_size=1,max_probability_difference=diff,label_changes=changes,auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)))
        (OUT/f'{mode}_full_reload.json').write_text(json.dumps(report,indent=2)+'\n');print('RELOAD',mode,json.dumps(report),flush=True);return
    plan=json.loads((ROOT/'docs/evidence/train_knn_v1/plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert file_sha256(ROOT/name)==digest,name
    for split,digest in plan['csv_sha256'].items():assert file_sha256(ROOT/f'datasets/glaucoma_first/{split}.csv')==digest
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    weights=json.loads((SIX/'manifest.json').read_text());weights['members']=weights['members'][:4]
    for member in weights['members']:
        source=SIX/member['file'];assert file_sha256(source)==member['sha256'];shutil.copy2(source,OUT/member['file'])
    (OUT/'weights.json').write_text(json.dumps(weights,indent=2)+'\n')
    loader=FourModelPredictor(OUT/'weights.json');anchor=loader.predictors[3]
    data=FundusDataset(ROOT/'datasets/glaucoma_first','train',512,jpeg_draft=True);data.transform=anchor.transform
    features=[];targets=[];head_logits=[];seen_names=[]
    with torch.inference_mode():
        for step,(images,y,batch_names) in enumerate(DataLoader(data,batch_size=16,shuffle=False,num_workers=4,pin_memory=True),1):
            images=images.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=anchor.autocast_dtype):
                feature=anchor.model.forward_head(anchor.model.forward_features(images),pre_logits=True);logit=anchor.model.get_classifier()(feature).flatten()
            features.append(F.normalize(feature.float(),dim=1).half().cpu());targets.append(y.to(torch.uint8));head_logits.append(logit.float().cpu());seen_names.extend(batch_names)
            if step%500==0:print('INDEX',step,'/3213',flush=True)
    assert seen_names==[r['Image_name'] for r in rows(ROOT/'datasets/glaucoma_first/train.csv')]
    vectors=torch.cat(features);y=torch.cat(targets);assert vectors.shape==(51396,768) and y.shape==(51396,) and torch.isfinite(vectors).all()
    bank=dict(vectors=vectors,labels=y,config=dict(split='train',rows=51396,feature_model_sha256=weights['members'][3]['sha256'],train_csv_sha256=file_sha256(ROOT/'datasets/glaucoma_first/train.csv'),storage='L2 features FP16; renormalize per vector in FP32 on load; fixed train-only bank'))
    torch.save(bank,OUT/'index.pt');torch.save(dict(names=seen_names,logits=torch.cat(head_logits),labels=y),OUT/'train_head_logits.pt')
    for mode in ('knn','mixed'):
        recipe=dict(mode=mode,k=200,temperature=.07,prior_smoothing=.001,loss_pos_weight=3,weights_manifest='weights.json',weights_manifest_sha256=file_sha256(OUT/'weights.json'),index='index.pt',index_sha256=file_sha256(OUT/'index.pt'),source_sha256=file_sha256(ROOT/'knn_inference.py'))
        (OUT/f'{mode}.json').write_text(json.dumps(recipe,indent=2)+'\n')
    del loader,anchor,data,features,targets,head_logits,vectors,y,bank;gc.collect();torch.cuda.empty_cache()
    predictor=KNNPredictor(OUT/'mixed.json');values={mode:[] for mode in ('baseline','knn','mixed')};anchor_diff=0.
    for step,row in enumerate(samples,1):
        image=predictor.loader.predictors[0].preprocess(image_root/row['actual_filename']).unsqueeze(0).cuda();result=predictor.evaluate_tensor(image,True);anchor_diff=max(anchor_diff,result['anchor_logit_difference']);assert anchor_diff==0
        for mode in values:values[mode].append(result[mode])
        if step%1000==0:print('VALID',step,flush=True)
    del predictor;gc.collect();torch.cuda.empty_cache()
    baseline=float(roc_auc_score(labels,values['baseline']));frozen=.9906119121514456;report=dict(status='PASS',images=5849,positives=191,batch_size=1,baseline_same_host_auroc=baseline,frozen_four_auroc=frozen,anchor_logit_difference=anchor_diff,index_bytes=(OUT/'index.pt').stat().st_size,index_train_rows=51396,source_scope='Official train only, no valid/test vectors in bank',candidates={})
    sample=json.loads((ROOT/'inputs/runtime_sample.json').read_text());actual={r['Image_name']:r['actual_filename'] for r in samples}
    for mode in ('knn','mixed'):
        with (OUT/f'{mode}_valid_predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['Image_name','True_Label','Probability']);w.writerows(zip(names,labels,values[mode]))
        predictor=KNNPredictor(OUT/f'{mode}.json')
        for row in sample['warmup']:predictor.predict_one(image_root/actual[row['Image_name']])
        times=[];difference=0.;lookup=dict(zip(names,values[mode]))
        for repeat in (1,2,3):
            with (OUT/f'{mode}_round{repeat}.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label'])
                for row in sample['measured']:
                    torch.cuda.synchronize();start=time.perf_counter();p,label=predictor.predict_one(image_root/actual[row['Image_name']]);write_prediction(w,f,row['Image_name'],p,label);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000);difference=max(difference,abs(p-lookup[row['Image_name']]))
        assert difference<=1e-6
        auc=float(roc_auc_score(labels,values[mode]));ap=float(average_precision_score(labels,values[mode]));size=sum(m['bytes'] for m in weights['members'])+(OUT/'index.pt').stat().st_size
        entry=dict(auroc=auc,ap=ap,delta_auroc=auc-max(baseline,frozen),mean_ms=statistics.mean(times),necessary_bytes_conservative=size,reload_sample_difference=difference,promising=auc-max(baseline,frozen)>=.0003 and statistics.mean(times)<100 and size<500000000)
        report['candidates'][mode]=entry;print('RESULT',mode,json.dumps(entry),flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()
        if entry['promising']:subprocess.run([sys.executable,'-u',str(Path(__file__)),'--reload',mode],cwd=ROOT,check=True)
    report['goal995_met']=any(e['promising'] and e['auroc']>=.995 for e in report['candidates'].values())
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
