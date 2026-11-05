"""Differential end-to-end10+5 benchmark, then complete5855 graph reproduction."""
import csv,json,sys,time,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from six_22k_crop_inference import Six22KCropPredictor
from inference_cuda_graph import GraphPredictor
from inference import image_paths,resolve_images,write_prediction,file_sha256


def check(reference,values,labels):
 a=np.array(reference);b=np.array(values)
 rank=int(np.count_nonzero(np.sign(a[:,None]-a[None,:])!=np.sign(b[:,None]-b[None,:])))
 return dict(max_probability_difference=float(np.abs(a-b).max()),label_changes=int(labels),pair_order_changes=rank)


def main():
 assert file_sha256(ROOT/'submission.csv')=='99623c06f05dfdb573062f6c5ab1ed3cc2cfb302dc94597331330479439acd46'
 rows=list(csv.DictReader((ROOT/'submission.csv').open()));lookup={r['Image_name'].casefold():r for r in rows}
 base=Six22KCropPredictor(ROOT/'model/manifest.json');graph=GraphPredictor(base)
 samples=ROOT/'datasets/cost_data_test';pictures=image_paths(samples/'pictures_100');warmup=image_paths(samples/'pictures_5');assert len(pictures)==10 and len(warmup)==5
 out=ROOT/'outputs/cuda_graph_v1';out.mkdir(parents=True,exist_ok=False)
 variants={'baseline':base.predict_one,'cuda_graph':graph.predict_one};times={k:[] for k in variants};checks={k:[] for k in variants}
 for f in variants.values():
  for p in warmup:f(p)
 for repeat in range(1,4):
  order=list(variants) if repeat%2 else list(reversed(variants))
  for name in order:
   values=[];labels=0;reference=[];durations=[]
   with (out/f'{name}_round{repeat}.csv').open('x',newline='') as stream:
    w=csv.writer(stream);w.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
    for p in pictures:
     r=lookup[p.name.casefold()];torch.cuda.synchronize();start=time.perf_counter();value,label=variants[name](p);write_prediction(w,stream,r['Image_name'],value,label);torch.cuda.synchronize();durations.append((time.perf_counter()-start)*1000)
     values.append(value);reference.append(float(r['Probability']));labels+=label!=int(r['Pred_Label'])
   times[name]+=durations;checks[name].append(check(reference,values,labels))
 report=dict(status='SAMPLE_ONLY',device=torch.cuda.get_device_name(),images=10,warmup_images=5,rounds=3,precision='Existing FP16autocast, unchanged FP32 aggregation',mean_ms={k:statistics.mean(v) for k,v in times.items()},checks=checks,official_T4_verified=False,official_100_image_gate_verified=False,test_used_for_training=False,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
 (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 exact=all(c['max_probability_difference']==c['label_changes']==c['pair_order_changes']==0 for c in checks['cuda_graph'])
 if not exact or report['mean_ms']['cuda_graph']>=report['mean_ms']['baseline']:
  print('No exact faster candidate; full export skipped',flush=True);return
 paths=resolve_images([r['Image_name'] for r in rows],ROOT/'datasets/glaucoma_first/test');values=[];labels=0
 output=out/'reproduced.csv'
 with output.open('x',newline='') as stream:
  w=csv.writer(stream);w.writerow(['Image_name','Probability','Pred_Label'])
  for i,(r,p) in enumerate(zip(rows,paths),1):
   value,label=graph.predict_one(p);write_prediction(w,stream,r['Image_name'],value,label);values.append(value);labels+=label!=int(r['Pred_Label'])
   if i%1000==0:print('FULL',i,flush=True)
 reference=np.array([float(r['Probability']) for r in rows]);actual=np.array(values)
 rank=(np.searchsorted(np.sort(reference),reference,side='left')!=np.searchsorted(np.sort(actual),actual,side='left'))|(np.searchsorted(np.sort(reference),reference,side='right')!=np.searchsorted(np.sort(actual),actual,side='right'))
 digest=file_sha256(output);original=file_sha256(ROOT/'submission.csv');result=dict(status='PASS' if digest==original else 'REVIEW',images=len(rows),max_probability_difference=float(abs(reference-actual).max()),label_changes=int(labels),rank_affected_images=int(rank.sum()),reproduced_sha256=digest,original_sha256=original,byte_identical=digest==original,official_T4_verified=False,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
 (out/'full_reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
