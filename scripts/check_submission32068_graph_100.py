"""Paired100 validation-image end-to-end study, never official test100/T4."""
import csv,json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'code'))
import torch,numpy as np
from six_22k_crop_inference import Six22KCropPredictor
from inference_cuda_graph import GraphPredictor
from inference import resolve_images,write_prediction


def main():
 base=Six22KCropPredictor(ROOT/'model/manifest.json');graph=GraphPredictor(base)
 sample=json.loads(Path('/home/ng/workspace/glaucoma_three22k_crop_native_20261102/outputs/three22k_crop_native_v1/runtime_sample.json').read_text())
 assert len(sample['measured'])==100 and len(sample['warmup'])==5
 images=ROOT/'datasets/glaucoma_first/valid';names=[r['Image_name'] for r in sample['measured']];paths=resolve_images(names,images);warm=resolve_images([r['Image_name'] for r in sample['warmup']],images)
 out=ROOT/'outputs/graph_100_valid_v1';out.mkdir(parents=True,exist_ok=False);variants={'baseline':base.predict_one,'cuda_graph':graph.predict_one};timings={k:[] for k in variants};values={k:[] for k in variants}
 for f in variants.values():
  for p in warm:f(p)
 for repeat in range(1,4):
  order=list(variants) if repeat%2 else list(reversed(variants))
  for name in order:
   probabilities=[]
   with (out/f'{name}_round{repeat}.csv').open('x',newline='') as stream:
    w=csv.writer(stream);w.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
    for image_name,path in zip(names,paths):
     torch.cuda.synchronize();start=time.perf_counter();value,label=variants[name](path);write_prediction(w,stream,image_name,value,label);torch.cuda.synchronize();timings[name].append((time.perf_counter()-start)*1000);probabilities.append(value)
   values[name].append(probabilities)
 a,b=np.array(values['baseline']),np.array(values['cuda_graph']);delta=float(abs(a-b).max());report=dict(status='PASS' if delta==0 else 'REVIEW',scope='Fixed100validationimages,5warmup,3round,batch1,read/preprocess/H2D/forward/sigmoid/CSVflush/sync included. Not officialtest100 or T4',images=100,device=torch.cuda.get_device_name(),mean_ms={k:statistics.mean(v) for k,v in timings.items()},round_mean_ms={k:[statistics.mean(v[i*100:(i+1)*100]) for i in range(3)] for k,v in timings.items()},max_probability_difference=delta,label_changes=int(np.count_nonzero((a>=.5)!=(b>=.5))),official_T4_verified=False,official_100_image_gate_verified=False,torch_threads=torch.get_num_threads(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
