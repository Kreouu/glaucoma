"""Instrument the unchanged submitted six-model path; official10 example only."""
import csv,json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import torch
from six_22k_crop_inference import Six22KCropPredictor
from inference import image_paths,write_prediction

@torch.inference_mode()
def main():
 predictor=Six22KCropPredictor(ROOT/'model/manifest.json');samples=ROOT/'datasets/cost_data_test'
 pictures=image_paths(samples/'pictures_100');warmup=image_paths(samples/'pictures_5');assert len(pictures)==10 and len(warmup)==5
 for p in warmup:predictor.predict_one(p)
 out=ROOT/'outputs/latency_profile_v1';out.mkdir(parents=True,exist_ok=False);parts=[]
 with (out/'predictions.csv').open('x') as stream:
  w=csv.writer(stream);w.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
  for path in pictures:
   torch.cuda.synchronize();t0=time.perf_counter();cpu=predictor.predictors[0].preprocess(path);t1=time.perf_counter()
   x=cpu.unsqueeze(0).cuda();torch.cuda.synchronize();t2=time.perf_counter()
   value=predictor.logits(x).mean(0).sigmoid().item();torch.cuda.synchronize();t3=time.perf_counter()
   write_prediction(w,stream,path.name,value,int(value>=.5));torch.cuda.synchronize();t4=time.perf_counter()
   parts.append(dict(preprocess_ms=(t1-t0)*1000,host_to_device_ms=(t2-t1)*1000,six_forward_ms=(t3-t2)*1000,write_ms=(t4-t3)*1000,total_ms=(t4-t0)*1000))
 report=dict(scope='Instrumented10example, extra synchronization overhead included; not official100/T4',device=torch.cuda.get_device_name(),mean={k:statistics.mean(r[k] for r in parts) for k in parts[0]},images=parts)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['mean']),flush=True)

if __name__=='__main__':main()
