"""Compare CPU thread counts for the exact graph predictor; example-only study."""
import csv,json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'code'))
import torch,numpy as np
from six_22k_crop_inference import Six22KCropPredictor
from inference_cuda_graph import GraphPredictor
from inference import image_paths,write_prediction


def main():
 default=torch.get_num_threads();counts=list(dict.fromkeys([1,2,4,default]))
 predictor=GraphPredictor(Six22KCropPredictor(ROOT/'model/manifest.json'));root=ROOT/'datasets/cost_data_test';pictures=image_paths(root/'pictures_100');warmup=image_paths(root/'pictures_5')
 lookup={r['Image_name'].casefold():r for r in csv.DictReader((ROOT/'submission.csv').open())};out=ROOT/'outputs/cpu_threads_v1';out.mkdir(parents=True,exist_ok=False);timings={n:[] for n in counts};checks={n:[] for n in counts}
 for repeat in range(3):
  for n in counts[repeat:]+counts[:repeat]:
   torch.set_num_threads(n)
   for p in warmup:predictor.predict_one(p)
   actual=[];reference=[];labels=0
   with (out/f'threads{n}_round{repeat+1}.csv').open('x') as stream:
    w=csv.writer(stream);w.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
    for p in pictures:
     row=lookup[p.name.casefold()];torch.cuda.synchronize();t=time.perf_counter();value,label=predictor.predict_one(p);write_prediction(w,stream,row['Image_name'],value,label);torch.cuda.synchronize();timings[n].append((time.perf_counter()-t)*1000);actual.append(value);reference.append(float(row['Probability']));labels+=label!=int(row['Pred_Label'])
   a,b=np.array(reference),np.array(actual);checks[n].append(dict(max_probability_difference=float(abs(a-b).max()),label_changes=labels,pair_order_changes=int(np.count_nonzero(np.sign(a[:,None]-a[None,:])!=np.sign(b[:,None]-b[None,:])))))
 report=dict(default_threads=default,mean_ms={n:statistics.mean(v) for n,v in timings.items()},checks=checks,scope='Exploratory10example thread study, extra warmup between configurations, not official100 or T4')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
