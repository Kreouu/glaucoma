"""Official downloaded example only; never claims the 100-image/T4 gate."""
import csv,json,sys,time,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from inference import Predictor,file_sha256,write_prediction,image_paths
OUT=ROOT/'outputs/extra_seeds_native_v1'

def main():
    samples=ROOT/'datasets/cost_data_test'
    pictures,warmup=image_paths(samples/'pictures_100'),image_paths(samples/'pictures_5')
    assert len(pictures)==10 and len(warmup)==5
    assert not {p.name.casefold() for p in pictures}&{p.name.casefold() for p in warmup}
    submission=OUT/'submission.csv'
    assert file_sha256(submission)=='99623c06f05dfdb573062f6c5ab1ed3cc2cfb302dc94597331330479439acd46'
    reference={r['Image_name'].casefold():r for r in csv.DictReader(submission.open())}
    evidence=json.loads((OUT/'native_logits.json').read_text());models=[]
    for member in evidence['members']:
        path=OUT/member['file'];assert file_sha256(path)==member['sha256'];models.append(Predictor(path))
    assert len(models)==6
    @torch.inference_mode()
    def predict(path):
        x=models[0].preprocess(path).unsqueeze(0).cuda()
        with torch.autocast('cuda',dtype=models[0].autocast_dtype):
            logits=torch.stack([p.model(x).flatten() for p in models]).float()
        value=logits.mean(0).sigmoid().item();return value,int(value>=.5)
    dest=ROOT/'outputs/official_timing_example_32068_v1';dest.mkdir(exist_ok=False)
    for p in warmup:predict(p)
    durations=[];errors=[];labels=0;rank_changes=0
    for repeat in range(3):
        values=[];expected=[]
        with (dest/f'round{repeat+1}.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['Image_name','Probability','Pred_Label']);f.flush()
            for p in pictures:
                r=reference[p.name.casefold()];torch.cuda.synchronize();start=time.perf_counter()
                value,label=predict(p);write_prediction(w,f,r['Image_name'],value,label);torch.cuda.synchronize()
                durations.append((time.perf_counter()-start)*1000);errors.append(abs(value-float(r['Probability'])));labels+=label!=int(r['Pred_Label']);values.append(value);expected.append(float(r['Probability']))
        for i in range(len(values)):
            for j in range(i):rank_changes+=((values[i]>values[j])-(values[i]<values[j]))!=((expected[i]>expected[j])-(expected[i]<expected[j]))
    report=dict(status='EXAMPLE_ONLY',measurement_images=10,warmup_images=5,rounds=3,batch_size=1,device=torch.cuda.get_device_name(),torch_version=torch.__version__,cudnn=torch.backends.cudnn.version(),mean_ms=statistics.mean(durations),round_mean_ms=[statistics.mean(durations[i*10:(i+1)*10]) for i in range(3)],max_probability_difference=max(errors),label_changes=labels,pair_order_changes=rank_changes,consistency_pass=max(errors)<=1e-3 and labels==rank_changes==0,official_100_image_gate_verified=False,official_T4_verified=False,submission_sha256=file_sha256(submission),sample_sha256={str(p.relative_to(samples)):file_sha256(p) for p in pictures+warmup})
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
