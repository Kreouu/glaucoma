"""Attribute batch-one latency; extra synchronizations make this diagnostic, not official cost.py."""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ema_variant_inference import EMAVariantPredictor
from inference import file_sha256,write_prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    predictor=EMAVariantPredictor(args.recipe)
    folder=ROOT/predictor.loader.predictors[0].config['data_root']/'valid'
    sample=json.loads((ROOT/'outputs/three_model_comparison_v1/runtime/sample.json').read_text())
    with args.recipe.with_name(args.recipe.stem+'_valid_predictions.csv').open() as stream:
        expected={r['Image_name']:float(r['Probability']) for r in csv.DictReader(stream)}
    for row in sample['warmup']:predictor.predict_one(folder/row['actual_filename'])
    stages=[];max_diff=0.
    with (args.output/'predictions.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
        with torch.inference_mode():
            for repeat in range(3):
                for row in sample['measured']:
                    torch.cuda.synchronize();start=time.perf_counter()
                    cpu=predictor.loader.predictors[0].preprocess(folder/row['actual_filename']).unsqueeze(0)
                    decoded=time.perf_counter();image=cpu.cuda();torch.cuda.synchronize();transferred=time.perf_counter()
                    logits=predictor.logits(image)
                    if predictor.recipe['horizontal_flip']:logits=torch.cat([logits,predictor.logits(image.flip(-1))],dim=0)
                    probability=predictor.aggregate(logits,predictor.recipe['aggregation']).item();torch.cuda.synchronize();forwarded=time.perf_counter()
                    label=int(probability>=.5);write_prediction(writer,stream,row['Image_name'],probability,label)
                    finished=time.perf_counter();max_diff=max(max_diff,abs(probability-expected[row['Image_name']]))
                    stages.append(dict(repeat=repeat+1,name=row['Image_name'],decode_preprocess_ms=(decoded-start)*1000,
                        transfer_ms=(transferred-decoded)*1000,model_aggregate_ms=(forwarded-transferred)*1000,
                        write_ms=(finished-forwarded)*1000,total_ms=(finished-start)*1000))
    assert max_diff<=1e-6
    means={key:statistics.mean(r[key] for r in stages) for key in stages[0] if key.endswith('_ms')}
    report=dict(status='PASS',recipe_sha256=file_sha256(args.recipe),script_sha256=file_sha256(Path(__file__)),
        device=torch.cuda.get_device_name(),images=100,warmup=5,rounds=3,batch_size=1,
        max_probability_difference=max_diff,mean_ms=means,official_T4_verified=False,
        scope='Extra synchronization at stage boundaries; diagnostic attribution on validation images, not official timing or independent baseline speed comparison')
    (args.output/'stages.json').write_text(json.dumps(stages,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
