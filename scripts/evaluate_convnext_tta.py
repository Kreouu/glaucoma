"""Fixed horizontal-flip probability averaging on the three frozen ConvNeXt-512 seeds."""
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import FundusDataset
from inference import Predictor, file_sha256


def scores(labels, probability):
    return dict(auroc=float(roc_auc_score(labels, probability)), ap=float(average_precision_score(labels, probability)))


def main():
    output = ROOT/'outputs/convnext_512_tta_v1'
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    originals=[]
    augmented=[]
    for seed in (42,43,44):
        run=ROOT/'outputs/convnext_512_w3_v1'/f'seed{seed}'
        predictor=Predictor(run/'best.pt')
        config=predictor.config
        dataset=FundusDataset(ROOT/config['data_root'],'valid',config['size'],None,seed,config['jpeg_draft'])
        loader=DataLoader(dataset,batch_size=16,num_workers=4,pin_memory=True)
        names=[];labels=[];base=[];flipped=[]
        with torch.inference_mode():
            for step,(images,target,batch_names) in enumerate(loader,1):
                images=images.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=predictor.autocast_dtype):
                    normal_logits=predictor.model(images).flatten()
                    flip_logits=predictor.model(images.flip(-1)).flatten()
                base.extend(normal_logits.float().sigmoid().cpu().tolist())
                flipped.extend(flip_logits.float().sigmoid().cpu().tolist())
                labels.extend(target.int().tolist());names.extend(batch_names)
                if step%100==0:print(f'TTA seed{seed} {step}/{len(loader)}',flush=True)
        with (run/'best_valid_predictions.csv').open() as stream:saved=list(csv.DictReader(stream))
        assert names==[r['Image_name'] for r in saved] and labels==[int(r['True_Label']) for r in saved]
        assert len(labels)==5849 and sum(labels)==191
        difference=float(np.abs(np.array(base)-[float(r['Probability']) for r in saved]).max())
        assert difference<=1e-6,difference
        average=(np.array(base)+np.array(flipped))/2
        assert np.isfinite(average).all()
        baseline=scores(labels,base);tta=scores(labels,average)
        with (output/f'seed{seed}_valid_predictions.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['Image_name','True_Label','Base','Flipped','TTA'])
            writer.writerows(zip(names,labels,base,flipped,average.tolist()))
        rows.append(dict(seed=seed,checkpoint_sha256=file_sha256(run/'best.pt'),baseline=baseline,tta=tta,
                         delta_auroc=tta['auroc']-baseline['auroc'],delta_ap=tta['ap']-baseline['ap'],
                         base_reload_max_difference=difference))
        originals.append(base);augmented.append(average)
        del predictor
        torch.cuda.empty_cache()
    result=dict(status='PASS',method='equal original/horizontal-flip probabilities; no transform/weight search',
                images=len(labels),positives=sum(labels),seeds=rows,
                mean_auroc_delta=statistics.mean(r['delta_auroc'] for r in rows),
                mean_ap_delta=statistics.mean(r['delta_ap'] for r in rows),
                equal_ensemble=scores(labels,np.mean(originals,axis=0)),
                equal_ensemble_tta=scores(labels,np.mean(augmented,axis=0)),
                limitations='Fixed validation set; runtime and batch-one deployment output still need verification; no test data used')
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
