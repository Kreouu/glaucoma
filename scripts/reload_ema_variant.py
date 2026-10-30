"""Independently reload one fixed EMA recipe and reproduce every validation prediction."""
import argparse,csv,json,sys
from pathlib import Path
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ema_variant_inference import EMAVariantPredictor
from inference import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recipe',type=Path,required=True);args=parser.parse_args()
    path=args.recipe
    with path.with_name(path.stem+'_valid_predictions.csv').open() as stream:rows=list(csv.DictReader(stream))
    with (ROOT/'outputs/convnext_512_w3_v1/seed43/valid_manifest.csv').open() as stream:samples=list(csv.DictReader(stream))
    assert [(r['Image_name'],r['True_Label']) for r in rows]==[(r['Image_name'],r['True_Label']) for r in samples]
    predictor=EMAVariantPredictor(path);folder=ROOT/predictor.loader.predictors[0].config['data_root']/'valid'
    labels=[int(r['True_Label']) for r in rows];assert len(rows)==5849 and sum(labels)==191
    values=[];difference=0.;changes=0
    for step,(row,sample) in enumerate(zip(rows,samples),1):
        probability,label=predictor.predict_one(folder/sample['actual_filename']);expected=float(row['Probability'])
        difference=max(difference,abs(probability-expected));changes+=int(label!=int(expected>=.5));values.append(probability)
        if step%1000==0:print(f'Reload {step}/5849',flush=True)
    assert difference<=1e-6 and changes==0
    result=dict(status='PASS',images=len(rows),positives=sum(labels),batch_size=1,recipe_sha256=file_sha256(path),
                max_probability_difference=difference,label_changes=changes,auroc=float(roc_auc_score(labels,values)),ap=float(average_precision_score(labels,values)))
    path.with_name(path.stem+'_full_reload.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
