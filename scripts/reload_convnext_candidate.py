"""Independently reload a fixed inference manifest and compare all saved validation outputs."""
import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ensemble_inference import ConvNeXtProbabilityPredictor
from inference import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    manifest=args.manifest
    reference=manifest.with_name(manifest.stem+'_valid_predictions.csv')
    with reference.open() as stream:rows=list(csv.DictReader(stream))
    predictor=ConvNeXtProbabilityPredictor(manifest)
    root=ROOT/predictor.predictors[0].config['data_root']/'valid'
    # Use the audited manifest to resolve filename case without reading test data.
    with (ROOT/'outputs/convnext_512_w3_v1/seed43/valid_manifest.csv').open() as stream:
        valid=list(csv.DictReader(stream))
    assert [(r['Image_name'],r['True_Label']) for r in rows]==[(r['Image_name'],r['True_Label']) for r in valid]
    assert len(rows)==5849 and sum(int(r['True_Label']) for r in rows)==191
    probabilities=[];difference=0.;label_changes=0
    for step,(row,sample) in enumerate(zip(rows,valid),1):
        probability,label=predictor.predict_one(root/sample['actual_filename'])
        expected=float(row['Probability'])
        difference=max(difference,abs(probability-expected))
        label_changes+=int(label!=int(expected>=.5));probabilities.append(probability)
        if step%1000==0:print(f'{manifest.stem} reload {step}/{len(rows)}',flush=True)
    assert difference<=1e-6 and label_changes==0,(difference,label_changes)
    labels=[int(r['True_Label']) for r in rows]
    result=dict(status='PASS',images=len(rows),positives=sum(labels),batch_size=1,
                manifest_sha256=file_sha256(manifest),reference_sha256=file_sha256(reference),
                max_probability_difference=difference,label_changes=label_changes,
                auroc=float(roc_auc_score(labels,probabilities)),ap=float(average_precision_score(labels,probabilities)))
    manifest.with_name(manifest.stem+'_full_reload.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
