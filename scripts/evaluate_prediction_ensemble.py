"""Predeclared equal-probability ensemble of ConvNeXt-512 seeds 42/43/44 on validation only."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'outputs/convnext_512_equal_ensemble_v1'
    output.mkdir(parents=True, exist_ok=False)
    tables = []
    sources = []
    for seed in (42, 43, 44):
        run = ROOT / 'outputs/convnext_512_w3_v1' / f'seed{seed}'
        path = run / 'best_valid_predictions.csv'
        with path.open() as stream:
            table = list(csv.DictReader(stream))
        if tables:
            assert [(r['Image_name'], r['True_Label']) for r in table] == [(r['Image_name'], r['True_Label']) for r in tables[0]]
        tables.append(table)
        sources.append(dict(seed=seed, predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            checkpoint_sha256=(run/'best.sha256').read_text().split()[0]))
    labels = np.array([int(r['True_Label']) for r in tables[0]])
    assert len(labels) == 5849 and labels.sum() == 191
    values = np.array([[float(r['Probability']) for r in table] for table in tables])
    prediction = values.mean(axis=0)
    baseline = values[1]
    auroc, ap = roc_auc_score(labels, prediction), average_precision_score(labels, prediction)
    original_auc, original_ap = roc_auc_score(labels, baseline), average_precision_score(labels, baseline)
    rng = np.random.default_rng(20261030)
    positive, negative = np.flatnonzero(labels), np.flatnonzero(1-labels)
    differences = []
    for _ in range(2000):
        indices = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative,len(negative))])
        differences.append(roc_auc_score(labels[indices], prediction[indices])-roc_auc_score(labels[indices],baseline[indices]))
    result = dict(status='VALIDATION_PREDICTIONS_ONLY', method='equal probability mean, seeds42/43/44, no weight search',
                  images=len(labels), positives=int(labels.sum()), sources=sources,
                  baseline_seed=43, baseline_auroc=float(original_auc), baseline_ap=float(original_ap),
                  ensemble_auroc=float(auroc), ensemble_ap=float(ap), delta_auroc=float(auroc-original_auc),
                  delta_ap=float(ap-original_ap), paired_stratified_bootstrap_95ci=np.quantile(differences,[.025,.975]).tolist(),
                  bootstrap_repeats=2000, bootstrap_seed=20261030,
                  limitations='Bootstrap over fixed validation images is not training-seed uncertainty; shared validation model selection can be optimistic. Deployment output, model size and runtime still need verification. No test data used.')
    with (output/'valid_predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Image_name','True_Label','Probability'])
        writer.writerows((r['Image_name'],r['True_Label'],format(float(p),'.17g')) for r,p in zip(tables[0],prediction))
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
