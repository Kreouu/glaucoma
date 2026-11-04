"""Complete UbuntuTrim three and WSL1e-4three equal six; saved predictions only."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ubuntu_trim_screen_v1'
EVIDENCE = ROOT / 'docs/evidence/ubuntu_trim_screen_v1'
paths = [OUT / f'inputs/seed{seed}.csv' for seed in (42,43,44)]
paths.append(ROOT / 'outputs/three22k_crop_native_v1/valid_predictions.csv')
values, reference = [], None
for path in paths:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r['Image_name'], int(r['True_Label'])) for r in rows]
    if reference is None:
        reference = keys
    assert keys == reference
    p = np.array([float(r['Probability']) for r in rows])
    assert np.all((p > 0) & (p < 1))
    values.append(np.log(p) - np.log1p(-p))
y = np.array([r[1] for r in reference])
assert len(y) == 5849 and y.sum() == 191


def metric(scores):
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    tp = np.cumsum(y[order])[ends]
    fp = ends + 1 - tp
    t, f = np.r_[0, tp] / sum(y), np.r_[0, fp] / sum(1 - y)
    return dict(auroc=float(np.sum((t[1:] + t[:-1]) * np.diff(f) / 2)),
                ap=float(np.sum(np.diff(np.r_[0, tp]) / sum(y) * tp / (ends + 1))))


three = np.mean(values[:3], axis=0)
six = (three + values[3]) / 2
result = dict(status='SCREEN_ONLY', images=5849, test_used=False,
              method='Complete UbuntuTrim three and WSL1e-4three; equal six logits; no subset/weight search; no crosshost causal trim claim; each member own preprocessing',
              limitations='Inverse logits from saved probabilities; native batch1 equivalence and runtime not verified',
              wsl_three=metric(values[3]), ubuntu_three=metric(three), six=metric(six),
              sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / 'screen.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
