"""Exercise the actual saved-checkpoint Predictor and verify backbone updates."""
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dataset import FundusDataset
from inference import Predictor
from model_eva02 import configure_preprocessing

run = Path(sys.argv[1])
predictor = Predictor(run / 'best.pt')
c = predictor.config
data = FundusDataset(Path(c['data_root']), 'valid', c['size'], c['valid_limit'], c['seed'], c['jpeg_draft'])
configure_preprocessing(data.transform)
with (run / 'valid_manifest.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(data) == len(rows) == 64
probs = []
for i in range(3):
    path = Path(c['data_root']) / 'valid' / rows[i]['actual_filename']
    assert torch.equal(data[i][0], predictor.preprocess(path))
    probability, label = predictor.predict_one(path)
    assert 0 <= probability <= 1 and label == int(probability >= .5)
    probs.append(probability)
initial = torch.load(c['pretrained_path'], map_location='cpu', weights_only=True)
key = 'patch_embed.proj.weight'
delta = float((predictor.model.state_dict()[key].cpu() - initial[key]).abs().max())
assert delta > 0
report = dict(status='PASS', saved_checkpoint_single_image_predictor_executed=True,
              validation_inference_preprocessing_exact=True, backbone_patch_weight_max_update=delta,
              probabilities=probs, scope='Three smoke images; BF16 native batch1 executes, not full deployment or T4 certification')
(ROOT / 'docs/evidence/eva02_small_v1/native_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
