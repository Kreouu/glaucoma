"""Strict retained initialization, official backbone equality and preprocessing parity."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import timm
from torchvision import transforms
from timm.models.eva import checkpoint_filter_fn
from dataset import FundusDataset, Letterbox
from inference import Predictor
from model_eva02 import build_model, configure_preprocessing, MODEL, MEAN, STD

torch.set_num_threads(2)
p = ROOT / 'pretrain_model/eva02_small_patch14_336_mim_in22k_ft_in1k.bin'
digest = hashlib.sha256(p.read_bytes()).hexdigest()
assert digest == 'c81fd4bc58267becf4ab3229c7e697f4d59547c44c91da4f7287f131d5195ee9'
state = torch.load(p, map_location='cpu', weights_only=True)
model = build_model(p).eval()
official = timm.create_model(MODEL, pretrained=False).eval()
filtered = checkpoint_filter_fn(state, official)
official.load_state_dict(filtered, strict=True)
kept = {k: v for k, v in filtered.items() if not k.startswith('head.')}
for k, v in kept.items():
    assert torch.equal(model.state_dict()[k], v), k
assert all(p.requires_grad for p in model.parameters())
assert list(official.pretrained_cfg['mean']) == MEAN and list(official.pretrained_cfg['std']) == STD
with torch.inference_mode():
    x = torch.randn(1, 3, 336, 336)
    difference = float((model.forward_features(x) - official.forward_features(x)).abs().max())
    output = model(x)
assert difference == 0 and output.shape == (1, 1) and torch.isfinite(output).all()
data = FundusDataset(ROOT / 'datasets/glaucoma_first', 'valid', 336, 64, 42, True)
configure_preprocessing(data.transform)
predictor = Predictor.__new__(Predictor)
predictor.config = dict(size=336, jpeg_draft=True)
predictor.transform = transforms.Compose([Letterbox(336), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
configure_preprocessing(predictor.transform)
for i in range(3):
    assert torch.equal(data[i][0], predictor.preprocess(data.samples[i][1]))
report = dict(status='PASS', retained_tensors=len(kept), official_feature_max_difference=difference,
              all_learned_parameters_trainable=True, parameters=sum(p.numel() for p in model.parameters()),
              buffers=[n for n, _ in model.named_buffers()], output_shape=list(output.shape),
              validation_inference_preprocessing_exact=True, pretrained_sha256=digest,
              scope='CPU initialization and actual JPEG transform equality; model buffers retained, EMA disabled; GPU training separate')
(ROOT / 'docs/evidence/eva02_small_v1/cpu_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
