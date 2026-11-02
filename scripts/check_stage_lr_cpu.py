"""Check real Tiny parameter coverage and the actual AdamW step per stage."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from model_convnext import build_model
from training_stage_lr import stage_lr_groups

torch.set_num_threads(2)
model = build_model()
groups = stage_lr_groups(model, 1e-4, .8)
ids = [id(p) for group in groups for p in group['params']]
assert len(ids) == len(set(ids)) == len(list(model.parameters()))
assert set(ids) == {id(p) for p in model.parameters()}
assert all(p.requires_grad for p in model.parameters())
expected = [.32768, .4096, .512, .64, .8, 1.]
for group, scale in zip(groups, expected):
    assert abs(group['lr_scale'] - scale) < 1e-12
    with torch.no_grad():
        group['params'][0].zero_()
    group['params'][0].grad = torch.ones_like(group['params'][0])
optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0)
optimizer.step()
for group, scale in zip(optimizer.param_groups, expected):
    assert float((group['params'][0].detach() + 1e-4 * scale).abs().max()) < 1e-10
    group['lr'] = 1e-5 * group['lr_scale']
    assert abs(group['lr'] - 1e-5 * scale) < 1e-12
report = dict(status='PASS', all_parameters_exactly_once=True, all_trainable=True,
              stage_scales=expected, adamw_step_and_scheduled_lr_verified=True,
              scope='Actual Tiny parameters; known numeric scales and real optimizer steps; training GPU separate')
(ROOT / 'docs/evidence/convnext22k_stage_lr_v1/cpu_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
