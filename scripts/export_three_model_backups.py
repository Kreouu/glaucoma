"""Preserve all six new best models without optimizer state, checking exact tensor equality."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import file_sha256
from scripts.audit_three_models import GROUPS


def main():
    output = ROOT / 'outputs/three_model_comparison_v1/model_backups'
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for family in ('inceptionnext', 'fastervit'):
        for seed in (42, 43, 44):
            original = ROOT / 'outputs' / GROUPS[family] / f'seed{seed}/best.pt'
            saved = torch.load(original, map_location='cpu', weights_only=True)
            backup = output / f'{family}_seed{seed}.pt'
            compact = {key: saved[key] for key in ('model', 'config', 'metrics')}
            torch.save(compact, backup)
            restored = torch.load(backup, map_location='cpu', weights_only=True)
            assert restored['config'] == saved['config'] and restored['metrics'] == saved['metrics']
            assert restored['model'].keys() == saved['model'].keys()
            assert all(torch.equal(t, restored['model'][name]) for name, t in saved['model'].items())
            records.append(dict(family=family, seed=seed, original=str(original.relative_to(ROOT)),
                original_sha256=file_sha256(original), backup=backup.name, backup_sha256=file_sha256(backup),
                bytes=backup.stat().st_size, tensor_equality='PASS',
                note='Exact model/config/metrics backup; omits optimizer/scaler, not a resume checkpoint'))
            del saved, compact, restored
    (output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
