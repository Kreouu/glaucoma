"""Reload a saved baseline and verify its validation predictions and provenance."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import FundusDataset
from model import build_model
from train import print_progress, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    actual_hash = sha256(run / 'best.pt')
    if actual_hash != (run / 'best.sha256').read_text().split()[0]:
        raise ValueError('Checkpoint SHA-256 mismatch')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config != json.loads((run / 'config.json').read_text()):
        raise ValueError('Checkpoint config differs from config.json')
    project = Path(__file__).resolve().parent
    sources = ['dataset.py', 'model.py']
    if config['model'] == 'efficientnet_b3':
        sources.append('model_b3.py')
    if config['model'] == 'swinv2_tiny':
        import timm
        if timm.__version__ != config['timm_version']:
            raise ValueError('timm version differs from training')
        sources.append('model_swinv2.py')
    for name in sources:
        if sha256(project / name) != config['source_sha256'][name]:
            raise ValueError(f'Inference source changed: {name}')
    root = Path(config['data_root'])
    for split in ('train', 'valid'):
        if sha256(root / f'{split}.csv') != config['csv_sha256'][split]:
            raise ValueError(f'CSV changed: {split}')
    data = FundusDataset(root, 'valid', config['size'], config['valid_limit'], config['seed'], config['jpeg_draft'])
    with (run / 'valid_manifest.csv').open(newline='') as stream:
        manifest = list(csv.DictReader(stream))
    observed = [(name, path.name, str(label)) for name, path, label in data.samples]
    expected = [(r['Image_name'], r['actual_filename'], r['True_Label']) for r in manifest]
    if observed != expected:
        raise ValueError('Validation sample manifest changed')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config['model'] == 'efficientnet_b3':
        from model_b3 import build_model as selected_builder
    else:
        selected_builder = build_model
    if config['model'] == 'swinv2_tiny':
        from model_swinv2 import build_model as swin_builder
        model = swin_builder(size=config['size'])
    else:
        model = selected_builder()
    model.load_state_dict(checkpoint['model'])
    model = model.cuda().eval()
    loader = DataLoader(data, batch_size=config['batch_size'], num_workers=config['workers'], pin_memory=True)
    names, labels, probabilities = [], [], []
    loss_total = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (images, target, batch_names) in enumerate(loader, 1):
            images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(images).flatten()
                loss = criterion(logits, target)
            loss_total += loss.item() * len(target)
            names.extend(batch_names)
            labels.extend(int(t) for t in target.cpu().tolist())
            probabilities.extend(logits.float().sigmoid().cpu().tolist())
            print_progress('Reload', checkpoint['metrics']['epoch'], step, len(loader), len(names), loss_total, started)
    with (run / 'best_valid_predictions.csv').open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    if names != [r['Image_name'] for r in saved] or labels != [int(r['True_Label']) for r in saved]:
        raise ValueError('Saved validation prediction names/labels differ')
    expected_probabilities = np.array([float(r['Probability']) for r in saved])
    if not np.isfinite(probabilities).all():
        raise ValueError('Non-finite probabilities')
    difference = float(np.max(np.abs(np.array(probabilities) - expected_probabilities)))
    if difference > 1e-6:
        raise ValueError(f'Validation probability difference exceeds tolerance: {difference}')
    metrics = dict(val_auroc=float(roc_auc_score(labels, probabilities)),
                   val_auprc=float(average_precision_score(labels, probabilities)))
    for name, value in metrics.items():
        if abs(value - checkpoint['metrics'][name]) > 1e-10:
            raise ValueError(f'Validation metric mismatch: {name}')
    result = dict(status='PASS', images=len(data), positives=sum(labels),
                  checkpoint_epoch=checkpoint['metrics']['epoch'], checkpoint_sha256=actual_hash,
                  max_probability_difference=difference, **metrics)
    (run / 'reload_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
