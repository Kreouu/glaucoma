"""Audit complete paired 512 model runs from checkpoints and saved validation predictions."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'convnext': 'convnext_512_w3_v1', 'inceptionnext': 'inceptionnext_512_w3_v1',
          'fastervit': 'fastervit_512_w3_v1'}
MODELS = {'convnext': 'convnext_tiny', 'inceptionnext': 'inception_next_tiny', 'fastervit': 'fastervit_1'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=GROUPS, default=list(GROUPS))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    reference_manifest = None
    for family in args.models:
        rows = []
        for seed in (42, 43, 44):
            run = ROOT / 'outputs' / GROUPS[family] / f'seed{seed}'
            config = json.loads((run / 'config.json').read_text())
            expected = dict(model=MODELS[family], seed=seed, size=512, epochs=5, batch_size=16,
                            workers=4, lr=3e-5, pos_weight=3, precision='fp16', lr_schedule='constant',
                            train_count=51396, valid_count=5849, train_limit=None, valid_limit=None)
            for key, value in expected.items():
                assert config[key] == value, (run, key, config[key], value)
            for split, digest in config['csv_sha256'].items():
                assert sha(ROOT / config['data_root'] / f'{split}.csv') == digest
            assert sha(ROOT / config['pretrained_path']) == config['pretrained_sha256']
            sources = {}
            for name, digest in config['source_sha256'].items():
                local = ROOT / name
                if local.is_file() and sha(local) == digest:
                    sources[name] = 'current_file'
                else:
                    source = subprocess.run(['git', 'show', f"{config['git_revision']}:{name}"],
                                            cwd=ROOT, check=True, capture_output=True).stdout
                    assert hashlib.sha256(source).hexdigest() == digest, (run, name)
                    sources[name] = config['git_revision']
            metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            assert [m['epoch'] for m in metrics] == [1, 2, 3, 4, 5]
            assert all(math.isfinite(m[k]) for m in metrics
                       for k in ('train_loss', 'valid_loss', 'val_auroc', 'val_auprc'))
            best = max(metrics, key=lambda m: m['val_auroc'])
            digest = sha(run / 'best.pt')
            assert digest == (run / 'best.sha256').read_text().split()[0]
            checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            assert checkpoint['config'] == config and checkpoint['metrics'] == best
            assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
            model_bytes = sum(t.numel() * t.element_size() for t in checkpoint['model'].values())
            reload = json.loads((run / 'reload_check.json').read_text())
            assert reload['status'] == 'PASS' and reload['checkpoint_sha256'] == digest
            assert reload['images'] == 5849 and reload['positives'] == 191
            assert reload['checkpoint_epoch'] == best['epoch'] and reload['max_probability_difference'] <= 1e-6
            with (run / 'best_valid_predictions.csv').open() as stream:
                predictions = list(csv.DictReader(stream))
            labels = [int(r['True_Label']) for r in predictions]
            probabilities = [float(r['Probability']) for r in predictions]
            names = [r['Image_name'] for r in predictions]
            assert len(names) == len(set(names)) == 5849 and sum(labels) == 191
            with (run / 'valid_manifest.csv').open() as stream:
                manifest = list(csv.DictReader(stream))
            assert [(r['Image_name'], int(r['True_Label'])) for r in manifest] == list(zip(names, labels))
            if reference_manifest is None:
                reference_manifest = manifest
            assert manifest == reference_manifest, run
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            auroc = float(roc_auc_score(labels, probabilities))
            ap = float(average_precision_score(labels, probabilities))
            assert auroc == best['val_auroc'] == reload['val_auroc']
            assert ap == best['val_auprc'] == reload['val_auprc']
            rows.append(dict(seed=seed, run_dir=str(run.relative_to(ROOT)), best_epoch=best['epoch'],
                             val_auroc=auroc, val_auprc=ap, checkpoint_sha256=digest,
                             training_seconds=sum(m['epoch_seconds'] for m in metrics),
                             peak_vram_mib=max(m['peak_vram_mib'] for m in metrics),
                             model_tensor_bytes=model_bytes, source_verification=sources,
                             reload=reload, metrics=metrics))
            del checkpoint
            print(f'{family} seed{seed} PASS AUROC={auroc:.9f} AP={ap:.9f}', flush=True)
        results[family] = dict(seeds=rows, auroc_mean=statistics.mean(r['val_auroc'] for r in rows),
                               auroc_std=statistics.stdev(r['val_auroc'] for r in rows),
                               ap_mean=statistics.mean(r['val_auprc'] for r in rows),
                               training_seconds_mean=statistics.mean(r['training_seconds'] for r in rows))
    if 'convnext' in results:
        for family, group in results.items():
            deltas = [r['val_auroc'] - b['val_auroc'] for r, b in
                      zip(group['seeds'], results['convnext']['seeds'])]
            group['paired_auroc_deltas_vs_convnext'] = deltas
            group['paired_auroc_mean_delta_vs_convnext'] = statistics.mean(deltas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PASS', scope='Fixed recipe, three paired seeds; not model ceiling',
                                           groups=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
