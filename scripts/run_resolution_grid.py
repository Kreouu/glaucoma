"""Compare 256/384 at positive weight 3 across seeds 42/43/44."""

import argparse
import csv
import fcntl
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_bce_grid import execute, inspect_run, sha256, write_json


BASELINES = {42: 'outputs/baseline_w3_20261028_010327',
             43: 'outputs/bce_grid_1_3_5_v1/w3_seed43',
             44: 'outputs/bce_grid_1_3_5_v1/w3_seed44'}


def summarize(batch, rows):
    with (batch / 'experiments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for size in (256, 384):
        selected = [r for r in rows if r['size'] == size]
        group = dict(size=size, completed_seeds=len(selected), planned_seeds=3)
        for key in ('val_auroc', 'val_auprc', 'training_seconds'):
            values = [r[key] for r in selected]
            group[key + '_mean'] = statistics.mean(values) if values else None
            group[key + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(group)
    with (batch / 'group_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    by_key = {(r['size'], r['seed']): r for r in rows}
    with (batch / 'paired_deltas.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'auroc_384_minus_256', 'auprc_384_minus_256'])
        for seed in (42, 43, 44):
            if (256, seed) in by_key and (384, seed) in by_key:
                old, new = by_key[256, seed], by_key[384, seed]
                writer.writerow([seed, new['val_auroc'] - old['val_auroc'], new['val_auprc'] - old['val_auprc']])
    write_json(batch / 'progress.json', dict(completed=len(rows), total=6, groups=groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    batch = ROOT / 'outputs/resolution_256_384_w3_v1'
    reference = json.loads((ROOT / BASELINES[42] / 'config.json').read_text())
    keys = ('epochs', 'batch_size', 'workers', 'jpeg_draft', 'lr', 'model', 'loss', 'optimizer',
            'weight_decay', 'amp', 'age_used', 'threshold', 'augmentation', 'preprocessing',
            'train_count', 'valid_count', 'train_limit', 'valid_limit', 'torch_version', 'torchvision_version')
    common = {k: reference[k] for k in keys}
    common['pos_weight'] = 3.0
    assert common['epochs'] == 5 and common['batch_size'] == 16 and common['workers'] == 4
    data = Path('datasets/glaucoma_first')
    pretrained = Path('pretrain_model/efficientnet_b0_rwightman-7f5810bc.pth')
    common['pretrained_sha256'] = sha256(pretrained)
    common['csv_sha256'] = {s: sha256(data / f'{s}.csv') for s in ('train', 'valid')}
    names = ('train.py', 'dataset.py', 'model.py', 'validate.py', 'scripts/run_bce_grid.py', 'scripts/run_resolution_grid.py')
    sources = {name: sha256(ROOT / name) for name in names}
    plan = dict(sizes=[256, 384], seeds=[42, 43, 44], common=common, source_sha256=sources)
    if (batch / 'plan.json').exists() and json.loads((batch / 'plan.json').read_text()) != plan:
        raise ValueError('Inputs/code differ from saved resolution plan')
    queue = []
    for size in (256, 384):
        for seed in (42, 43, 44):
            path = ROOT / BASELINES[seed] if size == 256 else batch / f'r384_seed{seed}'
            expected = dict(common, size=size, seed=seed)
            result = inspect_run(path, expected, sources, {sources['train.py']})
            if size == 256 and (not result or not result['verified']):
                raise ValueError(f'256 reference must already be verified: {path}')
            action = 'SKIP_VERIFIED' if result and result['verified'] else 'VALIDATE' if result else 'TRAIN'
            print(f'{action}: size={size} seed={seed} {path.relative_to(ROOT)}', flush=True)
            queue.append((path, expected, result))
    print(f'Plan: 6 results, {sum(r is None for _, _, r in queue)} new training runs.', flush=True)
    if args.dry_run:
        return

    def check_inputs():
        if any(sha256(ROOT / n) != d for n, d in sources.items()):
            raise RuntimeError('Source changed during resolution study')
        if sha256(pretrained) != common['pretrained_sha256'] or any(sha256(data / f'{s}.csv') != common['csv_sha256'][s] for s in ('train', 'valid')):
            raise RuntimeError('Data/weights changed during resolution study')

    batch.mkdir(parents=True, exist_ok=True)
    with (batch / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Resolution study is already running') from None
        write_json(batch / 'plan.json', plan)
        rows, current = [], None
        try:
            for path, expected, result in queue:
                current = str(path.relative_to(ROOT))
                check_inputs()
                if result is None:
                    if path.exists():
                        backup = path.with_name(path.name + '.interrupted_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                        path.rename(backup)
                        print(f'Preserved partial run: {backup}', flush=True)
                    execute([sys.executable, '-u', 'train.py', '--data-root', str(data), '--output-dir', str(path),
                             '--pretrained-path', str(pretrained), '--epochs', '5', '--batch-size', '16',
                             '--size', '384', '--workers', '4', '--lr', '0.0001', '--jpeg-draft',
                             '--seed', str(expected['seed']), '--pos-weight', '3'], batch / (path.name + '.train.log'))
                    result = inspect_run(path, expected, sources, {sources['train.py']})
                    if result is None:
                        raise RuntimeError(f'Incomplete training: {path}')
                if not result['verified']:
                    execute([sys.executable, '-u', 'validate.py', '--run-dir', str(path)], batch / (path.name + '.validate.log'))
                    result = inspect_run(path, expected, sources, {sources['train.py']})
                    if not result or not result['verified']:
                        raise RuntimeError(f'Validation failed: {path}')
                check_inputs()
                rows.append(dict(size=expected['size'], **result))
                summarize(batch, rows)
                print(f'Completed {len(rows)}/6 results', flush=True)
        except BaseException as error:
            write_json(batch / 'status.json', dict(status='STOPPED', experiment=current, error=f'{type(error).__name__}: {error}'))
            raise
        write_json(batch / 'status.json', dict(status='COMPLETE', results=6))
        print(f'COMPLETE: {batch}', flush=True)


if __name__ == '__main__':
    main()
