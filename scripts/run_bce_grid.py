"""Fixed 3 weights x 3 seeds experiment queue; never reads the test split."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (1, 3, 5)
SEEDS = (42, 43, 44)
LEGACY = {(1, 42): 'outputs/baseline_20261027_234436',
          (3, 42): 'outputs/baseline_w3_20261028_010327'}
LEGACY_TRAIN_SHA = 'df372b13250b7779d2018b46d5fccf4e635946cb5342ad8b36853de0416f8be9'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def inspect_run(path, expected, sources, allowed_train_hashes):
    if not (path / 'config.json').exists():
        return None
    config = json.loads((path / 'config.json').read_text())
    for key, value in expected.items():
        actual = config.get(key, 1.0 if key == 'pos_weight' else None)
        if actual != value:
            raise ValueError(f'{path}: config mismatch {key}: {actual!r} != {value!r}')
    for name in ('dataset.py', 'model.py'):
        if config['source_sha256'][name] != sources[name]:
            raise ValueError(f'{path}: source mismatch {name}')
    if config['source_sha256']['train.py'] not in allowed_train_hashes:
        raise ValueError(f'{path}: unrecognized training implementation')
    metrics_file = path / 'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_file.read_text().splitlines()] if metrics_file.exists() else []
    if [row['epoch'] for row in metrics] != list(range(1, len(metrics) + 1)) or len(metrics) > expected['epochs']:
        raise ValueError(f'{path}: invalid epoch sequence')
    for row in metrics:
        if not all(math.isfinite(row[k]) for k in ('train_loss', 'valid_loss', 'val_auroc', 'val_auprc')):
            raise ValueError(f'{path}: non-finite metric')
    if len(metrics) != expected['epochs'] or not (path / 'best.sha256').exists():
        return None
    for name in ('best_valid_predictions.csv', 'train_manifest.csv', 'valid_manifest.csv'):
        if not (path / name).is_file():
            raise ValueError(f'{path}: missing artifact {name}')
    digest = sha256(path / 'best.pt')
    if digest != (path / 'best.sha256').read_text().split()[0]:
        raise ValueError(f'{path}: checkpoint hash mismatch')
    best = max(metrics, key=lambda row: row['val_auroc'])
    verified = False
    report_path = path / 'reload_check.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        verified = (report['status'] == 'PASS' and report['checkpoint_sha256'] == digest
                    and report['checkpoint_epoch'] == best['epoch']
                    and report['images'] == expected['valid_count']
                    and report['max_probability_difference'] <= 1e-6
                    and all(abs(report[k] - best[k]) <= 1e-10 for k in ('val_auroc', 'val_auprc')))
        if not verified:
            raise ValueError(f'{path}: validation report mismatch')
    return dict(weight=expected['pos_weight'], seed=expected['seed'], run_dir=str(path.relative_to(ROOT)),
                best_epoch=best['epoch'], val_auroc=best['val_auroc'], val_auprc=best['val_auprc'],
                training_seconds=sum(row['epoch_seconds'] for row in metrics),
                checkpoint_sha256=digest, verified=verified)


def execute(command, log_path):
    print('RUN ' + ' '.join(command), flush=True)
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f'Command failed; inspect {log_path}')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
            raise
        finally:
            process.stdout.close()


def summarize(batch, rows):
    fields = ['weight', 'seed', 'run_dir', 'best_epoch', 'val_auroc', 'val_auprc',
              'training_seconds', 'checkpoint_sha256', 'verified']
    with (batch / 'experiments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for weight in WEIGHTS:
        selected = [r for r in rows if r['weight'] == weight]
        group = dict(weight=weight, completed_seeds=len(selected), planned_seeds=3)
        for key in ('val_auroc', 'val_auprc'):
            values = [r[key] for r in selected]
            group[key + '_mean'] = statistics.mean(values) if values else None
            group[key + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(group)
    with (batch / 'group_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    write_json(batch / 'progress.json', dict(completed=len(rows), total=9, groups=groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Check matching results and print queue without training')
    args = parser.parse_args()
    os.chdir(ROOT)
    data = Path('datasets/glaucoma_first')
    pretrained = Path('pretrain_model/efficientnet_b0_rwightman-7f5810bc.pth')
    batch = ROOT / 'outputs/bce_grid_1_3_5_v1'
    names = ('train.py', 'dataset.py', 'model.py', 'validate.py', 'scripts/run_bce_grid.py')
    sources = {name: sha256(ROOT / name) for name in names}
    common = dict(epochs=5, batch_size=16, size=256, workers=4, jpeg_draft=True, lr=1e-4,
                  model='efficientnet_b0', loss='BCEWithLogitsLoss', optimizer='AdamW',
                  weight_decay=0.01, amp=True, age_used=False, threshold=0.5,
                  augmentation='training horizontal flip p=0.5 only',
                  preprocessing='RGB, aspect-preserving letterbox, ImageNet normalization',
                  train_count=51396, valid_count=5849, train_limit=None, valid_limit=None,
                  torch_version='2.10.0+cu128', torchvision_version='0.25.0+cu128',
                  pretrained_sha256=sha256(pretrained),
                  csv_sha256={s: sha256(data / f'{s}.csv') for s in ('train', 'valid')})
    plan = dict(weights=WEIGHTS, seeds=SEEDS, common=common, source_sha256=sources)
    # JSON normalizes tuples to lists for subsequent comparisons.
    plan = json.loads(json.dumps(plan))
    plan_file = batch / 'plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError('Batch inputs/code differ from saved plan; use the original code/data or a new batch version')
    queue = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            expected = dict(common, pos_weight=weight, seed=seed)
            path = ROOT / LEGACY[(weight, seed)] if (weight, seed) in LEGACY else batch / f'w{weight}_seed{seed}'
            allowed = {sources['train.py']}
            if (weight, seed) == (1, 42):
                allowed.add(LEGACY_TRAIN_SHA)
            result = inspect_run(path, expected, sources, allowed)
            if (weight, seed) in LEGACY and result is None:
                raise ValueError(f'Expected historical baseline is incomplete: {path}')
            action = 'SKIP_VERIFIED' if result and result['verified'] else 'VALIDATE' if result else 'TRAIN'
            print(f'{action}: weight={weight} seed={seed} {path.relative_to(ROOT)}', flush=True)
            queue.append((path, expected, allowed, result))
    print(f'Plan: 9 experiments, {sum(r is None for _, _, _, r in queue)} need training.', flush=True)
    if args.dry_run:
        return

    def check_inputs():
        if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
            raise RuntimeError('Source code changed during batch')
        if sha256(pretrained) != common['pretrained_sha256'] or any(sha256(data / f'{s}.csv') != common['csv_sha256'][s] for s in ('train', 'valid')):
            raise RuntimeError('Data CSV or pretrained weights changed during batch')

    batch.mkdir(parents=True, exist_ok=True)
    with (batch / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This batch is already running') from None
        write_json(plan_file, plan)
        rows = []
        current = None
        try:
            for path, expected, allowed, result in queue:
                current = str(path.relative_to(ROOT))
                check_inputs()
                if result is None:
                    if path.exists():
                        backup = path.with_name(path.name + '.interrupted_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                        path.rename(backup)
                        print(f'Preserved partial run: {backup}', flush=True)
                    execute([sys.executable, '-u', 'train.py', '--data-root', str(data),
                             '--output-dir', str(path), '--pretrained-path', str(pretrained),
                             '--epochs', '5', '--batch-size', '16', '--size', '256', '--workers', '4',
                             '--lr', '0.0001', '--jpeg-draft', '--seed', str(expected['seed']),
                             '--pos-weight', str(expected['pos_weight'])], batch / (path.name + '.train.log'))
                    result = inspect_run(path, expected, sources, allowed)
                    if result is None:
                        raise RuntimeError(f'Training did not produce a complete run: {path}')
                if not result['verified']:
                    execute([sys.executable, '-u', 'validate.py', '--run-dir', str(path)], batch / (path.name + '.validate.log'))
                    result = inspect_run(path, expected, sources, allowed)
                    if not result or not result['verified']:
                        raise RuntimeError(f'Validation failed: {path}')
                check_inputs()
                rows.append(result)
                summarize(batch, rows)
                print(f'Completed {len(rows)}/9: weight={expected["pos_weight"]} seed={expected["seed"]}', flush=True)
        except BaseException as error:
            write_json(batch / 'status.json', dict(status='STOPPED', experiment=current,
                                                   error=f'{type(error).__name__}: {error}'))
            raise
        write_json(batch / 'status.json', dict(status='COMPLETE', experiments=9))
        print(f'COMPLETE: summary in {batch}; no test predictions generated.', flush=True)


if __name__ == '__main__':
    main()
