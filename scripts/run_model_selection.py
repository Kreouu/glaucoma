"""Bounded four-backbone search and three-seed confirmation; no platform upload."""

import argparse
import csv
import fcntl
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_bce_grid import execute, inspect_run, sha256, write_json

MODELS = {
    'convnext_tiny': ('convnext_tiny_fb_in1k.bin', 'model_convnext.py'),
    'efficientnet_b0': ('efficientnet_b0_rwightman-7f5810bc.pth', 'model.py'),
    'efficientnet_b3': ('efficientnet_b3_rwightman-b3899882.pth', 'model_b3.py'),
    'swinv2_tiny': ('swinv2_tiny_window8_256_ms_in1k.bin', 'model_swinv2.py'),
}
LEGACY = {
    ('efficientnet_b0', .0001, 3, seed): f'outputs/resolution_256_384_w3_v1/r384_seed{seed}'
    for seed in (42, 43, 44)
}
LEGACY.update({
    ('efficientnet_b3', .0001, 3, 43): 'outputs/b3_384_w3_seed43_v1',
    ('swinv2_tiny', .0001, 3, 43): 'outputs/swinv2_384_w3_seed43_v1/lr_0.0001',
    ('swinv2_tiny', .00003, 3, 43): 'outputs/swinv2_384_w3_seed43_v1/lr_0.00003',
    ('swinv2_tiny', .00003, 1, 43): 'outputs/swinv2_384_w1_seed43_v1',
})


def rank(row):
    return row['val_auroc'], row['val_auprc']


def extend_training(row, baseline_auroc):
    return row['best_epoch'] == 5 and row['val_auroc'] >= baseline_auroc - .005


def group_summary(rows):
    return dict(val_auroc_mean=statistics.mean(r['val_auroc'] for r in rows),
                val_auroc_std=statistics.stdev(r['val_auroc'] for r in rows),
                val_auprc_mean=statistics.mean(r['val_auprc'] for r in rows),
                val_auprc_std=statistics.stdev(r['val_auprc'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    batch = ROOT / 'outputs/model_selection_20261029_v1'
    data = Path('datasets/glaucoma_first')
    names = ['train.py', 'validate.py', 'predict.py', 'cost.py', 'inference.py', 'dataset.py',
             'model.py', 'model_b3.py', 'model_swinv2.py', 'model_convnext.py', 'amp_precision.py',
             'lr_schedule.py', 'scripts/run_bce_grid.py', 'scripts/run_model_selection.py']
    sources = {n: sha256(ROOT / n) for n in names}
    weights = {m: sha256(ROOT / 'pretrain_model' / spec[0]) for m, spec in MODELS.items()}
    csv_hashes = {s: sha256(data / f'{s}.csv') for s in ('train', 'valid')}
    legacy_configs = {v: sha256(ROOT / v / 'config.json') for v in LEGACY.values()}
    plan = dict(models=list(MODELS), learning_rates=[.0001, .00003], positive_weights=[1, 3],
                screen_seed=43, confirmation_seeds=[42, 43, 44], epochs=5, size=384,
                late_peak_rule='best epoch 5 and AUROC >= original B0 seed43 minus .005: restart 10 epochs',
                selection_rule='highest three-seed mean validation AUROC; AP then local latency break ties',
                checkpoint_rule='highest validation AUROC among the winning family three seeds',
                test_use='one fixed final inference after selection; no platform upload',
                sources=sources, pretrained_sha256=weights, csv_sha256=csv_hashes,
                legacy_configs=legacy_configs)
    if (batch / 'plan.json').exists() and json.loads((batch / 'plan.json').read_text()) != plan:
        raise ValueError('Frozen selection plan differs; do not overwrite this batch')
    print('Plan: 16 seed43 screening results; 8 additional confirmation results; reuse verified histories.', flush=True)
    if args.dry_run:
        for model in MODELS:
            for lr in plan['learning_rates']:
                for weight in (1, 3):
                    print(model, lr, weight, LEGACY.get((model, lr, weight, 43), 'TRAIN'), flush=True)
        return
    batch.mkdir(parents=True, exist_ok=True)

    def guard():
        for n, h in sources.items():
            if sha256(ROOT / n) != h:
                raise RuntimeError(f'Frozen source changed: {n}')
        for model, h in weights.items():
            if sha256(ROOT / 'pretrain_model' / MODELS[model][0]) != h:
                raise RuntimeError(f'Original weights changed: {model}')
        for split, h in csv_hashes.items():
            if sha256(data / f'{split}.csv') != h:
                raise RuntimeError(f'Data CSV changed: {split}')

    current = None
    rows = []

    def run(model, lr, weight, seed, epochs=5):
        nonlocal current
        guard()
        key = (model, lr, weight, seed)
        legacy = LEGACY.get(key) if epochs == 5 else None
        path = ROOT / legacy if legacy else batch / f'{model}_lr{lr:g}_w{weight}_e{epochs}_s{seed}'
        current = str(path.relative_to(ROOT))
        write_json(batch / 'status.json', dict(status='RUNNING', pid=os.getpid(), current=current,
                                             completed=len(rows), updated_at=datetime.now().isoformat()))
        expected = dict(model=model, lr=lr, pos_weight=float(weight), seed=seed, epochs=epochs,
                        size=384, batch_size=16, workers=4, jpeg_draft=True, weight_decay=.01,
                        amp=True, train_limit=None, valid_limit=None, train_count=51396, valid_count=5849,
                        csv_sha256=csv_hashes, pretrained_sha256=weights[model])
        allowed = {sources['train.py']}
        if legacy:
            c = json.loads((path / 'config.json').read_text())
            archived = subprocess.check_output(['git', 'show', c['git_revision'] + ':train.py'])
            import hashlib
            if hashlib.sha256(archived).hexdigest() != c['source_sha256']['train.py']:
                raise ValueError('Historical training source not backed by recorded commit')
            allowed.add(c['source_sha256']['train.py'])
        result = inspect_run(path, expected, sources, allowed)
        if result is None:
            if legacy:
                raise ValueError(f'Historical run incomplete: {path}')
            if path.exists():
                path.rename(path.with_name(path.name + '.interrupted_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            execute([sys.executable, '-u', 'train.py', '--model', model, '--precision', 'fp16',
                     '--data-root', str(data), '--output-dir', str(path),
                     '--pretrained-path', str(Path('pretrain_model') / MODELS[model][0]),
                     '--epochs', str(epochs), '--batch-size', '16', '--size', '384', '--workers', '4',
                     '--jpeg-draft', '--lr', str(lr), '--lr-schedule', 'constant',
                     '--pos-weight', str(weight), '--seed', str(seed)], Path(str(path) + '.train.log'))
            result = inspect_run(path, expected, sources, allowed)
        if not result:
            raise RuntimeError(f'Incomplete experiment: {path}')
        c = json.loads((path / 'config.json').read_text())
        if c.get('precision', 'fp16') != 'fp16' or c.get('lr_schedule', 'constant') != 'constant':
            raise ValueError('Screening requires fixed FP16 configuration')
        if c['source_sha256'][MODELS[model][1]] != sources[MODELS[model][1]]:
            raise ValueError('Model implementation differs from frozen selection plan')
        if not result['verified']:
            execute([sys.executable, '-u', 'validate.py', '--run-dir', str(path)], Path(str(path) + '.reload.log'))
            result = inspect_run(path, expected, sources, allowed)
        if not result or not result['verified']:
            raise RuntimeError('Full validation reload did not pass')
        result.update(model=model, lr=lr, epochs=epochs)
        rows.append(result)
        write_json(batch / 'experiments.json', rows)
        print('VERIFIED ' + json.dumps(result), flush=True)
        return result

    def timing(row, label):
        guard()
        out = batch / 'timing' / label
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = ROOT / row['run_dir'] / 'best.pt'
        if not (out / 'reference.csv').exists():
            execute([sys.executable, 'predict.py', '--checkpoint', str(checkpoint),
                     '--input-csv', 'outputs/inference_validation_001/input.csv',
                     '--images-dir', 'outputs/inference_validation_001/pictures_100',
                     '--output', str(out / 'reference.csv')], out / 'predict.log')
        if not (out / 'cost/report.json').exists():
            try:
                execute([sys.executable, 'cost.py', str(out / 'reference.csv'),
                         'outputs/inference_validation_001/pictures_100', 'outputs/inference_validation_001/pictures_5',
                         '--checkpoint', str(checkpoint), '--output-dir', str(out / 'cost')], out / 'cost.log')
            except RuntimeError:
                if not (out / 'cost/report.json').exists():
                    raise
        report = json.loads((out / 'cost/report.json').read_text())
        if report['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Timing checkpoint mismatch')
        return report

    with (batch / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(batch / 'plan.json', plan)
        try:
            selected = {}
            # ConvNeXt first supplies the missing architecture evidence promptly.
            for model in MODELS:
                screen = [run(model, lr, weight, 43) for lr in plan['learning_rates'] for weight in (1, 3)]
                selected[model] = max(screen, key=rank)
            original_b0 = next(r for r in rows if r['model'] == 'efficientnet_b0' and r['lr'] == .0001 and r['weight'] == 3)
            groups = []
            for model, choice in selected.items():
                if extend_training(choice, original_b0['val_auroc']):
                    choice = run(model, choice['lr'], choice['weight'], 43, epochs=10)
                confirmation = [choice] + [run(model, choice['lr'], choice['weight'], seed, choice['epochs']) for seed in (42, 44)]
                t = timing(choice, model)
                group = dict(model=model, lr=choice['lr'], weight=choice['weight'], epochs=choice['epochs'],
                             **group_summary(confirmation), seeds=confirmation, mean_ms=t['mean_ms'],
                             eligible=t['consistency_pass'] and t['within_100ms_on_this_machine'])
                groups.append(group)
                write_json(batch / 'groups.json', groups)
            eligible = [g for g in groups if g['eligible']]
            if not eligible:
                raise RuntimeError('No model passes local deployment checks')
            winner = max(eligible, key=lambda g: (g['val_auroc_mean'], g['val_auprc_mean'], -g['mean_ms']))
            checkpoint_choice = max(winner['seeds'], key=rank)
            # Export a compact inference checkpoint without modifying the training original.
            import torch
            final = batch / 'final'
            final.mkdir(exist_ok=True)
            parent = ROOT / checkpoint_choice['run_dir']
            if not (final / 'best.pt').exists():
                saved = torch.load(parent / 'best.pt', map_location='cpu', weights_only=True)
                torch.save({k: saved[k] for k in ('model', 'config', 'metrics')}, final / 'best.pt')
                for n in ('config.json', 'train_manifest.csv', 'valid_manifest.csv', 'best_valid_predictions.csv', 'metrics.jsonl'):
                    shutil.copy2(parent / n, final / n)
                (final / 'best.sha256').write_text(sha256(final / 'best.pt') + '  best.pt\n')
            if (final / 'best.pt').stat().st_size > 500_000_000:
                raise ValueError('Final model exceeds model-size limit')
            execute([sys.executable, '-u', 'validate.py', '--run-dir', str(final)], final / 'reload.log')
            exported = dict(checkpoint_choice, run_dir=str(final.relative_to(ROOT)))
            ft = timing(exported, 'final_export')
            if not ft['consistency_pass'] or not ft['within_100ms_on_this_machine']:
                raise RuntimeError('Final exported model failed timing/consistency gate')
            # The test split is accessed only now, after all model decisions are fixed.
            if not (final / 'submission.csv').exists():
                execute([sys.executable, 'predict.py', '--checkpoint', str(final / 'best.pt'),
                         '--input-csv', str(data / 'test.csv'), '--images-dir', str(data / 'test'),
                         '--output', str(final / 'submission.csv')], final / 'predict.log')
            snapshot = final / 'source'
            snapshot.mkdir(exist_ok=True)
            for name in names:
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            shutil.copytree(ROOT / 'licenses', final / 'licenses', dirs_exist_ok=True)
            (final / 'environment-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
            selection = dict(status='SELECTED', groups=groups, winning_model=winner['model'],
                             winning_group=winner, selected_run=checkpoint_choice, exported_sha256=sha256(final / 'best.pt'),
                             submission_sha256=sha256(final / 'submission.csv'), final_timing=ft,
                             caveats=['Finite search, not proof of architecture upper bound',
                                      'Seed43 used for hyperparameter choice; three-seed estimates retain selection bias',
                                      'Local RTX5060Ti checks only; official T4/designated images still pending',
                                      'No leaderboard-based tuning and no platform upload'])
            write_json(batch / 'selection.json', selection)
            guard()
            write_json(batch / 'status.json', dict(status='COMPLETE', winning_model=winner['model'],
                                                  pid=os.getpid(), finished_at=datetime.now().isoformat()))
        except BaseException as error:
            write_json(batch / 'status.json', dict(status='STOPPED', pid=os.getpid(), current=current,
                                                  error=f'{type(error).__name__}: {error}', updated_at=datetime.now().isoformat()))
            raise


if __name__ == '__main__':
    main()
