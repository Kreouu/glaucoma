"""Independent post-run audit of search coverage, scores, selection and final artifacts."""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_bce_grid import sha256, write_json
from scripts.run_model_selection import numerical_failure_reason


def main():
    root = ROOT / 'outputs/model_selection_20261029_v1'
    status = json.loads((root / 'status.json').read_text())
    if status['status'] != 'COMPLETE':
        raise RuntimeError('Selection is not complete; leave the running job alone')
    plan = json.loads((root / 'plan.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    experiments = json.loads((root / 'experiments.json').read_text())
    groups = json.loads((root / 'groups.json').read_text())
    for name, digest in plan['sources'].items():
        assert sha256(ROOT / name) == digest, name
    for split, digest in plan['csv_sha256'].items():
        assert sha256(ROOT / 'datasets/glaucoma_first' / f'{split}.csv') == digest, split
    expected_screen = {(m, lr, w) for m in plan['models'] for lr in plan['learning_rates']
                       for w in plan['positive_weights']}
    screen = {(r['model'], r['lr'], r['weight']) for r in experiments if r['seed'] == 43 and r['epochs'] == 5}
    assert screen == expected_screen and len(screen) == 16
    checked = {}
    for row in experiments:
        path = ROOT / row['run_dir']
        config = json.loads((path / 'config.json').read_text())
        for key, value in dict(model=row['model'], lr=row['lr'], pos_weight=float(row['weight']),
                               seed=row['seed'], epochs=row['epochs'], size=384, batch_size=16,
                               workers=4, train_limit=None, valid_limit=None).items():
            assert config[key] == value, (path, key)
        assert config.get('precision', 'fp16') == 'fp16'
        assert config.get('lr_schedule', 'constant') == 'constant'
        assert config['csv_sha256'] == plan['csv_sha256']
        assert config['pretrained_sha256'] == plan['pretrained_sha256'][row['model']]
        if row.get('rejected'):
            assert sha256(path / 'config.json') == row['config_sha256']
            log = ROOT / row['log_path']
            assert sha256(log) == row['log_sha256']
            assert numerical_failure_reason(log.read_text()) == row['reason']
            assert not row['verified']
            continue
        metrics = [json.loads(x) for x in (path / 'metrics.jsonl').read_text().splitlines()]
        assert [m['epoch'] for m in metrics] == list(range(1, row['epochs'] + 1))
        assert all(math.isfinite(m[k]) for m in metrics for k in ('train_loss', 'valid_loss', 'val_auroc', 'val_auprc'))
        best = max(metrics, key=lambda m: m['val_auroc'])
        digest = sha256(path / 'best.pt')
        assert digest == row['checkpoint_sha256'] == (path / 'best.sha256').read_text().split()[0]
        checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['config'] == config and checkpoint['metrics'] == best
        assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
        predictions = list(csv.DictReader((path / 'best_valid_predictions.csv').open()))
        labels = [int(r['True_Label']) for r in predictions]
        probabilities = [float(r['Probability']) for r in predictions]
        assert len(predictions) == 5849 and sum(labels) == 191
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        assert roc_auc_score(labels, probabilities) == row['val_auroc'] == best['val_auroc']
        assert average_precision_score(labels, probabilities) == row['val_auprc'] == best['val_auprc']
        reload = json.loads((path / 'reload_check.json').read_text())
        assert reload['status'] == 'PASS' and reload['checkpoint_sha256'] == digest
        assert reload['checkpoint_epoch'] == best['epoch'] and reload['images'] == 5849
        assert reload['max_probability_difference'] <= 1e-6
        checked[row['run_dir']] = digest
        del checkpoint
    assert {g['model'] for g in groups} == set(plan['models']) and len(groups) == 4
    reference_b0 = next(r for r in experiments if r['model'] == 'efficientnet_b0'
                        and r['seed'] == 43 and r['lr'] == .0001 and r['weight'] == 3 and r['epochs'] == 5)
    for group in groups:
        if 'reason' in group:
            assert not group['eligible']
            rejected = [r for r in experiments if r['model'] == group['model'] and r.get('rejected')]
            assert rejected
            continue
        best_screen = max((r for r in experiments if r['model'] == group['model']
                           and r['seed'] == 43 and r['epochs'] == 5 and not r.get('rejected')),
                          key=lambda r: (r['val_auroc'], r['val_auprc']))
        expected_epochs = (10 if best_screen['best_epoch'] == 5
                           and best_screen['val_auroc'] >= reference_b0['val_auroc'] - .005 else 5)
        assert group['lr'] == best_screen['lr'] and group['weight'] == best_screen['weight']
        assert group['epochs'] == expected_epochs
        runs = group['seeds']
        assert sorted(r['seed'] for r in runs) == [42, 43, 44]
        assert all(r['run_dir'] in checked for r in runs)
        assert all(r['model'] == group['model'] and r['lr'] == group['lr']
                   and r['weight'] == group['weight'] and r['epochs'] == group['epochs'] for r in runs)
        for metric in ('val_auroc', 'val_auprc'):
            values = [r[metric] for r in runs]
            assert group[metric + '_mean'] == statistics.mean(values)
            assert group[metric + '_std'] == statistics.stdev(values)
        timing_report = json.loads((root / 'timing' / group['model'] / 'cost/report.json').read_text())
        seed43 = next(r for r in runs if r['seed'] == 43)
        assert timing_report['checkpoint_sha256'] == checked[seed43['run_dir']]
        assert group['mean_ms'] == timing_report['mean_ms']
        assert group['eligible'] == (timing_report['consistency_pass'] and timing_report['mean_ms'] <= 100)
    winner = max((g for g in groups if g['eligible']),
                 key=lambda g: (g['val_auroc_mean'], g['val_auprc_mean'], -g['mean_ms']))
    assert winner == selection['winning_group'] and winner['model'] == selection['winning_model']
    parent_row = max(winner['seeds'], key=lambda r: (r['val_auroc'], r['val_auprc']))
    assert parent_row == selection['selected_run']
    final = root / 'final'
    original = torch.load(ROOT / parent_row['run_dir'] / 'best.pt', map_location='cpu', weights_only=True)
    exported = torch.load(final / 'best.pt', map_location='cpu', weights_only=True)
    assert set(exported) == {'model', 'config', 'metrics'}
    assert original['config'] == exported['config'] and original['metrics'] == exported['metrics']
    assert original['model'].keys() == exported['model'].keys()
    assert all(torch.equal(t, exported['model'][n]) for n, t in original['model'].items())
    assert (final / 'best.pt').stat().st_size <= 500_000_000
    assert sha256(final / 'best.pt') == selection['exported_sha256']
    for name, digest in plan['sources'].items():
        assert sha256(final / 'source' / name) == digest, name
    replay = json.loads((final / 'reload_check.json').read_text())
    assert replay['status'] == 'PASS' and replay['max_probability_difference'] <= 1e-6
    assert replay['checkpoint_sha256'] == selection['exported_sha256']
    timing = json.loads((root / 'timing/final_export/cost/report.json').read_text())
    assert timing == selection['final_timing'] and timing['consistency_pass']
    assert timing['mean_ms'] <= 100 and timing['checkpoint_sha256'] == selection['exported_sha256']
    with (ROOT / 'datasets/glaucoma_first/test.csv').open() as stream:
        expected_names = [r['Image_name'] for r in csv.DictReader(stream)]
    submission = list(csv.DictReader((final / 'submission.csv').open()))
    assert len(submission) == len(expected_names) == 5855
    assert [r['Image_name'] for r in submission] == expected_names
    assert len({r['Image_name'].casefold() for r in submission}) == 5855
    for row in submission:
        p = float(row['Probability'])
        assert math.isfinite(p) and 0 <= p <= 1 and int(row['Pred_Label']) == int(p >= .5)
    manifest = json.loads((final / 'submission.manifest.json').read_text())
    assert sha256(final / 'submission.csv') == manifest['submission_sha256'] == selection['submission_sha256']
    assert manifest['checkpoint_sha256'] == selection['exported_sha256']
    assert manifest['input_csv_sha256'] == sha256(ROOT / 'datasets/glaucoma_first/test.csv')
    for name, digest in manifest['source_sha256'].items():
        assert sha256(final / 'source' / name) == digest, name
    result = dict(status='PASS', screening_configurations=16, verified_runs=sum(not r.get('rejected') for r in experiments),
                  rejected_runs=sum(bool(r.get('rejected')) for r in experiments),
                  confirmed_families=sum('reason' not in g for g in groups), seeds_per_family=3, selected_model=winner['model'],
                  exported_model_tensor_equality=True, full_validation_reload=True,
                  local_timing_and_consistency=True, test_rows=5855,
                  exported_checkpoint_sha256=selection['exported_sha256'],
                  submission_sha256=selection['submission_sha256'],
                  official_T4_check='not performed; local selection does not certify official hardware')
    write_json(root / 'independent_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
