import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_bce_grid as grid


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.expected = dict(epochs=2, seed=42, pos_weight=1, valid_count=4)
        self.sources = {'dataset.py': 'data', 'model.py': 'model'}
        self.config = dict(self.expected, source_sha256={**self.sources, 'train.py': 'train'})
        self.save('config.json', self.config)
        self.metrics = [dict(epoch=i, train_loss=.2, valid_loss=.1, val_auroc=.8 + .01 * i,
                             val_auprc=.7, epoch_seconds=1) for i in (1, 2)]
        self.write_metrics(self.metrics)
        (self.run / 'best.pt').write_bytes(b'checkpoint fixture')
        digest = grid.sha256(self.run / 'best.pt')
        (self.run / 'best.sha256').write_text(digest + '  best.pt\n')
        for name in ('best_valid_predictions.csv', 'train_manifest.csv', 'valid_manifest.csv'):
            (self.run / name).write_text('fixture\n')
        self.report = dict(status='PASS', checkpoint_sha256=digest, checkpoint_epoch=2,
                           images=4, max_probability_difference=0, val_auroc=.82, val_auprc=.7)
        self.save('reload_check.json', self.report)

    def save(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def write_metrics(self, rows):
        (self.run / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def inspect(self):
        with patch.object(grid, 'ROOT', self.root):
            return grid.inspect_run(self.run, self.expected, self.sources, {'train'})

    def test_completed_legacy_weight_one_can_be_reused(self):
        self.config.pop('pos_weight')
        self.save('config.json', self.config)
        self.assertTrue(self.inspect()['verified'])

    def test_other_weight_is_not_accepted_as_legacy_default(self):
        self.config.pop('pos_weight')
        self.save('config.json', self.config)
        self.expected['pos_weight'] = 3
        with self.assertRaisesRegex(ValueError, 'pos_weight'):
            self.inspect()

    def test_partial_run_is_not_completed(self):
        self.write_metrics(self.metrics[:1])
        self.assertIsNone(self.inspect())

    def test_corrupt_checkpoint_stops(self):
        (self.run / 'best.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.inspect()

    def test_stale_validation_stops(self):
        self.report['checkpoint_epoch'] = 1
        self.save('reload_check.json', self.report)
        with self.assertRaisesRegex(ValueError, 'validation report'):
            self.inspect()

    def test_missing_validation_requires_validation(self):
        (self.run / 'reload_check.json').unlink()
        self.assertFalse(self.inspect()['verified'])

    def test_summary_keeps_single_seed_uncertainty_blank(self):
        row = self.inspect()
        grid.summarize(self.root, [row])
        with (self.root / 'group_summary.csv').open() as stream:
            groups = list(csv.DictReader(stream))
        self.assertEqual(groups[0]['completed_seeds'], '1')
        self.assertEqual(groups[0]['val_auroc_std'], '')
        self.assertEqual(groups[1]['completed_seeds'], '0')

    def test_failed_child_stops_and_keeps_log(self):
        log = self.root / 'child.log'
        with self.assertRaisesRegex(RuntimeError, 'Command failed'):
            grid.execute([sys.executable, '-c', 'print("failure evidence"); raise SystemExit(2)'], log)
        self.assertIn('failure evidence', log.read_text())

    def test_queue_validation_summary_skip_and_interruption_recovery(self):
        for name in ('train.py', 'dataset.py', 'model.py', 'validate.py', 'scripts/run_bce_grid.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('source fixture')
        data = self.root / 'datasets/glaucoma_first'
        data.mkdir(parents=True)
        for split in ('train', 'valid'):
            (data / f'{split}.csv').write_text('csv fixture')
        pretrained = self.root / 'pretrain_model/efficientnet_b0_rwightman-7f5810bc.pth'
        pretrained.parent.mkdir()
        pretrained.write_bytes(b'pretrained fixture')
        batch = self.root / 'outputs/bce_grid_1_3_5_v1'
        calls = []

        def execute(command, log_path):
            calls.append(command[2])
            if command[2] == 'train.py':
                run = Path(command[command.index('--output-dir') + 1])
                run.mkdir()
                plan = json.loads((batch / 'plan.json').read_text())
                config = dict(plan['common'], seed=int(command[command.index('--seed') + 1]),
                              pos_weight=float(command[command.index('--pos-weight') + 1]),
                              source_sha256=plan['source_sha256'])
                (run / 'config.json').write_text(json.dumps(config))
                rows = [dict(self.metrics[0], epoch=i) for i in range(1, 6)]
                (run / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                (run / 'best.pt').write_bytes(b'new checkpoint fixture')
                (run / 'best.sha256').write_text(grid.sha256(run / 'best.pt'))
                for name in ('best_valid_predictions.csv', 'train_manifest.csv', 'valid_manifest.csv'):
                    (run / name).write_text('fixture')
            else:
                run = Path(command[-1])
                report = dict(self.report, checkpoint_epoch=1, val_auroc=.81,
                              checkpoint_sha256=grid.sha256(run / 'best.pt'), images=5849)
                (run / 'reload_check.json').write_text(json.dumps(report))

        previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, previous_cwd)
        with patch.object(grid, 'ROOT', self.root), patch.object(grid, 'LEGACY', {}), \
                patch.object(grid, 'execute', execute), patch.object(sys, 'argv', ['grid']):
            grid.main()
            self.assertEqual(calls.count('train.py'), 9)
            self.assertEqual(calls.count('validate.py'), 9)
            self.assertEqual(json.loads((batch / 'status.json').read_text())['status'], 'COMPLETE')
            grid.main()
            self.assertEqual(len(calls), 18, 'Completed runs should not train or validate again')
            partial = batch / 'w1_seed43'
            (partial / 'metrics.jsonl').write_text(json.dumps(self.metrics[0]) + '\n')
            grid.main()
            self.assertEqual(len(calls), 20)
            self.assertEqual(len(list(batch.glob('w1_seed43.interrupted_*'))), 1)
            with (batch / 'group_summary.csv').open() as stream:
                self.assertTrue(all(r['completed_seeds'] == '3' for r in csv.DictReader(stream)))


if __name__ == '__main__':
    unittest.main()
