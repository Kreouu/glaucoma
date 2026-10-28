import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_resolution_grid import summarize


class ResolutionSummaryTests(unittest.TestCase):
    def test_pairs_match_seeds_not_input_order(self):
        rows = [dict(size=384, seed=43, val_auroc=.82, val_auprc=.72, training_seconds=20),
                dict(size=256, seed=42, val_auroc=.70, val_auprc=.60, training_seconds=10),
                dict(size=256, seed=43, val_auroc=.80, val_auprc=.70, training_seconds=10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summarize(root, rows)
            with (root / 'paired_deltas.csv').open() as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]['seed'], '43')
            self.assertAlmostEqual(float(pairs[0]['auroc_384_minus_256']), .02)
            progress = json.loads((root / 'progress.json').read_text())
            self.assertEqual(progress['completed'], 3)
            self.assertIsNone(progress['groups'][1]['val_auroc_std'])


if __name__ == '__main__':
    unittest.main()
