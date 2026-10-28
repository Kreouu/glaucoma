import unittest

from scripts.run_model_selection import extend_training, group_summary, rank, numerical_failure_reason


class ModelSelectionTests(unittest.TestCase):
    def test_expected_numeric_failure_is_recordable_but_other_errors_stop(self):
        self.assertEqual(numerical_failure_reason('RuntimeError: Non-finite validation output'),
                         'Non-finite validation output')
        self.assertIsNone(numerical_failure_reason('RuntimeError: CUDA out of memory'))
        self.assertIsNone(numerical_failure_reason('FileNotFoundError: missing image'))
    def test_late_peak_extension_only_for_competitive_unfinished_curve(self):
        self.assertTrue(extend_training(dict(best_epoch=5, val_auroc=.981), .983))
        self.assertFalse(extend_training(dict(best_epoch=4, val_auroc=.981), .983))
        self.assertFalse(extend_training(dict(best_epoch=5, val_auroc=.60), .983))

    def test_group_uses_all_seeds_not_highest_single_run(self):
        rows = [dict(val_auroc=v, val_auprc=.7) for v in (.97, .98, .99)]
        result = group_summary(rows)
        self.assertAlmostEqual(result['val_auroc_mean'], .98)
        self.assertAlmostEqual(result['val_auroc_std'], .01)

    def test_auroc_has_priority_over_ap(self):
        rows = [dict(val_auroc=.98, val_auprc=.8), dict(val_auroc=.981, val_auprc=.7)]
        self.assertEqual(max(rows, key=rank), rows[1])


if __name__ == '__main__':
    unittest.main()
