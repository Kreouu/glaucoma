import unittest

from cost import consistency


class ConsistencyTests(unittest.TestCase):
    def test_small_errors_can_still_reverse_ranking(self):
        changed, error = consistency([0.5001, 0.5002], [0.5002, 0.5001], [False, False])
        self.assertLess(float(error.max()), 1e-3)
        self.assertEqual(changed.tolist(), [True, True])

    def test_breaking_a_tie_changes_ranking(self):
        changed, _ = consistency([0.2, 0.2], [0.2, 0.20001], [False, False])
        self.assertEqual(changed.tolist(), [True, True])

    def test_small_order_preserving_error_is_accepted(self):
        changed, _ = consistency([0.2, 0.8], [0.2001, 0.8001], [False, False])
        self.assertFalse(changed.any())

    def test_label_change_or_large_error_is_rejected(self):
        changed, _ = consistency([0.2, 0.8], [0.2, 0.802], [True, False])
        self.assertEqual(changed.tolist(), [True, True])


if __name__ == '__main__':
    unittest.main()
