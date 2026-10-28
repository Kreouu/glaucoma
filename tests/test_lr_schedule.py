import unittest

from lr_schedule import warmup_cosine_lr


class WarmupCosineTests(unittest.TestCase):
    def test_endpoints_continuity_and_monotonic_segments(self):
        rates = [warmup_cosine_lr(i, 20, 4, 3e-5, 3e-7) for i in range(20)]
        self.assertAlmostEqual(rates[0], 3e-7)
        self.assertAlmostEqual(rates[3], 3e-5)
        self.assertAlmostEqual(rates[4], 3e-5)
        self.assertAlmostEqual(rates[-1], 3e-7)
        self.assertTrue(all(a < b for a, b in zip(rates[:3], rates[1:4])))
        self.assertTrue(all(a > b for a, b in zip(rates[4:-1], rates[5:])))
        self.assertTrue(all(3e-7 <= x <= 3e-5 for x in rates))

    def test_epoch_boundary_does_not_restart(self):
        # Five epochs, four batches each. The second epoch starts at peak,
        # and later epoch boundaries continue decaying rather than warming again.
        rates = [warmup_cosine_lr(i, 20, 4, 3e-5, 3e-7) for i in range(20)]
        self.assertGreater(rates[4], rates[8])
        self.assertGreater(rates[8], rates[12])
        self.assertGreater(rates[12], rates[16])

    def test_invalid_geometry_is_rejected(self):
        for step, total, warm in [(-1, 20, 4), (20, 20, 4), (0, 20, 1), (0, 20, 19)]:
            with self.assertRaises(ValueError):
                warmup_cosine_lr(step, total, warm, 3e-5, 3e-7)


if __name__ == '__main__':
    unittest.main()
